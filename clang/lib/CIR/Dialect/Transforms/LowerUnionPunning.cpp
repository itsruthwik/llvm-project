//====- LowerUnionPunning.cpp - Scalarize same-width union punning -*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CIR pre-pass that scalarizes same-bit-width union allocas so the downstream
// ThroughMLIR memref lowering never sees a `cir.get_member` on a `!cir.union`
// (a union has no memref storage model). This is the softfloat u64<->double
// type-punning idiom (`ullong_to_double`, `double_to_ullong`): a union alloca
// whose members are all scalar of the SAME bit width is a single storage cell
// whose bytes are reinterpreted between member types.
//
// Transform: replace the union alloca with a single scalar alloca of the
// union's first member (the shared storage). Rewrite each `cir.get_member[k]`
// access into a load/store on that storage, inserting a scalar
// `cir.cast(bitcast)` at the use site when the accessed member type differs
// from the storage type. Because all members are the same width there is no
// padding and the reinterpretation is exact; `--cir-to-mlir` lowers the scalar
// bitcast to `arith.bitcast`.
//
// SOUNDNESS: only same-bit-width scalar-member unions are handled. Anything
// else -- a mixed-width union, a non-scalar member, an escaping whole-union
// pointer, or a member address used by anything other than a direct load/store
// -- is honestly REJECTED with a named diagnostic (a hard error, never a silent
// miscompile).
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/TimeProfiler.h"
#include <memory>
#include <optional>

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_LOWERUNIONPUNNING
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

// Bit width of a scalar CIR type (integer or float). std::nullopt for anything
// that is not a same-storage scalar (pointers, arrays, nested records, ...).
static std::optional<unsigned> scalarBitWidth(mlir::Type t) {
  if (auto i = mlir::dyn_cast<cir::IntType>(t))
    return i.getWidth();
  if (auto f = mlir::dyn_cast<cir::FPTypeInterface>(t))
    return f.getWidth();
  return std::nullopt;
}

struct LowerUnionPunningPass
    : public impl::LowerUnionPunningBase<LowerUnionPunningPass> {
  LowerUnionPunningPass() = default;
  void runOnOperation() override;
};

void LowerUnionPunningPass::runOnOperation() {
  llvm::TimeTraceScope scope("Lower Union Punning");

  // Collect first: the rewrite erases ops (allocas, get_members, loads/stores),
  // which would invalidate an in-flight walk iterator.
  llvm::SmallVector<cir::AllocaOp> unionAllocas;
  getOperation()->walk([&](cir::AllocaOp alloca) {
    if (mlir::isa<cir::UnionType>(alloca.getAllocaType()))
      unionAllocas.push_back(alloca);
  });

  for (cir::AllocaOp alloca : unionAllocas) {
    auto unionTy = mlir::cast<cir::UnionType>(alloca.getAllocaType());

    auto members = unionTy.getMembers();
    if (members.empty())
      continue; // empty union: nothing to access, leave for downstream

    // Same-bit-width scalar-member gate.
    std::optional<unsigned> width0 = scalarBitWidth(members.front());
    if (!width0) {
      alloca.emitError("ThroughMLIR: union '")
          << unionTy << "' has a non-scalar member; only same-bit-width scalar "
             "(integer/float) unions can be scalarized for hardware synthesis";
      signalPassFailure();
      return;
    }
    bool badMember = false;
    for (mlir::Type m : members.drop_front()) {
      std::optional<unsigned> w = scalarBitWidth(m);
      if (!w || *w != *width0) {
        alloca.emitError("ThroughMLIR: union '")
            << unionTy
            << "' has members of differing bit width (or a non-scalar member); "
               "only same-bit-width scalar unions can be scalarized soundly "
               "(a mixed-width reinterpretation is not representable)";
        badMember = true;
        break;
      }
    }
    if (badMember) {
      signalPassFailure();
      return;
    }

    // Every use of the union alloca must be a `cir.get_member` (no whole-union
    // load/store/escape).
    bool escapes = false;
    for (mlir::Operation *user : alloca.getResult().getUsers()) {
      if (!mlir::isa<cir::GetMemberOp>(user)) {
        alloca.emitError("ThroughMLIR: union alloca '")
            << alloca.getName()
            << "' is used other than by member access (whole-union "
               "load/store/escape); not supported for hardware synthesis";
        escapes = true;
        break;
      }
    }
    if (escapes) {
      signalPassFailure();
      return;
    }

    mlir::Type storageTy = members.front();
    mlir::OpBuilder b(alloca);
    auto newAlloca = cir::AllocaOp::create(
        b, alloca.getLoc(), cir::PointerType::get(storageTy), alloca.getName(),
        alloca.getAlignmentAttr());
    mlir::Value storage = newAlloca.getResult();

    // Rewrite each member access.
    for (mlir::Operation *user :
         llvm::make_early_inc_range(alloca.getResult().getUsers())) {
      auto gm = mlir::cast<cir::GetMemberOp>(user);
      mlir::Type memberTy =
          mlir::cast<cir::PointerType>(gm.getResult().getType()).getPointee();

      for (mlir::Operation *guser :
           llvm::make_early_inc_range(gm.getResult().getUsers())) {
        if (auto ld = mlir::dyn_cast<cir::LoadOp>(guser)) {
          mlir::OpBuilder lb(ld);
          mlir::Value raw =
              cir::LoadOp::create(lb, ld.getLoc(), storageTy, storage);
          mlir::Value val = raw;
          if (memberTy != storageTy)
            val = cir::CastOp::create(lb, ld.getLoc(), memberTy,
                                      cir::CastKind::bitcast, raw);
          ld.getResult().replaceAllUsesWith(val);
          ld.erase();
        } else if (auto st = mlir::dyn_cast<cir::StoreOp>(guser)) {
          // The member pointer must be the store ADDRESS, not a stored value.
          if (st.getAddr() != gm.getResult()) {
            gm.emitError("ThroughMLIR: address of union member '")
                << gm.getName()
                << "' escapes (stored as a value); not supported";
            signalPassFailure();
            return;
          }
          mlir::OpBuilder sb(st);
          mlir::Value v = st.getValue();
          if (memberTy != storageTy)
            v = cir::CastOp::create(sb, st.getLoc(), storageTy,
                                    cir::CastKind::bitcast, v);
          cir::StoreOp::create(sb, st.getLoc(), v, storage);
          st.erase();
        } else {
          gm.emitError("ThroughMLIR: address of union member '")
              << gm.getName()
              << "' escapes (used other than by a direct load/store); not "
                 "supported for hardware synthesis";
          signalPassFailure();
          return;
        }
      }
      gm.erase();
    }
    alloca.erase();
  }
}

} // namespace

std::unique_ptr<mlir::Pass> mlir::createLowerUnionPunningPass() {
  return std::make_unique<LowerUnionPunningPass>();
}
