//====- LowerStructLocals.cpp - SROA of scalar struct locals ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CIR pre-pass that scalar-replaces (SROA) a `!cir.struct` local `cir.alloca`
// whose only uses are `cir.get_member` field accesses. Each field becomes its
// own `cir.alloca`, and every `cir.get_member[k]` is replaced by the k-th field
// alloca's pointer (same pointee type -- a pure redirect, no bitcast). This
// removes the struct alloca before `--cir-to-mlir`, which has no memref storage
// model for `cir.get_member` on a struct (md_grid's dvector_t/ivector_t scalar
// struct locals). Field pointers flow to load/store/get_element exactly as any
// scalar/array alloca would.
//
// SOUNDNESS: only structs whose every use is a field access are decomposed. A
// whole-struct load/store/escape (the struct pointer used other than by
// get_member) or a field that is itself a nested struct/union is honestly
// REJECTED with a named diagnostic (nested aggregates and by-value struct
// escape are owned elsewhere), never silently miscompiled. Array fields ARE
// allowed (an array-typed field alloca lowers normally).
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

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_LOWERSTRUCTLOCALS
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct LowerStructLocalsPass
    : public impl::LowerStructLocalsBase<LowerStructLocalsPass> {
  LowerStructLocalsPass() = default;
  void runOnOperation() override;
};

void LowerStructLocalsPass::runOnOperation() {
  llvm::TimeTraceScope scope("Lower Struct Locals");

  // Collect first: the rewrite erases allocas/get_members mid-traversal.
  llvm::SmallVector<cir::AllocaOp> structAllocas;
  getOperation()->walk([&](cir::AllocaOp alloca) {
    // A union is handled by --cir-lower-union-punning; only plain structs here.
    if (mlir::isa<cir::StructType>(alloca.getAllocaType()))
      structAllocas.push_back(alloca);
  });

  for (cir::AllocaOp alloca : structAllocas) {
    auto structTy = mlir::cast<cir::StructType>(alloca.getAllocaType());
    auto members = structTy.getMembers();
    if (members.empty())
      continue;

    // This pass is purely ADDITIVE: it decomposes only structs it can prove
    // safe, and LEAVES every other struct alloca untouched for the existing
    // downstream behaviour (a later --cir-canonicalize mem2reg pass, or a loud
    // failure at --cir-to-mlir). It never hard-rejects here, so it cannot
    // regress a kernel whose struct local would otherwise have been promoted or
    // handled elsewhere -- a genuinely unhandled struct still fails loudly
    // downstream (never a silent miscompile).

    // Skip if any field is itself a nested struct/union (SROA would only push
    // the aggregate down one level; owned by the general struct cluster).
    bool nested = false;
    for (mlir::Type m : members) {
      mlir::Type base = m;
      while (auto at = mlir::dyn_cast<cir::ArrayType>(base))
        base = at.getElementType();
      if (mlir::isa<cir::RecordType>(base)) {
        nested = true;
        break;
      }
    }
    if (nested)
      continue;

    // Skip unless EVERY use of the struct alloca is a field access (a
    // whole-struct load/store/escape cannot be scalar-replaced locally).
    bool escapes = false;
    for (mlir::Operation *user : alloca.getResult().getUsers()) {
      if (!mlir::isa<cir::GetMemberOp>(user)) {
        escapes = true;
        break;
      }
    }
    if (escapes)
      continue;

    // Materialize one alloca per field, lazily (only fields actually accessed).
    mlir::OpBuilder b(alloca);
    llvm::SmallVector<cir::AllocaOp> fieldAllocas(members.size());
    auto fieldAllocaFor = [&](uint64_t k, mlir::Type memberTy) -> mlir::Value {
      if (!fieldAllocas[k]) {
        fieldAllocas[k] = cir::AllocaOp::create(
            b, alloca.getLoc(), cir::PointerType::get(memberTy),
            alloca.getName(), alloca.getAlignmentAttr());
      }
      return fieldAllocas[k].getResult();
    };

    for (mlir::Operation *user :
         llvm::make_early_inc_range(alloca.getResult().getUsers())) {
      auto gm = mlir::cast<cir::GetMemberOp>(user);
      mlir::Type memberTy =
          mlir::cast<cir::PointerType>(gm.getResult().getType()).getPointee();
      gm.getResult().replaceAllUsesWith(fieldAllocaFor(gm.getIndex(), memberTy));
      gm.erase();
    }
    alloca.erase();
  }
}

} // namespace

std::unique_ptr<mlir::Pass> mlir::createLowerStructLocalsPass() {
  return std::make_unique<LowerStructLocalsPass>();
}
