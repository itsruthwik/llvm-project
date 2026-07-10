//===- LowerPointerCursors.cpp - (base, index) pointer-cursor lowering ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CIR pre-pass implementing the unified pointer-provenance model: a C pointer
// value is (base object, integer element index).
//
// Two additive rewrite stages, both firing only on shapes it can PROVE:
//
// (A) Pointer-cursor SLOTS. A `cir.alloca` of `!cir.ptr<T>` surviving mem2reg
//     (a fat-pointer slot) is rewritten into an INDEX slot when
//       - every use of the slot is a direct `cir.load` or a `cir.store`
//         addressing the slot (the slot's own address never escapes), and
//       - every value stored into the slot reduces to a single structurally
//         identical root (direct object: `cir.alloca` array / stable
//         `cir.get_global` / incoming pointer argument / `cir.get_member`
//         field pointer) plus an integer element offset expression, traced
//         through `array_to_ptrdecay` / `get_element` / `ptr_stride` and
//         through loads of other rewritable cursor slots.
//     Each load of the slot is replaced by a materialized pointer
//     `get_element(root, idx)` (or `ptr_stride(root, idx)` for a non-array
//     root such as a decayed pointer argument), so EVERY existing consumer
//     (deref load/store, chained get_element, call argument, pointer compare)
//     stays valid by construction. Each store becomes an index store; signed
//     index arithmetic supports negative strides.
//
// (B) SSA `cir.ptr_stride` CHAINS (`*p++` sequences mem2reg fully promoted):
//     a ptr_stride whose base or user is another ptr_stride would otherwise
//     take ThroughMLIR's untranslatable ptr-dialect fallback. Every chain
//     member whose provenance reduces to (root, offset expr) is collapsed to
//     a single `get_element(root, idx)` / root-based `ptr_stride`; dead chain
//     tails are erased.
//
// Same-root pointer compares whose operands were materialized by this pass
// are rewritten to integer index compares. A compare across PROVABLY distinct
// roots is left as a pointer compare: rewriting the slots makes the operand
// provenance visible to `--cir-to-mlir`'s cross-allocation named reject,
// which owns that OUT row (no double-reject here).
//
// SOUNDNESS / SCOPE (v1, per the ptr-cursor-provenance plan):
//   - whole-slot-or-nothing: any unprovable access leaves the slot untouched
//     for the existing downstream behaviour (loud failure, never silent);
//   - a slot whose stored values PROVABLY merge different base objects is
//     honestly rejected by name (different-base merge is OUT in v1);
//   - provenance is never traced through a partial N-D row `get_element`
//     (unit-changing re-bases) or element-size-changing casts: those stay
//     unresolvable and are owned by the downstream named rejects.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/TimeProfiler.h"
#include <memory>

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_LOWERPOINTERCURSORS
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// Provenance representation
//===----------------------------------------------------------------------===//

/// A root is a direct object producer: the SSA value of a `cir.alloca`
/// result, an incoming pointer block argument, a `cir.get_member` field
/// pointer, or a global identified by symbol name (so two `cir.get_global`s
/// of the same symbol are structurally identical).
struct RootKey {
  mlir::Value val;           // alloca result / block arg / get_member result
  mlir::StringAttr sym;      // global symbol (exclusive with val)
  mlir::Type globalResTy;    // result pointer type of the get_global

  bool valid() const { return val || sym; }
  bool operator==(const RootKey &o) const {
    if (sym || o.sym)
      return sym == o.sym;
    return val == o.val;
  }
  bool operator!=(const RootKey &o) const { return !(*this == o); }
};

/// One additive term of an element-offset expression.
struct Term {
  enum Kind { Const, SSA, SlotLoad } kind;
  int64_t cst = 0;          // Const
  mlir::Value ssa;          // SSA: an integer value from the original IR
  cir::AllocaOp slot;       // SlotLoad (analysis only): load of another slot
};

struct Prov {
  RootKey root;             // may be invalid while a SlotLoad term is pending
  llvm::SmallVector<Term, 4> terms;
  bool viaSlot = false;     // some term is a SlotLoad
};

/// Per-slot analysis state.
struct SlotInfo {
  cir::AllocaOp alloca;
  mlir::Type pointee;                       // T of the !cir.ptr<T> slot value
  llvm::SmallVector<cir::LoadOp, 4> loads;
  llvm::SmallVector<cir::StoreOp, 4> stores;
  llvm::SmallVector<Prov, 4> storeProvs;    // parallel to stores
  RootKey root;                             // resolved root (fixpoint)
  bool multiRoot = false;                   // stores name different roots
  // For a multi-root slot resolved as SEQUENTIAL REBINDING (straight-line
  // re-uses of one variable: `p = A; ...; p = B; ...`), the per-load root.
  llvm::DenseMap<mlir::Operation *, RootKey> accessRoot;
  cir::AllocaOp idxSlot;                    // created during rewrite
};

struct LowerPointerCursorsPass
    : public impl::LowerPointerCursorsBase<LowerPointerCursorsPass> {
  LowerPointerCursorsPass() = default;
  void runOnOperation() override;

private:
  void processFunction(cir::FuncOp fn);
  bool failed = false;

  /// Reduce a pointer SSA value of type !cir.ptr<pointee> to provenance.
  /// `candidates` maps slot-alloca results to their SlotInfo; a load of a
  /// candidate slot becomes a SlotLoad term (analysis) — during the rewrite
  /// phase candidate loads have already been replaced, so reduce only sees
  /// concrete address chains.
  std::optional<Prov>
  reduce(mlir::Value v, mlir::Type pointee,
         const llvm::DenseMap<mlir::Value, SlotInfo *> &candidates) const;

  bool intTypesCompatible(mlir::Type t) const {
    return mlir::isa<cir::IntType>(t);
  }
};

/// Return the integer constant of a `cir.const` int, if any.
static std::optional<int64_t> constIntValue(mlir::Value v) {
  if (auto c = v.getDefiningOp<cir::ConstantOp>())
    if (auto ia = mlir::dyn_cast<cir::IntAttr>(c.getValue()))
      return ia.getSInt();
  return std::nullopt;
}

std::optional<Prov> LowerPointerCursorsPass::reduce(
    mlir::Value v, mlir::Type pointee,
    const llvm::DenseMap<mlir::Value, SlotInfo *> &candidates) const {
  Prov p;
  while (true) {
    // Unit-consistency invariant: v must remain a pointer to `pointee`.
    auto pt = mlir::dyn_cast<cir::PointerType>(v.getType());
    if (!pt || pt.getPointee() != pointee)
      return std::nullopt;

    // Incoming pointer argument: a root.
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
      // Only a function entry argument is a stable root.
      if (!mlir::isa<cir::FuncOp>(arg.getOwner()->getParentOp()) ||
          !arg.getOwner()->isEntryBlock())
        return std::nullopt;
      p.root = RootKey{v, {}, {}};
      return p;
    }

    mlir::Operation *def = v.getDefiningOp();
    if (!def)
      return std::nullopt;

    if (auto stride = mlir::dyn_cast<cir::PtrStrideOp>(def)) {
      mlir::Value s = stride.getStride();
      if (!intTypesCompatible(s.getType()))
        return std::nullopt;
      if (auto c = constIntValue(s))
        p.terms.push_back(Term{Term::Const, *c, {}, {}});
      else
        p.terms.push_back(Term{Term::SSA, 0, s, {}});
      v = stride.getBase();
      continue;
    }

    if (auto ge = mlir::dyn_cast<cir::GetElementOp>(def)) {
      // ge result type == ptr<pointee> (checked above), so its index counts
      // `pointee` elements: an additive term. The BASE must then be a direct
      // root object (never a partial N-D row chain — unit-changing re-bases
      // stay unresolvable and are owned by the downstream named rejects).
      mlir::Value idx = ge.getIndex();
      if (!intTypesCompatible(idx.getType()))
        return std::nullopt;
      if (auto c = constIntValue(idx))
        p.terms.push_back(Term{Term::Const, *c, {}, {}});
      else
        p.terms.push_back(Term{Term::SSA, 0, idx, {}});
      mlir::Value base = ge.getBase();
      if (auto gg = base.getDefiningOp<cir::GetGlobalOp>()) {
        p.root = RootKey{{}, gg.getNameAttr().getAttr(), gg.getType()};
        return p;
      }
      if (base.getDefiningOp<cir::AllocaOp>() ||
          base.getDefiningOp<cir::GetMemberOp>()) {
        p.root = RootKey{base, {}, {}};
        return p;
      }
      return std::nullopt;
    }

    if (auto cast = mlir::dyn_cast<cir::CastOp>(def)) {
      if (cast.getKind() != cir::CastKind::array_to_ptrdecay)
        return std::nullopt; // element-size-changing / unknown cast: leave.
      mlir::Value src = cast.getSrc();
      auto srcPt = mlir::dyn_cast<cir::PointerType>(src.getType());
      if (!srcPt)
        return std::nullopt;
      auto arrTy = mlir::dyn_cast<cir::ArrayType>(srcPt.getPointee());
      if (!arrTy || arrTy.getElementType() != pointee)
        return std::nullopt;
      // The decayed array must itself be a direct root object.
      if (auto gg = src.getDefiningOp<cir::GetGlobalOp>()) {
        p.root = RootKey{{}, gg.getNameAttr().getAttr(), gg.getType()};
        return p;
      }
      if (src.getDefiningOp<cir::AllocaOp>() ||
          src.getDefiningOp<cir::GetMemberOp>() ||
          mlir::isa<mlir::BlockArgument>(src)) {
        p.root = RootKey{src, {}, {}};
        return p;
      }
      return std::nullopt;
    }

    if (auto load = mlir::dyn_cast<cir::LoadOp>(def)) {
      // Analysis-time only: a load of another candidate cursor slot.
      auto it = candidates.find(load.getAddr());
      if (it == candidates.end())
        return std::nullopt;
      p.terms.push_back(Term{Term::SlotLoad, 0, v, it->second->alloca});
      p.viaSlot = true;
      // Root is that slot's root, resolved by the caller's fixpoint.
      return p;
    }

    return std::nullopt;
  }
}

//===----------------------------------------------------------------------===//
// Rewrite helpers
//===----------------------------------------------------------------------===//

/// Materialize the root as a pointer SSA value at the builder's insertion
/// point (fresh get_global per use; direct value otherwise).
static mlir::Value materializeRoot(OpBuilder &b, mlir::Location loc,
                                   const RootKey &root) {
  if (root.sym)
    return cir::GetGlobalOp::create(b, loc, root.globalResTy,
                                    mlir::FlatSymbolRefAttr::get(root.sym));
  return root.val;
}

/// Materialize `root + idx` as a typed element pointer.
static mlir::Value materializePtr(OpBuilder &b, mlir::Location loc,
                                  const RootKey &root, mlir::Value idx,
                                  mlir::Type pointee) {
  mlir::Value rootV = materializeRoot(b, loc, root);
  auto rootPt = mlir::cast<cir::PointerType>(rootV.getType());
  auto resTy = cir::PointerType::get(pointee, rootPt.getAddrSpace());
  if (auto arrTy = mlir::dyn_cast<cir::ArrayType>(rootPt.getPointee())) {
    if (arrTy.getElementType() == pointee)
      return cir::GetElementOp::create(b, loc, resTy, rootV, idx);
  }
  // Non-array root (decayed pointer argument): a single root-based
  // ptr_stride, which ThroughMLIR lowers as a direct offset access.
  return cir::PtrStrideOp::create(b, loc, rootV.getType(), rootV, idx);
}

} // namespace

void LowerPointerCursorsPass::runOnOperation() {
  llvm::TimeTraceScope scope("Lower Pointer Cursors");
  llvm::SmallVector<cir::FuncOp> fns;
  getOperation()->walk([&](cir::FuncOp fn) { fns.push_back(fn); });
  for (cir::FuncOp fn : fns) {
    processFunction(fn);
    if (failed) {
      signalPassFailure();
      return;
    }
  }
}

void LowerPointerCursorsPass::processFunction(cir::FuncOp fn) {
  if (fn.getBody().empty())
    return; // declaration: nothing to do.
  mlir::MLIRContext *ctx = fn.getContext();

  //===------------------------------------------------------------------===//
  // Stage A analysis: collect candidate fat-pointer slots.
  //===------------------------------------------------------------------===//
  llvm::SmallVector<std::unique_ptr<SlotInfo>> slots;
  llvm::DenseMap<mlir::Value, SlotInfo *> candidates;

  fn->walk([&](cir::AllocaOp alloca) {
    auto ptrTy = mlir::dyn_cast<cir::PointerType>(alloca.getAllocaType());
    if (!ptrTy)
      return;
    auto info = std::make_unique<SlotInfo>();
    info->alloca = alloca;
    info->pointee = ptrTy.getPointee();
    // Slot legality: every use is a direct load, or a store ADDRESSING the
    // slot. Anything else (slot address stored as a value, passed to a call,
    // cast, compared, ...) means the access set is not enumerable: leave the
    // slot untouched for the existing downstream behaviour.
    for (mlir::Operation *user : alloca.getResult().getUsers()) {
      if (auto load = mlir::dyn_cast<cir::LoadOp>(user)) {
        info->loads.push_back(load);
        continue;
      }
      if (auto store = mlir::dyn_cast<cir::StoreOp>(user)) {
        if (store.getAddr() == alloca.getResult() &&
            store.getValue() != alloca.getResult()) {
          info->stores.push_back(store);
          continue;
        }
      }
      return; // address escape / unknown access: not a candidate.
    }
    if (info->stores.empty())
      return; // never written: nothing to model.
    // v1 scope: only CURSOR-LIKE slots are rewritten — the slot is reassigned
    // (>= 2 stores: increments, resets, rebinds) or its value participates in
    // a pointer compare (loop bounds like `end`). A trivial single-store slot
    // whose loads only feed dereferences (e.g. the argument-copy slot of every
    // pointer parameter) is handled by the existing lowering today and is
    // deliberately LEFT UNTOUCHED, keeping this pass's footprint additive.
    bool cursorLike = info->stores.size() >= 2;
    if (!cursorLike)
      for (cir::LoadOp l : info->loads) {
        for (mlir::Operation *u : l.getResult().getUsers())
          if (mlir::isa<cir::CmpOp>(u)) {
            cursorLike = true;
            break;
          }
        if (cursorLike)
          break;
      }
    if (!cursorLike)
      return;
    candidates[alloca.getResult()] = info.get();
    slots.push_back(std::move(info));
  });

  // Reduce every stored value; iteratively demote slots whose provenance
  // cannot be proven (whole-slot-or-nothing), re-checking slots that depended
  // on a demoted slot through a load-of-slot term.
  auto demote = [&](SlotInfo *s) { candidates.erase(s->alloca.getResult()); };
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &sp : slots) {
      SlotInfo *s = sp.get();
      if (!candidates.count(s->alloca.getResult()))
        continue;
      s->storeProvs.clear();
      s->root = RootKey{};
      s->multiRoot = false;
      bool ok = true;
      for (cir::StoreOp st : s->stores) {
        auto prov = reduce(st.getValue(), s->pointee, candidates);
        if (!prov) {
          ok = false;
          break;
        }
        s->storeProvs.push_back(*prov);
      }
      if (!ok) {
        demote(s);
        changed = true;
      }
    }
  }

  // Root fixpoint: resolve roots across slot-load references.
  changed = true;
  while (changed) {
    changed = false;
    for (auto &sp : slots) {
      SlotInfo *s = sp.get();
      if (!candidates.count(s->alloca.getResult()) || s->root.valid())
        continue;
      RootKey resolved;
      bool pending = false, brokenRef = false;
      for (Prov &prov : s->storeProvs) {
        RootKey r = prov.root;
        if (!r.valid()) {
          // Root comes through the referenced slot's own root. A store of
          // the slot's OWN current value (`p = p + k`) imposes no root
          // constraint; a reference to a demoted slot breaks provability.
          bool selfRef = false;
          for (Term &t : prov.terms)
            if (t.kind == Term::SlotLoad) {
              if (t.slot == s->alloca) {
                selfRef = true;
                break;
              }
              auto it = candidates.find(t.slot.getResult());
              if (it == candidates.end()) {
                brokenRef = true; // referenced slot demoted: leave this one
                break;
              }
              r = it->second->root;
            }
          if (brokenRef)
            break;
          if (selfRef)
            continue; // no constraint from a self-relative update
          if (!r.valid()) {
            pending = true; // referenced slot not resolved yet this round
            continue;
          }
        }
        if (!resolved.valid())
          resolved = r;
        else if (resolved != r) {
          s->multiRoot = true; // provable different-base merge
          break;
        }
      }
      if (s->multiRoot)
        continue;
      if (brokenRef) {
        demote(s);
        changed = true;
        continue;
      }
      if (!pending && resolved.valid()) {
        s->root = resolved;
        changed = true;
      }
    }
  }

  mlir::DominanceInfo dom(fn);

  // Resolve a multi-root slot as SEQUENTIAL REBINDING: one variable reused
  // for several bases in straight-line code (`p = key_P; loop; p = key_S;
  // loop;`). Sound criterion: every concrete-root ("rebind") store sits
  // directly in the function ENTRY block (which executes exactly once, so op
  // order gives exact reaching definitions), every other store is purely
  // self-relative (`p = p + k`), and every load / self-relative store has a
  // nearest preceding rebind store in the entry block — that store's root is
  // the unique root the access can observe. A store inside a nested region
  // (the `if (c) p = a; else p = b;` control-flow MERGE) fails this
  // resolution and stays a named reject (OUT in v1).
  mlir::Block *entry = &fn.getBody().front();
  auto resolveSeqRebinding = [&](SlotInfo *s) -> bool {
    llvm::SmallVector<std::pair<mlir::Operation *, RootKey>, 4> rebinds;
    llvm::SmallVector<mlir::Operation *, 4> selfStores;
    for (auto [st, prov] : llvm::zip(s->stores, s->storeProvs)) {
      if (prov.root.valid()) {
        if (st->getBlock() != entry)
          return false;
        // The root value itself must be defined before the rebind store.
        if (prov.root.val && !dom.properlyDominates(prov.root.val, st))
          return false;
        rebinds.push_back({st, prov.root});
      } else {
        for (Term &t : prov.terms)
          if (t.kind == Term::SlotLoad && t.slot != s->alloca)
            return false; // routed through another slot: not resolvable here
        selfStores.push_back(st);
      }
    }
    if (rebinds.empty())
      return false;
    // Hoist an access to its ancestor op in the entry block.
    auto ancestorIn = [&](mlir::Operation *op) -> mlir::Operation * {
      while (op && op->getBlock() != entry)
        op = op->getParentOp();
      return op;
    };
    auto rootBefore = [&](mlir::Operation *op) -> RootKey {
      mlir::Operation *a = ancestorIn(op);
      if (!a)
        return RootKey{};
      mlir::Operation *best = nullptr;
      RootKey r;
      for (auto &[st, rk] : rebinds)
        if (st != a && st->isBeforeInBlock(a) &&
            (!best || best->isBeforeInBlock(st))) {
          best = st;
          r = rk;
        }
      return best ? r : RootKey{};
    };
    llvm::DenseMap<mlir::Operation *, RootKey> roots;
    for (cir::LoadOp l : s->loads) {
      RootKey r = rootBefore(l);
      if (!r.valid())
        return false; // load before the first rebind: cannot resolve.
      roots[l] = r;
    }
    for (mlir::Operation *st : selfStores)
      if (!rootBefore(st).valid())
        return false; // self-update before any rebind: cannot resolve.
    s->accessRoot = std::move(roots);
    return true;
  };

  // Final validation sweep.
  for (auto &sp : slots) {
    SlotInfo *s = sp.get();
    if (!candidates.count(s->alloca.getResult()))
      continue;
    if (s->multiRoot) {
      if (!resolveSeqRebinding(s)) {
        // Every access is enumerable and every stored value reduces to a
        // concrete root, but the roots differ AND the stores are not
        // straight-line rebinds: a different-base merge, OUT / named reject
        // in v1 per the provenance plan.
        s->alloca.emitError()
            << "cir-lower-pointer-cursors: pointer variable '"
            << s->alloca.getName()
            << "' merges pointers into different base objects "
               "(different-base pointer merge is not supported; use a single "
               "base array or an explicit integer index)";
        failed = true;
        return;
      }
      continue;
    }
    if (!s->root.valid()) {
      demote(s); // cyclic slot references with no concrete root: leave.
      continue;
    }
    // The root value must dominate every rewrite point.
    if (s->root.val) {
      bool domOk = true;
      for (cir::LoadOp l : s->loads)
        domOk &= dom.properlyDominates(s->root.val, l);
      for (cir::StoreOp st : s->stores)
        domOk &= dom.properlyDominates(s->root.val, st);
      if (!domOk) {
        demote(s);
        continue;
      }
    }
  }

  // Nothing to do?
  llvm::SmallVector<SlotInfo *> rewriteSet;
  for (auto &sp : slots)
    if (candidates.count(sp->alloca.getResult()))
      rewriteSet.push_back(sp.get());

  //===------------------------------------------------------------------===//
  // Index type: the widest signed CIR integer type among all observed offset
  // contributions of the slots being rewritten (derived, not hardcoded).
  //===------------------------------------------------------------------===//
  unsigned idxWidth = 0;
  for (SlotInfo *s : rewriteSet)
    for (Prov &prov : s->storeProvs)
      for (Term &t : prov.terms) {
        if (t.kind == Term::SSA)
          idxWidth = std::max(
              idxWidth, mlir::cast<cir::IntType>(t.ssa.getType()).getWidth());
        else if (t.kind == Term::Const)
          idxWidth = std::max(idxWidth, 32u);
      }
  if (idxWidth == 0)
    idxWidth = 32;
  auto idxTy = cir::IntType::get(ctx, idxWidth, /*isSigned=*/true);

  auto toIdx = [&](OpBuilder &b, mlir::Location loc,
                   mlir::Value v) -> mlir::Value {
    if (v.getType() == idxTy)
      return v;
    return cir::CastOp::create(b, loc, idxTy, cir::CastKind::integral, v);
  };

  //===------------------------------------------------------------------===//
  // Stage A rewrite.
  //===------------------------------------------------------------------===//
  llvm::SetVector<mlir::Value> materialized; // pass-produced pointer values
  bool touched = !rewriteSet.empty();

  // 1) Index slots.
  for (SlotInfo *s : rewriteSet) {
    OpBuilder b(s->alloca);
    auto align = b.getI64IntegerAttr(idxWidth / 8);
    s->idxSlot = cir::AllocaOp::create(b, s->alloca.getLoc(),
                                       cir::PointerType::get(idxTy),
                                       s->alloca.getName(), align);
  }

  // 2) Replace every load of a rewritten slot with (index load + pointer
  //    materialization) at the load's own program point, so every consumer
  //    keeps its exact semantics.
  for (SlotInfo *s : rewriteSet) {
    for (cir::LoadOp l : s->loads) {
      OpBuilder b(l);
      mlir::Location loc = l.getLoc();
      mlir::Value idx =
          cir::LoadOp::create(b, loc, idxTy, s->idxSlot.getResult());
      const RootKey &root = s->multiRoot ? s->accessRoot[l] : s->root;
      mlir::Value ptr = materializePtr(b, loc, root, idx, s->pointee);
      materialized.insert(ptr);
      l.getResult().replaceAllUsesWith(ptr);
      l.erase();
    }
  }

  // 3) Rewrite stores: reduce the (post-substitution) stored chain and store
  //    the accumulated index instead. Empty candidate map: chains are now
  //    fully concrete.
  llvm::DenseMap<mlir::Value, SlotInfo *> noSlots;
  auto materializeIdx = [&](OpBuilder &b, mlir::Location loc,
                            const Prov &prov) -> mlir::Value {
    mlir::Value acc;
    for (const Term &t : prov.terms) {
      mlir::Value tv;
      if (t.kind == Term::Const) {
        tv = cir::ConstantOp::create(b, loc, cir::IntAttr::get(idxTy, t.cst));
      } else {
        tv = toIdx(b, loc, t.ssa);
      }
      acc = acc ? mlir::Value(cir::AddOp::create(b, loc, idxTy, acc, tv))
                : tv;
    }
    if (!acc)
      acc = cir::ConstantOp::create(b, loc, cir::IntAttr::get(idxTy, 0));
    return acc;
  };

  for (SlotInfo *s : rewriteSet) {
    for (cir::StoreOp st : s->stores) {
      auto prov = reduce(st.getValue(), s->pointee, noSlots);
      assert(prov && !prov->viaSlot &&
             "analysis proved this store reducible; post-substitution chain "
             "must reduce concretely");
      if (!prov) { // defensive in release builds: leave IR consistent
        failed = true;
        return;
      }
      OpBuilder b(st);
      mlir::Location loc = st.getLoc();
      mlir::Value idx = materializeIdx(b, loc, *prov);
      cir::StoreOp::create(b, loc, idx, s->idxSlot.getResult());
      st.erase();
    }
  }

  //===------------------------------------------------------------------===//
  // Stage B: collapse SSA ptr_stride chains (a ptr_stride whose base or user
  // is another ptr_stride, or which is dead) that reduce to a concrete root.
  // These shapes can only reach ThroughMLIR's untranslatable ptr-dialect
  // fallback today, so collapsing them is strictly additive.
  //===------------------------------------------------------------------===//
  llvm::SmallVector<cir::PtrStrideOp> chainOps;
  fn->walk([&](cir::PtrStrideOp op) {
    // Chain members, dead strides, and strides over a pointer this pass
    // materialized (e.g. `p[1]` on a rewritten cursor: ptr_stride over the
    // new get_element) are all collapsed to a single rooted access.
    bool chained = op.getBase().getDefiningOp<cir::PtrStrideOp>() != nullptr ||
                   op.use_empty() || materialized.contains(op.getBase());
    if (!chained)
      for (mlir::Operation *u : op->getUsers())
        if (mlir::isa<cir::PtrStrideOp>(u)) {
          chained = true;
          break;
        }
    if (chained)
      chainOps.push_back(op);
  });
  for (cir::PtrStrideOp op : chainOps) {
    if (op.use_empty()) {
      op.erase(); // pure address computation with no consumer.
      touched = true;
      continue;
    }
    auto ptrTy = mlir::cast<cir::PointerType>(op.getType());
    auto prov = reduce(op.getResult(), ptrTy.getPointee(), noSlots);
    if (!prov)
      continue; // unresolvable: downstream named rejects own it.
    OpBuilder b(op);
    mlir::Location loc = op.getLoc();
    // Derive the chain's index type independently (its own observed offsets).
    unsigned w = 0;
    for (Term &t : prov->terms)
      w = std::max(w, t.kind == Term::SSA
                          ? mlir::cast<cir::IntType>(t.ssa.getType()).getWidth()
                          : 32u);
    auto chainIdxTy = cir::IntType::get(ctx, w ? w : 32, /*isSigned=*/true);
    mlir::Value acc;
    for (Term &t : prov->terms) {
      mlir::Value tv;
      if (t.kind == Term::Const)
        tv = cir::ConstantOp::create(b, loc,
                                     cir::IntAttr::get(chainIdxTy, t.cst));
      else if (t.ssa.getType() == chainIdxTy)
        tv = t.ssa;
      else
        tv = cir::CastOp::create(b, loc, chainIdxTy, cir::CastKind::integral,
                                 t.ssa);
      acc = acc ? mlir::Value(cir::AddOp::create(b, loc, chainIdxTy, acc, tv))
                : tv;
    }
    if (!acc)
      acc = cir::ConstantOp::create(b, loc, cir::IntAttr::get(chainIdxTy, 0));
    mlir::Value ptr = materializePtr(b, loc, prov->root, acc,
                                     ptrTy.getPointee());
    materialized.insert(ptr);
    op.getResult().replaceAllUsesWith(ptr);
    op.erase();
    touched = true;
  }

  //===------------------------------------------------------------------===//
  // Same-root pointer compares touching pass-materialized pointers become
  // integer index compares. Cross-root compares are LEFT as pointer compares:
  // the materialized operands now expose their provenance, so the
  // cir-to-mlir cross-allocation named reject (which owns that OUT row)
  // fires on the provably-different-base relational case.
  //===------------------------------------------------------------------===//
  llvm::SmallVector<cir::CmpOp> cmps;
  fn->walk([&](cir::CmpOp cmp) {
    if (!mlir::isa<cir::PointerType>(cmp.getLhs().getType()))
      return;
    if (materialized.contains(cmp.getLhs()) ||
        materialized.contains(cmp.getRhs()))
      cmps.push_back(cmp);
  });
  for (cir::CmpOp cmp : cmps) {
    auto pointee =
        mlir::cast<cir::PointerType>(cmp.getLhs().getType()).getPointee();
    auto lp = reduce(cmp.getLhs(), pointee, noSlots);
    auto rp = reduce(cmp.getRhs(), pointee, noSlots);
    if (!lp || !rp || lp->root != rp->root)
      continue; // unresolvable or cross-root: leave for downstream.
    OpBuilder b(cmp);
    mlir::Location loc = cmp.getLoc();
    mlir::Value li = materializeIdx(b, loc, *lp);
    mlir::Value ri = materializeIdx(b, loc, *rp);
    auto newCmp = cir::CmpOp::create(b, loc, cmp.getKind(), li, ri);
    cmp.getResult().replaceAllUsesWith(newCmp.getResult());
    cmp.erase();
  }

  //===------------------------------------------------------------------===//
  // Cleanup: erase now-dead pure address computations (the old pointer
  // chains) and the retired fat-pointer slots. Runs ONLY when this pass
  // changed the function, so an untouched function stays byte-identical
  // (pre-existing dead ops are not this pass's to sweep).
  //===------------------------------------------------------------------===//
  if (!touched)
    return;
  bool erased = true;
  while (erased) {
    erased = false;
    llvm::SmallVector<mlir::Operation *> dead;
    fn->walk([&](mlir::Operation *op) {
      if (!mlir::isa<cir::PtrStrideOp, cir::GetElementOp, cir::CastOp,
                     cir::GetGlobalOp, cir::ConstantOp>(op))
        return;
      if (op->use_empty())
        dead.push_back(op);
    });
    for (mlir::Operation *op : dead) {
      op->erase();
      erased = true;
    }
  }
  for (SlotInfo *s : rewriteSet) {
    if (s->alloca.getResult().use_empty())
      s->alloca.erase();
  }
}

std::unique_ptr<mlir::Pass> mlir::createLowerPointerCursorsPass() {
  return std::make_unique<LowerPointerCursorsPass>();
}
