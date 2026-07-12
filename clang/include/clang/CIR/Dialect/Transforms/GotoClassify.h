//====- GotoClassify.h - Cross-scope goto classification --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared, header-only classifier for `cir.goto` operations. It is the single
// source of truth used by BOTH the `--cir-flatten-scope-goto` pre-pass (which
// flattens the one class it can lower soundly) and the VTR-HLS
// `--vtr-strict-c-features` gate (which must reject exactly the residual,
// unsupported classes and no more). Keeping the predicate in one inline
// function guarantees the two stay in lock-step: whatever the pass flattens,
// the gate must NOT reject, and vice-versa.
//
// Only ONE class is soundly lowerable today: the classic softfloat
// "roundAndPack" forward-exit idiom -- a `cir.goto` nested inside
// `cir.scope`/`cir.if`/loop regions that jumps FORWARD to a label sitting at
// the top of a block in the enclosing function-body region, where that label
// block is the *unconditional fall-through successor* (`cir.br`) of the block
// holding the goto's outermost ancestor. That is a single shared epilogue: on
// both the goto path and the natural fall-through path control reaches the same
// label block, so the goto can be replaced by a boolean flag + guard-climb
// (see FlattenScopeGoto.cpp), exactly like an early `return` that resumes at
// the epilogue instead of exiting the function.
//
// Everything else is rejected, by name:
//   * MultiTargetLadder -- the softfloat `subFloat64Sigs` shape: several nested
//     gotos into a chain of label blocks that do NOT all fall through from a
//     single entry block to one shared epilogue (the entry block is terminated
//     by a `cir.return`, not a `cir.br` to the target). Sound structured
//     lowering of that acyclic label ladder is possible in principle (the
//     blocks carry no cross-block SSA), but needs a dedicated structurizer and
//     is deliberately deferred rather than approximated.
//   * Backward   -- the target label block precedes the goto's entry block.
//   * IntoScope  -- the target label is not at function-body-region level
//     (it is nested inside a region that does not enclose the goto).
//   * Unsupported-- missing/duplicate label, cross-function, or a shape that
//     does not fit any of the above (defensive catch-all).
//   * NotNested  -- the goto is already a top-level terminator of a
//     function-body-region block; `--cir-goto-solver` lowers it to a same-region
//     `cir.br` and it never needs flattening. (Not an error.)
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_TRANSFORMS_GOTOCLASSIFY_H
#define CLANG_CIR_DIALECT_TRANSFORMS_GOTOCLASSIFY_H

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace cir {

enum class ScopeGotoClass {
  NotNested,         // top-level goto -- GotoSolver handles it, not an error
  CleanForwardExit,  // flattenable single-epilogue forward exit
  ForwardLadder,     // flattenable multi-target forward acyclic label chain
  MultiTargetLadder, // dispatch ladder that is NOT a clean forward chain (rejected)
  Backward,          // target block precedes the goto's entry block
  IntoScope,         // target label not at function-body-region level
  Unsupported        // missing label, cross-function, or unrecognised shape
};

// A short, stable name for a class, used verbatim in diagnostics so the
// pass and the gate emit identical, greppable wording.
inline llvm::StringRef scopeGotoClassName(ScopeGotoClass c) {
  switch (c) {
  case ScopeGotoClass::NotNested:
    return "top-level goto";
  case ScopeGotoClass::CleanForwardExit:
    return "clean forward-exit goto";
  case ScopeGotoClass::ForwardLadder:
    return "multi-target forward acyclic label chain (softfloat subFloat64Sigs)";
  case ScopeGotoClass::MultiTargetLadder:
    return "multi-target forward dispatch (not a clean forward chain)";
  case ScopeGotoClass::Backward:
    return "backward goto";
  case ScopeGotoClass::IntoScope:
    return "into-scope goto (target nested below function-body level)";
  case ScopeGotoClass::Unsupported:
    return "unsupported goto shape";
  }
  return "unsupported goto shape";
}

// Ordinal position of `blk` within `region` (blocks are visited in order); -1 if
// `blk` is not a top-level block of `region`.
inline int blockIndexInRegion(mlir::Region &region, mlir::Block *blk) {
  int idx = 0;
  for (mlir::Block &b : region) {
    if (&b == blk)
      return idx;
    ++idx;
  }
  return -1;
}

// Result of classifying a goto: the class plus (when meaningful) the resolved
// entry block, the target label op, and the target label block.
struct ScopeGotoInfo {
  ScopeGotoClass cls = ScopeGotoClass::Unsupported;
  mlir::Block *entryBlock = nullptr; // block of the goto's outermost ancestor
  mlir::Block *labelBlock = nullptr; // block that begins with the target label
  cir::LabelOp label = nullptr;      // the resolved target label op
};

// A ClangIR "label block" is a top-level block of the function-body region whose
// first op is a `cir.label`.
inline bool isLabelBlock(mlir::Block *b) {
  return !b->empty() && mlir::isa<cir::LabelOp>(b->front());
}

// The result of analysing a whole function for the ForwardLadder shape: the
// ordered label blocks and, per block, its single chain successor label block
// (nullptr when the block exits via `cir.return`).
struct ForwardLadderInfo {
  bool valid = false;
  llvm::SmallVector<mlir::Block *, 8> blocks;           // in region order
  llvm::DenseMap<mlir::Block *, mlir::Block *> successor; // null => returns
};

// Validate that ALL of `func`'s function-body-level label blocks form a single
// forward, acyclic chain that is reachable ONLY by goto/chain edges and carries
// no cross-block SSA -- the generalised softfloat `subFloat64Sigs` shape that
// FlattenScopeGoto can soundly structurize into a flag-guarded `cir.if`
// sequence. Conservative: ANY deviation returns {valid=false} so the caller
// falls back to the honest MultiTargetLadder reject.
inline ForwardLadderInfo analyzeForwardLadder(cir::FuncOp func) {
  ForwardLadderInfo out;
  mlir::Region &body = func.getBody();
  if (body.empty())
    return out;

  // Collect body-level label blocks, in order, with an index map.
  llvm::DenseMap<mlir::Block *, int> index;
  for (mlir::Block &b : body) {
    if (isLabelBlock(&b)) {
      index[&b] = static_cast<int>(out.blocks.size());
      out.blocks.push_back(&b);
    }
  }
  if (out.blocks.empty())
    return out;

  // Map label name -> label block (for goto-terminated chain edges).
  llvm::DenseMap<llvm::StringRef, mlir::Block *> byName;
  for (mlir::Block *b : out.blocks)
    byName[mlir::cast<cir::LabelOp>(b->front()).getLabel()] = b;

  for (mlir::Block *b : out.blocks) {
    int bi = index[b];

    // (1) No block arguments (a phi = genuine cross-block SSA merge).
    if (b->getNumArguments() != 0)
      return out;

    // (2) Every CFG predecessor (via cir.br) must itself be a label block in the
    // set -- i.e. the block is NOT the natural fall-through of ordinary code.
    for (mlir::Block *pred : b->getPredecessors())
      if (!index.count(pred))
        return out;

    // (3) No SSA value defined in this block may be used outside it (values from
    // block0 / args that dominate are fine; a def escaping to a sibling label
    // block would be non-dominating after restructuring).
    for (mlir::Operation &op : *b)
      for (mlir::Value res : op.getResults())
        for (mlir::Operation *user : res.getUsers())
          if (user->getBlock() != b)
            return out;

    // (4) Chain terminator: return (exit) or a FORWARD edge (br / top-level
    // goto) to a strictly-later label block.
    mlir::Operation *term = b->getTerminator();
    if (mlir::isa<cir::ReturnOp>(term)) {
      out.successor[b] = nullptr;
      continue;
    }
    mlir::Block *succ = nullptr;
    if (auto br = mlir::dyn_cast<cir::BrOp>(term)) {
      if (br->getNumSuccessors() != 1)
        return out;
      succ = br->getSuccessor(0);
    } else if (auto g = mlir::dyn_cast<cir::GotoOp>(term)) {
      auto it = byName.find(g.getLabel());
      succ = (it == byName.end()) ? nullptr : it->second;
    } else {
      return out; // any other terminator (switch, unreachable, ...) -> not a ladder
    }
    if (!succ || !index.count(succ) || index[succ] <= bi)
      return out; // must be a forward edge to another label block
    out.successor[b] = succ;
  }

  out.valid = true;
  return out;
}

// Classify `gotoOp`. `func` must be its enclosing cir.func.
inline ScopeGotoInfo classifyScopeGoto(cir::GotoOp gotoOp, cir::FuncOp func) {
  ScopeGotoInfo info;
  mlir::Region &body = func.getBody();
  if (body.empty())
    return info; // Unsupported (declaration)

  // Resolve the (unique) target label within this function.
  cir::LabelOp target;
  bool duplicate = false;
  func.walk([&](cir::LabelOp lab) {
    if (lab.getLabel() == gotoOp.getLabel()) {
      if (target)
        duplicate = true;
      target = lab;
    }
  });
  if (!target || duplicate)
    return info; // Unsupported
  info.label = target;
  info.labelBlock = target->getBlock();

  // The label must begin its block (ClangIR emits `cir.label` as the block's
  // first op for a jumped-to label).
  if (&info.labelBlock->front() != target.getOperation()) {
    info.cls = ScopeGotoClass::Unsupported;
    return info;
  }

  // R = the region that DIRECTLY contains the label block. It may be the
  // function body region or any enclosing `cir.scope` region (C block scope).
  mlir::Region *R = info.labelBlock->getParent();
  if (!R) {
    info.cls = ScopeGotoClass::Unsupported;
    return info;
  }

  // Find the goto's outermost ancestor whose block is directly in R; that block
  // is the entry block. R must be an ANCESTOR region of the goto; otherwise the
  // label is not on any path out of the goto's scopes (into-scope / sibling).
  mlir::Operation *runner = gotoOp.getOperation();
  while (runner->getBlock()->getParent() != R) {
    mlir::Operation *parent = runner->getParentOp();
    if (!parent || parent == func.getOperation())
      break;
    runner = parent;
  }
  if (runner->getBlock()->getParent() != R) {
    info.cls = ScopeGotoClass::IntoScope; // label not in an ancestor region
    return info;
  }
  info.entryBlock = runner->getBlock();
  if (runner == gotoOp.getOperation()) {
    info.cls = ScopeGotoClass::NotNested;
    return info;
  }

  // Order check: the target must come strictly after the entry block.
  int ei = blockIndexInRegion(*R, info.entryBlock);
  int li = blockIndexInRegion(*R, info.labelBlock);
  if (ei < 0 || li < 0) {
    info.cls = ScopeGotoClass::Unsupported;
    return info;
  }
  if (li <= ei) {
    info.cls = ScopeGotoClass::Backward;
    return info;
  }

  // Clean iff the entry block ends in an unconditional cir.br to the label
  // block -- i.e. the label is the single shared fall-through epilogue.
  mlir::Operation *term = info.entryBlock->getTerminator();
  if (mlir::isa<cir::BrOp>(term) && term->getNumSuccessors() == 1 &&
      term->getSuccessor(0) == info.labelBlock) {
    info.cls = ScopeGotoClass::CleanForwardExit;
    return info;
  }

  // Not a single clean epilogue. If the WHOLE function's label blocks form a
  // valid forward acyclic chain (reachable only by goto, no cross-block SSA) and
  // THIS goto targets one of them, it is a flattenable ForwardLadder. Otherwise
  // it is the honest MultiTargetLadder reject.
  ForwardLadderInfo ladder = analyzeForwardLadder(func);
  if (ladder.valid && info.labelBlock &&
      llvm::is_contained(ladder.blocks, info.labelBlock)) {
    info.cls = ScopeGotoClass::ForwardLadder;
    return info;
  }

  info.cls = ScopeGotoClass::MultiTargetLadder;
  return info;
}

} // namespace cir

#endif // CLANG_CIR_DIALECT_TRANSFORMS_GOTOCLASSIFY_H
