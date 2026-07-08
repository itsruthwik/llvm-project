//====- FlattenScopeGoto.cpp - CIR cross-scope forward-exit goto pre-pass -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites the classic softfloat "roundAndPack" forward-exit `goto` idiom into
// flag-guarded structured CIR, so `--cir-goto-solver` never has to emit a
// cross-region `cir.br` (which is illegal and is exactly how dfadd/dfsin/jpeg
// currently die in `--cir-to-mlir`).
//
// The one class handled -- CleanForwardExit (see GotoClassify.h) -- is a
// `cir.goto` nested inside `cir.scope`/`cir.if`/loop regions that jumps FORWARD
// to a label at the top of a block in the function-body region, where that
// label block is the unconditional `cir.br` fall-through successor of the
// block holding the goto's outermost ancestor (the "entry block"). Because both
// the goto path and the natural fall-through path reach the same label block,
// the goto is a single shared epilogue and can be replaced exactly like an
// early `return` that resumes at the epilogue:
//   * a function-scope `!cir.bool` flag alloca per target label (init false);
//   * each goto  =>  set flag=true, erase the goto, guard-climb every trailing
//     op at each enclosing level up to the entry block inside `cir.if(!flag)`
//     (the SAME guard-climb machinery as LowerReturn / LowerBreakContinue);
//   * every loop ENCLOSING a goto has its condition strengthened with
//     `cir.select if flag then false else C` and (for a `for` loop) its step
//     guarded, so the goto breaks out of ALL enclosing loops toward the
//     epilogue -- again identical to the early-return treatment.
// The entry block's `cir.br` to the label block is left intact, so on every
// path control reaches the epilogue; `--cir-goto-solver` then erases the now
// dead label and `--cir-canonicalize` merges the single-pred block back in.
//
// INVARIANT (correctness contract): no SSA value created here crosses a region
// boundary. All communication is through the flag memory slot; whole runs of
// ops are relocated into `cir.if(!flag)` regions, never splitting a def from a
// use -- the memory-slot-not-SSA lesson from the LowerReturn dominance fix.
//
// Any goto that is not CleanForwardExit is honestly rejected with a named
// diagnostic (backward, into-scope, multi-target ladder, unsupported). A
// top-level (NotNested) goto is left for `--cir-goto-solver`.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/GotoClassify.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/TimeProfiler.h"
#include <memory>

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_FLATTENSCOPEGOTO
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

// The single-block cond region terminator (cir.condition) of a for/while/do.
static cir::ConditionOp loopCondition(mlir::Operation *loop) {
  mlir::Region *cond = nullptr;
  if (auto f = dyn_cast<cir::ForOp>(loop))
    cond = &f.getCond();
  else if (auto w = dyn_cast<cir::WhileOp>(loop))
    cond = &w.getCond();
  else if (auto d = dyn_cast<cir::DoWhileOp>(loop))
    cond = &d.getCond();
  if (!cond || cond->empty())
    return nullptr;
  return dyn_cast<cir::ConditionOp>(cond->front().getTerminator());
}

struct FlattenScopeGotoPass
    : public impl::FlattenScopeGotoBase<FlattenScopeGotoPass> {
  FlattenScopeGotoPass() = default;
  void runOnOperation() override;

  bool failed = false;

  // Guard the ops that follow `runner` (within runner's block, up to the block
  // terminator) inside a fresh `cir.if(!flag) { ... }`. No-op if there is
  // nothing to guard. Lifted verbatim from LowerReturn.
  void wrapTrailing(mlir::Operation *runner, mlir::Value flagAddr,
                    mlir::Type boolTy) {
    mlir::Operation *first = runner->getNextNode();
    if (!first || first->hasTrait<mlir::OpTrait::IsTerminator>())
      return; // nothing after `runner` but the terminator
    mlir::OpBuilder b(runner->getContext());
    b.setInsertionPointAfter(runner);
    auto loc = runner->getLoc();
    mlir::Value flag = cir::LoadOp::create(b, loc, boolTy, flagAddr);
    mlir::Value notFlag = cir::NotOp::create(b, loc, boolTy, flag);
    auto ifnot = cir::IfOp::create(b, loc, notFlag, /*withElseRegion=*/false,
                                   [](mlir::OpBuilder &, mlir::Location) {});
    mlir::Block &then = ifnot.getThenRegion().back();
    b.setInsertionPointToEnd(&then);
    auto term = cir::YieldOp::create(b, loc);
    for (mlir::Operation *o = ifnot->getNextNode(); o;) {
      if (o->hasTrait<mlir::OpTrait::IsTerminator>())
        break;
      mlir::Operation *next = o->getNextNode();
      o->moveBefore(term);
      o = next;
    }
  }

  // Guard every op in `block` (except its terminator) inside `cir.if(!flag)`.
  // Used to skip a `for` step region when the goto fires. From LowerReturn.
  void guardBlock(mlir::Block *block, mlir::Value flagAddr, mlir::Type boolTy,
                  mlir::Location loc) {
    mlir::Operation *term = block->getTerminator();
    llvm::SmallVector<mlir::Operation *> ops;
    for (mlir::Operation &o : *block)
      if (&o != term)
        ops.push_back(&o);
    if (ops.empty())
      return;
    mlir::OpBuilder b(block, block->begin());
    mlir::Value flag = cir::LoadOp::create(b, loc, boolTy, flagAddr);
    mlir::Value notFlag = cir::NotOp::create(b, loc, boolTy, flag);
    auto ifnot = cir::IfOp::create(b, loc, notFlag, /*withElseRegion=*/false,
                                   [](mlir::OpBuilder &, mlir::Location) {});
    mlir::Block &then = ifnot.getThenRegion().back();
    mlir::OpBuilder tb(&then, then.end());
    auto yield = cir::YieldOp::create(tb, loc);
    for (mlir::Operation *o : ops)
      o->moveBefore(yield);
  }

  // Create a `!cir.bool` flag alloca initialised to false at the top of the
  // function body block.
  mlir::Value makeFlag(mlir::Block *bodyBlock, mlir::Type boolTy,
                       llvm::StringRef name, mlir::Location loc) {
    mlir::OpBuilder b(bodyBlock, bodyBlock->begin());
    auto boolPtrTy = cir::PointerType::get(boolTy);
    auto align = b.getI64IntegerAttr(1);
    mlir::Value addr = cir::AllocaOp::create(b, loc, boolPtrTy, name, align);
    mlir::Value falseVal = cir::ConstantOp::create(
        b, loc, cir::BoolAttr::get(&getContext(), false));
    cir::StoreOp::create(b, loc, falseVal, addr);
    return addr;
  }

  // Strengthen a loop so it terminates once `flag` is set (condition select +
  // for-step guard). Identical to the early-return treatment.
  void strengthenLoop(mlir::Operation *loop, mlir::Value flagAddr,
                      mlir::Type boolTy) {
    if (auto condOp = loopCondition(loop)) {
      mlir::OpBuilder b(condOp);
      auto loc = condOp.getLoc();
      mlir::Value flag = cir::LoadOp::create(b, loc, boolTy, flagAddr);
      mlir::Value falseVal = cir::ConstantOp::create(
          b, loc, cir::BoolAttr::get(&getContext(), false));
      mlir::Value orig = condOp.getCondition();
      mlir::Value sel =
          cir::SelectOp::create(b, loc, boolTy, flag, falseVal, orig);
      condOp.getConditionMutable().assign(sel);
    } else {
      loop->emitError("ThroughMLIR: loop enclosing a cross-scope 'goto' has no "
                      "structured condition region to strengthen");
      failed = true;
      return;
    }
    if (auto forOp = dyn_cast<cir::ForOp>(loop)) {
      mlir::Region &step = forOp.getStep();
      if (!step.empty() && step.hasOneBlock())
        guardBlock(&step.front(), flagAddr, boolTy, forOp.getLoc());
    }
  }

  // Materialize one clean goto IN PLACE (no climb yet): set flag=true, erase the
  // goto, re-terminate its block. Returns the flag-store op to climb from.
  mlir::Operation *materializeGoto(cir::GotoOp gotoOp, mlir::Value flagAddr) {
    mlir::OpBuilder b(gotoOp);
    auto loc = gotoOp.getLoc();
    mlir::Value trueVal =
        cir::ConstantOp::create(b, loc, cir::BoolAttr::get(&getContext(), true));
    mlir::Operation *flagStore =
        cir::StoreOp::create(b, loc, trueVal, flagAddr);
    mlir::Block *gotoBlock = gotoOp->getBlock();
    gotoOp->erase();
    if (gotoBlock->empty() ||
        !gotoBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::OpBuilder tb(gotoBlock, gotoBlock->end());
      cir::YieldOp::create(tb, loc);
    }
    return flagStore;
  }

  // Guard-climb from `anchor` up to (and including) `entryBlock`, guarding the
  // trailing ops at every ancestor level in `cir.if(!flag)`. The entry block's
  // own terminator (the `cir.br` to the epilogue) is never wrapped, so control
  // always reaches the epilogue.
  void climb(mlir::Operation *anchor, mlir::Value flagAddr, mlir::Type boolTy,
             mlir::Block *entryBlock) {
    mlir::Operation *runner = anchor;
    while (runner->getBlock() != entryBlock) {
      wrapTrailing(runner, flagAddr, boolTy);
      runner = runner->getParentOp();
      if (!runner)
        return; // defensive: classifier guarantees we reach entryBlock
    }
    wrapTrailing(runner, flagAddr, boolTy);
  }

  // Collect the loops enclosing `gotoOp` up to and including the outermost
  // ancestor sitting directly in `entryBlock`, recording a (loop, flag) pair
  // for each. A loop enclosing gotos to two DIFFERENT labels must be
  // strengthened for BOTH flags, so pairs (not a loop->flag map) are tracked.
  void collectEnclosingLoops(
      cir::GotoOp gotoOp, mlir::Block *entryBlock, mlir::Value flag,
      llvm::SmallSetVector<std::pair<mlir::Operation *, mlir::Value>, 4>
          &pairs) {
    for (mlir::Operation *a = gotoOp->getParentOp(); a; a = a->getParentOp()) {
      bool atEntry = (a->getBlock() == entryBlock);
      if (isa<cir::ForOp, cir::WhileOp, cir::DoWhileOp>(a))
        pairs.insert({a, flag});
      if (atEntry)
        break;
    }
  }

  void processFunc(cir::FuncOp func) {
    if (failed)
      return;
    mlir::Region &body = func.getBody();
    if (body.empty())
      return;

    // Collect every goto and classify. Reject the whole function on any
    // non-clean, non-top-level goto (honest, named diagnostic).
    llvm::SmallVector<std::pair<cir::GotoOp, ScopeGotoInfo>> clean;
    bool anyReject = false;
    func.walk([&](cir::GotoOp g) {
      ScopeGotoInfo info = classifyScopeGoto(g, func);
      switch (info.cls) {
      case ScopeGotoClass::NotNested:
        return; // GotoSolver handles it.
      case ScopeGotoClass::CleanForwardExit:
        clean.push_back({g, info});
        return;
      default:
        g.emitError("ThroughMLIR: unsupported 'goto' (")
            << scopeGotoClassName(info.cls)
            << "): only the single-epilogue cross-scope forward-exit idiom is "
               "flattened by --cir-flatten-scope-goto";
        anyReject = true;
        return;
      }
    });
    if (anyReject) {
      failed = true;
      return;
    }
    if (clean.empty())
      return; // churn guard.

    auto boolTy = cir::BoolType::get(&getContext());
    mlir::Block *bodyBlock = &body.front();

    // One flag per distinct target label (multiple gotos to the same epilogue
    // share it). Materialize ALL gotos first (memory-only), THEN climb, so a
    // climb that wraps another already-materialized goto's region moves a whole
    // op-run together and never splits a def from its use.
    llvm::DenseMap<mlir::Operation *, mlir::Value> flagFor; // label op -> flag
    llvm::SmallVector<mlir::Operation *> anchors;
    llvm::SmallVector<mlir::Value> anchorFlag;
    llvm::SmallVector<mlir::Block *> anchorEntry;
    llvm::SmallSetVector<std::pair<mlir::Operation *, mlir::Value>, 4>
        loopFlagPairs;

    for (auto &pr : clean) {
      cir::GotoOp g = pr.first;
      ScopeGotoInfo &info = pr.second;
      mlir::Value flag = flagFor.lookup(info.label.getOperation());
      if (!flag) {
        flag = makeFlag(bodyBlock, boolTy,
                        "goto_" + info.label.getLabel().str(), func.getLoc());
        flagFor[info.label.getOperation()] = flag;
      }
      // Record (loop, flag) pairs BEFORE mutation (climb reshapes the region).
      collectEnclosingLoops(g, info.entryBlock, flag, loopFlagPairs);
      anchorFlag.push_back(flag);
      anchorEntry.push_back(info.entryBlock);
      anchors.push_back(materializeGoto(g, flag));
    }
    for (size_t i = 0; i < anchors.size(); ++i)
      climb(anchors[i], anchorFlag[i], boolTy, anchorEntry[i]);

    // Strengthen each enclosing loop for every flag whose goto it encloses (a
    // loop shared by two labels is strengthened for both -- composed selects).
    for (auto &lf : loopFlagPairs) {
      strengthenLoop(lf.first, lf.second, boolTy);
      if (failed)
        return;
    }
  }
};

void FlattenScopeGotoPass::runOnOperation() {
  llvm::TimeTraceScope scope("Flatten Scope Goto");
  getOperation()->walk([&](cir::FuncOp func) {
    processFunc(func);
  });
  if (failed)
    signalPassFailure();
}

} // namespace

std::unique_ptr<mlir::Pass> mlir::createFlattenScopeGotoPass() {
  return std::make_unique<FlattenScopeGotoPass>();
}
