//====- LowerReturn.cpp - CIR nested-return pre-pass ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites a cir.func containing an early `cir.return` nested inside a loop,
// `cir.if`, or `cir.scope` into single-exit form, so the downstream one-shot
// --cir-to-mlir lowering (CIRReturnLowering) never has to place a func.return
// inside an scf region (which is illegal -- func.return must be parented by
// func.func). This is the return analogue of the "Option B" break/continue
// pre-pass: a normal CIR->CIR rewrite ahead of the structured lowering.
//
// Per function that has >=1 *nested* return (a return that is not the function
// body block's terminator):
//   * a function-scope `retval` alloca (only when the function is non-void) and
//     a `returning` !cir.bool flag alloca (init false at the top of the body);
//   * each nested `cir.return X`  =>  store X into retval (if any),
//     set returning=true, erase the return, and guard-climb every trailing op
//     at each enclosing level up to the function body inside `cir.if(!returning)`
//     -- the SAME guard-climb machinery as LowerBreakContinue;
//   * every loop ENCLOSING a nested return has its condition strengthened with
//     `%c = cir.select if returning then false else C` and (for a `for` loop)
//     its STEP body wrapped in `cir.if(!returning)`, so a return breaks out of
//     ALL enclosing loops (not just the innermost -- this is the difference from
//     break, which strengthens only its target loop) and skips the for-step;
//   * the function tail becomes a single `cir.return` that loads retval (or a
//     bare `cir.return` for void), read AFTER all guarded fall-through stores so
//     it always observes the correct value on every path.
//
// Returns already at function top level are left untouched (no churn on the
// common case). A nested return in a non-structured (multi-block) region that
// this pass cannot restructure soundly is honestly rejected rather than
// mis-lowered -- the Tier-1 reject in LowerCIRLoopToSCF is the fail-safe for any
// shape that still reaches the lowering.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/TimeProfiler.h"
#include <memory>

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_LOWERRETURN
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

struct LowerReturnPass : public impl::LowerReturnBase<LowerReturnPass> {
  LowerReturnPass() = default;
  void runOnOperation() override;

  bool failed = false;

  // Guard the ops that follow `runner` (within runner's block, up to the block
  // terminator) inside a fresh `cir.if(!flag) { ... }`. No-op if there is
  // nothing to guard. Lifted verbatim from LowerBreakContinue.
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
  // Used to skip a `for` step region on return. Lifted from LowerBreakContinue.
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

  // Create a typed alloca initialised (optionally) at the top of `bodyBlock`.
  mlir::Value makeAlloca(mlir::Block *bodyBlock, mlir::Type pointeeTy,
                         llvm::StringRef name, mlir::Location loc) {
    mlir::OpBuilder b(bodyBlock, bodyBlock->begin());
    auto ptrTy = cir::PointerType::get(pointeeTy);
    auto align = b.getI64IntegerAttr(1);
    return cir::AllocaOp::create(b, loc, ptrTy, name, align);
  }

  // Rewrite one nested return: store its value into retval, set returning=true,
  // erase it, and guard-climb the trailing ops up to `bodyBlock`.
  void rewriteNestedReturn(cir::ReturnOp ret, mlir::Value retval,
                           mlir::Value returningAddr, mlir::Type boolTy,
                           mlir::Block *bodyBlock) {
    mlir::OpBuilder b(ret);
    auto loc = ret.getLoc();
    if (retval && ret.getNumOperands() != 0)
      cir::StoreOp::create(b, loc, ret.getOperand(0), retval);
    mlir::Value trueVal =
        cir::ConstantOp::create(b, loc, cir::BoolAttr::get(&getContext(), true));
    cir::StoreOp::create(b, loc, trueVal, returningAddr);
    mlir::Block *retBlock = ret->getBlock();
    mlir::Operation *store = ret->getPrevNode();
    ret->erase();
    // The return was the block terminator; re-terminate the block.
    if (retBlock->empty() ||
        !retBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::OpBuilder tb(retBlock, retBlock->end());
      cir::YieldOp::create(tb, loc);
    }
    // Climb from the returning-store up to the function body block, guarding
    // the trailing ops at each ancestor level in `cir.if(!returning)`.
    mlir::Operation *runner = store;
    while (runner->getBlock() != bodyBlock) {
      wrapTrailing(runner, returningAddr, boolTy);
      runner = runner->getParentOp();
      if (!runner) // defensive: should always reach the body block
        return;
    }
    wrapTrailing(runner, returningAddr, boolTy);
  }

  // Strengthen an enclosing loop so it terminates once `returning` is set.
  void strengthenLoop(mlir::Operation *loop, mlir::Value returningAddr,
                      mlir::Type boolTy) {
    if (auto condOp = loopCondition(loop)) {
      mlir::OpBuilder b(condOp);
      auto loc = condOp.getLoc();
      mlir::Value flag = cir::LoadOp::create(b, loc, boolTy, returningAddr);
      mlir::Value falseVal = cir::ConstantOp::create(
          b, loc, cir::BoolAttr::get(&getContext(), false));
      mlir::Value orig = condOp.getCondition();
      mlir::Value sel =
          cir::SelectOp::create(b, loc, boolTy, flag, falseVal, orig);
      condOp.getConditionMutable().assign(sel);
    } else {
      loop->emitError("ThroughMLIR: loop enclosing a 'return' has no structured "
                      "condition region to strengthen");
      failed = true;
      return;
    }
    // A `for` step must be SKIPPED on return: wrap the step body in if(!ret).
    if (auto forOp = dyn_cast<cir::ForOp>(loop)) {
      mlir::Region &step = forOp.getStep();
      if (!step.empty() && step.hasOneBlock())
        guardBlock(&step.front(), returningAddr, boolTy, forOp.getLoc());
    }
  }

  void processFunc(cir::FuncOp func) {
    if (failed)
      return;
    mlir::Region &bodyRegion = func.getBody();
    if (bodyRegion.empty()) // declaration
      return;
    // The function body must be a single structured block for the guard-climb
    // to reach a well-defined tail; a multi-block body is rejected.
    if (!bodyRegion.hasOneBlock()) {
      // Only reject if there is actually a nested return to worry about.
      bool anyNested = false;
      mlir::Operation *anyTerm =
          bodyRegion.front().empty() ? nullptr : &bodyRegion.front().back();
      func.walk([&](cir::ReturnOp r) {
        if (r.getOperation() != anyTerm)
          anyNested = true;
      });
      if (anyNested) {
        func.emitError("ThroughMLIR: early 'return' in a non-structured "
                       "(multi-block) function body is not supported by the "
                       "CIR-to-MLIR lowering");
        failed = true;
      }
      return;
    }
    mlir::Block *bodyBlock = &bodyRegion.front();
    mlir::Operation *tailTerm = bodyBlock->empty() ? nullptr : &bodyBlock->back();
    auto tailRet = tailTerm ? dyn_cast<cir::ReturnOp>(tailTerm) : nullptr;

    // Collect nested returns (every return except the tail terminator).
    llvm::SmallVector<cir::ReturnOp> nested;
    func.walk([&](cir::ReturnOp r) {
      if (r.getOperation() != tailTerm)
        nested.push_back(r);
    });
    if (nested.empty())
      return; // churn guard: no nested return, leave the function untouched.

    auto boolTy = cir::BoolType::get(&getContext());
    auto fnType = func.getFunctionType();
    bool hasRet = !fnType.hasVoidReturn();
    auto loc = func.getLoc();

    // Function-scope retval (non-void) + returning flag, at the top of the body.
    mlir::Value retval;
    if (hasRet)
      retval = makeAlloca(bodyBlock, fnType.getReturnType(), "retval", loc);
    mlir::Value returningAddr = makeAlloca(bodyBlock, boolTy, "returning", loc);
    {
      mlir::OpBuilder b(returningAddr.getDefiningOp());
      b.setInsertionPointAfter(returningAddr.getDefiningOp());
      mlir::Value falseVal = cir::ConstantOp::create(
          b, loc, cir::BoolAttr::get(&getContext(), false));
      cir::StoreOp::create(b, loc, falseVal, returningAddr);
    }

    // Phase 1: capture the tail (fall-through) value into retval BEFORE the
    // nested guard-climbs, so the guard-climb wraps this store in if(!returning)
    // (the fall-through value must not overwrite retval once we are returning).
    if (hasRet && tailRet && tailRet.getNumOperands() != 0) {
      mlir::OpBuilder b(tailRet);
      cir::StoreOp::create(b, tailRet.getLoc(), tailRet.getOperand(0), retval);
    }

    // Collect the loops enclosing each nested return (all of them, deduped).
    llvm::SmallSetVector<mlir::Operation *, 4> enclosingLoops;
    for (auto r : nested)
      for (mlir::Operation *a = r->getParentOp();
           a && a != func.getOperation(); a = a->getParentOp())
        if (isa<cir::ForOp, cir::WhileOp, cir::DoWhileOp>(a))
          enclosingLoops.insert(a);

    // Phase 2: rewrite each nested return (store retval + set flag + climb).
    for (auto r : nested) {
      rewriteNestedReturn(r, retval, returningAddr, boolTy, bodyBlock);
      if (failed)
        return;
    }

    // Phase 3: strengthen every enclosing loop so a return breaks them all.
    for (mlir::Operation *loop : enclosingLoops) {
      strengthenLoop(loop, returningAddr, boolTy);
      if (failed)
        return;
    }

    // Phase 4: rebuild the single tail exit. The returned value is read from
    // retval AFTER all guarded fall-through stores, so it dominates the return
    // and observes the correct value on every path.
    if (tailRet) {
      if (hasRet) {
        mlir::OpBuilder b(tailRet);
        mlir::Value v = cir::LoadOp::create(b, tailRet.getLoc(),
                                            fnType.getReturnType(), retval);
        cir::ReturnOp::create(b, tailRet.getLoc(), mlir::ValueRange{v});
        tailRet.erase();
      }
      // void tail return needs no operand -- leave as-is.
    } else {
      // No structural tail return (control could fall off the end); append one.
      mlir::OpBuilder b(bodyBlock, bodyBlock->end());
      if (hasRet) {
        mlir::Value v =
            cir::LoadOp::create(b, loc, fnType.getReturnType(), retval);
        cir::ReturnOp::create(b, loc, mlir::ValueRange{v});
      } else {
        cir::ReturnOp::create(b, loc, mlir::ValueRange{});
      }
    }
  }
};

void LowerReturnPass::runOnOperation() {
  llvm::TimeTraceScope scope("Lower Return");
  getOperation()->walk([&](cir::FuncOp func) {
    processFunc(func);
  });
  if (failed)
    signalPassFailure();
}

} // namespace

std::unique_ptr<mlir::Pass> mlir::createLowerReturnPass() {
  return std::make_unique<LowerReturnPass>();
}
