//====- ControlFlowGuards.cpp - Shared CIR guard-climb / loop-exit utils --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/ControlFlowGuards.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace cir;

cir::ConditionOp cir::loopCondition(mlir::Operation *loop) {
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

void cir::wrapTrailing(mlir::Operation *runner, mlir::Value flagAddr,
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

void cir::guardBlock(mlir::Block *block, mlir::Value flagAddr, mlir::Type boolTy,
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

void cir::climbGuard(mlir::Operation *anchor, mlir::Value flagAddr,
                     mlir::Type boolTy, mlir::Block *stopBlock) {
  mlir::Operation *runner = anchor;
  while (runner->getBlock() != stopBlock) {
    wrapTrailing(runner, flagAddr, boolTy);
    runner = runner->getParentOp();
    if (!runner) // defensive: callers guarantee we reach stopBlock
      return;
  }
  wrapTrailing(runner, flagAddr, boolTy);
}

mlir::LogicalResult cir::strengthenLoopExit(mlir::Operation *loop,
                                            mlir::Value flagAddr,
                                            mlir::Type boolTy,
                                            llvm::StringRef noCondMsg) {
  auto condOp = loopCondition(loop);
  if (!condOp) {
    loop->emitError(noCondMsg);
    return mlir::failure();
  }
  mlir::MLIRContext *ctx = boolTy.getContext();
  mlir::Block *condBlock = condOp->getBlock();
  mlir::Value orig = condOp.getCondition();
  auto loc = condOp.getLoc();

  // Snapshot the ops that compute the original condition C: everything in the
  // cond block ahead of the `cir.condition` terminator. (Do this BEFORE we
  // insert the flag load / ternary so those are not swept in.)
  llvm::SmallVector<mlir::Operation *> condOps;
  for (mlir::Operation &o : *condBlock)
    if (&o != condOp.getOperation())
      condOps.push_back(&o);

  mlir::OpBuilder b(condOp);
  mlir::Value flag = cir::LoadOp::create(b, loc, boolTy, flagAddr);

  // Lazy short-circuit: %c = flag ? false : C, where C (and ALL of its
  // side-effecting computation) is EVALUATED ONLY in the false region. The
  // cir.ternary executes exactly one region, so C never re-executes on the
  // iteration where the exit flag is set -- matching real C, where a
  // break/return/goto means the loop condition is never re-evaluated. Using a
  // non-short-circuit cir.select over an already-computed C (the pre-fix shape)
  // silently fired C's side effects (`n--`, `*p++`, a call) one extra time.
  auto ternary = cir::TernaryOp::create(
      b, loc, flag,
      /*trueBuilder=*/
      [&](mlir::OpBuilder &tb, mlir::Location tl) {
        mlir::Value f =
            cir::ConstantOp::create(tb, tl, cir::BoolAttr::get(ctx, false));
        cir::YieldOp::create(tb, tl, f);
      },
      /*falseBuilder=*/
      [&](mlir::OpBuilder &fb, mlir::Location fl) {
        cir::YieldOp::create(fb, fl, orig);
      });

  // Sink C's computation into the false region, before its yield of `orig`, so
  // it runs only when the flag is clear. Whole ops move together (memory-slot
  // discipline) -- no def is split from its use.
  mlir::Operation *falseYield =
      ternary.getFalseRegion().front().getTerminator();
  for (mlir::Operation *o : condOps)
    o->moveBefore(falseYield);

  condOp.getConditionMutable().assign(ternary.getResult());

  // A `for` step must be SKIPPED once the flag is set: wrap it in if(!flag).
  if (auto forOp = dyn_cast<cir::ForOp>(loop)) {
    mlir::Region &step = forOp.getStep();
    if (!step.empty() && step.hasOneBlock())
      guardBlock(&step.front(), flagAddr, boolTy, forOp.getLoc());
  }
  return mlir::success();
}
