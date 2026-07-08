//====- LowerCIRLoopToSCF.cpp - Lowering from CIR Loop to SCF -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of CIR loop operations to SCF.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/LowerToMLIR.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace cir;
using namespace llvm;

namespace cir {

class SCFLoop {
public:
  SCFLoop(cir::ForOp op, mlir::ConversionPatternRewriter *rewriter)
      : forOp(op), rewriter(rewriter) {}

  int64_t getStep() { return step; }
  mlir::Value getLowerBound() { return lowerBound; }
  mlir::Value getUpperBound() { return upperBound; }
  bool isCanonical() { return canonical; }
  bool hasBreakOrContinue() { return hasBreakContinue; }

  // Returns true if successfully finds both step and induction variable.
  mlir::LogicalResult findStepAndIV();
  cir::CmpOp findCmpOp();
  mlir::Value findIVInitValue();
  void analysis();

  mlir::Value plusConstant(mlir::Value v, mlir::Location loc, int addend);
  void transferToSCFForOp();
  void transformToSCFWhileOp();
  void transformToCIRWhileOp(); // TODO

private:
  cir::ForOp forOp;
  cir::CmpOp cmpOp;
  mlir::Value ivAddr, lowerBound = nullptr, upperBound = nullptr;
  mlir::ConversionPatternRewriter *rewriter;
  int64_t step = 0;
  bool hasBreakContinue = false;
  bool canonical = true;
};

class SCFWhileLoop {
public:
  SCFWhileLoop(cir::WhileOp op, cir::WhileOp::Adaptor adaptor,
               mlir::ConversionPatternRewriter *rewriter)
      : whileOp(op), adaptor(adaptor), rewriter(rewriter) {}
  mlir::scf::WhileOp transferToSCFWhileOp();

private:
  cir::WhileOp whileOp;
  cir::WhileOp::Adaptor adaptor;
  mlir::scf::WhileOp scfWhileOp;
  mlir::ConversionPatternRewriter *rewriter;
};

class SCFDoLoop {
public:
  SCFDoLoop(cir::DoWhileOp op, cir::DoWhileOp::Adaptor adaptor,
            mlir::ConversionPatternRewriter *rewriter)
      : DoOp(op), adaptor(adaptor), rewriter(rewriter) {}
  void transferToSCFWhileOp();

private:
  cir::DoWhileOp DoOp;
  cir::DoWhileOp::Adaptor adaptor;
  mlir::ConversionPatternRewriter *rewriter;
};

static int64_t getConstant(cir::ConstantOp op) {
  auto attr = op.getValue();
  const auto intAttr = mlir::cast<cir::IntAttr>(attr);
  return intAttr.getValue().getSExtValue();
}

mlir::LogicalResult SCFLoop::findStepAndIV() {
  auto *stepBlock =
      (forOp.maybeGetStep() ? &forOp.maybeGetStep()->front() : nullptr);
  assert(stepBlock && "Can not find step block");

  // Try to match "iv = load addr; ++iv; store iv, addr; yield" to find step.
  // We should match the exact pattern, in case there's something unexpected:
  // we must rule out cases like `for (int i = 0; i < n; i++, printf("\n"))`.
  auto &oplist = stepBlock->getOperations();

  auto iterator = oplist.begin();

  // We might find constants at beginning. Skip them.
  // We could have hoisted them outside the for loop in previous passes, but
  // it hasn't been done yet.
  while (iterator != oplist.end() && isa<ConstantOp>(*iterator))
    ++iterator;

  if (iterator == oplist.end())
    return mlir::failure();

  auto load = dyn_cast<LoadOp>(*iterator);
  if (!load)
    return mlir::failure();

  // We assume this is the address of induction variable (IV). The operations
  // that come next will check if that's true.
  mlir::Value addr = load.getAddr();
  mlir::Value iv = load.getResult();

  // Then we try to match either "++IV" or "IV += n". Same for reversed loops.
  if (++iterator == oplist.end())
    return mlir::failure();

  mlir::Operation &arith = *iterator;

  // Upstream split cir::UnaryOp -> Inc/Dec and cir::BinOp -> Add/Sub (D1/D2),
  // so match the per-op step forms directly.
  if (auto inc = dyn_cast<IncOp>(arith)) {
    if (inc.getInput() != iv)
      return mlir::failure();
    step = 1;
  } else if (auto dec = dyn_cast<DecOp>(arith)) {
    if (dec.getInput() != iv)
      return mlir::failure();
    step = -1;
  } else if (auto add = dyn_cast<AddOp>(arith)) {
    if (add.getLhs() != iv)
      return mlir::failure();
    if (auto constValue = add.getRhs().getDefiningOp<cir::ConstantOp>();
        constValue && constValue.getValueAttr<cir::IntAttr>())
      step = getConstant(constValue);
  } else if (auto sub = dyn_cast<SubOp>(arith)) {
    if (sub.getLhs() != iv)
      return mlir::failure();
    if (auto constValue = sub.getRhs().getDefiningOp<cir::ConstantOp>();
        constValue && constValue.getValueAttr<cir::IntAttr>())
      step = getConstant(constValue);
    step = -step;
  } else {
    return mlir::failure();
  }

  // Check whether we immediately store this value into the appropriate place.
  if (++iterator == oplist.end())
    return mlir::failure();

  auto store = dyn_cast<StoreOp>(*iterator);
  if (!store || store.getAddr() != addr ||
      store.getValue() != arith.getResult(0))
    return mlir::failure();

  if (++iterator == oplist.end())
    return mlir::failure();

  // Finally, this should precede a yield with nothing in between.
  bool success = isa<YieldOp>(*iterator);

  // Remember to update analysis information.
  if (success)
    ivAddr = addr;

  return success ? mlir::success() : mlir::failure();
}

static bool isIVLoad(mlir::Operation *op, mlir::Value IVAddr) {
  if (!op)
    return false;
  if (isa<cir::LoadOp>(op)) {
    if (!op->getOperand(0))
      return false;
    if (op->getOperand(0) == IVAddr)
      return true;
  }
  return false;
}

cir::CmpOp SCFLoop::findCmpOp() {
  cmpOp = nullptr;
  for (auto *user : ivAddr.getUsers()) {
    if (user->getParentRegion() != &forOp.getCond())
      continue;
    if (auto loadOp = dyn_cast<cir::LoadOp>(*user)) {
      if (!loadOp->hasOneUse())
        continue;
      if (auto op = dyn_cast<cir::CmpOp>(*loadOp->user_begin())) {
        cmpOp = op;
        break;
      }
    }
  }
  if (!cmpOp)
    return nullptr;

  auto type = cmpOp.getLhs().getType();
  if (!mlir::isa<cir::IntType>(type))
    return nullptr;

  auto *lhsDefOp = cmpOp.getLhs().getDefiningOp();
  if (!lhsDefOp)
    return nullptr;
  if (!isIVLoad(lhsDefOp, ivAddr))
    return nullptr;

  if (cmpOp.getKind() != cir::CmpOpKind::le &&
      cmpOp.getKind() != cir::CmpOpKind::lt)
    return nullptr;

  return cmpOp;
}

mlir::Value SCFLoop::plusConstant(mlir::Value v, mlir::Location loc,
                                  int addend) {
  auto type = v.getType();
  auto c1 = mlir::arith::ConstantOp::create(
      *rewriter, loc, mlir::IntegerAttr::get(type, addend));
  return mlir::arith::AddIOp::create(*rewriter, loc, v, c1);
}

// Return IV initial value by searching the store before the loop.
// The operations before the loop have been transferred to MLIR.
// So we need to go through getRemappedValue to find the value.
mlir::Value SCFLoop::findIVInitValue() {
  auto remapAddr = rewriter->getRemappedValue(ivAddr);
  if (!remapAddr)
    return nullptr;
  if (auto castOp =
          mlir::dyn_cast<mlir::memref::CastOp>(remapAddr.getDefiningOp())) {
    remapAddr = castOp->getOperand(0);
    if (!remapAddr)
      return nullptr;
    // Alloca has two uses, one is the CastOp, and second is the StoreOp (which
    // bypasses the CastOp)
    if (remapAddr.getNumUses() > 2)
      return nullptr;
  } else {
    if (!remapAddr.hasOneUse())
      return nullptr;
  }
  for (auto user : remapAddr.getUsers()) {
    if (auto memrefStore = dyn_cast<mlir::memref::StoreOp>(user))
      return memrefStore->getOperand(0);
  }
  return nullptr;
}

// Forward decls: only break/continue that TARGET a given loop take it off the
// canonical path (definitions are below, after the SCFLoop members).
static bool loopHasTargetingBreak(mlir::Operation *loopOp);
static bool loopHasTargetingContinue(mlir::Operation *loopOp);

void SCFLoop::analysis() {
  // Only a break/continue that TARGETS this for-loop forces the non-canonical
  // while path. A break/continue belonging to an INNER nested loop is that
  // loop's concern and must not downgrade this (canonical) loop's lowering. A
  // coarse subtree walk would wrongly route e.g. `for(i){ while(p){ continue; }}`
  // through the fragile while-transform.
  hasBreakContinue =
      loopHasTargetingBreak(forOp) || loopHasTargetingContinue(forOp);
  if (hasBreakContinue) {
    canonical = false;
    return;
  }

  canonical = mlir::succeeded(findStepAndIV());
  if (!canonical)
    return;

  // If the IV is defined before the forOp (i.e. outside the surrounding
  // cir.scope) this is not a canonical loop as the IV would not have the
  // correct value after the forOp
  if (ivAddr.getDefiningOp()->getBlock() != forOp->getBlock()) {
    canonical = false;
    return;
  }

  cmpOp = findCmpOp();
  if (!cmpOp) {
    canonical = false;
    return;
  }

  auto ivInit = findIVInitValue();
  if (!ivInit) {
    canonical = false;
    return;
  }

  // The loop end value should be hoisted out of loop by -cir-mlir-scf-prepare.
  // So we could get the value by getRemappedValue.
  auto ivEndBound = rewriter->getRemappedValue(cmpOp.getRhs());
  // If the loop end bound is not loop invariant and can't be hoisted,
  // then this is not a canonical loop.
  if (!ivEndBound) {
    canonical = false;
    return;
  }

  if (step > 0) {
    lowerBound = ivInit;
    if (cmpOp.getKind() == cir::CmpOpKind::lt)
      upperBound = ivEndBound;
    else if (cmpOp.getKind() == cir::CmpOpKind::le)
      upperBound = plusConstant(ivEndBound, cmpOp.getLoc(), 1);
  }
  if (!lowerBound || !upperBound)
    canonical = false;
}

void SCFLoop::transferToSCFForOp() {
  auto ub = getUpperBound();
  auto lb = getLowerBound();
  auto loc = forOp.getLoc();
  auto type = lb.getType();
  auto step = mlir::arith::ConstantOp::create(
      *rewriter, loc, mlir::IntegerAttr::get(type, getStep()));
  auto scfForOp = mlir::scf::ForOp::create(*rewriter, loc, lb, ub, step);
  SmallVector<mlir::Value> bbArg;
  rewriter->eraseOp(&scfForOp.getBody()->back());
  rewriter->inlineBlockBefore(&forOp.getBody().front(), scfForOp.getBody(),
                              scfForOp.getBody()->end(), bbArg);
  scfForOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    // A break/continue that TARGETS this for-loop is routed to the while-loop
    // path by analysis() (hasBreakContinue => non-canonical), so it must never
    // reach the canonical scf.for lowering. A break/continue owned by an INNER
    // switch/loop nested in the body is legitimate: it targets that construct
    // and is consumed when that construct is lowered (e.g. an inner cir.switch's
    // case `break`). We must not assert on those, but we still descend so this
    // loop's IV loads inside the inner construct get replaced. An inner cir.if
    // IS supported the same way.
    if (isa<cir::BreakOp>(op) || isa<cir::ContinueOp>(op)) {
      bool isBreak = isa<cir::BreakOp>(op);
      for (mlir::Operation *anc = op->getParentOp(); anc;
           anc = anc->getParentOp()) {
        if (anc == scfForOp.getOperation())
          break; // reached the loop root without a nearer owner
        if (isa<cir::ForOp, cir::WhileOp, cir::DoWhileOp, mlir::scf::ForOp,
                mlir::scf::WhileOp>(anc))
          return mlir::WalkResult::advance(); // owned by an inner loop
        if (isBreak && isa<cir::SwitchOp>(anc))
          return mlir::WalkResult::advance(); // owned by an inner switch
      }
      llvm_unreachable(
          "a for-loop-targeting break/continue must be routed to the "
          "while-loop lowering path");
    }
    // Replace the IV usage to scf loop induction variable.
    if (isIVLoad(op, ivAddr)) {
      // Replace CIR IV load with scf.IV
      // (i.e. remove the load op and replace the uses of the result of the CIR
      // IV load with the scf.IV)
      rewriter->replaceOp(op, scfForOp.getInductionVar());
    }
    return mlir::WalkResult::advance();
  });

  // The body IV-loads have been replaced with the scf.for induction variable,
  // so the IV alloca + its memref.cast + initial store are now dead. We do NOT
  // erase them here. A conversion pattern must mutate only the matched op and
  // ops it created: the LLVM-23 one-shot dialect-conversion driver verifies, at
  // commit, that any erased op has no remaining uses, and the cir.for's
  // discarded cond/step regions still reference this alloca at that point.
  // The dead alloca/cast/store are removed by the cleanup run after
  // applyPartialConversion (see ConvertCIRToMLIRPass::runOnOperation).
}

void SCFLoop::transformToSCFWhileOp() {
  auto scfWhileOp = mlir::scf::WhileOp::create(
      *rewriter, forOp->getLoc(), forOp->getResultTypes(), mlir::ValueRange());
  rewriter->createBlock(&scfWhileOp.getBefore());
  rewriter->createBlock(&scfWhileOp.getAfter());

  rewriter->inlineBlockBefore(&forOp.getCond().front(),
                              scfWhileOp.getBeforeBody(),
                              scfWhileOp.getBeforeBody()->end());
  rewriter->inlineBlockBefore(&forOp.getBody().front(),
                              scfWhileOp.getAfterBody(),
                              scfWhileOp.getAfterBody()->end());
  // There will be a yield after the `for` body.
  // We should delete it.
  auto yield = mlir::cast<YieldOp>(scfWhileOp.getAfterBody()->back());
  rewriter->eraseOp(yield);

  rewriter->inlineBlockBefore(&forOp.getStep().front(),
                              scfWhileOp.getAfterBody(),
                              scfWhileOp.getAfterBody()->end());
}

void SCFLoop::transformToCIRWhileOp() {
  auto cirWhileOp = cir::WhileOp::create(
      *rewriter, forOp->getLoc(), forOp->getResultTypes(), mlir::ValueRange());
  rewriter->createBlock(&cirWhileOp.getCond());
  rewriter->createBlock(&cirWhileOp.getBody());

  mlir::Block &condFront = cirWhileOp.getCond().front();
  rewriter->inlineBlockBefore(&forOp.getCond().front(), &condFront,
                              condFront.end());

  mlir::Block &bodyFront = cirWhileOp.getBody().front();
  rewriter->inlineBlockBefore(&forOp.getBody().front(), &bodyFront,
                              bodyFront.end());

  // The last operation of `bodyFront` must be a terminator.
  // We need to place the step just before it.
  rewriter->inlineBlockBefore(&forOp.getStep().front(), &bodyFront,
                              --bodyFront.end());
  // This also introduces another terminator before the one of the body.
  // We need to erase it.
  auto &stepTerminator = *--bodyFront.end();
  rewriter->eraseOp(&stepTerminator);
}

mlir::scf::WhileOp SCFWhileLoop::transferToSCFWhileOp() {
  auto scfWhileOp = mlir::scf::WhileOp::create(*rewriter, whileOp->getLoc(),
                                               whileOp->getResultTypes(),
                                               adaptor.getOperands());
  rewriter->createBlock(&scfWhileOp.getBefore());
  rewriter->createBlock(&scfWhileOp.getAfter());
  rewriter->inlineBlockBefore(&whileOp.getCond().front(),
                              scfWhileOp.getBeforeBody(),
                              scfWhileOp.getBeforeBody()->end());
  rewriter->inlineBlockBefore(&whileOp.getBody().front(),
                              scfWhileOp.getAfterBody(),
                              scfWhileOp.getAfterBody()->end());
  return scfWhileOp;
}

void SCFDoLoop::transferToSCFWhileOp() {

  auto beforeBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::ValueRange args) {
    auto *newBlock = builder.getBlock();
    rewriter->mergeBlocks(&DoOp.getBody().front(), newBlock);
    auto *yieldOp = newBlock->getTerminator();
    rewriter->mergeBlocks(&DoOp.getCond().front(), newBlock,
                          yieldOp->getResults());
    rewriter->eraseOp(yieldOp);
  };

  auto afterBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::ValueRange args) {
    mlir::scf::YieldOp::create(*rewriter, loc, args);
  };

  mlir::scf::WhileOp::create(*rewriter, DoOp.getLoc(), DoOp->getResultTypes(),
                             adaptor.getOperands(), beforeBuilder,
                             afterBuilder);
}

// A `break` that targets THIS loop cannot be expressed structurally in scf
// without the fragile condition-flag transform, so we reject it with a clear
// diagnostic rather than emit fragile/incorrect control flow. A break enclosed
// by a nearer loop/switch belongs to that construct, not this loop. (Robust
// break support via CFG-based control-flow lowering is tracked separately.)
static bool loopHasTargetingBreak(mlir::Operation *loopOp) {
  bool found = false;
  loopOp->walk([&](BreakOp brk) {
    mlir::Operation *anc = brk->getParentOp();
    while (anc && anc != loopOp) {
      if (mlir::isa<ForOp, WhileOp, DoWhileOp, SwitchOp>(anc))
        return; // targets a nearer loop/switch, not this loop
      anc = anc->getParentOp();
    }
    if (anc == loopOp)
      found = true;
  });
  return found;
}

// A `return` anywhere inside a loop body would be lowered to a `func.return`
// nested inside an `scf` region, which is illegal (func.return must be parented
// by func.func). The `--cir-lower-return` pre-pass rewrites such nested returns
// into single-exit form before this lowering runs; if one still reaches here
// (e.g. a shape that pass declined), reject it honestly rather than emit
// malformed IR. This mirrors the return-inside-`switch`-case reject below.
static bool loopContainsReturn(mlir::Operation *loopOp) {
  bool found = false;
  loopOp->walk([&](cir::ReturnOp) { found = true; });
  return found;
}

// A `continue` targeting THIS loop. Unlike `break`, `continue` skips an
// enclosing `switch` (it targets the nearest enclosing loop), so switch is not
// in the shadowing set.
static bool loopHasTargetingContinue(mlir::Operation *loopOp) {
  bool found = false;
  loopOp->walk([&](ContinueOp cont) {
    mlir::Operation *anc = cont->getParentOp();
    while (anc && anc != loopOp) {
      if (mlir::isa<ForOp, WhileOp, DoWhileOp>(anc))
        return; // targets a nearer loop, not this one
      anc = anc->getParentOp();
    }
    if (anc == loopOp)
      found = true;
  });
  return found;
}

// True if this loop is enclosed by another (cir or scf) loop. The continue
// condition-flag machinery is only robust for a standalone flat loop; nested
// loops crash the one-shot conversion driver, so we reject them (see status).
static bool opIsNestedInLoop(mlir::Operation *loopOp) {
  for (mlir::Operation *p = loopOp->getParentOp(); p; p = p->getParentOp())
    if (mlir::isa<ForOp, WhileOp, DoWhileOp, mlir::scf::ForOp,
                  mlir::scf::WhileOp>(p))
      return true;
  return false;
}

class CIRForOpLowering : public mlir::OpConversionPattern<cir::ForOp> {
public:
  using OpConversionPattern<cir::ForOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ForOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (loopContainsReturn(op))
      return op.emitError(
          "ThroughMLIR: 'return' inside a loop is not supported by the "
          "CIR-to-MLIR lowering; it must be rewritten into single-exit form by "
          "the '--cir-lower-return' pre-pass");
    if (loopHasTargetingBreak(op))
      return op.emitError(
          "ThroughMLIR: 'break' inside a loop is not yet supported by the "
          "CIR-to-MLIR lowering; rewrite the loop to avoid 'break' (e.g. with a "
          "flag/condition)");
    // A 'continue' in a 'for' loop must STILL execute the loop step (e.g. i++):
    // C semantics jump to the step, then the condition. The current scf-based
    // lowering (transformToCIRWhileOp appends the step to the body end, then
    // rewriteContinue guards everything after the continue) incorrectly skips
    // the step on 'continue' -> the induction variable never advances ->
    // infinite loop. Reject it honestly rather than miscompile. (while/do
    // 'continue' is correct -- they have no separate step region. Robust
    // for+continue support via CFG-based control-flow lowering is tracked
    // separately and intentionally deferred.)
    if (loopHasTargetingContinue(op))
      return op.emitError(
          "ThroughMLIR: 'continue' inside a 'for' loop is not yet supported by "
          "the CIR-to-MLIR lowering (the loop step would be skipped on "
          "'continue', causing an infinite loop); rewrite as a 'while' loop "
          "with an explicit step, or avoid 'continue'");
    SCFLoop loop(op, &rewriter);
    loop.analysis();
    // Breaks and continues are handled in lowering of cir::WhileOp.
    // We can reuse the code by transforming this ForOp into WhileOp.
    if (loop.hasBreakOrContinue()) {
      loop.transformToCIRWhileOp();
      rewriter.eraseOp(op);
      return mlir::success();
    }

    if (!loop.isCanonical()) {
      loop.transformToSCFWhileOp();
      rewriter.eraseOp(op);
      return mlir::success();
    }

    loop.transferToSCFForOp();
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CIRWhileOpLowering : public mlir::OpConversionPattern<cir::WhileOp> {
  void rewriteContinueInIf(cir::IfOp ifOp, cir::ContinueOp continueOp,
                           mlir::scf::WhileOp whileOp,
                           mlir::ConversionPatternRewriter &rewriter) const {
    auto loc = ifOp->getLoc();

    rewriter.setInsertionPointToStart(whileOp.getAfterBody());
    auto boolTy = rewriter.getType<BoolType>();
    auto boolPtrTy = cir::PointerType::get(boolTy);
    auto alignment = rewriter.getI64IntegerAttr(4);
    auto condAlloca =
        AllocaOp::create(rewriter, loc, boolPtrTy, "condition", alignment);

    rewriter.setInsertionPoint(ifOp);
    auto negated = NotOp::create(rewriter, loc, boolTy, ifOp.getCondition());
    StoreOp::create(rewriter, loc, negated, condAlloca);

    // On each layer, surround everything after runner in its parent with a
    // guard: `if (!condAlloca)`.
    for (mlir::Operation *runner = ifOp; runner != whileOp;
         runner = runner->getParentOp()) {
      rewriter.setInsertionPointAfter(runner);
      auto cond = LoadOp::create(
          rewriter, loc, boolTy, condAlloca, /*isDeref=*/false,
          /*is_volatile=*/false, alignment,
          /*sync_scope=*/cir::SyncScopeKindAttr{},
          /*mem_order=*/cir::MemOrderAttr{});
      auto ifnot = IfOp::create(rewriter, loc, cond, /*withElseRegion=*/false,
                                [&](mlir::OpBuilder &, mlir::Location) {
                                  /* Intentionally left empty */
                                });

      auto &region = ifnot.getThenRegion();
      rewriter.setInsertionPointToEnd(&region.back());
      auto terminator = YieldOp::create(rewriter, loc);

      bool inserted = false;
      for (mlir::Operation *op = ifnot->getNextNode(); op;) {
        // Don't move terminators in.
        if (isa<YieldOp>(op) || isa<ReturnOp>(op))
          break;

        mlir::Operation *next = op->getNextNode();
        op->moveBefore(terminator);
        op = next;
        inserted = true;
      }
      // Don't retain `if (!condAlloca)` when it's empty.
      if (!inserted)
        rewriter.eraseOp(ifnot);
    }
    rewriter.setInsertionPoint(continueOp);
    mlir::scf::YieldOp::create(rewriter, continueOp->getLoc());
    rewriter.eraseOp(continueOp);
  }

  void rewriteContinue(mlir::scf::WhileOp whileOp,
                       mlir::ConversionPatternRewriter &rewriter) const {
    // Collect all ContinueOp inside this while.
    llvm::SmallVector<cir::ContinueOp> continues;
    whileOp->walk([&](mlir::Operation *op) {
      if (auto continueOp = dyn_cast<ContinueOp>(op))
        continues.push_back(continueOp);
    });

    if (continues.empty())
      return;

    for (auto continueOp : continues) {
      bool nested = false;
      // When there is another loop between this WhileOp and the ContinueOp,
      // we shouldn't change that loop instead.
      for (mlir::Operation *parent = continueOp->getParentOp();
           parent != whileOp; parent = parent->getParentOp()) {
        if (isa<WhileOp>(parent)) {
          nested = true;
          break;
        }
      }
      if (nested)
        continue;

      // When the ContinueOp is under an IfOp, a direct replacement of
      // `scf.yield` won't work: the yield would jump out of that IfOp instead.
      // We might need to change the WhileOp itself to achieve the same effect.
      bool rewritten = false;
      for (mlir::Operation *parent = continueOp->getParentOp();
           parent != whileOp; parent = parent->getParentOp()) {
        if (auto ifOp = dyn_cast<cir::IfOp>(parent)) {
          rewriteContinueInIf(ifOp, continueOp, whileOp, rewriter);
          rewritten = true;
          break;
        }
      }
      if (rewritten)
        continue;

      // Operations after this ContinueOp has to be removed.
      for (mlir::Operation *runner = continueOp->getNextNode(); runner;) {
        mlir::Operation *next = runner->getNextNode();
        runner->erase();
        runner = next;
      }

      // Blocks after this ContinueOp also has to be removed.
      for (mlir::Block *block = continueOp->getBlock()->getNextNode(); block;) {
        mlir::Block *next = block->getNextNode();
        block->erase();
        block = next;
      }
    }
  }

public:
  using OpConversionPattern<cir::WhileOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::WhileOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (loopContainsReturn(op))
      return op.emitError(
          "ThroughMLIR: 'return' inside a loop is not supported by the "
          "CIR-to-MLIR lowering; it must be rewritten into single-exit form by "
          "the '--cir-lower-return' pre-pass");
    if (loopHasTargetingBreak(op))
      return op.emitError(
          "ThroughMLIR: 'break' inside a loop is not yet supported by the "
          "CIR-to-MLIR lowering; rewrite the loop to avoid 'break' (e.g. with a "
          "flag/condition)");
    if (loopHasTargetingContinue(op) && opIsNestedInLoop(op))
      return op.emitError(
          "ThroughMLIR: 'continue' in a nested loop is not yet supported by the "
          "CIR-to-MLIR lowering; flatten the loop nesting or rewrite without "
          "'continue'");
    SCFWhileLoop loop(op, adaptor, &rewriter);
    auto whileOp = loop.transferToSCFWhileOp();
    rewriteContinue(whileOp, rewriter);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CIRDoOpLowering : public mlir::OpConversionPattern<cir::DoWhileOp> {
public:
  using OpConversionPattern<cir::DoWhileOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::DoWhileOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (loopContainsReturn(op))
      return op.emitError(
          "ThroughMLIR: 'return' inside a loop is not supported by the "
          "CIR-to-MLIR lowering; it must be rewritten into single-exit form by "
          "the '--cir-lower-return' pre-pass");
    if (loopHasTargetingBreak(op))
      return op.emitError(
          "ThroughMLIR: 'break' inside a loop is not yet supported by the "
          "CIR-to-MLIR lowering; rewrite the loop to avoid 'break' (e.g. with a "
          "flag/condition)");
    // Reject ANY 'continue' targeting this do-while (not just nested ones).
    // Unlike CIRWhileOpLowering, the do path does not run rewriteContinue:
    // SCFDoLoop::transferToSCFWhileOp merges the do body+cond into the scf.while
    // "before" region, so the while-style rewriteContinue (which assumes the
    // body is the scf "after" region) cannot be reused. Lowering it anyway would
    // leave a stray cir.continue that fails the op verifier once the cir.do is
    // erased. Reject honestly instead (consistent with 'for'+continue); flat
    // while+continue remains supported.
    if (loopHasTargetingContinue(op))
      return op.emitError(
          "ThroughMLIR: 'continue' inside a 'do-while' loop is not yet "
          "supported by the CIR-to-MLIR lowering; rewrite as a 'while' loop "
          "with an explicit condition, or avoid 'continue'");
    SCFDoLoop loop(op, adaptor, &rewriter);
    loop.transferToSCFWhileOp();
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CIRConditionOpLowering
    : public mlir::OpConversionPattern<cir::ConditionOp> {
public:
  using OpConversionPattern<cir::ConditionOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(cir::ConditionOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *parentOp = op->getParentOp();
    return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(parentOp)
        .Case<mlir::scf::WhileOp>([&](auto) {
          rewriter.replaceOpWithNewOp<mlir::scf::ConditionOp>(
              op, adaptor.getCondition(), parentOp->getOperands());
          return mlir::success();
        })
        .Default([](auto) { return mlir::failure(); });
  }
};

//===----------------------------------------------------------------------===//
// Switch lowering: cir.switch -> scf.index_switch
//===----------------------------------------------------------------------===//

// True if `brk` is a `break` whose nearest enclosing loop/switch is `sw` (i.e.
// it targets this switch, not an inner loop/switch nested inside a case).
static bool breakTargetsSwitch(cir::BreakOp brk, cir::SwitchOp sw) {
  for (mlir::Operation *anc = brk->getParentOp(); anc; anc = anc->getParentOp()) {
    if (anc == sw.getOperation())
      return true;
    if (mlir::isa<ForOp, WhileOp, DoWhileOp, SwitchOp>(anc))
      return false; // a nearer loop/switch owns this break
  }
  return false;
}

class CIRSwitchOpLowering : public mlir::OpConversionPattern<cir::SwitchOp> {
public:
  using OpConversionPattern<cir::SwitchOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::SwitchOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    if (op->getNumResults() != 0)
      return op.emitError("ThroughMLIR: 'switch' producing values is not "
                          "supported by the CIR-to-MLIR lowering");

    // Only the "simple form" (all cases directly in the switch body, no labels,
    // no code before the first case, no goto) can be lowered structurally.
    llvm::SmallVector<cir::CaseOp> cases;
    if (!op.isSimpleForm(cases))
      return op.emitError(
          "ThroughMLIR: non-simple 'switch' (labels, 'goto', a 'range' case, or "
          "code before the first case) is not supported by the CIR-to-MLIR "
          "lowering");

    // `cases` is in program order (collectCases walks the body pre-order).
    // Validate every case up front so we never emit partial IR.
    for (auto caseOp : cases) {
      if (caseOp.getKind() == cir::CaseOpKind::Range)
        return op.emitError("ThroughMLIR: 'case' ranges (GNU extension) are not "
                            "supported by the CIR-to-MLIR lowering");
      mlir::Block &b = caseOp.getCaseRegion().front();
      // A `return` inside a case would become a func.return nested in an scf
      // region, which is illegal; reject honestly rather than miscompile.
      for (mlir::Operation &inner : b)
        if (mlir::isa<cir::ReturnOp>(inner))
          return op.emitError("ThroughMLIR: 'return' inside a 'switch' case is "
                              "not supported by the CIR-to-MLIR lowering");
      // A break that targets this switch but is nested inside control flow (e.g.
      // `if (...) break;`) needs the guard-climb transform; not handled yet.
      // (A break directly in the case block, or one owned by an inner loop, is
      // fine.)
      bool nestedBreak = false;
      caseOp.getCaseRegion().walk([&](cir::BreakOp brk) {
        if (breakTargetsSwitch(brk, op) && brk->getParentOp() != caseOp)
          nestedBreak = true;
      });
      if (nestedBreak)
        return op.emitError(
            "ThroughMLIR: 'break' nested inside control flow within a 'switch' "
            "case is not yet supported by the CIR-to-MLIR lowering; use a "
            "top-level 'break' in each case");
    }

    // Build parallel arrays: for each distinct case value, the index into
    // `cases` where its fall-through chain starts. `anyof` contributes several
    // values that all start at the same case. `default` is tracked separately.
    llvm::SmallVector<int64_t> caseValues;
    llvm::SmallVector<unsigned> caseStart;
    int defaultStart = -1;
    for (unsigned i = 0; i < cases.size(); ++i) {
      cir::CaseOp c = cases[i];
      if (c.getKind() == cir::CaseOpKind::Default) {
        defaultStart = static_cast<int>(i);
        continue;
      }
      for (mlir::Attribute a : c.getValue()) {
        auto intAttr = mlir::cast<cir::IntAttr>(a);
        caseValues.push_back(intAttr.getValue().getSExtValue());
        caseStart.push_back(i);
      }
    }

    // Cast the (already type-converted) integer condition to `index`.
    mlir::Value condInt = adaptor.getCondition();
    bool isSigned = true;
    if (auto cirIntTy =
            mlir::dyn_cast<cir::IntType>(op.getCondition().getType()))
      isSigned = cirIntTy.isSigned();
    mlir::Value arg;
    auto idxTy = rewriter.getIndexType();
    if (isSigned)
      arg = mlir::arith::IndexCastOp::create(rewriter, loc, idxTy, condInt);
    else
      arg = mlir::arith::IndexCastUIOp::create(rewriter, loc, idxTy, condInt);

    auto switchOp = mlir::scf::IndexSwitchOp::create(
        rewriter, loc, mlir::TypeRange{}, arg, caseValues, caseValues.size());

    // The fall-through chain of statements executed when control enters a case:
    // the case's own ops, plus each following case's ops, up to and including
    // the first case that ends in a top-level `break` (or the last case).
    auto caseBreaks = [](cir::CaseOp c) {
      for (mlir::Operation &inner : c.getCaseRegion().front())
        if (mlir::isa<cir::BreakOp>(inner))
          return true;
      return false;
    };
    auto chainOf = [&](unsigned start) {
      llvm::SmallVector<unsigned> chain;
      for (unsigned i = start; i < cases.size(); ++i) {
        chain.push_back(i);
        if (caseBreaks(cases[i]))
          break;
      }
      return chain;
    };

    // Ordered list of arms: one per distinct case value, then the default.
    llvm::SmallVector<unsigned> armStart(caseStart.begin(), caseStart.end());
    bool hasDefault = defaultStart >= 0;
    if (hasDefault)
      armStart.push_back(static_cast<unsigned>(defaultStart));

    // Count how many arms include each case in their chain. A case body is
    // *moved* into its last consuming arm (sound under the one-shot conversion
    // driver, which cannot always reconcile clones of ops that still need
    // conversion — e.g. a nested cir.switch); earlier consumers *clone* it.
    llvm::DenseMap<unsigned, int> total, seen;
    for (unsigned s : armStart)
      for (unsigned idx : chainOf(s))
        total[idx]++;

    auto emitArm = [&](unsigned startIdx, mlir::Region &region) {
      mlir::Block *block = rewriter.createBlock(&region);
      rewriter.setInsertionPointToStart(block);
      auto yieldOp = mlir::scf::YieldOp::create(rewriter, loc);
      mlir::IRMapping map;
      for (unsigned idx : chainOf(startIdx)) {
        bool lastUse = (++seen[idx] == total[idx]);
        // Snapshot ops first: moving mutates the source block's op list.
        llvm::SmallVector<mlir::Operation *> body;
        for (mlir::Operation &inner : cases[idx].getCaseRegion().front()) {
          if (mlir::isa<cir::YieldOp>(inner))
            continue;
          if (mlir::isa<cir::BreakOp>(inner))
            break; // top-level break terminates the chain
          body.push_back(&inner);
        }
        for (mlir::Operation *inner : body) {
          if (lastUse)
            inner->moveBefore(yieldOp);
          else {
            rewriter.setInsertionPoint(yieldOp);
            rewriter.clone(*inner, map);
          }
        }
      }
    };

    for (unsigned i = 0; i < caseStart.size(); ++i)
      emitArm(caseStart[i], switchOp.getCaseRegions()[i]);

    if (hasDefault) {
      emitArm(static_cast<unsigned>(defaultStart), switchOp.getDefaultRegion());
    } else {
      // Synthesize an empty default (required by scf.index_switch).
      mlir::Block *block = rewriter.createBlock(&switchOp.getDefaultRegion());
      rewriter.setInsertionPointToStart(block);
      mlir::scf::YieldOp::create(rewriter, loc);
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

void populateCIRLoopToSCFConversionPatterns(mlir::RewritePatternSet &patterns,
                                            mlir::TypeConverter &converter) {
  patterns.add<CIRForOpLowering, CIRWhileOpLowering, CIRConditionOpLowering,
               CIRDoOpLowering, CIRSwitchOpLowering>(converter,
                                                     patterns.getContext());
}

} // namespace cir
