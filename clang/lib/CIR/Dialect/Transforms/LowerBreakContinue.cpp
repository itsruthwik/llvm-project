//====- LowerBreakContinue.cpp - CIR break/continue pre-pass ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites each cir.for/while/do that contains a loop-targeting cir.break or
// cir.continue into *break-free* structured CIR, so the downstream one-shot
// --cir-to-mlir lowering (LowerCIRLoopToSCF) never has to reason about
// break/continue inside a dialect-conversion driver. This is the "Option B"
// design settled in the wave-4 break/continue note: a normal CIR->CIR rewrite
// ahead of the structured lowering, exactly like GotoSolver / FlattenCFG.
//
// Per target loop that contains >=1 targeting break/continue:
//   * one `brk` and/or one `cont` !cir.bool alloca (one per loop, not per site);
//   * `break`  => set brk=true, guard-climb the trailing ops in `if(!brk)`,
//                 strengthen the loop condition with
//                     %c = cir.select if %brk then false else %C
//                 and (for a `for` loop) wrap the STEP region body in
//                 `if(!brk)` so the step is SKIPPED on break;
//   * `continue` => reset cont=false at the top of the body each iteration,
//                 set cont=true at the site, guard-climb the trailing ops in
//                 `if(!cont)`. The condition and (for a `for` loop) the STEP
//                 region are LEFT UNTOUCHED so the step still EXECUTES on
//                 continue -- this asymmetry (step skipped on break, executed
//                 on continue) is the historical for+continue miscompile (R-3).
//
// Loops are processed innermost-first so a break/continue is always rewritten
// against exactly its own target loop's flag; nested loops compose. A `break`
// nested in a `switch` targets the switch (P3's CIRSwitchOpLowering consumes
// it) and is left alone here. A `continue` that would have to climb out of a
// `cir.switch` is honestly rejected (no benchmark needs it).
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/ControlFlowGuards.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/TimeProfiler.h"
#include <memory>

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_LOWERBREAKCONTINUE
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

// The loop body region of a cir.for/while/do.
static mlir::Region *loopBodyRegion(mlir::Operation *loop) {
  if (auto f = dyn_cast<cir::ForOp>(loop))
    return &f.getBody();
  if (auto w = dyn_cast<cir::WhileOp>(loop))
    return &w.getBody();
  if (auto d = dyn_cast<cir::DoWhileOp>(loop))
    return &d.getBody();
  return nullptr;
}

// Does `brk` target `loop` (nearest enclosing loop, stopping at a switch)?
static bool breakTargets(cir::BreakOp brk, mlir::Operation *loop) {
  for (mlir::Operation *a = brk->getParentOp(); a; a = a->getParentOp()) {
    if (a == loop)
      return true;
    if (isa<cir::ForOp, cir::WhileOp, cir::DoWhileOp, cir::SwitchOp>(a))
      return false;
  }
  return false;
}

// Does `cont` target `loop` (nearest enclosing loop; a switch does NOT shadow)?
static bool continueTargets(cir::ContinueOp cont, mlir::Operation *loop) {
  for (mlir::Operation *a = cont->getParentOp(); a; a = a->getParentOp()) {
    if (a == loop)
      return true;
    if (isa<cir::ForOp, cir::WhileOp, cir::DoWhileOp>(a))
      return false;
  }
  return false;
}

// Does the climb from `op` up to `loop` cross a cir.switch case region? Such a
// continue (continue-inside-switch-inside-loop) needs a guard-climb through the
// switch boundary; we honestly reject it (no benchmark exercises it).
static bool climbsThroughSwitch(mlir::Operation *op, mlir::Operation *loop) {
  for (mlir::Operation *a = op->getParentOp(); a && a != loop;
       a = a->getParentOp())
    if (isa<cir::SwitchOp>(a))
      return true;
  return false;
}

struct LowerBreakContinuePass
    : public impl::LowerBreakContinueBase<LowerBreakContinuePass> {
  LowerBreakContinuePass() = default;
  void runOnOperation() override;

  bool failed = false;

  // Replace one break/continue op with `store true, flag` and guard-climb every
  // op after it (at each ancestor level up to the loop body) inside `if(!flag)`.
  void rewriteSite(mlir::Operation *site, mlir::Value flagAddr,
                   mlir::Type boolTy, mlir::Block *bodyBlock) {
    mlir::OpBuilder b(site);
    auto loc = site->getLoc();
    mlir::Value trueVal =
        cir::ConstantOp::create(b, loc, cir::BoolAttr::get(&getContext(), true));
    cir::StoreOp::create(b, loc, trueVal, flagAddr);
    mlir::Block *siteBlock = site->getBlock();
    mlir::Operation *store = site->getPrevNode();
    site->erase();
    // The break/continue was the block terminator; re-terminate the block.
    if (siteBlock->empty() ||
        !siteBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::OpBuilder tb(siteBlock, siteBlock->end());
      cir::YieldOp::create(tb, loc);
    }
    // Climb from the store up to the loop body block, guarding trailing ops.
    cir::climbGuard(store, flagAddr, boolTy, bodyBlock);
  }

  // Create a `!cir.bool` alloca initialised to false immediately before `loop`.
  mlir::Value makeFlag(mlir::Operation *loop, mlir::Type boolTy,
                       llvm::StringRef name) {
    mlir::OpBuilder b(loop);
    auto loc = loop->getLoc();
    auto boolPtrTy = cir::PointerType::get(boolTy);
    auto align = b.getI64IntegerAttr(1);
    mlir::Value addr = cir::AllocaOp::create(b, loc, boolPtrTy, name, align);
    mlir::Value falseVal = cir::ConstantOp::create(
        b, loc, cir::BoolAttr::get(&getContext(), false));
    cir::StoreOp::create(b, loc, falseVal, addr);
    return addr;
  }

  void processLoop(mlir::Operation *loop) {
    if (failed)
      return;
    auto boolTy = cir::BoolType::get(&getContext());
    mlir::Region *body = loopBodyRegion(loop);
    if (!body || body->empty())
      return;

    // Collect targeting continue/break sites (whole subtree, filtered).
    llvm::SmallVector<cir::ContinueOp> continues;
    llvm::SmallVector<cir::BreakOp> breaks;
    loop->walk([&](cir::ContinueOp c) {
      if (continueTargets(c, loop))
        continues.push_back(c);
    });
    loop->walk([&](cir::BreakOp brk) {
      if (breakTargets(brk, loop))
        breaks.push_back(brk);
    });
    if (continues.empty() && breaks.empty())
      return;

    // Only structured single-block loop bodies are handled; an unstructured
    // (multi-block) body (e.g. goto-solver left it non-structured) is rejected
    // rather than silently mis-lowered.
    if (!body->hasOneBlock()) {
      loop->emitError("ThroughMLIR: break/continue in a non-structured "
                      "(multi-block) loop body is not supported by the "
                      "CIR-to-MLIR lowering");
      failed = true;
      return;
    }
    mlir::Block *bodyBlock = &body->front();

    // continue-inside-switch-inside-loop crosses a switch case boundary: reject.
    for (auto c : continues)
      if (climbsThroughSwitch(c, loop)) {
        c.emitError("ThroughMLIR: 'continue' inside a 'switch' inside a loop is "
                    "not supported by the CIR-to-MLIR lowering");
        failed = true;
        return;
      }

    // --- continue first (body-local flag, condition/step untouched) ---
    if (!continues.empty()) {
      mlir::Value cont = makeFlag(loop, boolTy, "cont");
      // Reset cont=false at the very top of the body, every iteration.
      {
        mlir::OpBuilder b(bodyBlock, bodyBlock->begin());
        auto loc = loop->getLoc();
        mlir::Value falseVal = cir::ConstantOp::create(
            b, loc, cir::BoolAttr::get(&getContext(), false));
        cir::StoreOp::create(b, loc, falseVal, cont);
      }
      for (auto c : continues)
        rewriteSite(c, cont, boolTy, bodyBlock);
    }

    // --- break (loop-scope flag; condition strengthened; for-step guarded) ---
    if (!breaks.empty()) {
      mlir::Value brk = makeFlag(loop, boolTy, "brk");
      for (auto brkOp : breaks)
        rewriteSite(brkOp, brk, boolTy, bodyBlock);

      // Strengthen the loop condition so the loop terminates once brk is set --
      // re-evaluating the original condition lazily (side-effect-once) -- and
      // (for a `for`) skip the step on break. Shared with LowerReturn /
      // FlattenScopeGoto.
      if (mlir::failed(cir::strengthenLoopExit(
              loop, brk, boolTy,
              "ThroughMLIR: loop with 'break' has no structured condition "
              "region to strengthen"))) {
        failed = true;
        return;
      }
    }
  }
};

void LowerBreakContinuePass::runOnOperation() {
  llvm::TimeTraceScope scope("Lower Break/Continue");
  getOperation()->walk([&](cir::FuncOp func) {
    // Collect loops in pre-order, then process innermost-first (reverse).
    llvm::SmallVector<mlir::Operation *> loops;
    func.walk([&](mlir::Operation *op) {
      if (isa<cir::ForOp, cir::WhileOp, cir::DoWhileOp>(op))
        loops.push_back(op);
    });
    for (auto it = loops.rbegin(), e = loops.rend(); it != e; ++it) {
      processLoop(*it);
      if (failed)
        break;
    }
  });
  if (failed)
    signalPassFailure();
}

} // namespace

std::unique_ptr<mlir::Pass> mlir::createLowerBreakContinuePass() {
  return std::make_unique<LowerBreakContinuePass>();
}
