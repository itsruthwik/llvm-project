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
#include "clang/CIR/Dialect/Transforms/ControlFlowGuards.h"
#include "clang/CIR/Dialect/Transforms/GotoClassify.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
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

struct FlattenScopeGotoPass
    : public impl::FlattenScopeGotoBase<FlattenScopeGotoPass> {
  FlattenScopeGotoPass() = default;
  void runOnOperation() override;

  bool failed = false;

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

  // Structurize a validated ForwardLadder (see GotoClassify.h): the function's
  // body-level label blocks form a forward acyclic chain reachable only by goto.
  // Lower it to a flag-guarded `cir.if` sequence, gated by a shared `tookGoto`
  // flag, emitted just BEFORE block0's existing return (which stays as block0's
  // terminator and is the natural "no goto taken" path). Returns false (and sets
  // `failed`) if a dominance check makes the relocation unsafe.
  //
  //   dispatch gotos in block0  =>  store flag_L=true; store tookGoto=true;
  //                                 climb-guard remaining block0 ops by tookGoto
  //   before block0's return:   cir.if(tookGoto) {
  //                               cir.if(flag_L1){ <L1 ops>; store flag_succ=true }
  //                               ... (label blocks in region/topological order)
  //                               cir.if(flag_Ln){ <Ln ops ending in cir.return> }
  //                             }
  //   original ^bbN label blocks are erased.
  bool structurizeForwardLadder(cir::FuncOp func,
                                const cir::ForwardLadderInfo &ladder) {
    auto boolTy = cir::BoolType::get(&getContext());
    mlir::Region &body = func.getBody();
    mlir::Block *bodyBlock = &body.front();
    auto loc = func.getLoc();
    mlir::Operation *term0 = bodyBlock->getTerminator();

    // Dominance safety: every value used inside a label block but defined OUTSIDE
    // it must properly dominate block0's terminator (where the epilogue lands).
    mlir::DominanceInfo dom(func);
    for (mlir::Block *lb : ladder.blocks)
      for (mlir::Operation &op : *lb)
        for (mlir::Value v : op.getOperands()) {
          mlir::Operation *def = v.getDefiningOp();
          if (!def || def->getBlock() == lb)
            continue; // func/block arg, or local to this label block
          if (!dom.properlyDominates(def, term0))
            return false; // unsafe to relocate -> caller rejects
        }

    // Spill the natural return value to a slot BEFORE any mutation. climbGuard
    // will wrap block0's tail (including this store and the value's computation)
    // into a `cir.if(!tookGoto)` guard, after which the value no longer dominates
    // block0's terminator; we then reload the slot at the tail (the alloca
    // dominates, and the tail return is reached only on the !tookGoto path where
    // the guarded store has run). Works for ANY return operand (direct SSA or a
    // load), unlike reading a pre-existing load's slot.
    mlir::Value retSlot;
    if (auto ret0 = mlir::dyn_cast<cir::ReturnOp>(term0);
        ret0 && ret0.getNumOperands()) {
      mlir::Type rty = ret0.getOperand(0).getType();
      mlir::OpBuilder ab(bodyBlock, bodyBlock->begin());
      retSlot = cir::AllocaOp::create(ab, loc, cir::PointerType::get(rty),
                                      "ladder_natret", ab.getI64IntegerAttr(1));
      mlir::OpBuilder sb(term0);
      cir::StoreOp::create(sb, loc, ret0.getOperand(0), retSlot);
    }

    // One flag per label block + a shared tookGoto flag (all init false).
    llvm::DenseMap<mlir::Block *, mlir::Value> flagFor;
    for (mlir::Block *lb : ladder.blocks) {
      auto name = mlir::cast<cir::LabelOp>(lb->front()).getLabel();
      flagFor[lb] = makeFlag(bodyBlock, boolTy, ("ladder_" + name).str(), loc);
    }
    mlir::Value tookGoto = makeFlag(bodyBlock, boolTy, "ladder_took", loc);
    llvm::DenseMap<llvm::StringRef, mlir::Block *> byName;
    for (mlir::Block *lb : ladder.blocks)
      byName[mlir::cast<cir::LabelOp>(lb->front()).getLabel()] = lb;

    // Rewrite the DISPATCH gotos (those NOT a label block's own terminator).
    llvm::SmallVector<cir::GotoOp> dispatch;
    func.walk([&](cir::GotoOp g) {
      mlir::Block *gb = g->getBlock();
      if (cir::isLabelBlock(gb) && g.getOperation() == gb->getTerminator())
        return; // chain-edge goto, handled during epilogue build
      dispatch.push_back(g);
    });
    llvm::SmallSetVector<std::pair<mlir::Operation *, mlir::Value>, 4>
        loopFlagPairs;
    llvm::SmallVector<mlir::Operation *> anchors;
    llvm::SmallVector<mlir::Block *> anchorEntry;
    for (cir::GotoOp g : dispatch) {
      cir::ScopeGotoInfo info = cir::classifyScopeGoto(g, func);
      mlir::Block *labelBlock = info.label ? info.label->getBlock() : nullptr;
      if (!labelBlock || !flagFor.count(labelBlock)) {
        failed = true;
        return false;
      }
      mlir::Block *entry = info.entryBlock ? info.entryBlock : bodyBlock;
      // Record enclosing loops BEFORE any mutation (climb reshapes the region).
      collectEnclosingLoops(g, entry, tookGoto, loopFlagPairs);
      // set flag_target=true then tookGoto=true, in place.
      mlir::OpBuilder b(g);
      mlir::Value tv = cir::ConstantOp::create(
          b, g.getLoc(), cir::BoolAttr::get(&getContext(), true));
      cir::StoreOp::create(b, g.getLoc(), tv, flagFor[labelBlock]);
      mlir::Operation *anchor = materializeGoto(g, tookGoto); // sets tookGoto
      anchors.push_back(anchor);
      anchorEntry.push_back(entry);
    }
    for (size_t i = 0; i < anchors.size(); ++i)
      cir::climbGuard(anchors[i], tookGoto, boolTy, anchorEntry[i]);

    // Build the epilogue inside cir.if(tookGoto), just before block0's return.
    mlir::OpBuilder b(term0);
    mlir::Value tg = cir::LoadOp::create(b, loc, boolTy, tookGoto);
    auto ifTook = cir::IfOp::create(b, loc, tg, /*withElseRegion=*/false,
                                    [](mlir::OpBuilder &, mlir::Location) {});
    mlir::Block &tookThen = ifTook.getThenRegion().back();
    mlir::OpBuilder tb(&tookThen, tookThen.end());
    mlir::Operation *tookYield = cir::YieldOp::create(tb, loc);

    for (mlir::Block *lb : ladder.blocks) {
      mlir::OpBuilder gb(&tookThen, tookYield->getIterator());
      mlir::Value fl = cir::LoadOp::create(gb, loc, boolTy, flagFor[lb]);
      auto ifL = cir::IfOp::create(gb, loc, fl, /*withElseRegion=*/false,
                                   [](mlir::OpBuilder &, mlir::Location) {});
      mlir::Block &thenL = ifL.getThenRegion().back();
      mlir::OpBuilder lb2(&thenL, thenL.end());
      mlir::Operation *lYield = cir::YieldOp::create(lb2, loc);

      // Clone L's ops (except leading cir.label and trailing terminator) into
      // the guard, remapping intra-block SSA; external defs dominate.
      mlir::IRMapping map;
      mlir::Operation *term = lb->getTerminator();
      for (mlir::Operation &op : *lb) {
        if (mlir::isa<cir::LabelOp>(op) || &op == term)
          continue;
        lb2.setInsertionPoint(lYield);
        lb2.clone(op, map);
      }
      // Chain edge: set successor flag; return: clone the return.
      mlir::Block *succ = ladder.successor.lookup(lb);
      lb2.setInsertionPoint(lYield);
      if (succ) {
        mlir::Value tv = cir::ConstantOp::create(
            lb2, loc, cir::BoolAttr::get(&getContext(), true));
        cir::StoreOp::create(lb2, loc, tv, flagFor[succ]);
      } else if (auto ret = mlir::dyn_cast<cir::ReturnOp>(term)) {
        lb2.clone(*ret.getOperation(), map);
        lYield->erase(); // the cloned return terminates the guard region
      }
    }

    // block0's natural `cir.return` is on the "no goto taken" path. climbGuard
    // wrapped the ops computing its value into `cir.if(!tookGoto)`, so the value
    // no longer dominates the block terminator. Relocate the return INTO that
    // guard (as a nested early return, which lower-return consolidates) and give
    // block0 a `cir.unreachable` terminator (every real path returns earlier).
    if (retSlot) {
      auto ret = mlir::cast<cir::ReturnOp>(term0);
      mlir::Value rv = ret.getOperand(0);
      mlir::OpBuilder tb(term0);
      mlir::Value reload = cir::LoadOp::create(tb, loc, rv.getType(), retSlot);
      ret.setOperand(0, reload); // read the spilled value at the (dominating) tail
    }

    // Erase the now-dead original label blocks (terminators first to drop edges).
    for (mlir::Block *lb : ladder.blocks)
      while (!lb->empty())
        lb->back().erase();
    for (mlir::Block *lb : ladder.blocks)
      lb->erase();

    // Strengthen loops that enclosed a dispatch goto (break out toward epilogue).
    for (auto &lf : loopFlagPairs)
      if (mlir::failed(cir::strengthenLoopExit(
              lf.first, lf.second, boolTy,
              "ThroughMLIR: loop enclosing a cross-scope 'goto' has no "
              "structured condition region to strengthen"))) {
        failed = true;
        return false;
      }
    return true;
  }

  void processFunc(cir::FuncOp func) {
    if (failed)
      return;
    mlir::Region &body = func.getBody();
    if (body.empty())
      return;

    // A whole-function forward acyclic label chain (softfloat subFloat64Sigs) is
    // structurized in one shot; its dispatch gotos would each classify as
    // ForwardLadder. Handle it before the per-goto CleanForwardExit path.
    cir::ForwardLadderInfo ladder = cir::analyzeForwardLadder(func);
    if (ladder.valid) {
      // The dominance guard inside structurizeForwardLadder runs BEFORE any
      // mutation; a false result means the relocation is unsafe, so reject
      // honestly (no half-transformed IR is left behind).
      if (!structurizeForwardLadder(func, ladder)) {
        func.emitError("ThroughMLIR: unsupported 'goto' (")
            << cir::scopeGotoClassName(cir::ScopeGotoClass::MultiTargetLadder)
            << "): a value used across the label chain does not dominate the "
               "structurized epilogue";
        failed = true;
      }
      return;
    }

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
      cir::climbGuard(anchors[i], anchorFlag[i], boolTy, anchorEntry[i]);

    // Strengthen each enclosing loop for every flag whose goto it encloses (a
    // loop shared by two labels is strengthened for both -- composed lazy
    // ternaries). Shared with LowerBreakContinue / LowerReturn.
    for (auto &lf : loopFlagPairs) {
      if (mlir::failed(cir::strengthenLoopExit(
              lf.first, lf.second, boolTy,
              "ThroughMLIR: loop enclosing a cross-scope 'goto' has no "
              "structured condition region to strengthen"))) {
        failed = true;
        return;
      }
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
