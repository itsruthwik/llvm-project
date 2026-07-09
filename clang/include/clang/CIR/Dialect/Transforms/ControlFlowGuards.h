//====- ControlFlowGuards.h - Shared CIR guard-climb / loop-exit utils -*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Single source of truth for the flag-guarded "structured early-exit" machinery
// shared by the three CIR->CIR pre-passes that turn an unstructured exit
// (`break`/`continue`, early `return`, cross-scope forward `goto`) into
// break-free structured CIR ahead of the one-shot `--cir-to-mlir` lowering:
//
//   * LowerBreakContinue.cpp
//   * LowerReturn.cpp
//   * FlattenScopeGoto.cpp
//
// Every one of these passes rewrites an exit into `store true, flag` + a
// guard-climb that wraps the trailing ops at each enclosing level inside
// `cir.if(!flag)`, and (for any loop the exit escapes) strengthens the loop so
// it terminates once the flag is set. That machinery used to be copy-pasted
// verbatim across all three files (with "Lifted verbatim from ..." comments),
// which is exactly why a single correctness flaw lived in triplicate. It now
// lives here once.
//
// INVARIANT (the correctness contract of all three callers): no SSA value these
// helpers create ever crosses a region boundary it does not dominate. All
// exit-state communication is through the caller's memory `flag`/`retval`
// slots; these helpers only relocate WHOLE runs of ops into fresh
// `cir.if(!flag)` / `cir.ternary` regions, so a def is never split from its use.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_TRANSFORMS_CONTROLFLOWGUARDS_H
#define CLANG_CIR_DIALECT_TRANSFORMS_CONTROLFLOWGUARDS_H

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace cir {

// The single-block cond region terminator (cir.condition) of a
// cir.for/while/do, or null if `loop` has no structured condition region.
cir::ConditionOp loopCondition(mlir::Operation *loop);

// Guard the ops that follow `runner` (within runner's block, up to the block
// terminator) inside a fresh `cir.if(!flag) { ... }`. No-op if there is nothing
// after `runner` but the terminator. `boolTy` is `!cir.bool`.
void wrapTrailing(mlir::Operation *runner, mlir::Value flagAddr,
                  mlir::Type boolTy);

// Guard every op in `block` (except its terminator) inside `cir.if(!flag)`.
// Used to skip a `for` step region once the exit flag is set.
void guardBlock(mlir::Block *block, mlir::Value flagAddr, mlir::Type boolTy,
                mlir::Location loc);

// Guard-climb from `anchor` up to AND INCLUDING `stopBlock`, wrapping the
// trailing ops at each ancestor level in `cir.if(!flag)`. `anchor` is the
// flag-store op left in place of the erased exit.
void climbGuard(mlir::Operation *anchor, mlir::Value flagAddr, mlir::Type boolTy,
                mlir::Block *stopBlock);

// Strengthen `loop` so it terminates once `flag` is set. The original loop
// condition C is re-evaluated LAZILY -- only on the iterations where the flag is
// still clear -- so any side effect in C (`while(n-- > 0)`, `while(*p++)`, a
// call) fires exactly as often as in real C, never one extra time after the
// exit. For a `for` loop the step region is additionally skipped when the flag
// is set. On a loop that has no structured condition region to strengthen,
// emits `noCondMsg` on `loop` and returns failure.
//
// Composes: calling this more than once on the same loop (e.g. a loop enclosing
// gotos to two labels, or a loop escaped by both a return and a goto) nests the
// laziness so C is evaluated only when EVERY exit flag is clear.
mlir::LogicalResult strengthenLoopExit(mlir::Operation *loop,
                                       mlir::Value flagAddr, mlir::Type boolTy,
                                       llvm::StringRef noCondMsg);

} // namespace cir

#endif // CLANG_CIR_DIALECT_TRANSFORMS_CONTROLFLOWGUARDS_H
