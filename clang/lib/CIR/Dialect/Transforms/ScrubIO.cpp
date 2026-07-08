//====- ScrubIO.cpp - CIR printf/IO hardware scrub pass ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Commercial-HLS semantics: synthesis discards console/file OUTPUT. This CIR
// pass erases no-op output calls (printf/puts/putchar/fprintf/fputs/fputc),
// replacing any consumed i32 return value with a constant 0. The now-dead
// format-string globals are swept by the following --cir-canonicalize /
// --cir-to-mlir DCE. Erasing these calls at CIR level keeps them off the
// downstream printf-format-string lowering path (LowerCIRToMLIR), which only
// supports a narrow inline-format-string form and otherwise hard-errors.
//
// Only the C-SYNTHESIS side is scrubbed. The C reference used by the cosim
// harness is compiled/executed separately with the real libc, so printf output
// there is unaffected -- the numeric comparison is unchanged.
//
// INPUT (read) functions are NOT scrubbed: dropping a scanf/fgets/... whose
// effect (a value read from the environment into memory) is consumed would be a
// silent miscompile. Those are honestly rejected with a named diagnostic.
//
// EXTERNALITY GATE: the name lists below identify libc I/O only when the callee
// is an EXTERNAL DECLARATION (a prototype with no body in the module). A
// user-defined function that merely collides with one of these names (e.g.
// mpeg2's local `read` helper, or a hand-rolled `printf`) is ordinary code and
// is left completely untouched -- neither scrubbed nor rejected. This is the
// general mechanism; the name lists never special-case a specific program.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/TimeProfiler.h"
#include <memory>

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_SCRUBIO
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct ScrubIOPass : public impl::ScrubIOBase<ScrubIOPass> {
  ScrubIOPass() = default;
  void runOnOperation() override;
};

// Console/file OUTPUT sinks whose result (a byte/char count) is safe to drop.
static bool isOutputSink(llvm::StringRef n) {
  static const llvm::StringSet<> sinks = {
      "printf", "puts", "putchar", "fprintf", "fputs", "fputc", "putc"};
  return sinks.contains(n);
}

// Environment/file INPUT readers whose effect cannot be soundly dropped.
static bool isInputReader(llvm::StringRef n) {
  static const llvm::StringSet<> readers = {
      "scanf", "fscanf",  "sscanf", "gets",  "fgets",
      "fread", "getchar", "getc",   "fgetc", "read"};
  return readers.contains(n);
}

void ScrubIOPass::runOnOperation() {
  llvm::TimeTraceScope scope("Scrub IO");
  bool failed = false;

  getOperation()->walk([&](cir::FuncOp func) {
    if (failed)
      return;
    llvm::SmallVector<cir::CallOp> toErase;
    bool warned = false;

    func.walk([&](cir::CallOp call) {
      if (failed || call.isIndirect())
        return;
      auto callee = call.getCallee();
      if (!callee)
        return;
      llvm::StringRef name = *callee;

      // Externality gate: only libc I/O declarations (no body in the module)
      // are candidates. A user-defined function that shadows one of these
      // names has a body and is ordinary code -- leave it entirely untouched.
      auto resolved =
          mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
              call, call.getCalleeAttr());
      if (resolved && !resolved.isDeclaration())
        return;

      if (isInputReader(name)) {
        call.emitError("ThroughMLIR: input function '")
            << name
            << "' cannot be scrubbed for hardware synthesis (its result is "
               "consumed); remove the input dependency or provide it as a "
               "module input";
        failed = true;
        return;
      }

      if (!isOutputSink(name))
        return;

      if (!warned) {
        func.emitWarning()
            << func.getName() << ": '" << name
            << "' (and any other console/file output) discarded for hardware "
               "synthesis";
        warned = true;
      }

      // A consumed i32 return (byte count) becomes a constant 0.
      if (call.getNumResults() == 1 && !call.getResult().use_empty()) {
        if (auto intTy = dyn_cast<cir::IntType>(call.getResult().getType())) {
          mlir::OpBuilder b(call);
          mlir::Value zero = cir::ConstantOp::create(
              b, call.getLoc(), cir::IntAttr::get(intTy, 0));
          call.getResult().replaceAllUsesWith(zero);
        } else {
          // Non-integer consumed result: not soundly replaceable.
          call.emitError("ThroughMLIR: output function '")
              << name << "' has a consumed non-integer result; cannot scrub";
          failed = true;
          return;
        }
      }
      toErase.push_back(call);
    });

    for (auto call : toErase)
      call.erase();
  });

  if (failed)
    signalPassFailure();
}

} // namespace

std::unique_ptr<mlir::Pass> mlir::createScrubIOPass() {
  return std::make_unique<ScrubIOPass>();
}
