//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_PASSES_TD
#define CLANG_CIR_DIALECT_PASSES_TD

include "mlir/Pass/PassBase.td"

def CIRCanonicalize : Pass<"cir-canonicalize"> {
  let summary = "Performs CIR canonicalization";
  let description = [{
    Perform canonicalizations on CIR and removes some redundant operations.

    This pass performs basic cleanup and canonicalization transformations that
    are not intended to affect CIR-to-source fidelity and high-level code
    analysis passes. Example transformations performed in this pass include
    empty scope cleanup, trivial `try` cleanup, redundant branch cleanup, etc.
    Those more "heavyweight" transformations and those transformations that
    could significantly affect CIR-to-source fidelity are performed in the
    `cir-simplify` pass.
  }];

  let constructor = "mlir::createCIRCanonicalizePass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def CIRSimplify : Pass<"cir-simplify"> {
  let summary = "Performs CIR simplification and code optimization";
  let description = [{
    The pass performs semantics-preserving code simplifications and optimizations
    on CIR while maintaining strict program correctness.

    Unlike the `cir-canonicalize` pass, these transformations may reduce the IR's
    structural similarity to the original source code as a trade-off for improved
    code quality. This can affect debugging fidelity by altering intermediate
    representations of folded expressions, hoisted operations, and other
    optimized constructs.

    Example transformations include ternary expression folding and code hoisting
    while preserving program semantics.
  }];
  let constructor = "mlir::createCIRSimplifyPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def HoistAllocas : Pass<"cir-hoist-allocas"> {
  let summary = "Hoist allocas to the entry of the function";
  let description = [{
    This pass hoist all non-dynamic allocas to the entry of the function.
    This is helpful for later code generation.
  }];
  let constructor = "mlir::createHoistAllocasPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def CIRFlattenCFG : Pass<"cir-flatten-cfg"> {
  let summary = "Produces flatten CFG";
  let description = [{
    This pass transforms CIR by inlining all the nested regions. Thus,
    the following conditions are true after the pass applied:
    - there are no nested regions in any function body
    - all the blocks in a function belong to the parent region
    In other words, this pass removes such CIR operations like IfOp, LoopOp,
    ScopeOp and etc. and produces a flat CIR.
  }];
  let constructor = "mlir::createCIRFlattenCFGPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def GotoSolver : Pass<"cir-goto-solver"> {
  let summary = "Replaces goto operations with branches";
  let description = [{
    This pass transforms CIR and replaces goto-s with branch
    operations to the proper blocks.
  }];
  let constructor = "mlir::createGotoSolverPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def LowerBreakContinue : Pass<"cir-lower-break-continue"> {
  let summary = "Rewrite loop-targeting break/continue into break-free CIR";
  let description = [{
    Rewrites each `cir.for`/`cir.while`/`cir.do` that contains a loop-targeting
    `cir.break`/`cir.continue` into structured, break-free CIR using per-loop
    boolean flag allocas, `cir.if` guard-climbing, `cir.select` condition
    strengthening, and (for `break` in a `for` loop) a guarded step region so
    the step is skipped on break but still executes on continue. Runs before
    `--cir-to-mlir` so the structured loop lowering never sees break/continue.
  }];
  let constructor = "mlir::createLowerBreakContinuePass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def LowerReturn : Pass<"cir-lower-return"> {
  let summary = "Rewrite nested early returns into single-exit CIR";
  let description = [{
    Rewrites a `cir.func` that contains an early `cir.return` nested inside a
    loop, `cir.if`, or `cir.scope` into single-exit form so the downstream
    `--cir-to-mlir` lowering never has to place a `func.return` inside an `scf`
    region (which is illegal). Uses a function-scope `retval` alloca (for a
    non-void return) plus a `returning` boolean flag: each nested return stores
    its value into `retval`, sets `returning=true`, and guard-climbs the
    trailing ops at every enclosing level inside `cir.if(!returning)`; every
    enclosing loop's condition is strengthened with
    `cir.select if returning then false else C` (and a `for` step is guarded so
    it is skipped on return) so a return breaks out of ALL enclosing loops. The
    function tail becomes a single `cir.return` loading `retval`. Returns already
    at function top level are left untouched. Runs after
    `--cir-lower-break-continue` and before `--cir-to-mlir`.
  }];
  let constructor = "mlir::createLowerReturnPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def FlattenScopeGoto : Pass<"cir-flatten-scope-goto"> {
  let summary = "Flatten the single-epilogue cross-scope forward-exit goto idiom";
  let description = [{
    Rewrites the classic softfloat "roundAndPack" forward-exit `cir.goto` idiom
    into flag-guarded structured CIR, so `--cir-goto-solver` never emits an
    illegal cross-region `cir.br`. Handles only a `cir.goto` nested inside
    `cir.scope`/`cir.if`/loop regions that jumps FORWARD to a label at the top of
    a block in the function-body region where that label block is the
    unconditional `cir.br` fall-through successor of the goto's entry block (a
    single shared epilogue). Each such goto becomes a per-label `!cir.bool` flag
    plus the same guard-climb + loop condition strengthening as
    `--cir-lower-return`. Any other goto (backward, into-scope, multi-target
    ladder, unsupported) is honestly rejected with a named diagnostic; a
    top-level goto is left for `--cir-goto-solver`. Runs BEFORE
    `--cir-goto-solver` so the now dead label is swept.
  }];
  let constructor = "mlir::createFlattenScopeGotoPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def LowerUnionPunning : Pass<"cir-lower-union-punning"> {
  let summary = "Scalarize same-bit-width union allocas into single-storage "
                "load/store + bitcast (softfloat u64<->double type-punning)";
  let description = [{
    Decomposes a `cir.alloca` of a `!cir.union` whose members are ALL scalar
    (integer/float) of the SAME bit width into a single scalar-typed alloca of
    the union's first member (the shared storage). Every `cir.get_member[k]`
    access on that union is rewritten to a load/store on the shared storage,
    inserting a scalar `cir.cast(bitcast)` at each use site when the accessed
    member type differs from the storage type (the u64<->double reinterpretation
    at the heart of softfloat's `ullong_to_double`/`double_to_ullong`). Because
    all members are the same width there is no padding and the bit
    reinterpretation is exact; `--cir-to-mlir` lowers the scalar bitcast to
    `arith.bitcast`.

    SOUNDNESS: only same-bit-width scalar-member unions are handled. A union
    with mixed-width members, a non-scalar member, an escaping whole-union
    pointer, or a member address that escapes (used by anything other than a
    direct load/store) is honestly REJECTED with a named diagnostic rather than
    lowered, never silently miscompiled. Runs as a CIR pre-pass before
    `--cir-to-mlir` (a union has no memref storage model, so `cir.get_member`
    on it cannot otherwise be materialized).
  }];
  let constructor = "mlir::createLowerUnionPunningPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def LowerStructLocals : Pass<"cir-lower-struct-locals"> {
  let summary = "Scalar-replace (SROA) struct-local allocas whose only uses are "
                "field accesses into one alloca per field";
  let description = [{
    Decomposes a `cir.alloca` of a `!cir.struct` whose every use is a
    `cir.get_member` field access into one `cir.alloca` per field, replacing
    each `cir.get_member[k]` with the k-th field alloca's pointer (a pure
    pointer redirect -- no bitcast, since the field pointee type is unchanged).
    This removes the struct alloca before `--cir-to-mlir`, which has no memref
    storage model for `cir.get_member` on a struct (md_grid's scalar struct
    locals). Array fields are allowed; a whole-struct escape (the alloca used
    other than by field access) or a nested struct/union field is honestly
    REJECTED with a named diagnostic (owned by the general struct-lowering
    cluster), never silently miscompiled.
  }];
  let constructor = "mlir::createLowerStructLocalsPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def LowerPointerCursors : Pass<"cir-lower-pointer-cursors"> {
  let summary = "Lower provable pointer-cursor variables to (base, index) "
                "form before cir-to-mlir";
  let description = [{
    Implements the unified pointer-provenance model (a C pointer value =
    (base object, integer element index)) for the shapes it can PROVE:

    - A fat-pointer `cir.alloca` slot (`!cir.ptr<T>`) surviving mem2reg whose
      every use is a direct load/store and whose every stored value reduces to
      a single structurally identical root (array alloca / stable get_global /
      incoming pointer argument / struct-field pointer) plus an integer element
      offset is rewritten into an INDEX slot: loads become materialized
      `cir.get_element(root, idx)` pointers (so all consumers keep their exact
      semantics), stores become signed index arithmetic (negative strides
      supported), same-root pointer compares become integer index compares.
    - An SSA `cir.ptr_stride` chain (`*p++` sequences) that reduces to a
      concrete root is collapsed to a single `get_element(root, idx)`; such
      chains could previously only reach an untranslatable fallback lowering.

    SOUNDNESS: the pass is ADDITIVE and whole-slot-or-nothing. Any slot whose
    access set is not fully enumerable (address escape), whose stored values do
    not provably reduce (partial N-D row re-bases, element-size-changing
    casts, unknown producers), or whose provenance is otherwise unresolvable
    is LEFT UNTOUCHED for the existing downstream behaviour and its named
    rejects. A slot that provably merges pointers into different base objects
    (a different-base merge) is honestly rejected with a named diagnostic
    (OUT in v1). Cross-root pointer compares are left as pointer compares so
    the cir-to-mlir cross-allocation named reject, which owns that class,
    fires on the now-visible provenance.
  }];
  let constructor = "mlir::createLowerPointerCursorsPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def ScrubIO : Pass<"cir-scrub-io"> {
  let summary = "Discard console/file output calls for hardware synthesis";
  let description = [{
    Erases no-op output calls (printf/puts/putchar/fprintf/fputs/fputc),
    replacing any consumed i32 result with a constant 0, matching commercial
    HLS semantics. Input (read) functions whose result is consumed are honestly
    rejected instead of being silently dropped.
  }];
  let constructor = "mlir::createScrubIOPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def CXXABILowering : Pass<"cir-cxxabi-lowering", "mlir::ModuleOp"> {
  let summary = "Lower CIR according to C++ ABI requirements";
  let description = [{
    This pass lowers CIR operations and types that represent high-level C/C++
    constructs to a more fundamental form according to the target ABI
    requirements.

    See the description of the `TargetLowering` pass for the difference between
    this pass and the `TargetLowering` pass.
  }];
  let constructor = "mlir::createCXXABILoweringPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def TargetLowering : Pass<"cir-target-lowering", "mlir::ModuleOp"> {
  let summary = "Lower CIR to a target-specific form";
  let description = [{
    This pass lowers CIR operations from a target-agnostic form to a
    target-specific form without considering ABI requirements.

    CIR has three passes in its lowering pipeline that transform input CIR
    according to target-specific requirements, scheduled in the pipeline by the
    following order:

      1. The `TargetLowering` pass.
      2. The `CXXABILowering` pass.
      3. The `CallConvLowering` pass.

    The `TargetLowering` pass acts more like a legalization pass. It ensures
    every operation in CIR conforms to the target's constraints. An example
    would be the handling of synchronization scopes of atomic operations. The
    x86 family of targets only support a system-wide synchronization scope, thus
    any atomic operations with a different synchronization scope would be
    transformed to use the system-wide scope in this pass.

    The `CXXABILowering` pass and the `CallConvLowering` pass transform the
    CIR according to the target's ABI requirements. The former handles all
    ABI-related lowering except for calling convention handling, which is
    handled specifically in the latter. Example transformations that the
    `CXXABILowering` pass could make include:

      - Replace C/C++ types that have an ABI-defined layout with more
        fundamental types corresponding to the ABI requirements. For example,
        the layout of the pointer-to-data-member type in C++ is ABI-defined,
        thus the `CXXABILowering` pass would replace all occurrences of this
        type according to the ABI requirements. With a typical target ABI, it
        will be replaced by the `ptrdiff_t` type since this is the type most
        ABIs use for the layout of the pointer-to-data-member type.
      - Replace CIR operations that have ABI-dependent implementations with more
        fundamental operations. For example, the `dynamic_cast` operator in C++
        is implemented in an ABI-dependent way, typically by calling into a
        library function provided by the implementation. The `CXXABILowering`
        pass would thus replace all `cir.dyn_cast` operations with corresponding
        library function calls.

    The `CallConvLowering` pass is dedicated to handle calling conventions. It
    rewrites function signatures according to calling convention requirements,
    and updates function body and call sites accordingly. For example, when
    passing a large struct by value in C/C++, most ABIs require passing a
    pointer to a copy of the struct instead. Thus, the `CallConvLowering` pass
    would rewrite function signatures that take a value of a large struct type
    to take a pointer to the struct type instead, and update the function body
    and the call sites accordingly.
  }];
  let constructor = "mlir::createTargetLoweringPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def CIREHABILowering : Pass<"cir-eh-abi-lowering", "mlir::ModuleOp"> {
  let summary = "Lower flattened CIR EH operations to target-specific ABI form";
  let description = [{
    This pass lowers the ABI-agnostic exception handling operations produced by
    CFG flattening into an ABI-specific form. Currently only the Itanium C++
    ABI is implemented.

    For the Itanium ABI, the pass performs the following transformations:
      - Replaces `cir.eh.initiate` with `cir.eh.inflight_exception`
      - Replaces `cir.eh.dispatch` with `cir.eh.typeid` + comparison chains
      - Removes `cir.begin_cleanup` and `cir.end_cleanup` operations
      - Replaces `cir.begin_catch` with a call to `__cxa_begin_catch`
      - Replaces `cir.end_catch` with a call to `__cxa_end_catch`
      - Replaces `cir.resume` with `cir.resume.flat`
      - Sets the personality function attribute on functions that require EH

    If a non-Itanium ABI is specified, the pass emits a diagnostic indicating
    that the target is not yet implemented.
  }];
  let constructor = "mlir::createCIREHABILoweringPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def LoweringPrepare : Pass<"cir-lowering-prepare"> {
  let summary = "Lower to more fine-grained CIR operations before lowering to "
    "other dialects";
  let description = [{
    This pass does preparation work for lowering to other dialects. For example,
    it may expand the global variable initialziation in a more ABI-friendly form.
  }];
  let constructor = "mlir::createLoweringPreparePass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def IdiomRecognizer : Pass<"cir-idiom-recognizer", "mlir::ModuleOp"> {
  let summary = "Raise calls to C/C++ libraries to CIR operations";
  let description = [{
    This pass recognizes idiomatic C++ usage and captures information about C++
    standard library containers, library functions calls, and types into CIR
    operations, attributes and types.
  }];
  let constructor = "mlir::createIdiomRecognizerPass()";
  let dependentDialects = ["cir::CIRDialect"];
}

def CallConvLowering : Pass<"cir-call-conv-lowering", "mlir::ModuleOp"> {
  let summary = "Lower CIR function signatures and call sites to match target ABI";
  let description = [{
    This pass rewrites `cir.func` signatures and `cir.call` sites to match the
    target ABI's calling convention requirements (extension attributes, struct
    coercion, sret/byval indirect passing, struct flattening, etc.).

    Requirements on the input IR:

    - All C++-specific CIR types must already be lowered to ABI-relevant
      shapes.  In particular, member pointers and method types must have
      been rewritten by `CXXABILowering` so that the pass sees only the
      final ABI representation of each value.
    - The module must carry a `dlti.dl_spec` attribute so the classifier
      can query type sizes and alignments.

    Two driver modes select how each function's classification is computed:

    - `target=<name>` selects an ABI target.  Currently only `"test"` (the
      MLIR test target in `mlir/lib/ABI/Targets/Test/`) is supported.  Real
      targets (x86_64, AArch64, ...) will be added once the LLVM ABI library
      ships them.
    - `classification-attr=<name>` reads a `DictionaryAttr` named `<name>`
      from each `cir.func` and parses it via the test-target injection
      helper.  Used by tests to inject arbitrary classifications without
      depending on a real ABI target.

    Exactly one of the two options must be set.
  }];
  let constructor = "mlir::createCallConvLoweringPass()";
  let dependentDialects = ["cir::CIRDialect"];
  let options = [
    Option<"target", "target", "std::string", /*default=*/"\"\"",
           "Target whose ABI rules drive classification (currently: test)">,
    Option<"classificationAttr", "classification-attr", "std::string",
           /*default=*/"\"\"",
           "Function attribute name carrying a pre-built FunctionClassification "
           "DictionaryAttr (alternative to target=, used by tests)">,
  ];
}

#endif // CLANG_CIR_DIALECT_PASSES_TD
