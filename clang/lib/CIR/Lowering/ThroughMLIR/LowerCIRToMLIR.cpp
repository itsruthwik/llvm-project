//====- LowerCIRToMLIR.cpp - Lowering from CIR to MLIR --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of CIR operations to MLIR.
//
//===----------------------------------------------------------------------===//

#include "LowerToMLIRHelpers.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrAttrs.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Interfaces/CIRLoopOpInterface.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/CIR/LowerToMLIR.h"
#include "clang/CIR/LoweringHelpers.h"
#include "clang/CIR/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeProfiler.h"

using namespace cir;
using namespace llvm;

namespace cir {

class CIRReturnLowering : public mlir::OpConversionPattern<cir::ReturnOp> {
public:
  using OpConversionPattern<cir::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op,
                                                      adaptor.getOperands());
    return mlir::LogicalResult::success();
  }
};

struct ConvertCIRToMLIRPass
    : public mlir::PassWrapper<ConvertCIRToMLIRPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::BuiltinDialect, mlir::func::FuncDialect,
                    mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                    mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                    mlir::scf::SCFDialect, mlir::math::MathDialect,
                    mlir::ptr::PtrDialect, mlir::vector::VectorDialect,
                    mlir::LLVM::LLVMDialect, mlir::gpu::GPUDialect>();
  }
  void runOnOperation() final;

  StringRef getDescription() const override {
    return "Convert the CIR dialect module to MLIR standard dialects";
  }

  StringRef getArgument() const override { return "cir-to-mlir"; }
};

class CIRCallOpLowering : public mlir::OpConversionPattern<cir::CallOp> {
public:
  using OpConversionPattern<cir::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Type> types;
    if (mlir::failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), types)))
      return mlir::failure();

    if (!op.isIndirect()) {
      // Currently variadic functions are not supported by the builtin func
      // dialect. For now only basic call to printf are supported by using the
      // llvmir dialect.
      // TODO: remove this and add support for variadic function calls once
      // TODO: supported by the func dialect
      if (op.getCallee()->equals_insensitive("printf")) {
        SmallVector<mlir::Type> operandTypes =
            llvm::to_vector(adaptor.getOperands().getTypes());

        // Drop the initial memref operand type (we replace the memref format
        // string with equivalent llvm.mlir ops)
        operandTypes.erase(operandTypes.begin());

        // Check that the printf attributes can be used in llvmir dialect (i.e
        // they have integer/float type)
        if (!llvm::all_of(operandTypes, [](mlir::Type ty) {
              return mlir::LLVM::isCompatibleType(ty);
            })) {
          return op.emitError()
                 << "lowering of printf attributes having a type that is "
                    "converted to memref in cir-to-mlir lowering (e.g. "
                    "pointers) not supported yet";
        }

        // Currently only versions of printf are supported where the format
        // string is defined inside the printf ==> the lowering of the cir ops
        // will match:
        // %global = memref.get_global %frm_str
        // %* = memref.reinterpret_cast (%global, 0)
        if (auto reinterpret_castOP =
                adaptor.getOperands()[0]
                    .getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
          if (auto getGlobalOp =
                  reinterpret_castOP->getOperand(0)
                      .getDefiningOp<mlir::memref::GetGlobalOp>()) {
            mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();

            auto context = rewriter.getContext();

            // Find the memref.global op defining the frm_str
            auto globalOp = parentModule.lookupSymbol<mlir::memref::GlobalOp>(
                getGlobalOp.getNameAttr());

            rewriter.setInsertionPoint(globalOp);

            // Insert a equivalent llvm.mlir.global
            auto initialvalueAttr =
                mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(
                    globalOp.getInitialValueAttr());

            auto type = mlir::LLVM::LLVMArrayType::get(
                mlir::IntegerType::get(context, 8),
                initialvalueAttr.getNumElements());

            auto llvmglobalOp = mlir::LLVM::GlobalOp::create(
                rewriter, globalOp->getLoc(), type, true,
                mlir::LLVM::Linkage::Internal,
                "printf_format_" + globalOp.getSymName().str(),
                initialvalueAttr, 0);

            rewriter.setInsertionPoint(getGlobalOp);

            // Insert llvmir dialect ops to retrive the !llvm.ptr of the global
            auto globalPtrOp = mlir::LLVM::AddressOfOp::create(
                rewriter, getGlobalOp->getLoc(), llvmglobalOp);

            mlir::Value cst0 = mlir::LLVM::ConstantOp::create(
                rewriter, getGlobalOp->getLoc(), rewriter.getI8Type(),
                rewriter.getIndexAttr(0));
            auto gepPtrOp = mlir::LLVM::GEPOp::create(
                rewriter, getGlobalOp->getLoc(),
                mlir::LLVM::LLVMPointerType::get(context),
                llvmglobalOp.getType(), globalPtrOp,
                ArrayRef<mlir::Value>({cst0, cst0}));

            mlir::ValueRange operands = adaptor.getOperands();

            // Replace the old memref operand with the !llvm.ptr for the frm_str
            mlir::SmallVector<mlir::Value> newOperands;
            newOperands.push_back(gepPtrOp);
            newOperands.append(operands.begin() + 1, operands.end());

            // Create the llvmir dialect function type for printf
            auto llvmI32Ty = mlir::IntegerType::get(context, 32);
            auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(context);
            auto llvmFnType =
                mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                                  /*isVarArg=*/true);

            rewriter.setInsertionPoint(op);

            // Insert an llvm.call op with the updated operands to printf
            rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
                op, llvmFnType, op.getCalleeAttr(), newOperands);

            // Cleanup printf frm_str memref ops
            rewriter.eraseOp(reinterpret_castOP);
            rewriter.eraseOp(getGlobalOp);
            rewriter.eraseOp(globalOp);

            return mlir::LogicalResult::success();
          }
        }

        return op.emitError()
               << "lowering of printf function with Format-String"
                  "defined outside of printf is not supported yet";
      }

      rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
          op, op.getCalleeAttr(), types, adaptor.getOperands());
      return mlir::LogicalResult::success();

    } else {
      // A no-prototype / K&R direct call — ClangIR emits it as
      // `cir.get_global @foo : !cir.ptr<!cir.func<...>>` feeding an indirect
      // `cir.call` — is folded to a direct `func.call` by CIRGetGlobalOpLowering
      // (the func-pointer callee operand is unconvertible, so this pattern is
      // never even invoked for that shape). Any indirect call that DOES reach
      // here is a genuine runtime function pointer with no hardware realization.
      return op.emitError()
             << "lowering of indirect calls / function pointers is not "
                "supported (no hardware realization for a runtime function "
                "pointer)";
    }
  }
};

/// Given a type convertor and a data layout, convert the given type to a type
/// that is suitable for memory operations. For example, this can be used to
/// lower cir.bool accesses to i8.
static mlir::Type convertTypeForMemory(const mlir::TypeConverter &converter,
                                       mlir::Type type) {
  // TODO(cir): Handle other types similarly to clang's codegen
  // convertTypeForMemory
  if (isa<cir::BoolType>(type)) {
    // TODO: Use datalayout to get the size of bool
    return mlir::IntegerType::get(type.getContext(), 8);
  }

  return converter.convertType(type);
}

/// Emits the value from memory as expected by its users. Should be called when
/// the memory represetnation of a CIR type is not equal to its scalar
/// representation.
static mlir::Value emitFromMemory(mlir::ConversionPatternRewriter &rewriter,
                                  cir::LoadOp op, mlir::Value value) {

  // TODO(cir): Handle other types similarly to clang's codegen EmitFromMemory
  if (isa<cir::BoolType>(op.getType())) {
    // Create trunc of value from i8 to i1
    // TODO: Use datalayout to get the size of bool
    assert(value.getType().isInteger(8));
    return createIntCast(rewriter, value, rewriter.getI1Type());
  }

  return value;
}

/// Emits a value to memory with the expected scalar type. Should be called when
/// the memory represetnation of a CIR type is not equal to its scalar
/// representation.
static mlir::Value emitToMemory(mlir::ConversionPatternRewriter &rewriter,
                                cir::StoreOp op, mlir::Value value) {

  // TODO(cir): Handle other types similarly to clang's codegen EmitToMemory
  if (isa<cir::BoolType>(op.getValue().getType())) {
    // Create zext of value from i1 to i8
    // TODO: Use datalayout to get the size of bool
    return createIntCast(rewriter, value, rewriter.getI8Type());
  }

  return value;
}

class CIRAllocaOpLowering : public mlir::OpConversionPattern<cir::AllocaOp> {
public:
  using OpConversionPattern<cir::AllocaOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    mlir::Type mlirType =
        convertTypeForMemory(*getTypeConverter(), op.getAllocaType());

    // FIXME: Some types can not be converted yet (e.g. struct)
    if (!mlirType)
      return mlir::LogicalResult::failure();

    auto memreftype = mlir::dyn_cast<mlir::MemRefType>(mlirType);
    if (mlir::isa<cir::ArrayType>(op.getAllocaType())) {
      if (!memreftype)
        return mlir::LogicalResult::failure();
      rewriter.replaceOpWithNewOp<mlir::memref::AllocaOp>(
          op, memreftype, op.getAlignmentAttr());
    } else {
      auto targetType = mlir::cast<mlir::MemRefType>(
          getTypeConverter()->convertType(op.getResult().getType()));
      memreftype = mlir::MemRefType::get({1}, targetType.getElementType(),
                                         targetType.getLayout(),
                                         targetType.getMemorySpace());
      auto allocaOp = mlir::memref::AllocaOp::create(
          rewriter, op.getLoc(), memreftype, op.getAlignmentAttr());
      // Cast from memref<1xMlirType> to memref<?xMlirType>
      // This is needed since Typeconverter produces memref<?xMlirType> for
      // non-array cir.ptrs, The cast will be eliminated later in
      // load/store-lowering.
      auto castOp = mlir::memref::CastOp::create(rewriter, op.getLoc(),
                                                 targetType, allocaOp);
      rewriter.replaceOp(op, castOp);
    }
    return mlir::LogicalResult::success();
  }
};

// Find base and indices from memref.reinterpret_cast
// and put it into eraseList.
// During dialect conversion a `cir.get_element` on an N-D array is lowered to a
// `memref.reinterpret_cast` whose result type carries a DYNAMIC offset (see
// CIRGetElementOpLowering), which differs from the converter's static-offset
// result type for the get_element and therefore has a
// builtin.unrealized_conversion_cast inserted on top of it. To keep the
// reinterpret_cast CHAIN walkable (so the multi-index base/indices can be
// reconstructed), peel any such single-input memref-to-memref materialization.
static mlir::Value lookThroughMemrefMaterialization(mlir::Value v) {
  while (auto ucc = v.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (ucc.getInputs().size() != 1 ||
        !mlir::isa<mlir::MemRefType>(ucc.getInputs()[0].getType()))
      break;
    v = ucc.getInputs()[0];
  }
  return v;
}

// Collapse a chain of `cir.get_element` ops (partial indexing of an N-D array,
// e.g. `m[i][j]` = get_element(get_element(%base,%i),%j)) into a single
// multi-index access on the N-D base memref. Walks the ORIGINAL (pre-conversion)
// cir value chain — each get_element indexes exactly one array dimension — so
// the reconstructed index list is exactly the row-major subscript sequence and
// the address semantics (declared dims, row-major stride) are preserved by
// construction. The per-op `memref.reinterpret_cast` lowerings of the
// intermediate get_elements become dead and are reconciled away. Returns true
// and fills base/indices when `addr` is rooted in such a chain over a ranked
// memref whose rank matches the number of subscripts; false otherwise (callers
// then fall back to findBaseAndIndices for ptr_stride / single-level cases).
static bool collapseGetElementChain(mlir::Value addr, mlir::Value &base,
                                    SmallVector<mlir::Value> &indices,
                                    mlir::ConversionPatternRewriter &rewriter,
                                    mlir::Location loc) {
  SmallVector<mlir::Value> cirIndices;
  mlir::Value cur = addr;
  while (auto ge = cur.getDefiningOp<cir::GetElementOp>()) {
    cirIndices.push_back(ge.getIndex());
    cur = ge.getBase();
  }
  if (cirIndices.empty())
    return false;

  mlir::Value root = rewriter.getRemappedValue(cur);
  if (!root)
    return false;
  auto mrt = mlir::dyn_cast<mlir::MemRefType>(root.getType());
  if (!mrt || mrt.getRank() != static_cast<int64_t>(cirIndices.size()))
    return false;

  std::reverse(cirIndices.begin(), cirIndices.end());
  auto indexType = rewriter.getIndexType();
  indices.clear();
  for (mlir::Value idx : cirIndices) {
    mlir::Value conv = rewriter.getRemappedValue(idx);
    if (!conv)
      return false;
    if (conv.getType() != indexType)
      conv = mlir::arith::IndexCastOp::create(rewriter, loc, indexType, conv);
    indices.push_back(conv);
  }
  base = root;
  return true;
}

static bool findBaseAndIndices(mlir::Value addr, mlir::Value &base,
                               SmallVector<mlir::Value> &indices,
                               SmallVector<mlir::Operation *> &eraseList,
                               mlir::ConversionPatternRewriter &rewriter) {
  addr = lookThroughMemrefMaterialization(addr);
  while (mlir::Operation *addrOp =
             addr.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
    indices.push_back(addrOp->getOperand(1));
    addr = lookThroughMemrefMaterialization(addrOp->getOperand(0));
    eraseList.push_back(addrOp);
  }
  if (auto castOp = addr.getDefiningOp<mlir::memref::CastOp>()) {
    auto castInput = castOp->getOperand(0);
    if (castInput.getDefiningOp<mlir::memref::AllocaOp>() ||
        castInput.getDefiningOp<mlir::memref::GetGlobalOp>()) {
      // AllocaOp and GetGlobalOp-lowerings produce 1-element memrefs
      indices.push_back(
          mlir::arith::ConstantIndexOp::create(rewriter, castOp.getLoc(), 0));
      addr = castInput;
      eraseList.push_back(castOp);
    }
  }
  base = addr;
  if (indices.size() == 0) {
    auto memrefType = mlir::cast<mlir::MemRefType>(base.getType());
    auto rank = memrefType.getRank();
    indices.reserve(rank);
    for (unsigned d = 0; d < rank; ++d) {
      mlir::Value zero = mlir::arith::ConstantIndexOp::create(
          rewriter, base.getLoc(), /*value=*/0);
      indices.push_back(zero);
    }
    return false;
  }
  std::reverse(indices.begin(), indices.end());
  return true;
}

// If the memref.reinterpret_cast has multiple users (i.e the original
// cir.ptr_stride op has multiple users), only erase the operation after the
// last load or store has been generated.
static void eraseIfSafe(mlir::Value oldAddr, mlir::Value newAddr,
                        SmallVector<mlir::Operation *> &eraseList,
                        mlir::ConversionPatternRewriter &rewriter) {

  unsigned oldUsedNum =
      std::distance(oldAddr.getUses().begin(), oldAddr.getUses().end());
  unsigned newUsedNum = 0;
  // Count the uses of the newAddr (the result of the original base alloca) in
  // load/store ops using an forwarded offset from the current
  // memref.reinterpret_cast op
  for (auto *user : newAddr.getUsers()) {
    if (auto loadOpUser = mlir::dyn_cast_or_null<mlir::memref::LoadOp>(*user)) {
      if (!loadOpUser.getIndices().empty()) {
        if (auto reinterpretOp =
                mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(
                    eraseList.back())) {
          auto strideVal = loadOpUser.getIndices()[0];
          if (strideVal == reinterpretOp.getOffsets()[0])
            ++newUsedNum;
        } else if (auto castOp =
                       mlir::dyn_cast<mlir::memref::CastOp>(eraseList.back()))
          ++newUsedNum;
      }
    } else if (auto storeOpUser =
                   mlir::dyn_cast_or_null<mlir::memref::StoreOp>(*user)) {
      if (!storeOpUser.getIndices().empty()) {
        if (auto reinterpretOp =
                mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(
                    eraseList.back())) {
          auto strideVal = storeOpUser.getIndices()[0];
          if (strideVal == reinterpretOp.getOffsets()[0])
            ++newUsedNum;
        } else if (auto castOp =
                       mlir::dyn_cast<mlir::memref::CastOp>(eraseList.back()))
          ++newUsedNum;
      }
    }
  }
  // If all load/store ops are using forwarded offsets from the current
  // memref.(reinterpret_)cast ops, erase them
  if (oldUsedNum == newUsedNum) {
    for (auto op : eraseList)
      rewriter.eraseOp(op);
  }
}

static mlir::LogicalResult
prepareReinterpretMetadata(mlir::MemRefType type, mlir::Value source,
                           mlir::ConversionPatternRewriter &rewriter,
                           llvm::SmallVectorImpl<mlir::OpFoldResult> &sizes,
                           llvm::SmallVectorImpl<mlir::OpFoldResult> &strides,
                           mlir::Operation *anchorOp) {
  sizes.clear();
  strides.clear();

  auto sourceType = llvm::dyn_cast<mlir::MemRefType>(source.getType());
  int64_t sourceRank = sourceType ? sourceType.getRank() : 0;

  for (auto [dimIdx, dim] : llvm::enumerate(type.getShape())) {
    if (mlir::ShapedType::isDynamic(dim)) {
      // A dynamic result dimension must be given a dynamic size *Value*.
      // Pushing a static kDynamic index attr here (with no matching operand)
      // builds a malformed memref.reinterpret_cast whose getMixedSizes()
      // asserts in getMixedValues — which crashes the reinterpret_cast constant
      // folder as soon as the offset is a constant (e.g. backprop's constant
      // strides), while kernels with non-constant offsets merely skip the fold.
      // Materialize the size from the source memref (memref.dim). The result
      // type stays dynamic, so the folder's constifyIndexValues keeps this
      // dynamic — no result-type mismatch. These reinterpret_casts are erased
      // after load/store rewiring, so the exact dim is a well-formedness
      // carrier, not a semantic size.
      if (!sourceType || sourceRank == 0) {
        anchorOp->emitError(
            "cannot materialize dynamic reinterpret_cast size: source is not a "
            "ranked memref");
        return mlir::failure();
      }
      int64_t srcDim = std::min<int64_t>(dimIdx, sourceRank - 1);
      mlir::Value dimVal = mlir::memref::DimOp::create(
          rewriter, anchorOp->getLoc(), source, srcDim);
      sizes.push_back(dimVal);
    } else {
      sizes.push_back(rewriter.getIndexAttr(dim));
    }
  }

  llvm::SmallVector<int64_t, 4> strideValues;
  int64_t layoutOffset = 0;
  if (mlir::failed(type.getStridesAndOffset(strideValues, layoutOffset))) {
    anchorOp->emitError("expected strided memref layout");
    return mlir::failure();
  }

  for (int64_t stride : strideValues) {
    if (mlir::ShapedType::isDynamic(stride)) {
      anchorOp->emitError("dynamic memref strides are not supported yet");
      return mlir::failure();
    }
    strides.push_back(rewriter.getIndexAttr(stride));
  }

  return mlir::success();
}

class CIRLoadOpLowering : public mlir::OpConversionPattern<cir::LoadOp> {
public:
  using OpConversionPattern<cir::LoadOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value base;
    SmallVector<mlir::Value> indices;
    SmallVector<mlir::Operation *> eraseList;
    mlir::memref::LoadOp newLoad;
    if (collapseGetElementChain(op.getAddr(), base, indices, rewriter,
                                op.getLoc())) {
      newLoad =
          mlir::memref::LoadOp::create(rewriter, op.getLoc(), base, indices);
    } else {
      bool eraseIntermediateOp = findBaseAndIndices(
          adaptor.getAddr(), base, indices, eraseList, rewriter);
      newLoad =
          mlir::memref::LoadOp::create(rewriter, op.getLoc(), base, indices);
      if (eraseIntermediateOp)
        eraseIfSafe(op.getAddr(), base, eraseList, rewriter);
    }

    // Convert adapted result to its original type if needed.
    mlir::Value result = emitFromMemory(rewriter, op, newLoad.getResult());
    rewriter.replaceOp(op, result);
    return mlir::LogicalResult::success();
  }
};

class CIRStoreOpLowering : public mlir::OpConversionPattern<cir::StoreOp> {
public:
  using OpConversionPattern<cir::StoreOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value base;
    SmallVector<mlir::Value> indices;
    SmallVector<mlir::Operation *> eraseList;

    // Convert adapted value to its memory type if needed.
    mlir::Value value = emitToMemory(rewriter, op, adaptor.getValue());
    if (collapseGetElementChain(op.getAddr(), base, indices, rewriter,
                                op.getLoc())) {
      rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, value, base,
                                                         indices);
    } else {
      bool eraseIntermediateOp = findBaseAndIndices(
          adaptor.getAddr(), base, indices, eraseList, rewriter);
      rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, value, base,
                                                         indices);
      if (eraseIntermediateOp)
        eraseIfSafe(op.getAddr(), base, eraseList, rewriter);
    }

    return mlir::LogicalResult::success();
  }
};

/// Converts CIR unary math ops (e.g., cir::SinOp) to their MLIR equivalents
/// (e.g., math::SinOp) using a generic template to avoid redundant boilerplate
/// matchAndRewrite definitions.

template <typename CIROp, typename MLIROp>
class CIRUnaryMathOpLowering : public mlir::OpConversionPattern<CIROp> {
public:
  using mlir::OpConversionPattern<CIROp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CIROp op,
                  typename mlir::OpConversionPattern<CIROp>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<MLIROp>(op, adaptor.getSrc());
    return mlir::LogicalResult::success();
  }
};

using CIRASinOpLowering =
    CIRUnaryMathOpLowering<cir::ASinOp, mlir::math::AsinOp>;
using CIRSinOpLowering = CIRUnaryMathOpLowering<cir::SinOp, mlir::math::SinOp>;
using CIRExp2OpLowering =
    CIRUnaryMathOpLowering<cir::Exp2Op, mlir::math::Exp2Op>;
using CIRExpOpLowering = CIRUnaryMathOpLowering<cir::ExpOp, mlir::math::ExpOp>;
using CIRRoundOpLowering =
    CIRUnaryMathOpLowering<cir::RoundOp, mlir::math::RoundOp>;
using CIRLog2OpLowering =
    CIRUnaryMathOpLowering<cir::Log2Op, mlir::math::Log2Op>;
using CIRLogOpLowering = CIRUnaryMathOpLowering<cir::LogOp, mlir::math::LogOp>;
using CIRLog10OpLowering =
    CIRUnaryMathOpLowering<cir::Log10Op, mlir::math::Log10Op>;
using CIRCeilOpLowering =
    CIRUnaryMathOpLowering<cir::CeilOp, mlir::math::CeilOp>;
using CIRFloorOpLowering =
    CIRUnaryMathOpLowering<cir::FloorOp, mlir::math::FloorOp>;
using CIRAbsOpLowering = CIRUnaryMathOpLowering<cir::AbsOp, mlir::math::AbsIOp>;
using CIRFAbsOpLowering =
    CIRUnaryMathOpLowering<cir::FAbsOp, mlir::math::AbsFOp>;
using CIRSqrtOpLowering =
    CIRUnaryMathOpLowering<cir::SqrtOp, mlir::math::SqrtOp>;
using CIRCosOpLowering = CIRUnaryMathOpLowering<cir::CosOp, mlir::math::CosOp>;
using CIRATanOpLowering =
    CIRUnaryMathOpLowering<cir::ATanOp, mlir::math::AtanOp>;
using CIRACosOpLowering =
    CIRUnaryMathOpLowering<cir::ACosOp, mlir::math::AcosOp>;
using CIRTanOpLowering = CIRUnaryMathOpLowering<cir::TanOp, mlir::math::TanOp>;

class CIRShiftOpLowering : public mlir::OpConversionPattern<cir::ShiftOp> {
public:
  using mlir::OpConversionPattern<cir::ShiftOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(cir::ShiftOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto cirAmtTy = mlir::dyn_cast<cir::IntType>(op.getAmount().getType());
    auto cirValTy = mlir::dyn_cast<cir::IntType>(op.getValue().getType());

    // Operands could also be vector type
    auto cirAmtVTy = mlir::dyn_cast<cir::VectorType>(op.getAmount().getType());
    auto cirValVTy = mlir::dyn_cast<cir::VectorType>(op.getValue().getType());
    auto targetTy = getTypeConverter()->convertType(op.getType());
    mlir::Value amt = adaptor.getAmount();
    mlir::Value val = adaptor.getValue();

    assert(((cirValTy && cirAmtTy) || (cirAmtVTy && cirValVTy)) &&
           "shift input type must be integer or vector type, otherwise NYI");

    assert((cirValTy == op.getType() || cirValVTy == op.getType()) &&
           "inconsistent operands' types NYI");

    // Ensure shift amount is the same type as the value. Some undefined
    // behavior might occur in the casts below as per [C99 6.5.7.3].
    // Vector type shift amount needs no cast as type consistency is expected to
    // already be enforced at CIRGen.
    if (cirAmtTy)
      amt = createIntCast(rewriter, amt, targetTy, cirAmtTy.isSigned());

    // Lower to the proper arithmetic shift operation.
    if (op.getIsShiftleft())
      rewriter.replaceOpWithNewOp<mlir::arith::ShLIOp>(op, targetTy, val, amt);
    else {
      bool isUnSigned =
          cirValTy ? !cirValTy.isSigned()
                   : !mlir::cast<cir::IntType>(cirValVTy.getElementType())
                          .isSigned();
      if (isUnSigned)
        rewriter.replaceOpWithNewOp<mlir::arith::ShRUIOp>(op, targetTy, val,
                                                          amt);
      else
        rewriter.replaceOpWithNewOp<mlir::arith::ShRSIOp>(op, targetTy, val,
                                                          amt);
    }

    return mlir::success();
  }
};

template <typename CIROp, typename MLIROp>
class CIRCountZerosBitOpLowering : public mlir::OpConversionPattern<CIROp> {
public:
  using mlir::OpConversionPattern<CIROp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(CIROp op,
                  typename mlir::OpConversionPattern<CIROp>::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<MLIROp>(op, adaptor.getInput());
    return mlir::LogicalResult::success();
  }
};

using CIRBitClzOpLowering =
    CIRCountZerosBitOpLowering<cir::BitClzOp, mlir::math::CountLeadingZerosOp>;
using CIRBitCtzOpLowering =
    CIRCountZerosBitOpLowering<cir::BitCtzOp, mlir::math::CountTrailingZerosOp>;

class CIRBitClrsbOpLowering
    : public mlir::OpConversionPattern<cir::BitClrsbOp> {
public:
  using OpConversionPattern<cir::BitClrsbOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitClrsbOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto inputTy = adaptor.getInput().getType();
    auto zero = getConst(rewriter, op.getLoc(), inputTy, 0);
    auto isNeg = mlir::arith::CmpIOp::create(
        rewriter, op.getLoc(),
        mlir::arith::CmpIPredicateAttr::get(rewriter.getContext(),
                                            mlir::arith::CmpIPredicate::slt),
        adaptor.getInput(), zero);

    auto negOne = getConst(rewriter, op.getLoc(), inputTy, -1);
    auto flipped = mlir::arith::XOrIOp::create(rewriter, op.getLoc(),
                                               adaptor.getInput(), negOne);

    auto select = mlir::arith::SelectOp::create(rewriter, op.getLoc(), isNeg,
                                                flipped, adaptor.getInput());

    auto clz =
        mlir::math::CountLeadingZerosOp::create(rewriter, op->getLoc(), select);

    auto one = getConst(rewriter, op.getLoc(), inputTy, 1);
    auto res = mlir::arith::SubIOp::create(rewriter, op.getLoc(), clz, one);
    rewriter.replaceOp(op, res);

    return mlir::LogicalResult::success();
  }
};

class CIRBitFfsOpLowering : public mlir::OpConversionPattern<cir::BitFfsOp> {
public:
  using OpConversionPattern<cir::BitFfsOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitFfsOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto inputTy = adaptor.getInput().getType();
    auto ctz = mlir::math::CountTrailingZerosOp::create(rewriter, op.getLoc(),
                                                        adaptor.getInput());

    auto one = getConst(rewriter, op.getLoc(), inputTy, 1);
    auto ctzAddOne =
        mlir::arith::AddIOp::create(rewriter, op.getLoc(), ctz, one);

    auto zero = getConst(rewriter, op.getLoc(), inputTy, 0);
    auto isZero = mlir::arith::CmpIOp::create(
        rewriter, op.getLoc(),
        mlir::arith::CmpIPredicateAttr::get(rewriter.getContext(),
                                            mlir::arith::CmpIPredicate::eq),
        adaptor.getInput(), zero);

    auto res = mlir::arith::SelectOp::create(rewriter, op.getLoc(), isZero,
                                             zero, ctzAddOne);
    rewriter.replaceOp(op, res);

    return mlir::LogicalResult::success();
  }
};

class CIRBitPopcountOpLowering
    : public mlir::OpConversionPattern<cir::BitPopcountOp> {
public:
  using mlir::OpConversionPattern<cir::BitPopcountOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitPopcountOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::math::CtPopOp>(op, adaptor.getInput());
    return mlir::LogicalResult::success();
  }
};

class CIRBitParityOpLowering
    : public mlir::OpConversionPattern<cir::BitParityOp> {
public:
  using OpConversionPattern<cir::BitParityOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BitParityOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto count =
        mlir::math::CtPopOp::create(rewriter, op.getLoc(), adaptor.getInput());
    auto countMod2 = mlir::arith::AndIOp::create(
        rewriter, op.getLoc(), count,
        getConst(rewriter, op.getLoc(), count.getType(), 1));
    rewriter.replaceOp(op, countMod2);
    return mlir::LogicalResult::success();
  }
};

class CIRConstantOpLowering
    : public mlir::OpConversionPattern<cir::ConstantOp> {
public:
  using OpConversionPattern<cir::ConstantOp>::OpConversionPattern;

private:
  // This code is in a separate function rather than part of matchAndRewrite
  // because it is recursive.  There is currently only one level of recursion;
  // when lowing a vector attribute the attributes for the elements also need
  // to be lowered.
  mlir::TypedAttr
  lowerCirAttrToMlirAttr(mlir::Attribute cirAttr,
                         mlir::ConversionPatternRewriter &rewriter) const {
    assert(mlir::isa<mlir::TypedAttr>(cirAttr) &&
           "Can't lower a non-typed attribute");
    auto mlirType = getTypeConverter()->convertType(
        mlir::cast<mlir::TypedAttr>(cirAttr).getType());
    if (auto vecAttr = mlir::dyn_cast<cir::ConstVectorAttr>(cirAttr)) {
      assert(mlir::isa<mlir::VectorType>(mlirType) &&
             "MLIR type for CIR vector attribute is not mlir::VectorType");
      assert(mlir::isa<mlir::ShapedType>(mlirType) &&
             "mlir::VectorType is not a mlir::ShapedType ??");
      SmallVector<mlir::Attribute> mlirValues;
      for (auto elementAttr : vecAttr.getElts()) {
        mlirValues.push_back(
            this->lowerCirAttrToMlirAttr(elementAttr, rewriter));
      }
      return mlir::DenseElementsAttr::get(
          mlir::cast<mlir::ShapedType>(mlirType), mlirValues);
    } else if (auto zeroAttr = mlir::dyn_cast<cir::ZeroAttr>(cirAttr)) {
      (void)zeroAttr;
      return rewriter.getZeroAttr(mlirType);
    } else if (auto complexAttr =
                   mlir::dyn_cast<cir::ConstComplexAttr>(cirAttr)) {
      auto vecType = mlir::dyn_cast<mlir::VectorType>(mlirType);
      assert(vecType && "complex attribute lowered type should be a vector");
      SmallVector<mlir::Attribute, 2> elements{
          this->lowerCirAttrToMlirAttr(complexAttr.getReal(), rewriter),
          this->lowerCirAttrToMlirAttr(complexAttr.getImag(), rewriter)};
      return mlir::DenseElementsAttr::get(vecType, elements);
    } else if (auto boolAttr = mlir::dyn_cast<cir::BoolAttr>(cirAttr)) {
      return rewriter.getIntegerAttr(mlirType, boolAttr.getValue());
    } else if (auto floatAttr = mlir::dyn_cast<cir::FPAttr>(cirAttr)) {
      return rewriter.getFloatAttr(mlirType, floatAttr.getValue());
    } else if (auto intAttr = mlir::dyn_cast<cir::IntAttr>(cirAttr)) {
      return rewriter.getIntegerAttr(mlirType, intAttr.getValue());
    } else {
      llvm_unreachable("NYI: unsupported attribute kind lowering to MLIR");
      return {};
    }
  }

public:
  mlir::LogicalResult
  matchAndRewrite(cir::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
        op, getTypeConverter()->convertType(op.getType()),
        this->lowerCirAttrToMlirAttr(op.getValue(), rewriter));
    return mlir::LogicalResult::success();
  }
};

class CIRFuncOpLowering : public mlir::OpConversionPattern<cir::FuncOp> {
public:
  using OpConversionPattern<cir::FuncOp>::OpConversionPattern;

  void lowerFuncAttributesToMLIR(
      cir::FuncOp func, SmallVectorImpl<mlir::NamedAttribute> &result) const {
    if (auto symVisibilityAttr = func.getSymVisibilityAttr())
      result.push_back(
          mlir::NamedAttribute("sym_visibility", symVisibilityAttr));
    // D7: upstream removed extra_attrs/ExtraFuncAttributesAttr and the
    // OpenCLKernelAttr; vtrhls targets func/scf/memref (no GPU), so the
    // GPU-kernel metadata branch is dropped.
  }

  mlir::LogicalResult
  matchAndRewrite(cir::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto fnType = op.getFunctionType();

    if (fnType.isVarArg()) {
      // TODO: once the func dialect supports variadic functions rewrite this
      // For now only insert special handling of printf via the llvmir dialect
      if (op.getSymName().equals_insensitive("printf")) {
        auto *context = rewriter.getContext();
        // Create a llvmir dialect function declaration for printf, the
        // signature is: i32 (!llvm.ptr, ...)
        auto llvmI32Ty = mlir::IntegerType::get(context, 32);
        auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(context);
        auto llvmFnType =
            mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                              /*isVarArg=*/true);
        auto printfFunc = mlir::LLVM::LLVMFuncOp::create(rewriter, op.getLoc(),
                                                         "printf", llvmFnType);
        rewriter.replaceOp(op, printfFunc);
      } else {
        rewriter.eraseOp(op);
        return op.emitError() << "lowering of variadic functions (except "
                                 "printf) not supported yet";
      }
    } else {
      mlir::TypeConverter::SignatureConversion signatureConversion(
          fnType.getNumInputs());

      for (const auto &argType : enumerate(fnType.getInputs())) {
        auto convertedType = typeConverter->convertType(argType.value());
        if (!convertedType)
          return mlir::failure();
        signatureConversion.addInputs(argType.index(), convertedType);
      }

      SmallVector<mlir::NamedAttribute, 4> passThroughAttrs;
      lowerFuncAttributesToMLIR(op, passThroughAttrs);

      mlir::Type resultType =
          getTypeConverter()->convertType(fnType.getReturnType());
      auto fn = mlir::func::FuncOp::create(
          rewriter, op.getLoc(), op.getName(),
          rewriter.getFunctionType(signatureConversion.getConvertedTypes(),
                                   resultType ? mlir::TypeRange(resultType)
                                              : mlir::TypeRange()),
          passThroughAttrs);

      if (failed(rewriter.convertRegionTypes(&op.getBody(), *typeConverter,
                                             &signatureConversion)))
        return mlir::failure();
      rewriter.inlineRegionBefore(op.getBody(), fn.getBody(), fn.end());

      rewriter.eraseOp(op);
    }
    return mlir::LogicalResult::success();
  }
};

// Upstream CIR split the monolithic cir::BinOp / cir::UnaryOp (with
// BinOpKind / UnaryOpKind enums) into per-op classes. So these are now one
// OpConversionPattern per CIR op instead of a switch on a kind enum (D1/D2).

// cir.add / cir.sub carry an optional 'saturated' unit flag (clamp to the
// type's range instead of wrapping); the other binary ops do not. ThroughMLIR
// lowers to plain wrapping arith.*, which would SILENTLY MISCOMPILE a saturating
// op, so detect and reject it with an honest diagnostic. (CIRGen does not emit
// saturating add/sub today, so this is defensive against a future gap.)
static bool binOpIsSaturated(mlir::Operation *op) {
  if (auto add = mlir::dyn_cast<cir::AddOp>(op))
    return add.getSaturated();
  if (auto sub = mlir::dyn_cast<cir::SubOp>(op))
    return sub.getSaturated();
  return false;
}

// Direct 1:1 binary lowering: cir.<op> %a, %b -> arith.<MLIROp> %a, %b.
// Integer Add/Sub/Mul/And/Or/Xor and FP FAdd/FSub/FMul/FDiv/FRem.
template <typename CIROp, typename MLIROp>
class CIRBinOpLowering : public mlir::OpConversionPattern<CIROp> {
public:
  using mlir::OpConversionPattern<CIROp>::OpConversionPattern;
  using OpAdaptor = typename CIROp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(CIROp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type mlirType = this->getTypeConverter()->convertType(op.getType());
    if (!mlirType)
      return op.emitError("CIRBinOpLowering: unconvertible result type");
    if (binOpIsSaturated(op))
      return op.emitError("CIRBinOpLowering: saturating add/sub is not yet "
                          "supported by the CIR-to-MLIR lowering");
    rewriter.replaceOpWithNewOp<MLIROp>(op, mlirType, adaptor.getLhs(),
                                        adaptor.getRhs());
    return mlir::success();
  }
};

// Integer divide/remainder: signedness taken from the CIR operand type
// (upstream cir::DivOp/RemOp are integer-only; FP uses FDivOp/FRemOp).
template <typename CIROp, typename SIntOp, typename UIntOp>
class CIRSignedAwareIntBinOpLowering : public mlir::OpConversionPattern<CIROp> {
public:
  using mlir::OpConversionPattern<CIROp>::OpConversionPattern;
  using OpAdaptor = typename CIROp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(CIROp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type mlirType = this->getTypeConverter()->convertType(op.getType());
    if (!mlirType)
      return op.emitError("CIRSignedAwareIntBinOpLowering: unconvertible result type");
    mlir::Type type = op.getLhs().getType();
    if (auto vecType = mlir::dyn_cast<cir::VectorType>(type))
      type = vecType.getElementType();
    bool isUnsigned = false;
    if (auto intTy = mlir::dyn_cast<cir::IntType>(type))
      isUnsigned = intTy.isUnsigned();
    if (isUnsigned)
      rewriter.replaceOpWithNewOp<UIntOp>(op, mlirType, adaptor.getLhs(),
                                          adaptor.getRhs());
    else
      rewriter.replaceOpWithNewOp<SIntOp>(op, mlirType, adaptor.getLhs(),
                                          adaptor.getRhs());
    return mlir::success();
  }
};

// Unary ++/-- : input +/- Delta. In this CIR version cir.inc/cir.dec are
// integer(-or-vector-of-int)-only; floating-point ++/-- is emitted as
// cir.fadd/cir.fsub, not Inc/Dec. The FloatType branch below is therefore
// defensive/unreachable today but kept correct in case that constraint widens.
template <typename CIROp, int64_t Delta>
class CIRIncDecOpLowering : public mlir::OpConversionPattern<CIROp> {
public:
  using mlir::OpConversionPattern<CIROp>::OpConversionPattern;
  using OpAdaptor = typename CIROp::Adaptor;

  mlir::LogicalResult
  matchAndRewrite(CIROp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type type = this->getTypeConverter()->convertType(op.getType());
    mlir::Value input = adaptor.getInput();
    if (mlir::isa<mlir::FloatType>(type)) {
      auto imm = mlir::arith::ConstantOp::create(
          rewriter, op.getLoc(),
          mlir::FloatAttr::get(type, static_cast<double>(Delta)));
      rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(op, type, input, imm);
      return mlir::success();
    }
    if (mlir::isa<mlir::IntegerType>(type)) {
      auto imm = mlir::arith::ConstantOp::create(
          rewriter, op.getLoc(), mlir::IntegerAttr::get(type, Delta));
      rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, type, input, imm);
      return mlir::success();
    }
    return op.emitError("CIRIncDecOpLowering: unsupported type ") << type;
  }
};

// Unary integer negation: 0 - input.
class CIRMinusOpLowering : public mlir::OpConversionPattern<cir::MinusOp> {
public:
  using OpConversionPattern<cir::MinusOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::MinusOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type type = getTypeConverter()->convertType(op.getType());
    // cir.minus also accepts vector-of-int; the scalar IntegerAttr path below
    // would hard-cast (assert/crash) on a vector type. Fail honestly instead.
    if (!mlir::isa<mlir::IntegerType>(type))
      return op.emitError("CIRMinusOpLowering: only scalar integer negation is "
                          "supported (vector operands not yet handled)");
    auto zero = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(), mlir::IntegerAttr::get(type, 0));
    rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, type, zero,
                                                     adaptor.getInput());
    return mlir::success();
  }
};

// Unary floating-point negation.
class CIRFNegOpLowering : public mlir::OpConversionPattern<cir::FNegOp> {
public:
  using OpConversionPattern<cir::FNegOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::FNegOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::NegFOp>(op, adaptor.getInput());
    return mlir::success();
  }
};

// Bitwise NOT: input ^ -1.
class CIRNotOpLowering : public mlir::OpConversionPattern<cir::NotOp> {
public:
  using OpConversionPattern<cir::NotOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::NotOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type type = getTypeConverter()->convertType(op.getType());
    // cir.not also accepts vector-of-int; the scalar IntegerAttr path below
    // would hard-cast (assert/crash) on a vector type. Fail honestly instead.
    if (!mlir::isa<mlir::IntegerType>(type))
      return op.emitError("CIRNotOpLowering: only scalar integer bitwise-not is "
                          "supported (vector operands not yet handled)");
    auto minusOne = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(), mlir::IntegerAttr::get(type, -1));
    rewriter.replaceOpWithNewOp<mlir::arith::XOrIOp>(op, type, minusOne,
                                                     adaptor.getInput());
    return mlir::success();
  }
};

class CIRCmpOpLowering : public mlir::OpConversionPattern<cir::CmpOp> {
public:
  using OpConversionPattern<cir::CmpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::CmpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto type = op.getLhs().getType();

    if (auto ty = mlir::dyn_cast<cir::IntType>(type)) {
      auto kind = convertCmpKindToCmpIPredicate(op.getKind(), ty.isSigned());
      rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(
          op, kind, adaptor.getLhs(), adaptor.getRhs());
    } else if (auto ty = mlir::dyn_cast<cir::FPTypeInterface>(type)) {
      auto kind = convertCmpKindToCmpFPredicate(op.getKind());
      rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
          op, kind, adaptor.getLhs(), adaptor.getRhs());
    } else if (mlir::isa<cir::PointerType>(type)) {
      auto lhs = adaptor.getLhs();
      auto rhs = adaptor.getRhs();
      if (!mlir::isa<mlir::MemRefType>(lhs.getType()) ||
          !mlir::isa<mlir::MemRefType>(rhs.getType()))
        return op.emitError(
            "ThroughMLIR: pointer comparison operands did not lower to memref "
            "(nested pointer or unsupported pointer representation)");
      // Lower a pointer comparison to an index comparison on the physical
      // address:  addr = aligned_base_pointer + element_offset.
      // extract_aligned_pointer_as_index gives the underlying buffer base;
      // extract_strided_metadata recovers the reinterpret offset (in elements),
      // which is dropped by the aligned-pointer op alone. For SAME-ALLOCATION
      // pointers (the sound, observed HLS idiom — a wrap-around cursor
      // `if (p >= end) p = start`) the identical base cancels and the compare
      // reduces to the element offsets, which is exactly correct. Adding the
      // base back in keeps distinct allocations distinct. CROSS-ALLOCATION
      // pointer comparisons are not physically meaningful after synthesis
      // (distinct BRAMs); this op-local pattern cannot do the aliasing analysis
      // to reject them and must leave that to a downstream check rather than
      // fabricate a gate here.
      auto loc = op.getLoc();
      auto addrOf = [&](mlir::Value p) -> mlir::Value {
        mlir::Value base =
            mlir::memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc,
                                                                 p);
        auto meta =
            mlir::memref::ExtractStridedMetadataOp::create(rewriter, loc, p);
        return mlir::arith::AddIOp::create(rewriter, loc, base,
                                           meta.getOffset());
      };
      auto kind =
          convertCmpKindToCmpIPredicate(op.getKind(), /*isSigned=*/false);
      rewriter.replaceOpWithNewOp<mlir::arith::CmpIOp>(op, kind, addrOf(lhs),
                                                       addrOf(rhs));
    } else {
      return op.emitError() << "unsupported type for CmpOp: " << type;
    }

    return mlir::LogicalResult::success();
  }
};

class CIRBrOpLowering : public mlir::OpConversionPattern<cir::BrOp> {
public:
  using mlir::OpConversionPattern<cir::BrOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(op, op.getDest(),
                                                    adaptor.getDestOperands());
    return mlir::LogicalResult::success();
  }
};

class CIRScopeOpLowering : public mlir::OpConversionPattern<cir::ScopeOp> {
public:
  using mlir::OpConversionPattern<cir::ScopeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ScopeOp scopeOp, [[maybe_unused]] OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Empty scope: just remove it.
    // TODO: Remove this logic once CIR uses MLIR infrastructure to remove
    // trivially dead operations
    if (scopeOp.isEmpty()) {
      rewriter.eraseOp(scopeOp);
      return mlir::success();
    }

    // Check if the scope is empty (no operations)
    auto &scopeRegion = scopeOp.getScopeRegion();
    if (scopeRegion.empty() ||
        (scopeRegion.front().empty() ||
         (scopeRegion.front().getOperations().size() == 1 &&
          isa<cir::YieldOp>(scopeRegion.front().front())))) {
      // Drop empty scopes
      rewriter.eraseOp(scopeOp);
      return mlir::LogicalResult::success();
    }

    // For scopes without results, use memref.alloca_scope
    if (scopeOp.getNumResults() == 0) {
      auto allocaScope = mlir::memref::AllocaScopeOp::create(
          rewriter, scopeOp.getLoc(), mlir::TypeRange{});
      rewriter.inlineRegionBefore(scopeOp.getScopeRegion(),
                                  allocaScope.getBodyRegion(),
                                  allocaScope.getBodyRegion().end());
      rewriter.eraseOp(scopeOp);
    } else {
      // For scopes with results, use scf.execute_region
      SmallVector<mlir::Type> types;
      if (mlir::failed(getTypeConverter()->convertTypes(
              scopeOp->getResultTypes(), types)))
        return mlir::failure();
      auto exec =
          mlir::scf::ExecuteRegionOp::create(rewriter, scopeOp.getLoc(), types);
      rewriter.inlineRegionBefore(scopeOp.getScopeRegion(), exec.getRegion(),
                                  exec.getRegion().end());
      rewriter.replaceOp(scopeOp, exec.getResults());
    }
    return mlir::LogicalResult::success();
  }
};

struct CIRBrCondOpLowering : public mlir::OpConversionPattern<cir::BrCondOp> {
  using mlir::OpConversionPattern<cir::BrCondOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::BrCondOp brOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        brOp, adaptor.getCond(), brOp.getDestTrue(),
        adaptor.getDestOperandsTrue(), brOp.getDestFalse(),
        adaptor.getDestOperandsFalse());

    return mlir::success();
  }
};

class CIRTernaryOpLowering : public mlir::OpConversionPattern<cir::TernaryOp> {
public:
  using OpConversionPattern<cir::TernaryOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::TernaryOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    SmallVector<mlir::Type> resultTypes;
    if (mlir::failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      resultTypes)))
      return mlir::failure();

    auto ifOp = mlir::scf::IfOp::create(rewriter, op.getLoc(), resultTypes,
                                        adaptor.getCond(), true);
    auto *thenBlock = &ifOp.getThenRegion().front();
    auto *elseBlock = &ifOp.getElseRegion().front();
    rewriter.inlineBlockBefore(&op.getTrueRegion().front(), thenBlock,
                               thenBlock->end());
    rewriter.inlineBlockBefore(&op.getFalseRegion().front(), elseBlock,
                               elseBlock->end());

    rewriter.replaceOp(op, ifOp);
    return mlir::success();
  }
};

class CIRYieldOpLowering : public mlir::OpConversionPattern<cir::YieldOp> {
public:
  using OpConversionPattern<cir::YieldOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(cir::YieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *parentOp = op->getParentOp();
    return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(parentOp)
        .Case<mlir::scf::IfOp, mlir::scf::ForOp, mlir::scf::WhileOp,
              mlir::scf::ExecuteRegionOp>([&](auto) {
          rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(
              op, adaptor.getOperands());
          return mlir::success();
        })
        .Case<mlir::memref::AllocaScopeOp>([&](auto) {
          rewriter.replaceOpWithNewOp<mlir::memref::AllocaScopeReturnOp>(
              op, adaptor.getOperands());
          return mlir::success();
        })
        .Default([](auto) { return mlir::failure(); });
  }
};

class CIRIfOpLowering : public mlir::OpConversionPattern<cir::IfOp> {
public:
  using mlir::OpConversionPattern<cir::IfOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::IfOp ifop, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto newIfOp =
        mlir::scf::IfOp::create(rewriter, ifop->getLoc(),
                                ifop->getResultTypes(), adaptor.getCondition());
    auto *thenBlock = rewriter.createBlock(&newIfOp.getThenRegion());
    rewriter.inlineBlockBefore(&ifop.getThenRegion().front(), thenBlock,
                               thenBlock->end());
    if (!ifop.getElseRegion().empty()) {
      auto *elseBlock = rewriter.createBlock(&newIfOp.getElseRegion());
      rewriter.inlineBlockBefore(&ifop.getElseRegion().front(), elseBlock,
                                 elseBlock->end());
    }
    rewriter.replaceOp(ifop, newIfOp);
    return mlir::success();
  }
};

// Detect the type-punned array-global idiom that Clang emits for a large,
// partially-initialized C array (e.g. `static const int tbl[256] = {8,7,...};`
// where only a prefix is spelled out): the initializer is stored as an
// anonymous packed record `{ array<T x k>, #cir.zero : array<T x (N-k)> }` and
// every use recovers the flat `array<T x N>` through a `cir.cast(bitcast)`.
//
// Returns the common target array type when the global is record-typed and
// EVERY `cir.get_global` user feeds only bitcasts to the SAME
// `!cir.ptr<!cir.array<T x N>>`. Returns a null type otherwise — multiple
// distinct pun targets, any non-bitcast consumer (e.g. `cir.get_member`, which
// belongs to the struct-lowering cluster), or a non-record global all fall
// through to the normal lowering / a loud failure downstream. No RecordType
// TypeConverter rule is added here; this is the one narrow record exception.
static cir::ArrayType detectPunnedArrayGlobal(cir::GlobalOp global) {
  if (!mlir::isa<cir::RecordType>(global.getSymType()))
    return {};
  auto moduleOp = global->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp)
    return {};
  cir::ArrayType target;
  bool sawUse = false;
  bool valid = true;
  moduleOp.walk([&](cir::GetGlobalOp gg) {
    if (gg.getName() != global.getSymName())
      return;
    for (auto *user : gg->getUsers()) {
      auto castOp = mlir::dyn_cast<cir::CastOp>(user);
      if (!castOp || castOp.getKind() != cir::CastKind::bitcast) {
        valid = false;
        return;
      }
      auto ptrTy = mlir::dyn_cast<cir::PointerType>(castOp.getType());
      if (!ptrTy) {
        valid = false;
        return;
      }
      auto arrTy = mlir::dyn_cast<cir::ArrayType>(ptrTy.getPointee());
      if (!arrTy) {
        valid = false;
        return;
      }
      if (target && target != arrTy) {
        valid = false;
        return;
      }
      target = arrTy;
      sawUse = true;
    }
  });
  if (!valid || !sawUse)
    return {};
  return target;
}

// Self-contained variant of detectPunnedArrayGlobal that inspects a
// `cir.get_global`'s OWN uses. Used from CIRGetGlobalOpLowering because by the
// time a get_global is converted its referenced `cir.global` may already have
// been rewritten to a `memref.global` (pattern application order), so a symbol
// lookup for the original `cir::GlobalOp` can miss. Returns the common punned
// array type when every user is a bitcast to the same `!cir.ptr<array<T x N>>`.
static cir::ArrayType punnedTargetOfGetGlobal(cir::GetGlobalOp op) {
  if (!mlir::isa<cir::RecordType>(op.getType().getPointee()))
    return {};
  cir::ArrayType target;
  for (auto *user : op->getUsers()) {
    auto castOp = mlir::dyn_cast<cir::CastOp>(user);
    if (!castOp || castOp.getKind() != cir::CastKind::bitcast)
      return {};
    auto ptrTy = mlir::dyn_cast<cir::PointerType>(castOp.getType());
    if (!ptrTy)
      return {};
    auto arrTy = mlir::dyn_cast<cir::ArrayType>(ptrTy.getPointee());
    if (!arrTy)
      return {};
    if (target && target != arrTy)
      return {};
    target = arrTy;
  }
  return target;
}

// Flatten the `#cir.const_record` initializer of a type-punned array global
// into a `DenseElementsAttr` of shape [N] (N = `target` array size). The record
// members must all be `cir.const_array` / `#cir.zero` arrays (or scalar leaves)
// of the SAME converted element type as `target`; anything else (a nested
// aggregate, a mismatched element type, or a total leaf count != N, which would
// imply padding) yields std::nullopt so the caller can fail loudly. The
// leaf-count == N check with a uniform element type is exactly the bit-exact
// total-size gate for this homogeneous case.
static std::optional<mlir::Attribute>
flattenPunnedRecordInit(mlir::Attribute initAttr, cir::ArrayType target,
                        const mlir::TypeConverter *converter) {
  auto rec = mlir::dyn_cast<cir::ConstRecordAttr>(initAttr);
  if (!rec)
    return std::nullopt;
  mlir::Type convElt = converter->convertType(target.getElementType());
  if (!convElt)
    return std::nullopt;
  int64_t N = target.getSize();

  auto leafCount = [](mlir::Type t) -> int64_t {
    int64_t c = 1;
    while (auto at = mlir::dyn_cast<cir::ArrayType>(t)) {
      c *= at.getSize();
      t = at.getElementType();
    }
    return c;
  };
  auto scalarBase = [](mlir::Type t) -> mlir::Type {
    while (auto at = mlir::dyn_cast<cir::ArrayType>(t))
      t = at.getElementType();
    return t;
  };

  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(convElt)) {
    llvm::SmallVector<llvm::APInt> vals;
    vals.reserve(N);
    for (mlir::Attribute m : rec.getMembers()) {
      if (auto ca = mlir::dyn_cast<cir::ConstArrayAttr>(m)) {
        auto de = lowerConstArrayAttr(ca, converter);
        if (!de)
          return std::nullopt;
        auto dea = mlir::dyn_cast<mlir::DenseIntElementsAttr>(*de);
        if (!dea || dea.getElementType() != convElt)
          return std::nullopt;
        for (const llvm::APInt &v : dea.getValues<llvm::APInt>())
          vals.push_back(v);
      } else if (auto z = mlir::dyn_cast<cir::ZeroAttr>(m)) {
        auto zt = mlir::cast<mlir::TypedAttr>(z).getType();
        if (converter->convertType(scalarBase(zt)) != convElt)
          return std::nullopt;
        for (int64_t i = 0, e = leafCount(zt); i < e; ++i)
          vals.push_back(llvm::APInt(intTy.getWidth(), 0));
      } else if (auto ia = mlir::dyn_cast<cir::IntAttr>(m)) {
        if (converter->convertType(
                mlir::cast<mlir::TypedAttr>(ia).getType()) != convElt)
          return std::nullopt;
        vals.push_back(ia.getValue());
      } else {
        return std::nullopt;
      }
    }
    if (static_cast<int64_t>(vals.size()) != N)
      return std::nullopt;
    return mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get({N}, convElt), llvm::ArrayRef(vals));
  }

  if (auto fltTy = mlir::dyn_cast<mlir::FloatType>(convElt)) {
    llvm::SmallVector<llvm::APFloat> vals;
    vals.reserve(N);
    for (mlir::Attribute m : rec.getMembers()) {
      if (auto ca = mlir::dyn_cast<cir::ConstArrayAttr>(m)) {
        auto de = lowerConstArrayAttr(ca, converter);
        if (!de)
          return std::nullopt;
        auto dea = mlir::dyn_cast<mlir::DenseFPElementsAttr>(*de);
        if (!dea || dea.getElementType() != convElt)
          return std::nullopt;
        for (const llvm::APFloat &v : dea.getValues<llvm::APFloat>())
          vals.push_back(v);
      } else if (auto z = mlir::dyn_cast<cir::ZeroAttr>(m)) {
        auto zt = mlir::cast<mlir::TypedAttr>(z).getType();
        if (converter->convertType(scalarBase(zt)) != convElt)
          return std::nullopt;
        for (int64_t i = 0, e = leafCount(zt); i < e; ++i)
          vals.push_back(llvm::APFloat::getZero(fltTy.getFloatSemantics()));
      } else {
        return std::nullopt;
      }
    }
    if (static_cast<int64_t>(vals.size()) != N)
      return std::nullopt;
    return mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get({N}, convElt), llvm::ArrayRef(vals));
  }

  return std::nullopt;
}

// Flatten a homogeneous-leaf-scalar `#cir.const_record` (a struct global
// initializer) into a flat `DenseElementsAttr` of shape [N]. This is the S4
// const-record path: derive the single leaf scalar type + total flat element
// count from the record's field TYPES, then reuse flattenPunnedRecordInit to
// gather values (it already validates uniform element type + the leaf-count ==
// N size gate). Returns std::nullopt (a REJECT the caller turns into a loud
// diagnostic) for a union, a heterogeneous leaf type, a nested aggregate
// member, or -- the core soundness gate -- any ABI-size mismatch between the
// record's declared size and the flattened leaf bytes (implicit padding /
// mis-flattening; never silently zero-padded).
//
// Implemented here (not in LoweringHelpers.cpp) to reuse the sibling
// flattenPunnedRecordInit machinery and keep the DataLayout query local to the
// one pattern that needs it; the bare-cir.constant-holding-a-record path
// (a separate crash site) is out of scope for this increment.
static std::optional<mlir::Attribute>
lowerConstRecordAttr(cir::ConstRecordAttr rec,
                     const mlir::TypeConverter *converter,
                     const mlir::DataLayout &dataLayout) {
  auto recTy = mlir::dyn_cast<cir::RecordType>(rec.getType());
  if (!recTy || recTy.isUnion())
    return std::nullopt;

  auto leafCount = [](mlir::Type t) -> int64_t {
    int64_t c = 1;
    while (auto at = mlir::dyn_cast<cir::ArrayType>(t)) {
      c *= at.getSize();
      t = at.getElementType();
    }
    return c;
  };
  auto scalarBase = [](mlir::Type t) -> mlir::Type {
    while (auto at = mlir::dyn_cast<cir::ArrayType>(t))
      t = at.getElementType();
    return t;
  };

  // Derive the unified leaf scalar type and total flat element count from the
  // field types. A member that is not (an array of) a single scalar -- e.g. a
  // nested record or a pointer -- is rejected (struct-cluster territory).
  mlir::Type leafCir;
  int64_t N = 0;
  for (mlir::Type fieldTy : recTy.getMembers()) {
    mlir::Type base = scalarBase(fieldTy);
    if (!mlir::isa<cir::IntType, cir::BoolType, cir::FPTypeInterface>(base))
      return std::nullopt;
    if (!leafCir)
      leafCir = base;
    else if (converter->convertType(base) != converter->convertType(leafCir))
      return std::nullopt; // heterogeneous leaf types
    N += leafCount(fieldTy);
  }
  if (!leafCir || N == 0)
    return std::nullopt;

  mlir::Type convElt = converter->convertType(leafCir);
  if (!convElt)
    return std::nullopt;

  // Core soundness gate: the flattened leaf byte count must equal the record's
  // ABI size. A mismatch means implicit alignment padding or a mis-flattened
  // nested type -- either way the flat tensor would be wrong, so reject rather
  // than fabricate a padded (silent-wrong) initializer.
  unsigned leafBits = 0;
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(convElt))
    leafBits = it.getWidth();
  else if (auto ft = mlir::dyn_cast<mlir::FloatType>(convElt))
    leafBits = ft.getWidth();
  else
    return std::nullopt;
  uint64_t expectedBits = static_cast<uint64_t>(N) * leafBits;
  uint64_t declaredBits = dataLayout.getTypeSizeInBits(recTy);
  if (declaredBits != expectedBits)
    return std::nullopt;

  // Reuse the homogeneous flattener with a synthetic [N x leaf] target; it
  // re-validates element types and enforces vals.size() == N.
  auto target = cir::ArrayType::get(leafCir, N);
  return flattenPunnedRecordInit(rec, target, converter);
}

// True when any `cir.get_global` of this global feeds a `cir.cast bitcast`
// (a type-punned reinterpretation). Such a global is owned by the punned-array
// path (detectPunnedArrayGlobal): if that path accepted it, it already
// rewrote the global above; if it BAILED (e.g. two consumers pun to
// conflicting array types), the global MUST remain a loud failure and must NOT
// be silently flattened by the homogeneous const-record path -- a flat
// memref<N x leaf> would honor only one consumer's element type and drop the
// other. So the const-record flatten vetoes whenever a bitcast consumer exists.
static bool recordGlobalHasBitcastConsumer(cir::GlobalOp global) {
  auto moduleOp = global->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp)
    return false;
  bool found = false;
  moduleOp.walk([&](cir::GetGlobalOp gg) {
    if (gg.getName() != global.getSymName())
      return;
    for (auto *user : gg->getUsers())
      if (auto c = mlir::dyn_cast<cir::CastOp>(user))
        if (c.getKind() == cir::CastKind::bitcast)
          found = true;
  });
  return found;
}

class CIRGlobalOpLowering : public mlir::OpConversionPattern<cir::GlobalOp> {
public:
  using OpConversionPattern<cir::GlobalOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(cir::GlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp)
      return mlir::failure();

    mlir::OpBuilder b(moduleOp.getContext());

    // Narrow type-punned array-global exception: a record-typed global whose
    // every get_global user is a bitcast to the same flat array type lowers
    // directly to that array's memref, folding the record away. See
    // detectPunnedArrayGlobal. Fails loudly if the initializer cannot be
    // flattened to a homogeneous array (that would be a struct-cluster shape).
    if (auto punTarget = detectPunnedArrayGlobal(op)) {
      auto convElt = getTypeConverter()->convertType(punTarget.getElementType());
      if (!convElt)
        return op.emitError("ThroughMLIR: cannot convert element type of "
                            "type-punned array global @")
               << op.getSymName();
      auto memrefType = mlir::MemRefType::get({static_cast<int64_t>(punTarget.getSize())}, convElt);
      mlir::Attribute initialValue;
      if (auto init = op.getInitialValue()) {
        auto flat =
            flattenPunnedRecordInit(*init, punTarget, getTypeConverter());
        if (!flat)
          return op.emitError("ThroughMLIR: type-punned array global @")
                 << op.getSymName()
                 << " has an initializer that cannot be flattened to a "
                    "homogeneous array (non-uniform element type, padding, or "
                    "nested aggregate); this belongs to the struct-lowering "
                    "cluster";
        initialValue = *flat;
      }
      mlir::IntegerAttr punAlignment =
          op.getAlignment()
              ? mlir::IntegerAttr::get(b.getI64Type(), op.getAlignment().value())
              : mlir::IntegerAttr();
      std::string punVisibility = op.isPrivate() ? "private" : "public";
      rewriter.replaceOpWithNewOp<mlir::memref::GlobalOp>(
          op, b.getStringAttr(op.getSymName()),
          /*sym_visibility=*/b.getStringAttr(punVisibility),
          /*type=*/memrefType, initialValue,
          /*constant=*/op.getConstant(),
          /*alignment=*/punAlignment);
      return mlir::success();
    }

    // Homogeneous-leaf const-record global (S4). A `!cir.record` global whose
    // initializer is a `#cir.const_record` of a single leaf scalar type
    // flattens to a flat `memref<N x leaf>` (mirrors the punned-array path, but
    // driven by the record's own shape rather than a bitcast consumer). This
    // composes with -- does not double-claim -- detectPunnedArrayGlobal above:
    // that fires only when every consumer is a bitcast-to-array, which returns
    // before reaching here. A record that cannot be soundly flattened
    // (heterogeneous leaves, implicit padding, nested aggregate, union) is a
    // loud diagnostic, never the terminal llvm_unreachable.
    if (auto init = op.getInitialValue()) {
      if (auto recAttr = mlir::dyn_cast<cir::ConstRecordAttr>(*init);
          recAttr && !recordGlobalHasBitcastConsumer(op)) {
        mlir::DataLayout dl(moduleOp);
        auto flat = lowerConstRecordAttr(recAttr, getTypeConverter(), dl);
        if (!flat)
          return op.emitError("ThroughMLIR: const-record global @")
                 << op.getSymName()
                 << " cannot be flattened to a homogeneous array (non-uniform "
                    "leaf type, implicit alignment padding, nested aggregate, "
                    "or union); this belongs to the struct-lowering cluster";
        auto dense = mlir::cast<mlir::DenseElementsAttr>(*flat);
        auto tt = mlir::cast<mlir::RankedTensorType>(dense.getType());
        auto memrefType =
            mlir::MemRefType::get(tt.getShape(), tt.getElementType());
        mlir::IntegerAttr recAlignment =
            op.getAlignment()
                ? mlir::IntegerAttr::get(b.getI64Type(), op.getAlignment().value())
                : mlir::IntegerAttr();
        std::string recVisibility = op.isPrivate() ? "private" : "public";
        rewriter.replaceOpWithNewOp<mlir::memref::GlobalOp>(
            op, b.getStringAttr(op.getSymName()),
            /*sym_visibility=*/b.getStringAttr(recVisibility),
            /*type=*/memrefType, dense,
            /*constant=*/op.getConstant(),
            /*alignment=*/recAlignment);
        return mlir::success();
      }
    }

    const auto CIRSymType = op.getSymType();
    auto convertedType = convertTypeForMemory(*getTypeConverter(), CIRSymType);
    if (!convertedType)
      return mlir::failure();
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(convertedType);
    if (!memrefType) {
      auto maybeAddrSpace = getTypeConverter()->convertTypeAttribute(
          CIRSymType, op.getAddrSpaceAttr());
      mlir::Attribute addrSpace = maybeAddrSpace.value_or(mlir::Attribute());
      memrefType = mlir::MemRefType::get(
          {1}, convertedType, mlir::MemRefLayoutAttrInterface(), addrSpace);
    }
    // Add an optional alignment to the global memref.
    mlir::IntegerAttr memrefAlignment =
        op.getAlignment()
            ? mlir::IntegerAttr::get(b.getI64Type(), op.getAlignment().value())
            : mlir::IntegerAttr();
    // Add an optional initial value to the global memref.
    mlir::Attribute initialValue = mlir::Attribute();
    std::optional<mlir::Attribute> init = op.getInitialValue();
    if (init.has_value()) {
      if (auto constArr = mlir::dyn_cast<cir::ConstArrayAttr>(init.value())) {
        init = lowerConstArrayAttr(constArr, getTypeConverter());
        if (init.has_value())
          initialValue = init.value();
        else
          return op.emitError("ThroughMLIR: array global @")
                 << op.getSymName()
                 << " has an initializer that cannot be lowered to a dense "
                    "constant (unsupported element attribute)";
      } else if (mlir::isa<cir::ConstComplexAttr>(init.value())) {
        // D6: upstream removed lowerConstComplexAttr; complex global-init
        // lowering is not yet ported. Fail loudly via a diagnostic (stays loud
        // even in assertions-off builds) rather than llvm_unreachable.
        return op.emitError("ThroughMLIR: lowering of a complex global "
                            "initializer is not yet implemented");
      } else if (auto zeroAttr = mlir::dyn_cast<cir::ZeroAttr>(init.value())) {
        (void)zeroAttr;
        // Build the zero splat with getZeroAttr so the fill value carries the
        // element type's real bit width. A raw `DenseIntElementsAttr::get(rtt,
        // 0)` deduces a 32-bit APInt and asserts (isValidIntOrFloat) on wider
        // element types, e.g. a `#cir.zero : !cir.array<!u64i x N>` global.
        auto elementType =
            memrefType.getShape().size() ? memrefType.getElementType()
                                         : convertedType;
        auto rtt = memrefType.getShape().size()
                       ? mlir::RankedTensorType::get(memrefType.getShape(),
                                                     elementType)
                       : mlir::RankedTensorType::get({1}, elementType);
        if (mlir::isa<mlir::IntegerType, mlir::FloatType>(elementType))
          initialValue = rewriter.getZeroAttr(rtt);
        else
          initialValue = mlir::Attribute();
      } else if (auto intAttr = mlir::dyn_cast<cir::IntAttr>(init.value())) {
        auto rtt = mlir::RankedTensorType::get({1}, convertedType);
        initialValue = mlir::DenseIntElementsAttr::get(rtt, intAttr.getValue());
      } else if (auto fltAttr = mlir::dyn_cast<cir::FPAttr>(init.value())) {
        auto rtt = mlir::RankedTensorType::get({1}, convertedType);
        initialValue = mlir::DenseFPElementsAttr::get(rtt, fltAttr.getValue());
      } else if (auto boolAttr = mlir::dyn_cast<cir::BoolAttr>(init.value())) {
        auto rtt = mlir::RankedTensorType::get({1}, convertedType);
        initialValue =
            mlir::DenseIntElementsAttr::get(rtt, (char)boolAttr.getValue());
      } else
        return op.emitError("ThroughMLIR: global @")
               << op.getSymName() << " has an unsupported initializer kind ("
               << init.value()
               << "); const-record/array globals flatten earlier, other kinds "
                  "(e.g. indexed global views) are not yet implemented";
    }

    // Add symbol visibility
    std::string sym_visibility = op.isPrivate() ? "private" : "public";

    rewriter.replaceOpWithNewOp<mlir::memref::GlobalOp>(
        op, b.getStringAttr(op.getSymName()),
        /*sym_visibility=*/b.getStringAttr(sym_visibility),
        /*type=*/memrefType, initialValue,
        /*constant=*/op.getConstant(),
        /*alignment=*/memrefAlignment);

    return mlir::success();
  }
};

class CIRGetGlobalOpLowering
    : public mlir::OpConversionPattern<cir::GetGlobalOp> {
public:
  using OpConversionPattern<cir::GetGlobalOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::GetGlobalOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // FIXME(cir): Premature DCE to avoid lowering stuff we're not using.
    // CIRGen should mitigate this and not emit the get_global.
    if (op->getUses().empty()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }
    // A get_global of a FUNCTION symbol (`!cir.ptr<!cir.func<...>>`) has no
    // memref storage model. Its only sound use for hardware is as the callee of
    // an indirect `cir.call` that ClangIR emitted for a no-prototype / K&R
    // direct call — CIRCallOpLowering folds that to a direct `func.call`,
    // reading the callee symbol from this (original) op. When every use is such
    // a call callee, erase this op (the call fold drops the operand); the call
    // pattern is order-independent because it inspects the original op. Any
    // OTHER use (stored/compared/passed) is a genuine function pointer with no
    // hardware realization and is rejected by CIRCallOpLowering / left to fail.
    if (mlir::isa<cir::FuncType>(op.getType().getPointee())) {
      // `!cir.ptr<!cir.func<...>>` has no memref storage model, so a consuming
      // indirect `cir.call` can never have its adaptor built (its callee
      // operand is unconvertible) — meaning CIRCallOpLowering is never even
      // invoked for it. Fold the whole no-prototype/K&R shape HERE: resolve the
      // callee symbol to a DEFINED function and rewrite each indirect call to a
      // direct `func.call`, then erase this get_global. Any other use is a
      // genuine runtime function pointer with no hardware realization.
      bool allUsesAreCallCallee = llvm::all_of(op->getUses(), [&](auto &use) {
        auto call = mlir::dyn_cast<cir::CallOp>(use.getOwner());
        return call && call.isIndirect() &&
               use.getOperandNumber() == 0; // callee operand
      });
      if (!allUsesAreCallCallee)
        return op.emitError()
               << "get_global of function symbol @" << op.getName()
               << " that is not a direct call callee (a runtime function "
                  "pointer) has no hardware realization";

      mlir::Operation *sym =
          mlir::SymbolTable::lookupNearestSymbolFrom(op, op.getNameAttr());
      bool isDefined = false;
      if (auto cf = mlir::dyn_cast_or_null<cir::FuncOp>(sym))
        isDefined = !cf.isDeclaration();
      else if (auto ff = mlir::dyn_cast_or_null<mlir::func::FuncOp>(sym))
        isDefined = !ff.isDeclaration();
      if (!isDefined)
        return op.emitError()
               << "indirect call to function symbol @" << op.getName()
               << " with no in-module definition is not supported (no "
                  "hardware realization for a runtime function pointer)";

      for (mlir::Operation *user :
           llvm::make_early_inc_range(op->getUsers())) {
        auto call = mlir::cast<cir::CallOp>(user);
        llvm::SmallVector<mlir::Type> resTypes;
        if (mlir::failed(getTypeConverter()->convertTypes(
                call.getResultTypes(), resTypes)))
          return mlir::failure();
        llvm::SmallVector<mlir::Value> args;
        for (mlir::Value arg : call.getArgOperands())
          args.push_back(rewriter.getRemappedValue(arg));
        rewriter.setInsertionPoint(call);
        auto newCall = mlir::func::CallOp::create(
            rewriter, call.getLoc(), op.getNameAttr(), resTypes, args);
        rewriter.replaceOp(call, newCall.getResults());
      }
      rewriter.eraseOp(op);
      return mlir::success();
    }
    // Mirror of the type-punned array-global exception in CIRGlobalOpLowering:
    // when the referenced global is punned, produce a get_global of the flat
    // array memref directly so the recovering bitcast becomes an identity on
    // the converted operand (handled by the bitcast case in CIRCastOpLowering).
    if (auto punTarget = punnedTargetOfGetGlobal(op)) {
      auto convElt =
          getTypeConverter()->convertType(punTarget.getElementType());
      if (!convElt)
        return op.emitError("ThroughMLIR: cannot convert element type of "
                            "type-punned array global @")
               << op.getName();
      auto memrefType = mlir::MemRefType::get(
          {static_cast<int64_t>(punTarget.getSize())}, convElt);
      auto getGlobalOp = mlir::memref::GetGlobalOp::create(
          rewriter, op.getLoc(), memrefType, op.getName());
      // Replace the punning bitcast users directly rather than the get_global:
      // the get_global's result type is `!cir.ptr<record>`, which has NO
      // conversion (no RecordType rule — struct-cluster territory), so replacing
      // it would leave the consuming bitcast with an unconvertible operand and
      // strand it behind unresolved materializations. Each bitcast's *result*
      // type `!cir.ptr<array<T x N>>` DOES convert to the same memref<N x T>, so
      // replacing the bitcasts is a clean same-converted-type substitution and
      // folds the record pun away entirely.
      for (mlir::Operation *user :
           llvm::make_early_inc_range(op->getUsers())) {
        // Guaranteed a bitcast by punnedTargetOfGetGlobal.
        rewriter.replaceOp(user, getGlobalOp.getResult());
      }
      rewriter.eraseOp(op);
      return mlir::success();
    }
    auto globalOpType =
        convertTypeForMemory(*getTypeConverter(), op.getType().getPointee());
    if (!globalOpType)
      return mlir::failure();
    auto resultType = getTypeConverter()->convertType(op.getType());
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(globalOpType);
    if (!memrefType) {
      mlir::MemRefType resultTypeMemref =
          mlir::cast<mlir::MemRefType>(resultType);
      memrefType = mlir::MemRefType::get({1}, resultTypeMemref.getElementType(),
                                         resultTypeMemref.getLayout(),
                                         resultTypeMemref.getMemorySpace());
    }

    auto symbol = op.getName();
    auto getGlobalOp = mlir::memref::GetGlobalOp::create(rewriter, op.getLoc(),
                                                         memrefType, symbol);

    if (isa<cir::ArrayType>(op.getType().getPointee())) {
      rewriter.replaceOp(op, getGlobalOp);
    } else {
      // Cast from memref<1xmlirType> to memref<?xmlirType>. This is needed
      // since Typeconverter produces memref<?xmlirType> for non-array cir.ptrs.
      // The cast will be eliminated later in load/store-lowering.
      auto castOp = mlir::memref::CastOp::create(rewriter, op.getLoc(),
                                                 resultType, getGlobalOp);
      rewriter.replaceOp(op, castOp);
    }
    return mlir::success();
  }
};

class CIRComplexCreateOpLowering
    : public mlir::OpConversionPattern<cir::ComplexCreateOp> {
public:
  using OpConversionPattern<cir::ComplexCreateOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ComplexCreateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto vecType = mlir::cast<mlir::VectorType>(
        getTypeConverter()->convertType(op.getType()));
    auto zeroAttr = rewriter.getZeroAttr(vecType);
    mlir::Value result =
        mlir::arith::ConstantOp::create(rewriter, loc, vecType, zeroAttr)
            .getResult();
    SmallVector<int64_t, 1> realIdx{0};
    SmallVector<int64_t, 1> imagIdx{1};
    result = mlir::vector::InsertOp::create(rewriter, loc, adaptor.getReal(),
                                            result, realIdx)
                 .getResult();
    result = mlir::vector::InsertOp::create(rewriter, loc, adaptor.getImag(),
                                            result, imagIdx)
                 .getResult();
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class CIRComplexRealOpLowering
    : public mlir::OpConversionPattern<cir::ComplexRealOp> {
public:
  using OpConversionPattern<cir::ComplexRealOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ComplexRealOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SmallVector<int64_t, 1> idx{0};
    rewriter.replaceOpWithNewOp<mlir::vector::ExtractOp>(
        op, adaptor.getOperand(), idx);
    return mlir::success();
  }
};

class CIRComplexImagOpLowering
    : public mlir::OpConversionPattern<cir::ComplexImagOp> {
public:
  using OpConversionPattern<cir::ComplexImagOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::ComplexImagOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    SmallVector<int64_t, 1> idx{1};
    rewriter.replaceOpWithNewOp<mlir::vector::ExtractOp>(
        op, adaptor.getOperand(), idx);
    return mlir::success();
  }
};

class CIRVectorCreateLowering
    : public mlir::OpConversionPattern<cir::VecCreateOp> {
public:
  using OpConversionPattern<cir::VecCreateOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecCreateOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto vecTy = mlir::dyn_cast<cir::VectorType>(op.getType());
    assert(vecTy && "result type of cir.vec.create op is not VectorType");
    auto elementTy = typeConverter->convertType(vecTy.getElementType());
    auto loc = op.getLoc();
    auto zeroElement = rewriter.getZeroAttr(elementTy);
    mlir::Value vectorVal = mlir::arith::ConstantOp::create(
        rewriter, loc,
        mlir::DenseElementsAttr::get(
            mlir::VectorType::get(vecTy.getSize(), elementTy), zeroElement));
    assert(vecTy.getSize() == op.getElements().size() &&
           "cir.vec.create op count doesn't match vector type elements count");
    for (uint64_t i = 0; i < vecTy.getSize(); ++i) {
      SmallVector<int64_t, 1> position{static_cast<int64_t>(i)};
      vectorVal = mlir::vector::InsertOp::create(rewriter, loc,
                                                 adaptor.getElements()[i],
                                                 vectorVal, position)
                      .getResult();
    }
    rewriter.replaceOp(op, vectorVal);
    return mlir::success();
  }
};

class CIRVectorInsertLowering
    : public mlir::OpConversionPattern<cir::VecInsertOp> {
public:
  using OpConversionPattern<cir::VecInsertOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecInsertOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value index = adaptor.getIndex();
    if (!mlir::isa<mlir::IndexType>(index.getType()))
      index = mlir::arith::IndexCastOp::create(rewriter, op.getLoc(),
                                               rewriter.getIndexType(), index);
    SmallVector<mlir::OpFoldResult, 1> position{index};
    auto newVec = mlir::vector::InsertOp::create(
        rewriter, op.getLoc(), adaptor.getValue(), adaptor.getVec(), position);
    rewriter.replaceOp(op, newVec.getResult());
    return mlir::success();
  }
};

class CIRVectorExtractLowering
    : public mlir::OpConversionPattern<cir::VecExtractOp> {
public:
  using OpConversionPattern<cir::VecExtractOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecExtractOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value index = adaptor.getIndex();
    if (!mlir::isa<mlir::IndexType>(index.getType()))
      index = mlir::arith::IndexCastOp::create(rewriter, op.getLoc(),
                                               rewriter.getIndexType(), index);
    SmallVector<mlir::OpFoldResult, 1> position{index};
    auto extracted = mlir::vector::ExtractOp::create(
        rewriter, op.getLoc(), adaptor.getVec(), position);
    rewriter.replaceOp(op, extracted.getResult());
    return mlir::success();
  }
};

class CIRVectorCmpOpLowering : public mlir::OpConversionPattern<cir::VecCmpOp> {
public:
  using OpConversionPattern<cir::VecCmpOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::VecCmpOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(mlir::isa<cir::VectorType>(op.getType()) &&
           mlir::isa<cir::VectorType>(op.getLhs().getType()) &&
           mlir::isa<cir::VectorType>(op.getRhs().getType()) &&
           "Vector compare with non-vector type");
    auto elementType =
        mlir::cast<cir::VectorType>(op.getLhs().getType()).getElementType();
    mlir::Value bitResult;
    if (auto intType = mlir::dyn_cast<cir::IntType>(elementType)) {
      bitResult = mlir::arith::CmpIOp::create(
          rewriter, op.getLoc(),
          convertCmpKindToCmpIPredicate(op.getKind(), intType.isSigned()),
          adaptor.getLhs(), adaptor.getRhs());
    } else if (mlir::isa<cir::FPTypeInterface>(elementType)) {
      bitResult = mlir::arith::CmpFOp::create(
          rewriter, op.getLoc(), convertCmpKindToCmpFPredicate(op.getKind()),
          adaptor.getLhs(), adaptor.getRhs());
    } else {
      return op.emitError() << "unsupported type for VecCmpOp: " << elementType;
    }
    rewriter.replaceOpWithNewOp<mlir::arith::ExtSIOp>(
        op, typeConverter->convertType(op.getType()), bitResult);
    return mlir::success();
  }
};

class CIRCastOpLowering : public mlir::OpConversionPattern<cir::CastOp> {
public:
  using OpConversionPattern<cir::CastOp>::OpConversionPattern;

  inline mlir::Type convertTy(mlir::Type ty) const {
    return getTypeConverter()->convertType(ty);
  }

  mlir::LogicalResult
  matchAndRewrite(cir::CastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (isa<cir::VectorType>(op.getSrc().getType()))
      llvm_unreachable("CastOp lowering for vector type is not supported yet");
    auto src = adaptor.getSrc();
    auto dstType = op.getType();
    using CIR = cir::CastKind;
    switch (op.getKind()) {
    case CIR::array_to_ptrdecay: {
      auto newDstType = llvm::cast<mlir::MemRefType>(convertTy(dstType));
      llvm::SmallVector<mlir::OpFoldResult> sizes, strides;
      if (mlir::failed(prepareReinterpretMetadata(newDstType, src, rewriter,
                                                  sizes, strides,
                                                  op.getOperation())))
        return mlir::failure();
      rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
          op, newDstType, src, rewriter.getIndexAttr(0), sizes, strides);
      return mlir::success();
    }
    case CIR::int_to_bool: {
      auto zero =
          cir::ConstantOp::create(rewriter, src.getLoc(), op.getSrc().getType(),
                                  cir::IntAttr::get(op.getSrc().getType(), 0));
      rewriter.replaceOpWithNewOp<cir::CmpOp>(
          op, cir::BoolType::get(getContext()), cir::CmpOpKind::ne, op.getSrc(),
          zero);
      return mlir::success();
    }
    case CIR::integral: {
      auto newDstType = convertTy(dstType);
      auto srcType = op.getSrc().getType();
      cir::IntType srcIntType = mlir::cast<cir::IntType>(srcType);
      auto newOp =
          createIntCast(rewriter, src, newDstType, srcIntType.isSigned());
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    }
    case CIR::floating: {
      auto newDstType = convertTy(dstType);
      auto srcTy = op.getSrc().getType();
      auto dstTy = op.getType();

      if (!mlir::isa<cir::FPTypeInterface>(dstTy) ||
          !mlir::isa<cir::FPTypeInterface>(srcTy))
        return op.emitError() << "NYI cast from " << srcTy << " to " << dstTy;

      auto getFloatWidth = [](mlir::Type ty) -> unsigned {
        return mlir::cast<cir::FPTypeInterface>(ty).getWidth();
      };

      if (getFloatWidth(srcTy) > getFloatWidth(dstTy))
        rewriter.replaceOpWithNewOp<mlir::arith::TruncFOp>(op, newDstType, src);
      else
        rewriter.replaceOpWithNewOp<mlir::arith::ExtFOp>(op, newDstType, src);
      return mlir::success();
    }
    case CIR::float_to_bool: {
      auto kind = mlir::arith::CmpFPredicate::UNE;

      // Check if float is not equal to zero.
      auto zeroFloat = mlir::arith::ConstantOp::create(
          rewriter, op.getLoc(), src.getType(),
          mlir::FloatAttr::get(src.getType(), 0.0));

      rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(op, kind, src,
                                                       zeroFloat);
      return mlir::success();
    }
    case CIR::bool_to_int: {
      auto dstTy = mlir::cast<cir::IntType>(op.getType());
      auto newDstType = mlir::cast<mlir::IntegerType>(convertTy(dstTy));
      auto newOp = createIntCast(rewriter, src, newDstType);
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    }
    case CIR::bool_to_float: {
      auto dstTy = op.getType();
      auto newDstType = convertTy(dstTy);
      rewriter.replaceOpWithNewOp<mlir::arith::UIToFPOp>(op, newDstType, src);
      return mlir::success();
    }
    case CIR::int_to_float: {
      auto dstTy = op.getType();
      auto newDstType = convertTy(dstTy);
      if (mlir::cast<cir::IntType>(op.getSrc().getType()).isSigned())
        rewriter.replaceOpWithNewOp<mlir::arith::SIToFPOp>(op, newDstType, src);
      else
        rewriter.replaceOpWithNewOp<mlir::arith::UIToFPOp>(op, newDstType, src);
      return mlir::success();
    }
    case CIR::float_to_int: {
      auto dstTy = op.getType();
      auto newDstType = convertTy(dstTy);
      if (mlir::cast<cir::IntType>(op.getType()).isSigned())
        rewriter.replaceOpWithNewOp<mlir::arith::FPToSIOp>(op, newDstType, src);
      else
        rewriter.replaceOpWithNewOp<mlir::arith::FPToUIOp>(op, newDstType, src);
      return mlir::success();
    }
    case CIR::bitcast: {
      auto dstConv = convertTy(dstType);
      if (!dstConv)
        return op.emitError("ThroughMLIR: unsupported bitcast — destination "
                            "type ")
               << dstType << " has no MLIR conversion";
      // Same converted type: an identity forward. This is the common case for
      // the type-punned array global (the get_global already lowered to the
      // flat array memref, so recovering `ptr<rec>`->`ptr<array<T x N>>`
      // converts to the same memref).
      if (src.getType() == dstConv) {
        rewriter.replaceOp(op, src);
        return mlir::success();
      }
      // Scalar same-bit-width reinterpretation (e.g. the u64<->double punning
      // that `--cir-lower-union-punning` leaves behind): a value-level
      // int<->float bitcast between equal-width scalars lowers directly to
      // `arith.bitcast`. Bit-exact by construction; the width equality is the
      // soundness gate.
      {
        auto srcScalar = src.getType();
        auto isScalar = [](mlir::Type t) {
          return mlir::isa<mlir::IntegerType, mlir::FloatType>(t);
        };
        auto scalarWidth = [](mlir::Type t) -> unsigned {
          if (auto i = mlir::dyn_cast<mlir::IntegerType>(t))
            return i.getWidth();
          return mlir::cast<mlir::FloatType>(t).getWidth();
        };
        if (isScalar(srcScalar) && isScalar(dstConv) &&
            scalarWidth(srcScalar) == scalarWidth(dstConv)) {
          rewriter.replaceOpWithNewOp<mlir::arith::BitcastOp>(op, dstConv, src);
          return mlir::success();
        }
      }
      // Any bitcast whose converted operand and result are DIFFERENT memrefs
      // (a genuine byte-level reinterpretation, e.g. array<i64 x 2> viewed as
      // array<i32 x 4>) is honestly rejected. Such a view changes element-wise
      // indexing, and the downstream load/store base-finding
      // (findBaseAndIndices) is not plumbed to walk a reinterpret_cast that
      // sits between two allocations of differing element type; emitting one
      // would either crash that walk or fabricate a mis-indexed access. No
      // benchmark in the ptr/array cluster needs this (every observed pun is a
      // record<->flat-array identity handled above), so it is surfaced rather
      // than shipped half-working. Extend here only with an end-to-end numeric
      // check for the new shape.
      return op.emitError("ThroughMLIR: unsupported bitcast from ")
             << op.getSrc().getType() << " to " << dstType
             << " (only same-converted-type forwarding is supported; a "
                "byte-level memref reinterpretation across differing element "
                "types is not yet implemented)";
    }
    default:
      break;
    }
    return mlir::failure();
  }
};

class CIRGetElementOpLowering
    : public mlir::OpConversionPattern<cir::GetElementOp> {
  using mlir::OpConversionPattern<cir::GetElementOp>::OpConversionPattern;

  bool isLoadStoreOrGetProducer(cir::GetElementOp op) const {
    for (auto *user : op->getUsers()) {
      if (!op->isBeforeInBlock(user))
        return false;
      if (isa<cir::LoadOp, cir::StoreOp, cir::GetElementOp>(*user))
        continue;
      return false;
    }
    return true;
  }

  // Rewrite
  //        cir.get_element(%base[%index])
  // to
  //        memref.reinterpret_cast (%base, %stride)
  //
  // MemRef Dialect doesn't have GEP-like operation. memref.reinterpret_cast
  // only been used to propagate %base and %index to memref.load/store and
  // should be erased after the conversion.
  mlir::LogicalResult
  matchAndRewrite(cir::GetElementOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Only rewrite if all users are load/stores.
    if (!isLoadStoreOrGetProducer(op))
      return mlir::failure();

    // Cast the index to the index type, if needed.
    auto index = adaptor.getIndex();
    auto indexType = rewriter.getIndexType();
    if (index.getType() != indexType)
      index = mlir::arith::IndexCastOp::create(rewriter, op.getLoc(), indexType,
                                               index);

    // Convert the destination type.
    auto dstType =
        cast<mlir::MemRefType>(getTypeConverter()->convertType(op.getType()));

    // For an N-D array, the multi-index access is reconstructed at the
    // load/store site by collapseGetElementChain, which walks the ORIGINAL
    // cir.get_element chain and emits a single memref.load/store on the N-D
    // base (exact row-major semantics, proven by the transpose-read numeric
    // guard). The reinterpret_cast this pattern emits for an intermediate row
    // slice then has no live consumer and is reconciled away — so a
    // rank-reducing view is not materialized here for the common case. A
    // get_element whose result genuinely ESCAPES as a value (stored into a
    // pointer variable, passed to a call) with a dynamic index cannot be a
    // valid identity-offset reinterpret_cast and is left to fail as an honest
    // hard error rather than a silent mis-indexed access (e.g. CHStone sha's
    // `char *p = &buf[i]` — a pointer-aliasing gap owned elsewhere).
    //
    // Peel any type-mismatch materialization off the base so a chained
    // get_element sees its parent reinterpret_cast directly.
    mlir::Value base = lookThroughMemrefMaterialization(adaptor.getBase());

    // Replace the GetElementOp with a memref.reinterpret_cast.
    llvm::SmallVector<mlir::OpFoldResult> sizes, strides;
    if (mlir::failed(prepareReinterpretMetadata(dstType, base, rewriter, sizes,
                                                strides, op.getOperation())))
      return mlir::failure();
    rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
        op, dstType, base,
        /*offset=*/index,
        /*sizes=*/sizes,
        /*strides=*/strides);

    return mlir::success();
  }
};

class CIRPtrStrideOpLowering
    : public mlir::OpConversionPattern<cir::PtrStrideOp> {
public:
  using mlir::OpConversionPattern<cir::PtrStrideOp>::OpConversionPattern;

  // Return true if PtrStrideOp is produced by cast with array_to_ptrdecay kind
  // and they are in the same block.
  inline bool isCastArrayToPtrConsumer(cir::PtrStrideOp op) const {
    auto castOp = op->getOperand(0).getDefiningOp<cir::CastOp>();
    if (!castOp)
      return false;
    if (castOp.getKind() != cir::CastKind::array_to_ptrdecay)
      return false;
    if (!castOp->hasOneUse())
      return false;
    if (!castOp->isBeforeInBlock(op))
      return false;
    return true;
  }

  // Return true if all the PtrStrideOp users are load, store or cast
  // with array_to_ptrdecay kind and they are in the same block.
  inline bool isLoadStoreOrCastArrayToPtrProduer(cir::PtrStrideOp op) const {
    if (op.use_empty())
      return false;
    for (auto *user : op->getUsers()) {
      if (!op->isBeforeInBlock(user))
        return false;
      if (isa<cir::LoadOp, cir::StoreOp, cir::GetElementOp>(*user))
        continue;
      auto castOp = dyn_cast<cir::CastOp>(*user);
      if (castOp && (castOp.getKind() == cir::CastKind::array_to_ptrdecay))
        continue;
      return false;
    }
    return true;
  }

  inline mlir::Type convertTy(mlir::Type ty) const {
    return getTypeConverter()->convertType(ty);
  }

  // Rewrite
  //        cir.ptr_stride(%base, %stride)
  // to
  //        memref.reinterpret_cast (%base, %stride)
  //
  mlir::LogicalResult rewritePtrStrideToReinterpret(
      cir::PtrStrideOp op, mlir::Value base, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const {
    auto ptrType = op.getType();
    auto memrefType = llvm::cast<mlir::MemRefType>(convertTy(ptrType));
    auto stride = adaptor.getStride();
    auto indexType = rewriter.getIndexType();
    // Generate casting if the stride is not index type.
    if (stride.getType() != indexType)
      stride = mlir::arith::IndexCastOp::create(rewriter, op.getLoc(),
                                                indexType, stride);

    // Build the sizes/strides metadata from the result memref so it stays
    // consistent with the result rank. Passing empty ValueRanges here left the
    // static size/stride arrays empty while the result type carried a rank >= 1,
    // and the reinterpret_cast constant folder then aborts in getMixedValues
    // ("expected the rank of dynamic values to match ...") — the exact crash on
    // a rank>1 ptr_stride (e.g. backprop's 2-D array access). The offset is the
    // dynamic stride; the sizes and strides come from the result layout.
    llvm::SmallVector<mlir::OpFoldResult> sizes, strides;
    if (mlir::failed(prepareReinterpretMetadata(memrefType, base, rewriter,
                                                sizes, strides,
                                                op.getOperation())))
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::memref::ReinterpretCastOp>(
        op, memrefType, base, /*offset=*/mlir::OpFoldResult(stride), sizes,
        strides);

    return mlir::success();
  }

  // Rewrite
  //        %0 = cir.cast array_to_ptrdecay %base
  //        cir.ptr_stride(%0, %stride)
  // to
  //        memref.reinterpret_cast (%base, %stride)
  //
  // MemRef Dialect doesn't have GEP-like operation. memref.reinterpret_cast
  // only been used to propogate %base and %stride to memref.load/store and
  // should be erased after the conversion.
  mlir::LogicalResult
  rewriteArrayDecay(cir::PtrStrideOp op, OpAdaptor adaptor,
                    mlir::ConversionPatternRewriter &rewriter) const {
    auto baseDefiningOp =
        adaptor.getBase().getDefiningOp<mlir::memref::ReinterpretCastOp>();
    if (!baseDefiningOp)
      return mlir::failure();

    if (mlir::failed(rewritePtrStrideToReinterpret(
            op, baseDefiningOp->getOperand(0), adaptor, rewriter)))
      return mlir::failure();

    rewriter.eraseOp(baseDefiningOp);
    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewrite(cir::PtrStrideOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (isLoadStoreOrCastArrayToPtrProduer(op)) {
      if (isCastArrayToPtrConsumer(op))
        return rewriteArrayDecay(op, adaptor, rewriter);
      else if (!isa<cir::ArrayType>(op.getType().getPointee()))
        return rewritePtrStrideToReinterpret(op, adaptor.getBase(), adaptor,
                                             rewriter);
    }

    auto base = adaptor.getBase();
    auto stride = adaptor.getStride();

    auto ptrType = op.getType();

    int mulSize = 1;
    auto innerMostPointee = ptrType.getPointee();
    while (auto t1 = mlir::dyn_cast<cir::ArrayType>(innerMostPointee)) {
      mulSize *= t1.getSize();
      innerMostPointee = t1.getElementType();
    }

    auto elementType = convertTy(innerMostPointee);

    auto ptrPtrType = mlir::ptr::PtrType::get(
        rewriter.getContext(),
        mlir::ptr::GenericSpaceAttr::get(op->getContext()));

    mlir::Value elemSizeVal = mlir::ptr::TypeOffsetOp::create(
        rewriter, op.getLoc(), rewriter.getIndexType(), elementType);

    mlir::Value strideIndex = mlir::arith::IndexCastOp::create(
        rewriter, op.getLoc(), rewriter.getIndexType(), stride);

    mlir::Value offsetInnerMost = mlir::arith::MulIOp::create(
        rewriter, op.getLoc(), strideIndex, elemSizeVal);

    mlir::Value offset;
    if (mulSize > 1) {
      mlir::Value mulSizeConst =
          mlir::arith::ConstantIndexOp::create(rewriter, op->getLoc(), mulSize);
      offset = mlir::arith::MulIOp::create(rewriter, op.getLoc(),
                                           offsetInnerMost, mulSizeConst);
    } else {
      offset = offsetInnerMost;
    }

    auto t1 = mlir::cast<mlir::MemRefType>(base.getType());
    auto t2 =
        mlir::MemRefType::get(t1.getShape(), t1.getElementType(),
                              t1.getLayout(), ptrPtrType.getMemorySpace());

    auto ptrMetaType = mlir::ptr::PtrMetadataType::get(t2);

    auto fixedBase = mlir::memref::MemorySpaceCastOp::create(
        rewriter, op->getLoc(), t2, base);

    auto getMetadataOp = mlir::ptr::GetMetadataOp::create(
        rewriter, op->getLoc(), ptrMetaType, fixedBase);

    auto toPtrOp = mlir::ptr::ToPtrOp::create(rewriter, op->getLoc(),
                                              ptrPtrType, fixedBase);

    auto ptrAddOp = mlir::ptr::PtrAddOp::create(rewriter, op.getLoc(),
                                                ptrPtrType, toPtrOp, offset);

    auto fromPtrOp = mlir::ptr::FromPtrOp::create(rewriter, op.getLoc(), t2,
                                                  ptrAddOp, getMetadataOp);

    auto memrefCastOp = mlir::memref::MemorySpaceCastOp::create(
        rewriter, op.getLoc(), t1, fromPtrOp);

    rewriter.replaceOp(op, memrefCastOp);
    return mlir::success();
  }
};

class CIRSelectOpLowering : public mlir::OpConversionPattern<cir::SelectOp> {
public:
  using OpConversionPattern<cir::SelectOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::SelectOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(
        op, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return mlir::success();
  }
};

class CIRUnreachableOpLowering
    : public mlir::OpConversionPattern<cir::UnreachableOp> {
public:
  using OpConversionPattern<cir::UnreachableOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::UnreachableOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(op);
    return mlir::success();
  }
};

class CIRTrapOpLowering : public mlir::OpConversionPattern<cir::TrapOp> {
public:
  using OpConversionPattern<cir::TrapOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::TrapOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPointAfter(op);
    auto trapIntrinsicName = rewriter.getStringAttr("llvm.trap");
    mlir::LLVM::CallIntrinsicOp::create(rewriter, op.getLoc(),
                                        trapIntrinsicName,
                                        /*args=*/mlir::ValueRange());
    mlir::LLVM::UnreachableOp::create(rewriter, op.getLoc());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class CIRCopyOpLowering : public mlir::OpConversionPattern<cir::CopyOp> {
public:
  using OpConversionPattern<cir::CopyOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(cir::CopyOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::memref::CopyOp>(op, adaptor.getSrc(),
                                                      adaptor.getDst());
    return mlir::success();
  }
};

void populateCIRToMLIRConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::TypeConverter &converter) {
  patterns.add<CIRReturnLowering, CIRBrOpLowering>(patterns.getContext());

  patterns
      .add<CIRATanOpLowering, CIRCmpOpLowering, CIRCallOpLowering,
           CIRBinOpLowering<cir::AddOp, mlir::arith::AddIOp>,
           CIRBinOpLowering<cir::SubOp, mlir::arith::SubIOp>,
           CIRBinOpLowering<cir::MulOp, mlir::arith::MulIOp>,
           CIRBinOpLowering<cir::AndOp, mlir::arith::AndIOp>,
           CIRBinOpLowering<cir::OrOp, mlir::arith::OrIOp>,
           CIRBinOpLowering<cir::XorOp, mlir::arith::XOrIOp>,
           CIRBinOpLowering<cir::FAddOp, mlir::arith::AddFOp>,
           CIRBinOpLowering<cir::FSubOp, mlir::arith::SubFOp>,
           CIRBinOpLowering<cir::FMulOp, mlir::arith::MulFOp>,
           CIRBinOpLowering<cir::FDivOp, mlir::arith::DivFOp>,
           CIRBinOpLowering<cir::FRemOp, mlir::arith::RemFOp>,
           CIRSignedAwareIntBinOpLowering<cir::DivOp, mlir::arith::DivSIOp,
                                  mlir::arith::DivUIOp>,
           CIRSignedAwareIntBinOpLowering<cir::RemOp, mlir::arith::RemSIOp,
                                  mlir::arith::RemUIOp>,
           CIRSignedAwareIntBinOpLowering<cir::MaxOp, mlir::arith::MaxSIOp,
                                  mlir::arith::MaxUIOp>,
           CIRSignedAwareIntBinOpLowering<cir::MinOp, mlir::arith::MinSIOp,
                                  mlir::arith::MinUIOp>,
           CIRIncDecOpLowering<cir::IncOp, 1>,
           CIRIncDecOpLowering<cir::DecOp, -1>, CIRMinusOpLowering,
           CIRFNegOpLowering, CIRNotOpLowering, CIRLoadOpLowering,
           CIRConstantOpLowering, CIRStoreOpLowering, CIRAllocaOpLowering,
           CIRFuncOpLowering, CIRBrCondOpLowering, CIRTernaryOpLowering,
           CIRYieldOpLowering, CIRCosOpLowering, CIRGlobalOpLowering,
           CIRGetGlobalOpLowering, CIRComplexCreateOpLowering,
           CIRComplexRealOpLowering, CIRComplexImagOpLowering,
           CIRCastOpLowering, CIRPtrStrideOpLowering, CIRSelectOpLowering,
           CIRGetElementOpLowering, CIRSqrtOpLowering, CIRCeilOpLowering,
           CIRExp2OpLowering, CIRExpOpLowering, CIRFAbsOpLowering,
           CIRAbsOpLowering, CIRFloorOpLowering, CIRLog10OpLowering,
           CIRLog2OpLowering, CIRLogOpLowering, CIRRoundOpLowering,
           CIRSinOpLowering, CIRTanOpLowering, CIRShiftOpLowering,
           CIRBitClzOpLowering, CIRBitCtzOpLowering, CIRBitPopcountOpLowering,
           CIRBitClrsbOpLowering, CIRBitFfsOpLowering, CIRBitParityOpLowering,
           CIRIfOpLowering, CIRScopeOpLowering, CIRVectorCreateLowering,
           CIRVectorInsertLowering, CIRVectorExtractLowering,
           CIRVectorCmpOpLowering, CIRACosOpLowering, CIRASinOpLowering,
           CIRUnreachableOpLowering, CIRTrapOpLowering, CIRCopyOpLowering>(
          converter, patterns.getContext());
}

static mlir::Attribute
convertCIRLangAddrSpaceToGPU(cir::LangAddressSpaceAttr addrSpace) {
  auto context = addrSpace.getContext();
  switch (addrSpace.getValue()) {
  case cir::LangAddressSpace::OffloadPrivate:
    return mlir::gpu::AddressSpaceAttr::get(context,
                                            mlir::gpu::AddressSpace::Private);
  case cir::LangAddressSpace::OffloadLocal:
    return mlir::gpu::AddressSpaceAttr::get(context,
                                            mlir::gpu::AddressSpace::Workgroup);
  case cir::LangAddressSpace::OffloadGlobal:
    return mlir::gpu::AddressSpaceAttr::get(context,
                                            mlir::gpu::AddressSpace::Global);
  case cir::LangAddressSpace::OffloadConstant:
  case cir::LangAddressSpace::OffloadGeneric:
  case cir::LangAddressSpace::Default:
    return mlir::Attribute();
  }
}

static mlir::TypeConverter prepareTypeConverter() {
  mlir::TypeConverter converter;
  converter.addConversion([&](cir::PointerType type) -> mlir::Type {
    auto ty = convertTypeForMemory(converter, type.getPointee());
    // FIXME: The pointee type might not be converted (e.g. struct)
    if (!ty)
      return nullptr;
    if (isa<cir::ArrayType>(type.getPointee()))
      return ty;
    auto maybeAddrSpace =
        converter.convertTypeAttribute(type, type.getAddrSpace());
    mlir::Attribute addrSpace = maybeAddrSpace.value_or(mlir::Attribute());
    return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, ty,
                                 mlir::MemRefLayoutAttrInterface(), addrSpace);
  });
  converter.addConversion(
      [&](mlir::IntegerType type) -> mlir::Type { return type; });
  converter.addConversion(
      [&](mlir::FloatType type) -> mlir::Type { return type; });
  converter.addConversion([&](cir::VoidType type) -> mlir::Type { return {}; });
  converter.addConversion([&](cir::IntType type) -> mlir::Type {
    // arith dialect ops doesn't take signed integer -- drop cir sign here
    return mlir::IntegerType::get(
        type.getContext(), type.getWidth(),
        mlir::IntegerType::SignednessSemantics::Signless);
  });
  converter.addConversion([&](cir::BoolType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), 1);
  });
  converter.addConversion([&](cir::SingleType type) -> mlir::Type {
    return mlir::Float32Type::get(type.getContext());
  });
  converter.addConversion([&](cir::DoubleType type) -> mlir::Type {
    return mlir::Float64Type::get(type.getContext());
  });
  converter.addConversion([&](cir::FP80Type type) -> mlir::Type {
    return mlir::Float80Type::get(type.getContext());
  });
  converter.addConversion([&](cir::LongDoubleType type) -> mlir::Type {
    return converter.convertType(type.getUnderlying());
  });
  converter.addConversion([&](cir::FP128Type type) -> mlir::Type {
    return mlir::Float128Type::get(type.getContext());
  });
  converter.addConversion([&](cir::FP16Type type) -> mlir::Type {
    return mlir::Float16Type::get(type.getContext());
  });
  converter.addConversion([&](cir::BF16Type type) -> mlir::Type {
    return mlir::BFloat16Type::get(type.getContext());
  });
  converter.addConversion([&](cir::ArrayType type) -> mlir::Type {
    SmallVector<int64_t> shape;
    mlir::Type curType = type;
    while (auto arrayType = dyn_cast<cir::ArrayType>(curType)) {
      shape.push_back(arrayType.getSize());
      curType = arrayType.getElementType();
    }
    auto elementType = converter.convertType(curType);
    // FIXME: The element type might not be converted (e.g. struct)
    if (!elementType)
      return nullptr;
    return mlir::MemRefType::get(shape, elementType);
  });
  converter.addConversion([&](cir::VectorType type) -> mlir::Type {
    auto ty = converter.convertType(type.getElementType());
    return mlir::VectorType::get(type.getSize(), ty);
  });
  converter.addConversion([&](cir::ComplexType type) -> mlir::Type {
    auto elemTy = converter.convertType(type.getElementType());
    if (!elemTy)
      return nullptr;
    return mlir::VectorType::get(2, elemTy);
  });
  // D8: cir::OpaqueType removed upstream — no type conversion needed.
  converter.addTypeAttributeConversion(
      [](mlir::Type, cir::TargetAddressSpaceAttr memorySpaceAttr) {
        auto targetMemorySpace = memorySpaceAttr.getValue();
        return mlir::IntegerAttr::get(
            mlir::IntegerType::get(memorySpaceAttr.getContext(), 64),
            targetMemorySpace);
      });
  converter.addTypeAttributeConversion(
      [](mlir::Type, cir::LangAddressSpaceAttr memorySpaceAttr) {
        return convertCIRLangAddrSpaceToGPU(memorySpaceAttr);
      });
  return converter;
}

//===----------------------------------------------------------------------===//
// SCF preparation (ported from the incubator SCFPrepare pass): hoist
// loop-invariant bound ops out of cir.for's cond region and canonicalize the IV
// to the comparison LHS. Run as a greedy pre-pass at the start of the
// conversion so the canonical-for lowering can build scf.for from out-of-loop
// operands and discard the cond region cleanly. Folded in here (rather than a
// separate cir-opt invocation) to keep `cir-opt --cir-to-mlir` self-contained.
//===----------------------------------------------------------------------===//
namespace {

static mlir::Value scfPrepFindIVAddr(mlir::Block *step) {
  mlir::Value ivAddr = nullptr;
  for (mlir::Operation &op : *step) {
    if (auto loadOp = mlir::dyn_cast<cir::LoadOp>(op))
      ivAddr = loadOp.getAddr();
    else if (auto storeOp = mlir::dyn_cast<cir::StoreOp>(op))
      if (ivAddr != storeOp.getAddr())
        return nullptr;
  }
  return ivAddr;
}

static cir::CmpOp scfPrepFindLoopCmpAndIV(mlir::Block *cond, mlir::Value ivAddr,
                                          mlir::Value &iv) {
  mlir::Operation *ivLoadOp = nullptr;
  for (mlir::Operation &op : *cond)
    if (auto loadOp = mlir::dyn_cast<cir::LoadOp>(op))
      if (loadOp.getAddr() == ivAddr) {
        ivLoadOp = &op;
        break;
      }
  if (!ivLoadOp || !ivLoadOp->hasOneUse())
    return nullptr;
  iv = ivLoadOp->getResult(0);
  return mlir::dyn_cast<cir::CmpOp>(*ivLoadOp->user_begin());
}

// cir.cmp(gt, %bound, %iv) -> cir.cmp(lt, %iv, %bound) so the RHS is the bound.
struct SCFPrepCanonicalizeIVtoCmpLHS
    : public mlir::OpRewritePattern<cir::ForOp> {
  using mlir::OpRewritePattern<cir::ForOp>::OpRewritePattern;

  static cir::CmpOpKind swapCmpKind(cir::CmpOpKind kind) {
    switch (kind) {
    case cir::CmpOpKind::gt: return cir::CmpOpKind::lt;
    case cir::CmpOpKind::ge: return cir::CmpOpKind::le;
    case cir::CmpOpKind::lt: return cir::CmpOpKind::gt;
    case cir::CmpOpKind::le: return cir::CmpOpKind::ge;
    default: break;
    }
    return kind;
  }

  mlir::LogicalResult
  matchAndRewrite(cir::ForOp op, mlir::PatternRewriter &rewriter) const final {
    mlir::Block *cond = &op.getCond().front();
    mlir::Block *step =
        op.maybeGetStep() ? &op.maybeGetStep()->front() : nullptr;
    if (!step)
      return mlir::failure();
    mlir::Value ivAddr = scfPrepFindIVAddr(step);
    if (!ivAddr)
      return mlir::failure();
    mlir::Value iv = nullptr;
    cir::CmpOp loopCmp = scfPrepFindLoopCmpAndIV(cond, ivAddr, iv);
    if (!loopCmp || !iv || loopCmp.getLhs() == iv)
      return mlir::failure();
    rewriter.setInsertionPointAfter(loopCmp.getOperation());
    auto newCmp =
        cir::CmpOp::create(rewriter, loopCmp.getLoc(), loopCmp.getType(),
                           swapCmpKind(loopCmp.getKind()), iv, loopCmp.getLhs());
    rewriter.replaceOp(loopCmp, newCmp.getResult());
    return mlir::success();
  }
};

// Hoist the loop-invariant bound computation out of cir.for's cond region.
struct SCFPrepHoistLoopInvariant : public mlir::OpRewritePattern<cir::ForOp> {
  using mlir::OpRewritePattern<cir::ForOp>::OpRewritePattern;

  bool isLoopInvariantLoad(mlir::Operation *op, cir::ForOp forOp) const {
    auto load = mlir::dyn_cast<cir::LoadOp>(op);
    if (!load)
      return false;
    mlir::Value loadAddr = load.getAddr();
    auto result = forOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *o) {
      if (auto store = mlir::dyn_cast<cir::StoreOp>(o))
        if (store.getAddr() == loadAddr)
          return mlir::WalkResult::interrupt();
      return mlir::WalkResult::advance();
    });
    return !result.wasInterrupted();
  }

  bool isLoopInvariantOp(mlir::Operation *op, cir::ForOp forOp,
                         llvm::SmallVector<mlir::Operation *> &initOps) const {
    if (!op)
      return false;
    if (mlir::isa<cir::ConstantOp>(op) || isLoopInvariantLoad(op, forOp)) {
      initOps.push_back(op);
      return true;
    }
    // D1: upstream split cir::BinOp into per-op classes; match the integer
    // binops a loop-bound expression can be built from.
    if (mlir::isa<cir::AddOp, cir::SubOp, cir::MulOp, cir::DivOp, cir::RemOp,
                  cir::ShiftOp, cir::AndOp, cir::OrOp, cir::XorOp>(op) &&
        isLoopInvariantOp(op->getOperand(0).getDefiningOp(), forOp, initOps) &&
        isLoopInvariantOp(op->getOperand(1).getDefiningOp(), forOp, initOps)) {
      initOps.push_back(op);
      return true;
    }
    if (mlir::isa<cir::CastOp>(op) &&
        isLoopInvariantOp(op->getOperand(0).getDefiningOp(), forOp, initOps)) {
      initOps.push_back(op);
      return true;
    }
    return false;
  }

  mlir::LogicalResult
  matchAndRewrite(cir::ForOp forOp,
                  mlir::PatternRewriter &rewriter) const final {
    mlir::Block *cond = &forOp.getCond().front();
    mlir::Block *step =
        forOp.maybeGetStep() ? &forOp.maybeGetStep()->front() : nullptr;
    if (!step)
      return mlir::failure();
    mlir::Value ivAddr = scfPrepFindIVAddr(step);
    if (!ivAddr)
      return mlir::failure();
    mlir::Value iv = nullptr;
    cir::CmpOp loopCmp = scfPrepFindLoopCmpAndIV(cond, ivAddr, iv);
    if (!loopCmp || !iv)
      return mlir::failure();
    llvm::SmallVector<mlir::Operation *> initOps;
    if (!isLoopInvariantOp(loopCmp.getRhs().getDefiningOp(), forOp, initOps))
      return mlir::failure();
    for (mlir::Operation *op : initOps)
      rewriter.moveOpBefore(op, forOp);
    return mlir::success();
  }
};

} // namespace

void ConvertCIRToMLIRPass::runOnOperation() {
  mlir::ModuleOp theModule = getOperation();

  // Up-front validation: a `cir.return` nested inside a loop would be lowered to
  // a `func.return` sitting inside an `scf` region, which is illegal
  // (func.return must be parented by func.func). The `--cir-lower-return`
  // pre-pass rewrites such nested returns into single-exit form; if one still
  // reaches here (e.g. a shape that pass declined), reject it up front with a
  // single clean diagnostic rather than emitting partial/malformed IR mid
  // conversion. This mirrors the return-inside-`switch`-case reject in the loop
  // lowering, but runs before applyFullConversion so it never cascades.
  {
    bool badReturn = false;
    theModule.walk([&](cir::ReturnOp ret) {
      for (mlir::Operation *a = ret->getParentOp(); a; a = a->getParentOp()) {
        if (mlir::isa<cir::FuncOp>(a))
          break;
        if (mlir::isa<cir::ForOp, cir::WhileOp, cir::DoWhileOp>(a)) {
          ret.emitError(
              "ThroughMLIR: 'return' inside a loop is not supported by the "
              "CIR-to-MLIR lowering; it must be rewritten into single-exit form "
              "by the '--cir-lower-return' pre-pass");
          badReturn = true;
          break;
        }
      }
    });
    if (badReturn) {
      signalPassFailure();
      return;
    }
  }

  // SCF preparation (see patterns above): hoist loop bounds out of the cond
  // region + canonicalize the IV to the cmp LHS, before the conversion, so the
  // canonical-for lowering can build scf.for from out-of-loop operands.
  {
    mlir::RewritePatternSet scfPrep(&getContext());
    scfPrep.add<SCFPrepCanonicalizeIVtoCmpLHS, SCFPrepHoistLoopInvariant>(
        &getContext());
    llvm::SmallVector<mlir::Operation *, 16> forOps;
    theModule->walk([&](mlir::Operation *op) {
      if (mlir::isa<cir::ForOp>(op))
        forOps.push_back(op);
    });
    if (mlir::failed(mlir::applyOpPatternsGreedily(forOps, std::move(scfPrep))))
      signalPassFailure();
  }

  auto converter = prepareTypeConverter();

  mlir::RewritePatternSet patterns(&getContext());

  populateCIRLoopToSCFConversionPatterns(patterns, converter);
  populateCIRToMLIRConversionPatterns(patterns, converter);

  mlir::ConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
  target
      .addLegalDialect<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                       mlir::memref::MemRefDialect, mlir::func::FuncDialect,
                       mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
                       mlir::ptr::PtrDialect, mlir::math::MathDialect,
                       mlir::vector::VectorDialect, mlir::LLVM::LLVMDialect>();
  auto *context = patterns.getContext();

  // We cannot mark cir dialect as illegal before conversion.
  // The conversion of WhileOp relies on partially preserving operations from
  // cir dialect, for example the `cir.continue`. If we marked cir as illegal
  // here, then MLIR would think any remaining `cir.continue` indicates a
  // failure, which is not what we want.

  patterns.add<CIRCastOpLowering, CIRIfOpLowering, CIRScopeOpLowering,
               CIRYieldOpLowering>(converter, context);

  if (mlir::failed(mlir::applyPartialConversion(theModule, target,
                                                std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  // The conversion intentionally leaves dead IR rather than have a pattern
  // hand-erase ops it did not match (the LLVM-23 one-shot driver rejects that)
  // — most notably each loop's IV alloca + initial store, now that body
  // IV-loads use the scf.for induction variable. Clean it up here, after the
  // conversion has fully committed, on legalized standard-dialect IR.
  mlir::PassManager cleanup(&getContext());
  cleanup.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(cleanup.run(theModule)))
    signalPassFailure();
}

// Step 3b: the CIR->LLVM "tail" drivers (lowerFromCIRToMLIRToLLVMDialect /
// lowerFromCIRToMLIRToLLVMIR / lowerDirectlyFromCIRToLLVMIR) are omitted in this
// port (Step 7) — the HLS flow only uses --cir-to-mlir (CIR -> standard MLIR).
// The upstream CIR->LLVM route is cir::direct::lowerDirectlyFromCIRToLLVMIR;
// these ThroughMLIR variants had no external consumers.

std::unique_ptr<mlir::Pass> createConvertCIRToMLIRPass() {
  return std::make_unique<ConvertCIRToMLIRPass>();
}

mlir::ModuleOp lowerFromCIRToMLIR(mlir::ModuleOp theModule,
                                  mlir::MLIRContext *mlirCtx) {
  llvm::TimeTraceScope scope("Lower CIR To MLIR");

  mlir::PassManager pm(mlirCtx);
  pm.addPass(createConvertCIRToMLIRPass());

  auto result = !mlir::failed(pm.run(theModule));
  if (!result)
    report_fatal_error(
        "The pass manager failed to lower CIR to MLIR standard dialects!");
  // Now that we ran all the lowering passes, verify the final output.
  if (theModule.verify().failed())
    report_fatal_error(
        "Verification of the final MLIR in standard dialects failed!");

  return theModule;
}

} // namespace cir
