//===- MpirOps.cpp ----------------------------------------------------===//
// Mpir ops implementation.
//===----------------------------------------------------------------------===//

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mplang/Dialect/Mpir/MpirOps.h"

using namespace mlir;
using namespace mpir;

//===----------------------------------------------------------------------===//
// PEvalOp verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult mpir::PEvalOp::verify() {
  auto calleeAttr = getCalleeAttr();
  auto fnTypeAttr = getFnTypeAttr();
  auto fnAttrsAttr = getFnAttrsAttr();

  // Must have exactly one of callee or fn_type
  if (!calleeAttr && !fnTypeAttr) {
    return emitOpError("must specify either 'callee' or 'fn_type'");
  }
  if (calleeAttr && fnTypeAttr) {
    return emitOpError("cannot specify both 'callee' and 'fn_type'");
  }

  // fn_attrs should only be used with fn_type (Mode 2)
  if (fnAttrsAttr && calleeAttr) {
    return emitOpError("'fn_attrs' can only be used with 'fn_type' (external backend mode), not with 'callee'");
  }

  // TODO: If callee is specified, verify symbol exists and signature matches
  // TODO: Add pmask validation when pmask attributes are available

  return success();
}

//===----------------------------------------------------------------------===//
// PEvalDynOp verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult mpir::PEvalDynOp::verify() {
  auto calleeAttr = getCalleeAttr();
  auto fnTypeAttr = getFnTypeAttr();
  auto fnAttrsAttr = getFnAttrsAttr();

  // Must have exactly one of callee or fn_type
  if (!calleeAttr && !fnTypeAttr) {
    return emitOpError("must specify either 'callee' or 'fn_type'");
  }
  if (calleeAttr && fnTypeAttr) {
    return emitOpError("cannot specify both 'callee' and 'fn_type'");
  }

  // fn_attrs should only be used with fn_type (Mode 2)
  if (fnAttrsAttr && calleeAttr) {
    return emitOpError("'fn_attrs' can only be used with 'fn_type' (external backend mode), not with 'callee'");
  }

  // TODO: If callee is specified, verify symbol exists and signature matches
  // TODO: Add pmask validation when pmask attributes are available

  return success();
}

//===----------------------------------------------------------------------===//
// ConvOp verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult mpir::ConvOp::verify() {
  // Must have at least one input
  if (getInputs().empty()) {
    return emitOpError("requires at least one input");
  }

  // All inputs must have the same type
  Type firstType = getInputs().front().getType();
  for (auto input : getInputs()) {
    if (input.getType() != firstType) {
      return emitOpError("all inputs must have the same type, but got ")
             << firstType << " and " << input.getType();
    }
  }

  // Result type must match input type
  if (getResult().getType() != firstType) {
    return emitOpError("result type must match input types");
  }

  // TODO: Add pmask disjointness check when pmask attributes are available

  return success();
}

//===----------------------------------------------------------------------===//
// PEvalOp custom assembly format
// Simplified: use declarative format with custom handling
//===----------------------------------------------------------------------===//

// We'll use a simpler approach - let TableGen generate most of it
// and just customize what we need via assemblyFormat in ODS

//===----------------------------------------------------------------------===//
// PEvalDynOp custom assembly format
// Simplified: use declarative format with custom handling
//===----------------------------------------------------------------------===//

// We'll use a simpler approach - let TableGen generate most of it
// and just customize what we need via assemblyFormat in ODS

//===----------------------------------------------------------------------===//
// ShuffleStaticOp verifier (TODO: implement pmask checks)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::mpir::ShuffleStaticOp::verify() {
  // TODO: Check src_ranks validity against input pmask
  // TODO: Verify input/output types match (element type, shape)
  return success();
}

//===----------------------------------------------------------------------===//
// ShuffleDynOp verifier (TODO: implement inner type checks)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::mpir::ShuffleDynOp::verify() {
  // TODO: Check inner type consistency between input and output
  return success();
}

//===----------------------------------------------------------------------===//
// UniformCondOp verifier (TODO: implement uniform predicate checks)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::mpir::UniformCondOp::verify() {
  // TODO: Check condition is scalar i1 type (boolean scalar, not tensor<i1>)
  // TODO: Verify both branches return same types
  // TODO: Check branch result types match op result types
  // TODO: Add compile-time warning if verify_uniform=false (unsafe path)
  return success();
}

//===----------------------------------------------------------------------===//
// UniformWhileOp verifier (TODO: implement uniform condition checks)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::mpir::UniformWhileOp::verify() {
  // TODO: Check condition region returns scalar MP<i1> (not tensor<i1>)
  // TODO: Verify body region yields match init args types
  // TODO: Check regions have correct terminator ops (ConditionOp, YieldOp)
  // TODO: Add compile-time warning if verify_uniform=false (unsafe path)
  return success();
}

#define GET_OP_CLASSES
#include "MpirOps.cpp.inc"
