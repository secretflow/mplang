//===- MPLANGOps.cpp ----------------------------------------------------===//
// Minimal ops glue to include generated implementation.
//===----------------------------------------------------------------------===//

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mplang/Dialect/MPLANG/MPLANGOps.h"

using namespace mlir;
using namespace mplang;

//===----------------------------------------------------------------------===//
// ConvOp verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult mplang::ConvOp::verify() {
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

#define GET_OP_CLASSES
#include "MPLANGOps.cpp.inc"
