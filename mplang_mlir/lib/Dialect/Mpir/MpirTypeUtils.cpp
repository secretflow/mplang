//===- MpirTypeUtils.cpp - Mpir Type Utility Functions -----------*- C++ -*-===//
//
// Implementation of utility functions for Mpir type system.
//
//===----------------------------------------------------------------------===//

#include "mplang/Dialect/Mpir/MpirTypeUtils.h"
#include "mplang/Dialect/Mpir/MpirTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::mpir;

//===----------------------------------------------------------------------===//
// Pmask Extraction and Manipulation
//===----------------------------------------------------------------------===//

std::optional<int64_t> mlir::mpir::extractPmask(Type type) {
  if (auto mpType = type.dyn_cast<MPType>()) {
    return mpType.getPmask();
  }
  return std::nullopt;
}

std::optional<int64_t> mlir::mpir::extractPmaskRecursive(Type type) {
  // Handle MPType directly
  if (auto mpType = type.dyn_cast<MPType>()) {
    return mpType.getPmask();
  }

  // MPDynamicType has no static pmask
  if (type.isa<MPDynamicType>()) {
    return std::nullopt;
  }

  return std::nullopt;
}

bool mlir::mpir::arePmasksDisjoint(int64_t pmask1, int64_t pmask2) {
  return (pmask1 & pmask2) == 0;
}

bool mlir::mpir::arePmasksDisjoint(llvm::ArrayRef<int64_t> pmasks) {
  if (pmasks.size() <= 1)
    return true;

  // Check all pairs for disjointness
  for (size_t i = 0; i < pmasks.size(); ++i) {
    for (size_t j = i + 1; j < pmasks.size(); ++j) {
      if (!arePmasksDisjoint(pmasks[i], pmasks[j])) {
        return false;
      }
    }
  }
  return true;
}

int64_t mlir::mpir::computeUnionPmask(llvm::ArrayRef<int64_t> pmasks) {
  int64_t result = 0;
  for (int64_t pmask : pmasks) {
    result |= pmask;
  }
  return result;
}

bool mlir::mpir::isPmaskSubset(int64_t pmask1, int64_t pmask2) {
  return (pmask1 & pmask2) == pmask1;
}

int64_t mlir::mpir::countParties(int64_t pmask) {
  // Use __builtin_popcountll for 64-bit popcount
  return __builtin_popcountll(static_cast<uint64_t>(pmask));
}

//===----------------------------------------------------------------------===//
// Type Checking Predicates
//===----------------------------------------------------------------------===//

bool mlir::mpir::isScalarBoolean(Type type) {
  if (auto intType = type.dyn_cast<IntegerType>()) {
    return intType.getWidth() == 1;
  }
  return false;
}

bool mlir::mpir::isMPScalarBoolean(Type type) {
  auto mpType = type.dyn_cast<MPType>();
  if (!mpType)
    return false;

  return isScalarBoolean(mpType.getInnerType());
}

bool mlir::mpir::isBooleanTensor(Type type) {
  auto tensorType = type.dyn_cast<RankedTensorType>();
  if (!tensorType)
    return false;

  return isScalarBoolean(tensorType.getElementType());
}

bool mlir::mpir::isMPType(Type type) {
  return type.isa<MPType>();
}

bool mlir::mpir::isMPDynamicType(Type type) {
  return type.isa<MPDynamicType>();
}

bool mlir::mpir::isEncodedType(Type type) {
  return type.isa<EncodedType>();
}

Type mlir::mpir::extractInnerType(Type type) {
  if (auto mpType = type.dyn_cast<MPType>()) {
    return mpType.getInnerType();
  }
  if (auto mpDynType = type.dyn_cast<MPDynamicType>()) {
    return mpDynType.getInnerType();
  }
  return Type();
}

bool mlir::mpir::haveSameShape(Type type1, Type type2) {
  // Extract inner types if wrapped in MP
  Type inner1 = extractInnerType(type1);
  Type inner2 = extractInnerType(type2);

  // Use original types if not MP-wrapped
  if (!inner1)
    inner1 = type1;
  if (!inner2)
    inner2 = type2;

  // Check if both are tensors
  auto tensor1 = inner1.dyn_cast<RankedTensorType>();
  auto tensor2 = inner2.dyn_cast<RankedTensorType>();

  if (tensor1 && tensor2) {
    // Compare element type and shape
    if (tensor1.getElementType() != tensor2.getElementType())
      return false;
    if (tensor1.getShape() != tensor2.getShape())
      return false;
    return true;
  }

  // Check if both are tuples (for tables)
  auto tuple1 = inner1.dyn_cast<TupleType>();
  auto tuple2 = inner2.dyn_cast<TupleType>();

  if (tuple1 && tuple2) {
    auto types1 = tuple1.getTypes();
    auto types2 = tuple2.getTypes();

    if (types1.size() != types2.size())
      return false;

    // Recursively check each element
    for (size_t i = 0; i < types1.size(); ++i) {
      if (!haveSameShape(types1[i], types2[i]))
        return false;
    }
    return true;
  }

  // For other types, use direct equality
  return inner1 == inner2;
}

//===----------------------------------------------------------------------===//
// Region and Terminator Utilities
//===----------------------------------------------------------------------===//

llvm::SmallVector<Type> mlir::mpir::getRegionReturnTypes(Region &region) {
  llvm::SmallVector<Type> returnTypes;

  if (region.empty())
    return returnTypes;

  // Walk all blocks looking for return operations
  for (Block &block : region) {
    if (block.empty())
      continue;

    Operation *terminator = block.getTerminator();
    if (!terminator)
      continue;

    // Check if it's a return operation (has "return" in name)
    if (terminator->getName().getStringRef().contains("return")) {
      // Get operand types
      for (Value operand : terminator->getOperands()) {
        returnTypes.push_back(operand.getType());
      }
      break; // Assume all returns have same types
    }
  }

  return returnTypes;
}
