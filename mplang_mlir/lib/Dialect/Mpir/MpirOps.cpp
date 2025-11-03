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

  // Check rmask constraint: rmask ⊆ input pmask union (only if there are MP inputs)
  auto rmaskAttr = getRmaskAttr();
  if (rmaskAttr) {
    int64_t rmask = rmaskAttr.getInt();

    // Compute union of input pmasks
    int64_t inputPmaskUnion = 0;
    bool hasMPInputs = false;
    for (Value arg : getArgs()) {
      Type argType = arg.getType();
      if (auto mpType = argType.dyn_cast<mpir::MPType>()) {
        inputPmaskUnion |= mpType.getPmask();
        hasMPInputs = true;
      }
      // Non-MP types don't contribute to pmask union
    }

    // Check if rmask is subset of input pmask union (only if there are MP inputs)
    // For operations without MP inputs (e.g., keygen), rmask can be freely specified
    if (hasMPInputs && (rmask & inputPmaskUnion) != rmask) {
      return emitOpError("rmask ")
             << rmask << " is not a subset of input pmask union "
             << inputPmaskUnion << ". rmask can only execute on parties "
             << "that have input data.";
    }
  }

  // TODO: If callee is specified, verify symbol exists and signature matches

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

  // Extract input types and pmasks
  SmallVector<Type> inputTypes;
  SmallVector<int64_t> inputPmasks;

  for (auto input : getInputs()) {
    Type inputType = input.getType();
    inputTypes.push_back(inputType);

    // Extract pmask if MP type
    if (auto mpType = inputType.dyn_cast<mpir::MPType>()) {
      inputPmasks.push_back(mpType.getPmask());
    } else {
      return emitOpError("all inputs must be MP types, got ") << inputType;
    }
  }

  // Check 1: All inputs must have same inner type (ignoring pmask)
  if (!inputTypes.empty()) {
    auto firstMpType = inputTypes[0].cast<mpir::MPType>();
    Type firstInnerType = firstMpType.getInnerType();

    for (size_t i = 1; i < inputTypes.size(); ++i) {
      auto mpType = inputTypes[i].cast<mpir::MPType>();
      Type innerType = mpType.getInnerType();

      if (innerType != firstInnerType) {
        return emitOpError("all inputs must have same inner type, got MP<")
               << firstInnerType << "> and MP<" << innerType << ">";
      }
    }
  }

  // Check 2: Input pmasks must be pairwise disjoint
  for (size_t i = 0; i < inputPmasks.size(); ++i) {
    for (size_t j = i + 1; j < inputPmasks.size(); ++j) {
      int64_t intersection = inputPmasks[i] & inputPmasks[j];
      if (intersection != 0) {
        return emitOpError("input pmasks must be disjoint, but pmask ")
               << inputPmasks[i] << " and pmask " << inputPmasks[j]
               << " overlap (intersection = " << intersection << ")";
      }
    }
  }

  // Check 3: Result pmask must equal union of input pmasks
  Type resultType = getResult().getType();
  if (auto resultMpType = resultType.dyn_cast<mpir::MPType>()) {
    int64_t resultPmask = resultMpType.getPmask();

    // Compute expected pmask (union)
    int64_t expectedPmask = 0;
    for (int64_t pmask : inputPmasks) {
      expectedPmask |= pmask;
    }

    if (resultPmask != expectedPmask) {
      return emitOpError("result pmask must equal union of input pmasks. ")
             << "Expected " << expectedPmask << ", got " << resultPmask;
    }

    // Check 4: Result inner type must match input inner types
    Type resultInnerType = resultMpType.getInnerType();
    auto firstMpType = inputTypes[0].cast<mpir::MPType>();
    Type firstInnerType = firstMpType.getInnerType();

    if (resultInnerType != firstInnerType) {
      return emitOpError("result inner type must match input inner types, got MP<")
             << resultInnerType << "> but inputs have MP<" << firstInnerType << ">";
    }
  } else {
    return emitOpError("result must be MP type, got ") << resultType;
  }

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
// UniformCondOp verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::mpir::UniformCondOp::verify() {
  // Check 1: Condition must be MP<i1, pmask> (scalar boolean)
  Type condType = getCondition().getType();

  if (auto mpType = condType.dyn_cast<mpir::MPType>()) {
    Type innerType = mpType.getInnerType();
    if (auto intType = innerType.dyn_cast<IntegerType>()) {
      if (intType.getWidth() != 1) {
        return emitOpError("condition must have inner type i1 (boolean), got i")
               << intType.getWidth();
      }
    } else {
      return emitOpError("condition must be MP<i1, pmask>, got MP<")
             << innerType << ", pmask>";
    }
  } else if (auto mpDynType = condType.dyn_cast<mpir::MPDynamicType>()) {
    Type innerType = mpDynType.getInnerType();
    if (auto intType = innerType.dyn_cast<IntegerType>()) {
      if (intType.getWidth() != 1) {
        return emitOpError("condition must have inner type i1 (boolean), got i")
               << intType.getWidth();
      }
    } else {
      return emitOpError("condition must be MPDynamic<i1>, got MPDynamic<")
             << innerType << ">";
    }
  } else {
    return emitOpError("condition must be MP<i1, pmask> or MPDynamic<i1>, got ")
           << condType;
  }

  // Check 2: Both branches must be present and non-empty
  if (getThenRegion().empty()) {
    return emitOpError("then region must not be empty");
  }
  if (getElseRegion().empty()) {
    return emitOpError("else region must not be empty");
  }

  // Check 3: Get return types from both branches
  Region &thenRegion = getThenRegion();
  Region &elseRegion = getElseRegion();

  // Helper to get return operation types
  auto getReturnTypes = [](Region &region) -> SmallVector<Type> {
    SmallVector<Type> types;
    if (region.empty())
      return types;

    for (Block &block : region) {
      if (block.empty())
        continue;
      Operation *terminator = block.getTerminator();
      if (!terminator)
        continue;

      // Check if it's a return operation
      if (isa<mpir::ReturnOp>(terminator)) {
        for (Value operand : terminator->getOperands()) {
          types.push_back(operand.getType());
        }
        break;
      }
    }
    return types;
  };

  SmallVector<Type> thenTypes = getReturnTypes(thenRegion);
  SmallVector<Type> elseTypes = getReturnTypes(elseRegion);

  // Check 4: Both branches must return same number of values
  if (thenTypes.size() != elseTypes.size()) {
    return emitOpError("then branch returns ")
           << thenTypes.size() << " values, but else branch returns "
           << elseTypes.size() << " values";
  }

  // Check 5: Both branches must return same types
  for (size_t i = 0; i < thenTypes.size(); ++i) {
    if (thenTypes[i] != elseTypes[i]) {
      return emitOpError("then branch returns type ")
             << thenTypes[i] << " at position " << i
             << ", but else branch returns type " << elseTypes[i];
    }
  }

  // Check 6: Branch return types must match op result types
  if (thenTypes.size() != getResults().size()) {
    return emitOpError("branches return ")
           << thenTypes.size() << " values, but op declares "
           << getResults().size() << " results";
  }

  for (size_t i = 0; i < thenTypes.size(); ++i) {
    if (thenTypes[i] != getResults()[i].getType()) {
      return emitOpError("branch returns type ")
             << thenTypes[i] << " at position " << i
             << ", but op result has type " << getResults()[i].getType();
    }
  }

  // Check 7: Note if verify_uniform is disabled (unsafe path)
  // Note: Don't emit warning here as it would trigger verification recursion
  // TODO: Consider emitting warning through a separate diagnostic pass

  return success();
}

//===----------------------------------------------------------------------===//
// UniformWhileOp verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::mpir::UniformWhileOp::verify() {
  // Check 1: Both regions must be present and non-empty
  if (getBefore().empty()) {
    return emitOpError("condition region (before) must not be empty");
  }
  if (getAfter().empty()) {
    return emitOpError("body region (after) must not be empty");
  }

  Region &condRegion = getBefore();
  Region &bodyRegion = getAfter();

  // Check 2: Condition region must have ConditionOp terminator
  bool hasConditionTerminator = false;
  Type conditionType;

  for (Block &block : condRegion) {
    if (block.empty())
      continue;
    Operation *terminator = block.getTerminator();
    if (!terminator)
      continue;

    if (auto condOp = dyn_cast<mpir::ConditionOp>(terminator)) {
      hasConditionTerminator = true;
      conditionType = condOp.getCondition().getType();
      break;
    }
  }

  if (!hasConditionTerminator) {
    return emitOpError("condition region must have mpir.condition terminator");
  }

  // Check 3: Condition must be MP<i1, pmask> (scalar boolean)
  if (auto mpType = conditionType.dyn_cast<mpir::MPType>()) {
    Type innerType = mpType.getInnerType();
    if (auto intType = innerType.dyn_cast<IntegerType>()) {
      if (intType.getWidth() != 1) {
        return emitOpError("condition must have inner type i1 (boolean), got i")
               << intType.getWidth();
      }
    } else {
      return emitOpError("condition must be MP<i1, pmask>, got MP<")
             << innerType << ", pmask>";
    }
  } else if (auto mpDynType = conditionType.dyn_cast<mpir::MPDynamicType>()) {
    Type innerType = mpDynType.getInnerType();
    if (auto intType = innerType.dyn_cast<IntegerType>()) {
      if (intType.getWidth() != 1) {
        return emitOpError("condition must have inner type i1 (boolean), got i")
               << intType.getWidth();
      }
    } else {
      return emitOpError("condition must be MPDynamic<i1>, got MPDynamic<")
             << innerType << ">";
    }
  } else {
    return emitOpError("condition must be MP<i1, pmask> or MPDynamic<i1>, got ")
           << conditionType;
  }

  // Check 4: Body region must have YieldOp terminator
  bool hasYieldTerminator = false;
  SmallVector<Type> yieldTypes;

  for (Block &block : bodyRegion) {
    if (block.empty())
      continue;
    Operation *terminator = block.getTerminator();
    if (!terminator)
      continue;

    if (auto yieldOp = dyn_cast<mpir::YieldOp>(terminator)) {
      hasYieldTerminator = true;
      for (Value result : yieldOp.getResults()) {
        yieldTypes.push_back(result.getType());
      }
      break;
    }
  }

  if (!hasYieldTerminator) {
    return emitOpError("body region must have mpir.yield terminator");
  }

  // Check 5: Init args and yield types must match
  if (getInitArgs().size() != yieldTypes.size()) {
    return emitOpError("loop initialized with ")
           << getInitArgs().size() << " values, but body yields "
           << yieldTypes.size() << " values";
  }

  for (size_t i = 0; i < getInitArgs().size(); ++i) {
    Type initType = getInitArgs()[i].getType();
    Type yieldType = yieldTypes[i];
    if (initType != yieldType) {
      return emitOpError("init arg at position ")
             << i << " has type " << initType
             << ", but body yields type " << yieldType;
    }
  }

  // Check 6: Result types must match init arg types
  if (getResults().size() != getInitArgs().size()) {
    return emitOpError("loop initialized with ")
           << getInitArgs().size() << " values, but declares "
           << getResults().size() << " results";
  }

  for (size_t i = 0; i < getInitArgs().size(); ++i) {
    Type initType = getInitArgs()[i].getType();
    Type resultType = getResults()[i].getType();
    if (initType != resultType) {
      return emitOpError("init arg at position ")
             << i << " has type " << initType
             << ", but result has type " << resultType;
    }
  }

  // Check 7: Note if verify_uniform is disabled (unsafe path)
  // Note: Don't emit warning here as it would trigger verification recursion
  // TODO: Consider emitting warning through a separate diagnostic pass

  return success();
}

#define GET_OP_CLASSES
#include "MpirOps.cpp.inc"
