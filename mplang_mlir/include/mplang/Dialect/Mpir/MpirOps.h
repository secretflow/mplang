//===- MpirOps.h -------------------------------------------------------===//
// Wrapper header for generated Mpir ops declarations.
//===----------------------------------------------------------------------===//

#ifndef MPLANG_DIALECT_MPIR_MPIROPS_H
#define MPLANG_DIALECT_MPIR_MPIROPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"

// Include types before ops, as ops reference type classes
#include "mplang/Dialect/Mpir/MpirTypes.h"

// Generate op class declarations.
#define GET_OP_CLASSES
#include "MpirOps.h.inc"

#endif // MPLANG_DIALECT_MPIR_MPIROPS_H
