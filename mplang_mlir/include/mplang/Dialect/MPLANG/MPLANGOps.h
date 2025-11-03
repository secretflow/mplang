//===- MPLANGOps.h -----------------------------------------------------===//
// Wrapper header for generated MPLANG ops declarations.
//===----------------------------------------------------------------------===//

#ifndef MPLANG_DIALECT_MPLANG_MPLANGOPS_H
#define MPLANG_DIALECT_MPLANG_MPLANGOPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"

// Generate op class declarations.
#define GET_OP_CLASSES
#include "MPLANGOps.h.inc"

#endif // MPLANG_DIALECT_MPLANG_MPLANGOPS_H
