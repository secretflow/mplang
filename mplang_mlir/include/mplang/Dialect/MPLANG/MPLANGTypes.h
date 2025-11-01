//===- MPLANGTypes.h ---------------------------------------------------===//
// Wrapper header for generated MPLANG types declarations.
//===----------------------------------------------------------------------===//

#ifndef MPLANG_DIALECT_MPLANG_MPLANGTYPES_H
#define MPLANG_DIALECT_MPLANG_MPLANGTYPES_H

#include "mlir/IR/Types.h"

// Generate typedef class declarations.
#define GET_TYPEDEF_CLASSES
#include "MPLANGTypes.h.inc"

#endif // MPLANG_DIALECT_MPLANG_MPLANGTYPES_H
