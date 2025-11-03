//===- MpirTypes.h ---------------------------------------------------===//
// Wrapper header for generated Mpir types declarations.
//===----------------------------------------------------------------------===//

#ifndef MPLANG_DIALECT_MPIR_MPLANGTYPES_H
#define MPLANG_DIALECT_MPIR_MPLANGTYPES_H

#include "mlir/IR/Types.h"

// Generate typedef class declarations.
#define GET_TYPEDEF_CLASSES
#include "MpirTypes.h.inc"

namespace mlir {
class Dialect;
}

namespace mpir {
// Registers all Mpir types with the given dialect. This ensures the
// addTypes<...>() template is instantiated in a TU that includes the
// generated type storage definitions, avoiding incomplete-type issues.
void registerMpirTypes(::mlir::Dialect &dialect);
}

#endif // MPLANG_DIALECT_MPIR_MPLANGTYPES_H
