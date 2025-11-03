//===- MplangTypes.h ---------------------------------------------------===//
// Wrapper header for generated Mplang types declarations.
//===----------------------------------------------------------------------===//

#ifndef MPLANG_DIALECT_MPLANG_MPLANGTYPES_H
#define MPLANG_DIALECT_MPLANG_MPLANGTYPES_H

#include "mlir/IR/Types.h"

// Generate typedef class declarations.
#define GET_TYPEDEF_CLASSES
#include "MplangTypes.h.inc"

namespace mlir {
class Dialect;
}

namespace mplang {
// Registers all Mplang types with the given dialect. This ensures the
// addTypes<...>() template is instantiated in a TU that includes the
// generated type storage definitions, avoiding incomplete-type issues.
void registerMplangTypes(::mlir::Dialect &dialect);
}

#endif // MPLANG_DIALECT_MPLANG_MPLANGTYPES_H
