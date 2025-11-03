//===- MpirTypes.h ---------------------------------------------------===//
// Wrapper header for generated Mpir types declarations.
//===----------------------------------------------------------------------===//

#ifndef MPLANG_DIALECT_MPIR_MPIRTYPES_H
#define MPLANG_DIALECT_MPIR_MPIRTYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"

// Generate typedef class declarations.
#define GET_TYPEDEF_CLASSES
#include "MpirTypes.h.inc"

namespace mlir {
class Dialect;

namespace mpir {
// Registers all Mpir types with the given dialect. This ensures the
// addTypes<...>() template is instantiated in a TU that includes the
// generated type storage definitions, avoiding incomplete-type issues.
void registerMpirTypes(::mlir::Dialect &dialect);
} // namespace mpir
} // namespace mlir

#endif // MPLANG_DIALECT_MPIR_MPIRTYPES_H
