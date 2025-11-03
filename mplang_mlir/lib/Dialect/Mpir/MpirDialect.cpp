//===- MpirDialect.cpp -----------------------------------------------===//
// Mpir dialect registration glue.
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"

// Public wrappers for generated decls referenced in addOperations/addTypes
#include "mplang/Dialect/Mpir/MpirOps.h"
#include "mplang/Dialect/Mpir/MpirTypes.h"

#include "MpirDialect.h.inc"

using namespace mlir;

// Include generated type class implementations in this TU so that when
// addTypes<...>() is instantiated, the Storage classes are complete and
// satisfy static assertions in TypeUniquer.
#define GET_TYPEDEF_CLASSES
#include "MpirTypes.cpp.inc"

void ::mlir::mpir::MpirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MpirOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "MpirTypes.cpp.inc"
      >();
}

// Define the dialect constructor/destructor and type ID.
#include "MpirDialect.cpp.inc"
