//===- MplangDialect.cpp -----------------------------------------------===//
// Mplang dialect registration glue.
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinAttributes.h" // for IntegerAttr used by generated types
#include "mlir/IR/BuiltinTypes.h"

// Public wrappers for generated decls referenced in addOperations/addTypes
#include "mplang/Dialect/Mplang/MplangOps.h"
#include "mplang/Dialect/Mplang/MplangTypes.h"

#include "MplangDialect.h.inc"

using namespace mlir;
using namespace mplang;

// Include generated type class implementations in this TU so that when
// addTypes<...>() is instantiated, the Storage classes are complete and
// satisfy static assertions in TypeUniquer.
#define GET_TYPEDEF_CLASSES
#include "MplangTypes.cpp.inc"

void MplangDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MplangOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "MplangTypes.cpp.inc"
      >();
}

// Define the dialect constructor/destructor and type ID.
#include "MplangDialect.cpp.inc"
