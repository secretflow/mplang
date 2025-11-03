//===- MPLANGDialect.cpp -----------------------------------------------===//
// Minimal MPLANG dialect registration glue.
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinAttributes.h" // for IntegerAttr used by generated types
#include "mlir/IR/BuiltinTypes.h"

// Public wrappers for generated decls referenced in addOperations/addTypes
#include "mplang/Dialect/MPLANG/MPLANGOps.h"
#include "mplang/Dialect/MPLANG/MPLANGTypes.h"

#include "MPLANGDialect.h.inc"

using namespace mlir;
using namespace mplang;

// Include generated type class implementations in this TU so that when
// addTypes<...>() is instantiated, the Storage classes are complete and
// satisfy static assertions in TypeUniquer.
#define GET_TYPEDEF_CLASSES
#include "MPLANGTypes.cpp.inc"

void MplangDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "MPLANGOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "MPLANGTypes.cpp.inc"
      >();
}

// Define the dialect constructor/destructor and type ID.
#include "MPLANGDialect.cpp.inc"
