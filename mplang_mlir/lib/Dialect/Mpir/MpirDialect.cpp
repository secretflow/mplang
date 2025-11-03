//===- MpirDialect.cpp -----------------------------------------------===//
// Mpir dialect registration glue.
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinAttributes.h" // for IntegerAttr used by generated types
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"

// Public wrappers for generated decls referenced in addOperations/addTypes
#include "mplang/Dialect/Mpir/MpirOps.h"
#include "mplang/Dialect/Mpir/MpirTypes.h"

#include "MpirDialect.h.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Custom parser/printer for TensorType shape
//===----------------------------------------------------------------------===//

namespace {
/// Parse tensor shape: dimensions separated by 'x', e.g., "10x20x30"
/// For scalar (rank 0), allow empty or just the element type
static mlir::ParseResult parseTensorShape(mlir::AsmParser &parser,
                                           llvm::SmallVectorImpl<int64_t> &shape) {
  // Try to parse first dimension
  int64_t dim;
  auto result = parser.parseOptionalInteger(dim);
  if (!result.has_value() || result.value().failed()) {
    // No dimension parsed, this is a scalar tensor (rank 0)
    return mlir::success();
  }

  shape.push_back(dim);

  // Parse additional dimensions: x<dim>
  while (mlir::succeeded(parser.parseOptionalKeyword("x"))) {
    if (mlir::failed(parser.parseInteger(dim))) {
      return mlir::failure();
    }
    shape.push_back(dim);
  }

  return mlir::success();
}

/// Print tensor shape: dimensions separated by 'x'
static void printTensorShape(mlir::AsmPrinter &printer, llvm::ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    // Scalar tensor - don't print anything before element type
    return;
  }

  llvm::interleaveComma(shape, printer, [&](int64_t dim) {
    printer << dim;
  });
  printer << "x";
}
} // namespace

// Include generated type class implementations in this TU so that when
// addTypes<...>() is instantiated, the Storage classes are complete and
// satisfy static assertions in TypeUniquer.
#define GET_TYPEDEF_CLASSES
#include "MpirTypes.cpp.inc"

void mpir::MpirDialect::initialize() {
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
