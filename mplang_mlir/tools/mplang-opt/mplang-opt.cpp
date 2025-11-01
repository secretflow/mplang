//===- mplang-opt.cpp ---------------------------------------------------===//
// A tiny opt-like driver that registers the MPLANG dialect and runs mlir-opt.
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"

#include "MPLANGDialect.h.inc"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mplang::MPLANGDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MPLANG optimizer", registry));
}
