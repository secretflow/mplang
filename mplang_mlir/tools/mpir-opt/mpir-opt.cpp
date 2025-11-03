//===- mpir-opt.cpp ---------------------------------------------------===//
// A tiny opt-like driver that registers the Mpir dialect and runs mlir-opt.
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "MpirDialect.h.inc"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::mpir::MpirDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MPLANG optimizer", registry));
}
