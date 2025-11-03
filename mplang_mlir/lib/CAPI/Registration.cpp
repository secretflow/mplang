//===- Registration.cpp - C API for Mpir dialect registration ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mplang-c/Registration.h"
#include "mplang/Dialect/Mpir/MpirDialect.h"

#include "mlir/CAPI/IR.h"

void mlirContextRegisterMpirDialect(MlirContext context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::mpir::MpirDialect>();
  unwrap(context)->appendDialectRegistry(registry);
}

void mlirContextLoadMpirDialect(MlirContext context) {
  unwrap(context)->loadDialect<mlir::mpir::MpirDialect>();
}

MlirStringRef mlirMpirDialectGetNamespace() {
  return wrap(mlir::mpir::MpirDialect::getDialectNamespace());
}
