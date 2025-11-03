//===- Registration.cpp - C API for Mplang dialect registration ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mplang-c/Registration.h"
#include "mplang/Dialect/Mplang/MplangDialect.h"

#include "mlir/CAPI/IR.h"

void mlirContextRegisterMplangDialect(MlirContext context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::mplang::MplangDialect>();
  unwrap(context)->appendDialectRegistry(registry);
}

void mlirContextLoadMplangDialect(MlirContext context) {
  unwrap(context)->loadDialect<mlir::mplang::MplangDialect>();
}

MlirStringRef mlirMplangDialectGetNamespace() {
  return wrap(mlir::mplang::MplangDialect::getDialectNamespace());
}
