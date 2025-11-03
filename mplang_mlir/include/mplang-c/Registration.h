//===- Registration.h - C API for Mpir dialect registration --*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C API for Mpir dialect registration.
//
//===----------------------------------------------------------------------===//

#ifndef MPLANG_C_REGISTRATION_H
#define MPLANG_C_REGISTRATION_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Register the Mpir dialect with the given context.
MLIR_CAPI_EXPORTED void mlirContextRegisterMplangDialect(MlirContext context);

/// Load the Mpir dialect into the given context.
MLIR_CAPI_EXPORTED void mlirContextLoadMplangDialect(MlirContext context);

/// Get the namespace string for the Mpir dialect.
MLIR_CAPI_EXPORTED MlirStringRef mlirMplangDialectGetNamespace(void);

#ifdef __cplusplus
}
#endif

#endif // MPLANG_C_REGISTRATION_H
