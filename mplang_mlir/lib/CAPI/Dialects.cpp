//===- Dialects.cpp - CAPI for Mplang dialect registration ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mplang-c/Dialects.h"
#include "mplang/Dialect/Mplang/MplangDialect.h"

#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Mplang, mplang,
                                       mlir::mplang::MplangDialect)
