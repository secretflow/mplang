//===- MplangExtension.cpp - Mplang dialect Python extension -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mplang-c/Dialects.h"
#include "mplang-c/Registration.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_mplang, m) {
  m.doc() = "Mplang dialect Python bindings";

  //===--------------------------------------------------------------------===//
  // Mplang dialect
  //===--------------------------------------------------------------------===//
  auto mplangDialect =
      mlir_dialect_submodule(m, "mplang", mlirMplangDialectGetNamespace());

  // Register dialect with context on import
  mplangDialect.def(
      "register_dialect",
      [](py::object context) {
        MlirContext ctx = py::cast<MlirContext>(context);
        mlirContextRegisterMplangDialect(ctx);
      },
      "Register the Mplang dialect with a context.");

  mplangDialect.def(
      "load_dialect",
      [](py::object context) {
        MlirContext ctx = py::cast<MlirContext>(context);
        mlirContextLoadMplangDialect(ctx);
      },
      "Load the Mplang dialect into a context.");
}
