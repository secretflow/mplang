//===- MPLANGDialect.td -----------------------------------------*- tablegen -*-===//
// Minimal MPLANG dialect ODS declaration.
//===----------------------------------------------------------------------===//

include "mlir/IR/DialectBase.td"

def MPLANG_Dialect : Dialect {
  let name = "mplang";
  let cppNamespace = "mplang";
  let summary = "MPLang SPMD dialect (distributed types and SPMD ops)";
}
