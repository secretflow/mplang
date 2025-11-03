//===- MplangDialect.td -----------------------------------------*- tablegen -*-===//
// Mplang SPMD dialect ODS declaration.
//===----------------------------------------------------------------------===//

include "mlir/IR/DialectBase.td"

def Mplang_Dialect : Dialect {
  let name = "mplang";
  let cppNamespace = "mplang";
  let summary = "MPLang SPMD dialect (distributed types and SPMD ops)";
}
