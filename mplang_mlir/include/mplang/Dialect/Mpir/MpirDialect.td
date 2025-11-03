//===- MpirDialect.td -------------------------------------------*- tablegen -*-===//
// Mpir SPMD dialect ODS declaration.
//===----------------------------------------------------------------------===//

include "mlir/IR/DialectBase.td"

def Mpir_Dialect : Dialect {
  let name = "mpir";
  let cppNamespace = "mpir";
  let summary = "Multi-Party Intermediate Representation SPMD dialect";
}
