//===- MPLANGTypes.td -------------------------------------------*- tablegen -*-===//
// Minimal MPLANG type ODS declarations.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "MPLANGDialect.td"

def MPLANG_TensorType : TypeDef<MPLANG_Dialect, "Tensor"> {
  let mnemonic = "tensor"; // !mplang.tensor
  let summary = "Distributed tensor type with pmask & attrs";
}

def MPLANG_TableType : TypeDef<MPLANG_Dialect, "Table"> {
  let mnemonic = "table"; // !mplang.table
  let summary = "Distributed table type with pmask & attrs";
}
