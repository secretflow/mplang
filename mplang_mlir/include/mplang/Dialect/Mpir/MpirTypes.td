//===- MpirTypes.td -------------------------------------------*- tablegen -*-===//
// Mplang type ODS declarations.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "MpirDialect.td"

// TableType: multi-party table with columns (similar to DataFrame)
def MPIR_TableType : TypeDef<Mpir_Dialect, "Table"> {
  let typeName = "mplang.table";
  let summary = "Table type with schema (tuple of column types)";
  let description = [{
    A table characterized by:
    - schema: a tuple type describing column types

    Example:
      !mpir.table<schema = tuple<f32, i64>>
  }];

  let parameters = (
    ins
      TypeParameter<"::mlir::Type", "tuple schema type">:$schema
  );
}
