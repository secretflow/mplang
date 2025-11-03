//===- MPLANGTypes.td -------------------------------------------*- tablegen -*-===//
// Minimal MPLANG type ODS declarations.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "MPLANGDialect.td"

// TableType: multi-party table with columns (similar to DataFrame)
def MPLANG_TableType : TypeDef<Mplang_Dialect, "Table"> {
  let typeName = "mplang.table";
  let summary = "Table type with schema (tuple of column types)";
  let description = [{
    A table characterized by:
    - schema: a tuple type describing column types

    Example:
      !mplang.table<schema = tuple<f32, i64>>
  }];

  let parameters = (
    ins
      TypeParameter<"::mlir::Type", "tuple schema type">:$schema
  );
}
