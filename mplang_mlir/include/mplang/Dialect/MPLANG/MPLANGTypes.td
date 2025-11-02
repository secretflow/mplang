//===- MPLANGTypes.td -------------------------------------------*- tablegen -*-===//
// Minimal MPLANG type ODS declarations.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "MPLANGDialect.td"

// A distributed tensor carrying element type, (possibly dynamic) shape, and
// an optional party-execution mask (pmask). We keep attributes minimal for now.
def MPLANG_TensorType : TypeDef<MPLANG_Dialect, "Tensor"> {
  let typeName = "mplang.tensor";
  let summary = "Distributed tensor type with element type, shape, and optional pmask";
  let description = [{
    A distributed tensor characterized by:
    - element: element type (e.g., f32, i64)
    - shape: a list of dims (use -1 for dynamic)
    - pmask: optional IntegerAttr mask indicating participating parties

    Example (generic assembly may vary based on auto-generated printer):
      !mplang.tensor<element = f32, shape = [2, -1], pmask = 3>
  }];

  let parameters = (
    ins
  TypeParameter<"::mlir::Type", "element type">:$element,
      ArrayRefParameter<"int64_t">:$shape,
      OptionalParameter<"::mlir::IntegerAttr">:$pmask
  );
}

// A distributed table with a schema (tuple of column types) and optional pmask.
def MPLANG_TableType : TypeDef<MPLANG_Dialect, "Table"> {
  let typeName = "mplang.table";
  let summary = "Distributed table type with schema and optional pmask";
  let description = [{
    A distributed table characterized by:
    - schema: a tuple type describing column types
    - pmask: optional IntegerAttr mask indicating participating parties

    Example:
      !mplang.table<schema = tuple<f32, i64>, pmask = 0x3>
  }];

  let parameters = (
    ins
  TypeParameter<"::mlir::Type", "tuple schema type">:$schema,
      OptionalParameter<"::mlir::IntegerAttr">:$pmask
  );
}
