// Copyright 2025 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===- MPIRTypes.td - MPIR Types ----------------------------*- tablegen -*-===//
// Minimal type scaffolding; real C++ impl to be added with MLIR wiring.
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "MPIR.td"

// NOTE:
// For bootstrap, we encode pmask as a 64-bit integer attribute where -1 denotes
// dynamic. A dedicated attribute/type can be added later.

def MPIR_TensorType : TypeDef<MPIR_Dialect, "Tensor"> {
  let summary = "Tensor with pmask";
  let description = [{
    MPIR tensor wraps an element type, a shape vector (use -1 for dynamic dims),
    and a pmask (int64; -1 represents dynamic pmask).
  }];
  let typeName = "mpir.tensor";
  let parameters = (ins
    TypeParameter<"Type", "Element type">:$elementType,
    ArrayRefParameter<"int64_t">:$shape,
    AttrParameter<"IntegerAttr", "party mask (-1 dynamic)">:$pmask
  );
}

def MPIR_TableType : TypeDef<MPIR_Dialect, "Table"> {
  let summary = "Table type with pmask";
  let description = [{
    Placeholder table type carrying a schema tag and pmask. The schema is a
    string tag for bootstrap; real impl will use structured column types.
  }];
  let typeName = "mpir.table";
  let parameters = (ins
    AttrParameter<"StringAttr", "schema tag">:$schema,
    AttrParameter<"IntegerAttr", "party mask (-1 dynamic)">:$pmask
  );
}
