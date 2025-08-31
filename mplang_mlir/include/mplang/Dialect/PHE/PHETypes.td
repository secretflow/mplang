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
//===- PHETypes.td - PHE Types ----------------------------*- tablegen -*-===//
// Minimal types for PHE dialect.
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "PHE.td"

// Ciphertext type with underlying element type and optional shape.
// For bootstrap, keep it simple: element type + shape vector (use -1 for dynamic).

def PHE_CipherType : TypeDef<PHE_Dialect, "Cipher"> {
  let summary = "Homomorphic ciphertext type";
  let description = [{
    Represents an encrypted tensor with element type and shape. Parameters are
    simplified for bootstrap; real impl would include scheme/params.
  }];
  let typeName = "phe.cipher";
  let parameters = (ins
    TypeParameter<"Type", "Element type">:$elementType,
    ArrayRefParameter<"int64_t">:$shape
  );
}
