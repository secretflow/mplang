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
//===- PHE.td - PHE Dialect -------------------------------*- tablegen -*-===//
// Minimal PHE dialect scaffold (homomorphic encryption ops/types)
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

def PHE_Dialect : Dialect {
  let name = "phe";
  let cppNamespace = "mplang";
  let summary = "Homomorphic encryption dialect";
  let description = [{
    The phe dialect models homomorphic encryption types and operations, to be
    targeted by mpir lowering or used directly.
  }];
}
