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
//===- PHEOps.td - PHE Ops -------------------------------*- tablegen -*-===//
// Minimal ops for homomorphic encryption dialect.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "PHETypes.td"

class PHE_Op<string mnemonic, list<Trait> traits = []>
  : Op<PHE_Dialect, mnemonic, traits>;

// Encrypt plaintext into ciphertext
def PHE_Encrypt : PHE_Op<"encrypt"> {
  let summary = "Encrypt plaintext to ciphertext";
  let arguments = (ins AnyType:$input);
  let results = (outs PHE_CipherType:$result);
}

// Decrypt ciphertext into plaintext
def PHE_Decrypt : PHE_Op<"decrypt"> {
  let summary = "Decrypt ciphertext to plaintext";
  let arguments = (ins PHE_CipherType:$input, TypeAttr:$out_type);
  let results = (outs AnyType:$result);
}

// Homomorphic add
def PHE_Add : PHE_Op<"add"> {
  let summary = "Ciphertext add";
  let arguments = (ins PHE_CipherType:$lhs, PHE_CipherType:$rhs);
  let results = (outs PHE_CipherType:$result);
}

// Homomorphic mul
def PHE_Mul : PHE_Op<"mul"> {
  let summary = "Ciphertext mul";
  let arguments = (ins PHE_CipherType:$lhs, PHE_CipherType:$rhs);
  let results = (outs PHE_CipherType:$result);
}
