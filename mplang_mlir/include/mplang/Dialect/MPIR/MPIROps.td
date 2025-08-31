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
//===- MPIROps.td - MPIR Ops ------------------------------*- tablegen -*-===//
// Ops aligned to expr/ast.py semantics (no send/recv). SIMT-like control.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "MPIRTypes.td"

class MPIR_Op<string mnemonic, list<Trait> traits = []>
  : Op<MPIR_Dialect, mnemonic, traits>;

// eval: evaluate a PFunction on args, producing N results.
// For bootstrap, reference the PFunction by string name; verifier ensures arity.
def MPIR_Eval : MPIR_Op<"eval"> {
  let summary = "Evaluate a multi-party function (PFunction)";
  let description = [{
    Evaluate the referenced PFunction with provided arguments. Result arity and
    types are determined by the function signature. pmask rules follow the
    semantics in mplang.expr.ast.EvalExpr.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs, StrAttr:$pfunc);
  let results = (outs Variadic<AnyType>:$results);
}

// tuple: aggregate multiple single-output values into a multi-result tuple.
def MPIR_Tuple : MPIR_Op<"tuple"> {
  let summary = "Tuple construction from single-result values";
  let description = [{
    Flattens inputs into multiple results. Each input must be single-result.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
}

// pconv: convergence of multiple variables; validates dtype/shape and pmask union.
def MPIR_PConv : MPIR_Op<"pconv"> {
  let summary = "Convergence of variables under pmask union constraints";
  let description = [{
    Implements ConvExpr: ensures same dtype/shape across inputs, and computes
    output pmask as union (or dynamic if any dynamic present).
  }];
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);
}

// shfl.static: static shuffle (pull model): pmask + src_ranks attributes.
def MPIR_ShflStatic : MPIR_Op<"shfl_static"> {
  let summary = "Static shuffle with destination pmask and source ranks (pull)";
  let description = [{
    Implements ShflSExpr. pmask is int64 (-1 dynamic). src_ranks is i64 array.
  }];
  let arguments = (ins AnyType:$input,
                       I64Attr:$pmask,
                       ArrayAttr:$src_ranks);
  let results = (outs AnyType:$result);
}

// shfl: dynamic shuffle with index operand.
def MPIR_Shfl : MPIR_Op<"shfl"> {
  let summary = "Dynamic shuffle by index (pull)";
  let description = [{
    Implements ShflExpr with dynamic source index.
  }];
  let arguments = (ins AnyType:$input, AnyType:$index);
  let results = (outs AnyType:$result);
}

// if: SIMT-like masked conditional; regions carry then/else bodies with same signature.
def MPIR_If : MPIR_Op<"if"> {
  let summary = "SIMT-style conditional with pmask-aware semantics";
  let description = [{
    Branch selection respects multi-party masks; then/else must return same types.
  }];
  let arguments = (ins AnyType:$pred, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
}

// while: SIMT-style while; carries cond/body regions.
def MPIR_While : MPIR_Op<"while"> {
  let summary = "SIMT-style while loop with pmask-aware semantics";
  let description = [{
    Result types equal initial state; cond/body regions enforce signature.
  }];
  let arguments = (ins Variadic<AnyType>:$init);
  let results = (outs Variadic<AnyType>:$results);
}

//===----------------------------------------------------------------------===//
// Region terminators
//===----------------------------------------------------------------------===//

def MPIR_Yield : MPIR_Op<"yield", [Terminator]> {
  let summary = "Yield values from an mpir region";
  let description = [{
    Terminator for mpir regions (then/else/body) yielding values to the parent
    op. The number and types of operands must match the enclosing region's
    expected yields.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs);
}

def MPIR_CondYield : MPIR_Op<"cond_yield", [Terminator]> {
  let summary = "Yield loop continuation predicate from mpir.while cond region";
  let description = [{
    Terminator for the mpir.while cond region. The single i1 operand determines
    whether the loop continues. Additional carried values are not modeled yet.
  }];
  let arguments = (ins I1:$cond);
  let results = (outs);
}
