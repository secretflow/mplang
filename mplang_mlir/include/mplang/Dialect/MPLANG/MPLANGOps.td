//===- MPLANGOps.td ---------------------------------------------*- tablegen -*-===//
// MPLANG SPMD operations: call, shuffle, convergence.
// All ops describe multi-party behavior in a homogeneous/isomorphic IR.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "MPLANGDialect.td"

//===----------------------------------------------------------------------===//
// PCallStaticOp - Static mask private call
//===----------------------------------------------------------------------===//

def MPLANG_PCallStaticOp : Op<Mplang_Dialect, "pcall", [Pure, CallOpInterface]> {
  let summary = "SPMD private call with static execution mask";
  let description = [{
    Calls a single-party (private) function on a statically-specified subset of parties (mask).

    Single-party semantics: If this party's rank is in the mask, execute the call;
    otherwise, this is a no-op (does not execute, no side effects).

    IR is homogeneous: all parties see the same op in their IR, but only those in
    the mask actually invoke the callee at runtime.

    Attributes:
      - callee: Symbol reference to the called function
      - mask: i64 bit-mask indicating which parties execute (bit i = party i)

    Verifier checks:
      - Callee symbol exists and signature matches
      - Mask is non-negative
      - (Optional) If arg pmasks are statically known, validate consistency

    Example:
      %0 = mplang.pcall @foo(%arg0, %arg1) {mask = 7 : i64}
           : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
      // Parties 0,1,2 execute @foo; party 3+ see this op but don't execute
  }];

  let arguments = (ins
    SymbolRefAttr:$callee,
    I64Attr:$mask,
    Variadic<AnyType>:$args
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = "$callee `(` $args `)` attr-dict `:` functional-type($args, $results)";

  let extraClassDeclaration = [{
    /// CallOpInterface methods
    mlir::Operation::operand_range getArgOperands() { return getArgs(); }
    mlir::MutableOperandRange getArgOperandsMutable() { return getArgsMutable(); }
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<mlir::SymbolRefAttr>());
    }
  }];
}

//===----------------------------------------------------------------------===//
// PCallDynOp - Dynamic mask private call
//===----------------------------------------------------------------------===//

def MPLANG_PCallDynOp : Op<Mplang_Dialect, "pcall_dyn", [Pure, CallOpInterface]> {
  let summary = "SPMD private call with dynamic execution (args presence determines execution)";
  let description = [{
    Calls a single-party (private) function where execution is determined by argument
    availability at runtime.

    Single-party semantics: If this party has valid arguments (determined by runtime
    pmask intersection), execute the call; otherwise, no-op.

    This models the Python AST's EvalExpr behavior where mask is deduced from args.

    Attributes:
      - callee: Symbol reference to the called function

    Verifier checks:
      - Callee symbol exists and signature matches

    Example:
      %0 = mplang.pcall_dyn @bar(%arg0, %arg1)
           : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
      // Each party executes if it has both arg0 and arg1 at runtime
  }];

  let arguments = (ins
    SymbolRefAttr:$callee,
    Variadic<AnyType>:$args
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = "$callee `(` $args `)` attr-dict `:` functional-type($args, $results)";

  let extraClassDeclaration = [{
    /// CallOpInterface methods
    mlir::Operation::operand_range getArgOperands() { return getArgs(); }
    mlir::MutableOperandRange getArgOperandsMutable() { return getArgsMutable(); }
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("callee");
    }
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<mlir::SymbolRefAttr>());
    }
  }];
}

//===----------------------------------------------------------------------===//
// ShuffleStaticOp - Static shuffle with explicit source routing
//===----------------------------------------------------------------------===//

def MPLANG_ShuffleStaticOp : Op<Mplang_Dialect, "shfl", [Pure]> {
  let summary = "SPMD shuffle with static mask and source ranks (pull model)";
  let description = [{
    Redistributes data using a "pull" model: each receiving party specifies which
    source party it pulls data from.

    Semantics (aligned with mplang/core/expr/ast.py ShflSExpr):
    - Output mask: bit-mask of parties that will receive output
    - src_ranks: array of source ranks, length = popcount(mask)
    - The i-th party in output mask pulls data from src_ranks[i]

    Single-party semantics: If this party is in output mask at position i, pull
    data from src_ranks[i]; otherwise, no-op.

    Pull model rationale: Guarantees each output party receives exactly one value,
    maintaining SPMD semantic predictability. Push model would be ambiguous.

    Attributes:
      - mask: i64 output party mask
      - src_ranks: DenseI64ArrayAttr, length must equal popcount(mask)

    Verifier checks:
      - len(src_ranks) == popcount(mask)
      - All src_ranks in valid range and present in input pmask (if statically known)
      - Input/output element types and shapes match

    Example:
      %0 = mplang.shfl %input {mask = 5 : i64, src_ranks = array<i64: 1, 3>}
           : tensor<10xf32> -> tensor<10xf32>
      // mask=5=0b101: parties 0 and 2 receive output
      // Party 0 pulls from rank 1, party 2 pulls from rank 3
  }];

  let arguments = (ins
    AnyType:$input,
    I64Attr:$mask,
    DenseI64ArrayAttr:$src_ranks
  );
  let results = (outs AnyType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// ShuffleDynOp - Dynamic shuffle
//===----------------------------------------------------------------------===//

def MPLANG_ShuffleDynOp : Op<Mplang_Dialect, "shfl_dyn", [Pure]> {
  let summary = "SPMD shuffle with dynamic mask and source ranks";
  let description = [{
    Dynamic version of shuffle where mask and source routing are runtime values.

    Single-party semantics: If this party is in the runtime output mask, pull data
    from the corresponding source rank in src_ranks tensor; otherwise, no-op.

    Operands:
      - input: value to shuffle
      - mask: i64 scalar, runtime output party mask
      - src_ranks: 1D tensor<Nxi64> where N = popcount(mask), source ranks for each output party

    Verifier checks:
      - src_ranks is 1D tensor of i64
      - Element types and shapes match between input and result
      - Runtime: len(src_ranks) must equal popcount(mask)

    Example:
      %mask = arith.constant 5 : i64
      %ranks = tensor.from_elements %c1, %c3 : tensor<2xi64>
      %0 = mplang.shfl_dyn %input, %mask, %ranks
           : tensor<10xf32>, i64, tensor<2xi64> -> tensor<10xf32>
  }];

  let arguments = (ins
    AnyType:$input,
    I64:$mask,
    1DTensorOf<[I64]>:$src_ranks
  );
  let results = (outs AnyType:$result);

  let assemblyFormat = "$input `,` $mask `,` $src_ranks attr-dict `:` type($input) `,` type($mask) `,` type($src_ranks) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// ConvOp - Multi-party convergence (disjoint merge)
//===----------------------------------------------------------------------===//

def MPLANG_ConvOp : Op<Mplang_Dialect, "conv", [Pure]> {
  let summary = "SPMD convergence: merge disjoint-masked values";
  let description = [{
    Logical merge of multiple values with disjoint pmasks into a single distributed
    value with union pmask. This is purely a compile-time/analysis-time construct.

    Semantics (aligned with mplang/core/expr/ast.py ConvExpr):
    - Takes N inputs, all with identical dtype/shape (for tensors) or schema (for tables)
    - Input pmasks must be pairwise disjoint (no party holds multiple inputs)
    - Output pmask = union of all input pmasks
    - Each party retains its local data (no cross-party data movement)

    Single-party semantics: Pure identity/no-op. If this party holds input i, the
    output is just input i. No computation occurs.

    This op exists for:
    1. Type checking: ensure merged values are compatible
    2. Mask analysis: explicitly track pmask unions
    3. IR readability: make convergence points explicit

    Can be eliminated in optimization passes (canonicalize to identity/select).

    Verifier checks:
    - All inputs have same element type and shape/schema
    - (If pmasks statically known) all input pmasks are pairwise disjoint
    - At least one input

    Example:
      // Party 0 has %a (pmask=0b001), party 2 has %b (pmask=0b100)
      %merged = mplang.conv(%a, %b)
                : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
      // Result has pmask=0b101; party 0 sees %a, party 2 sees %b
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs
  );
  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $inputs `)` attr-dict `:` `(` type($inputs) `)` `->` type($result)";

  let hasVerifier = 1;
}
