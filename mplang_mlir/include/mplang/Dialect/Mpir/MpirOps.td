//===- MpirOps.td ---------------------------------------------*- tablegen -*-===//
// Mpir SPMD operations: call, shuffle, convergence.
// All ops describe multi-party behavior in a homogeneous/isomorphic IR.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "MpirDialect.td"

//===----------------------------------------------------------------------===//
// PEvalOp - Multi-party evaluation with static execution mask
//===----------------------------------------------------------------------===//

def MPIR_PEvalOp : Op<Mpir_Dialect, "peval", [Pure]> {
  let summary = "SPMD multi-party evaluation with static execution mask";
  let description = [{
    Evaluates a single-party function on a statically-specified subset of parties (mask).
    Supports two modes:

    Mode 1 (MLIR function): Use 'callee' to reference an MLIR function symbol
      %0 = mpir.peval @my_func(%arg0) {mask = 7 : i64}
           : (tensor<10xf32>) -> tensor<10xf32>

    Mode 2 (External backend): Use 'fn_type' to specify an external kernel
      %1 = mpir.peval (%arg0) {fn_type = "phe.encrypt", mask = 7 : i64}
           : (tensor<10xf32>) -> tensor<10xf32>

    Single-party semantics: If this party's rank is in the mask, execute the function;
    otherwise, this is a no-op (does not execute, no side effects).

    IR is homogeneous: all parties see the same op in their IR, but only those in
    the mask actually invoke the function at runtime.

    Attributes:
      - callee: (Optional) Symbol reference to MLIR function (Mode 1)
      - fn_type: (Optional) External backend identifier like "basic.add", "phe.encrypt" (Mode 2)
      - fn_name: (Optional) Function name for debugging/display
      - fn_attrs: (Optional) Dictionary of attributes passed to backend function (Mode 2 only)
                  This corresponds to PFunction.attrs in Python (e.g., scheme, key_size, etc.)
      - mask: (Required) i64 bit-mask indicating which parties execute (bit i = party i)

    At least one of 'callee' or 'fn_type' must be specified.

    Verifier checks:
      - Exactly one of callee or fn_type is present
      - If callee exists, verify symbol exists and signature matches
      - fn_attrs should only be used with fn_type (Mode 2)

    Examples:
      // Mode 1: Call MLIR function @compute on parties 0,1,2
      %0 = mpir.peval @compute(%x, %y) {mask = 7 : i64}
           : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>

      // Mode 2: Call PHE keygen with custom attributes
      %pk, %sk = mpir.peval () {
        fn_type = "phe.keygen",
        fn_attrs = {scheme = "paillier", key_size = 2048 : i64},
        mask = 1 : i64
      } : () -> (!mpir.public_key, !mpir.private_key)

      // Mode 2: Call PHE encrypt
      %ct = mpir.peval (%data, %pk) {
        fn_type = "phe.encrypt",
        fn_attrs = {fxp_bits = 12 : i64},
        mask = 1 : i64
      } : (tensor<10xf32>, !mpir.public_key) -> !mpir.ciphertext
  }];

  let arguments = (ins
    OptionalAttr<SymbolRefAttr>:$callee,
    OptionalAttr<StrAttr>:$fn_type,
    OptionalAttr<StrAttr>:$fn_name,
    OptionalAttr<DictionaryAttr>:$fn_attrs,
    I64Attr:$mask,
    Variadic<AnyType>:$args
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    ( $callee^ )?
    `(` $args `)`
    attr-dict
    `:` functional-type($args, $results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// PEvalDynOp - Multi-party evaluation with dynamic execution
//===----------------------------------------------------------------------===//

def MPIR_PEvalDynOp : Op<Mpir_Dialect, "peval_dyn", [Pure]> {
  let summary = "SPMD multi-party evaluation with dynamic execution (mask inferred from args)";
  let description = [{
    Evaluates a single-party function where execution is determined by argument
    availability at runtime. Supports two modes:

    Mode 1 (MLIR function): Use 'callee' to reference an MLIR function symbol
      %0 = mpir.peval_dyn @my_func(%arg0) : (tensor<10xf32>) -> tensor<10xf32>

    Mode 2 (External backend): Use 'fn_type' to specify an external kernel
      %1 = mpir.peval_dyn (%arg0) {fn_type = "phe.decrypt"}
           : (tensor<10xf32>) -> tensor<10xf32>

    Single-party semantics: If this party has valid arguments (determined by runtime
    pmask intersection), execute the function; otherwise, no-op.

    This models the Python AST's EvalExpr behavior where mask is deduced from args.

    Attributes:
      - callee: (Optional) Symbol reference to MLIR function (Mode 1)
      - fn_type: (Optional) External backend identifier like "basic.add", "phe.decrypt" (Mode 2)
      - fn_name: (Optional) Function name for debugging/display
      - fn_attrs: (Optional) Dictionary of attributes passed to backend function (Mode 2 only)
                  This corresponds to PFunction.attrs in Python

    At least one of 'callee' or 'fn_type' must be specified.

    Verifier checks:
      - Exactly one of callee or fn_type is present
      - If callee exists, verify symbol exists and signature matches
      - fn_attrs should only be used with fn_type (Mode 2)

    Examples:
      // Mode 1: Call MLIR function, execution based on arg availability
      %0 = mpir.peval_dyn @compute(%x, %y)
           : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>

      // Mode 2: Call basic.read with path attribute
      %1 = mpir.peval_dyn (%rank) {
        fn_type = "basic.read",
        fn_attrs = {path = "/data/file.csv"}
      } : (i64) -> tensor<?x?xf32>

      // Mode 2: Call PHE decrypt
      %2 = mpir.peval_dyn (%ct, %sk) {fn_type = "phe.decrypt"}
           : (!mpir.ciphertext, !mpir.private_key) -> tensor<10xf32>
  }];

  let arguments = (ins
    OptionalAttr<SymbolRefAttr>:$callee,
    OptionalAttr<StrAttr>:$fn_type,
    OptionalAttr<StrAttr>:$fn_name,
    OptionalAttr<DictionaryAttr>:$fn_attrs,
    Variadic<AnyType>:$args
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    ( $callee^ )?
    `(` $args `)`
    attr-dict
    `:` functional-type($args, $results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ShuffleStaticOp - Static shuffle with explicit source routing
//===----------------------------------------------------------------------===//

def MPIR_ShuffleStaticOp : Op<Mpir_Dialect, "shfl", [Pure]> {
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
      %0 = mpir.shfl %input {mask = 5 : i64, src_ranks = array<i64: 1, 3>}
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

def MPIR_ShuffleDynOp : Op<Mpir_Dialect, "shfl_dyn", [Pure]> {
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
      %0 = mpir.shfl_dyn %input, %mask, %ranks
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

def MPIR_ConvOp : Op<Mpir_Dialect, "conv", [Pure]> {
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
      %merged = mpir.conv(%a, %b)
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
