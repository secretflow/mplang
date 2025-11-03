//===- MpirOps.td ---------------------------------------------*- tablegen -*-===//
// Mpir SPMD operations: call, shuffle, convergence.
// All ops describe multi-party behavior in a homogeneous/isomorphic IR.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "MpirTypes.td"  // This will transitively include MpirDialect.td

//===----------------------------------------------------------------------===//
// PEvalOp - Multi-party evaluation with static execution mask
//===----------------------------------------------------------------------===//

def MPIR_PEvalOp : Op<Mpir_Dialect, "peval", [Pure]> {
  let summary = "SPMD multi-party evaluation with static execution mask";
  let description = [{
    Evaluates a single-party function on a statically-specified subset of parties.
    Supports two modes:

    Mode 1 (MLIR function): Use 'callee' to reference an MLIR function symbol
      %0 = mpir.peval @my_func(%arg0) {rmask = 7 : i64}
           : (!mpir.mp<tensor<10xf32>, 7>) -> !mpir.mp<tensor<10xf32>, 7>

    Mode 2 (External backend): Use 'fn_type' to specify an external kernel
      %1 = mpir.peval (%arg0) {fn_type = "phe.encrypt", rmask = 1 : i64}
           : (!mpir.mp<tensor<10xf32>, 1>)
           -> !mpir.mp<!mpir.encrypted<tensor<10xf32>, "paillier">, 1>

    Key design (New Type System):
    - Input/output pmask: Encoded in the TYPE via MP<T, pmask> wrapper
      * Semantic property: "who holds this data"
    - rmask (runtime mask): Op ATTRIBUTE for execution constraint
      * Operational property: "which parties should execute"
      * If rmask is specified, only parties in rmask execute
      * Output pmask typically equals rmask (narrowing from input pmask)
    - Type composition: MP<Encrypted<Tensor>> for encrypted multi-party data

    Single-party semantics: If this party's rank is in rmask, execute the function;
    otherwise, no-op (does not execute, no side effects).

    IR is homogeneous: all parties see the same op, but only those in rmask execute.

    Attributes:
      - callee: (Optional) Symbol reference to MLIR function (Mode 1)
      - fn_type: (Optional) External backend identifier like "basic.add", "phe.encrypt" (Mode 2)
      - fn_name: (Optional) Function name for debugging/display
      - fn_attrs: (Optional) Dictionary of attributes for backend function (Mode 2)
                  Corresponds to PFunction.attrs in Python (e.g., scheme, key_size)
      - rmask: (Optional) i64 bit-mask indicating which parties execute (bit i = party i)
               If omitted, all parties holding input data (input pmask) execute

    At least one of 'callee' or 'fn_type' must be specified.

    Verifier checks:
      - Exactly one of callee or fn_type is present
      - If callee exists, verify symbol exists and signature matches
      - fn_attrs should only be used with fn_type (Mode 2)
      - If rmask specified: rmask should be subset of input pmask union

    Examples:
      // Mode 1: Call MLIR function @compute on parties 0,1,2
      %0 = mpir.peval @compute(%x, %y) {rmask = 7 : i64}
           : (!mpir.mp<tensor<10xf32>, 7>, !mpir.mp<tensor<10xf32>, 7>)
           -> !mpir.mp<tensor<10xf32>, 7>

      // Mode 2: Call PHE keygen (only party 0 generates keys)
      %pk = mpir.peval () {
        fn_type = "phe.keygen",
        fn_attrs = {scheme = "paillier", key_size = 2048 : i64},
        rmask = 1 : i64
      } : () -> !mpir.mp<!mpir.encrypted<tensor<0xi8>, "paillier_pubkey">, 1>

      // Mode 2: Call basic.add (parties 0,1 compute on plaintext)
      %result = mpir.peval (%a, %b) {fn_type = "basic.add", rmask = 3 : i64}
           : (!mpir.mp<tensor<10xf32>, 3>, !mpir.mp<tensor<10xf32>, 3>)
           -> !mpir.mp<tensor<10xf32>, 3>

      // Mode 2: PHE homomorphic add (party 1 operates on ciphertext)
      %ct_sum = mpir.peval (%ct1, %ct2) {fn_type = "phe.add", rmask = 2 : i64}
           : (!mpir.mp<!mpir.encrypted<tensor<10xf32>, "paillier">, 2>,
              !mpir.mp<!mpir.encrypted<tensor<10xf32>, "paillier">, 2>)
           -> !mpir.mp<!mpir.encrypted<tensor<10xf32>, "paillier">, 2>
  }];

  let arguments = (ins
    OptionalAttr<SymbolRefAttr>:$callee,
    OptionalAttr<StrAttr>:$fn_type,
    OptionalAttr<StrAttr>:$fn_name,
    OptionalAttr<DictionaryAttr>:$fn_attrs,
    OptionalAttr<I64Attr>:$rmask,
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
      %0 = mpir.peval_dyn @my_func(%arg0)
           : (!mpir.tensor<10xf32, 3>) -> !mpir.tensor<10xf32, 3>

    Mode 2 (External backend): Use 'fn_type' to specify an external kernel
      %1 = mpir.peval_dyn (%arg0, %sk) {fn_type = "phe.decrypt"}
           : (!mpir.mp<!mpir.encrypted<tensor<10xf32>, "paillier">, 1>,
              !mpir.mp<!mpir.encrypted<tensor<0xi8>, "paillier_privkey">, 1>)
           -> !mpir.mp<tensor<10xf32>, 1>

    Key design (New Type System):
    - Execution determined by input pmasks (union of all input pmasks)
    - Output pmask: Either statically known OR dynamic (runtime-determined)
    - Dynamic pmask representation: Use !mpir.mp_dynamic<T> for runtime-determined pmask
      (avoids ambiguity with "all parties" representation)

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
           : (!mpir.mp<tensor<10xf32>, 3>, !mpir.mp<tensor<10xf32>, 3>)
           -> !mpir.mp<tensor<10xf32>, 3>

      // Mode 2: Call basic.read with path attribute (dynamic output pmask)
      %1 = mpir.peval_dyn (%rank) {
        fn_type = "basic.read",
        fn_attrs = {path = "/data/file.csv"}
      } : (i64) -> !mpir.mp_dynamic<tensor<10x20xf32>>
      // mp_dynamic indicates runtime-determined pmask

      // Mode 2: Call PHE decrypt
      %2 = mpir.peval_dyn (%ct, %sk) {fn_type = "phe.decrypt"}
           : (!mpir.mp<!mpir.encrypted<tensor<10xf32>, "paillier">, 1>,
              !mpir.mp<!mpir.encrypted<tensor<0xi8>, "paillier_privkey">, 1>)
           -> !mpir.mp<tensor<10xf32>, 1>
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
  let summary = "SPMD shuffle with static source ranks (pull model)";
  let description = [{
    Redistributes data using a "pull" model: each receiving party specifies which
    source party it pulls data from.

    Key design (New Type System):
    - Input pmask: Encoded in input type via MP<T, pmask>
    - Output pmask: Encoded in result type via MP<T, pmask>
    - src_ranks: Attribute specifying routing (length = popcount(output_pmask))
    - Works with both plaintext and encrypted data: MP<Tensor> and MP<Encrypted<Tensor>>

    Semantics (aligned with mplang/core/expr/ast.py ShflSExpr):
    - The i-th party in output pmask pulls data from src_ranks[i]
    - Example: output_pmask=0b110 (parties 1,2), src_ranks=[0, 1]
      * Party 1 pulls from rank 0
      * Party 2 pulls from rank 1

    Single-party semantics: If this party is in output pmask at position i, pull
    data from src_ranks[i]; otherwise, no-op.

    Pull model rationale: Guarantees each output party receives exactly one value,
    maintaining SPMD semantic predictability. Push model would be ambiguous.

    Attributes:
      - src_ranks: DenseI64ArrayAttr, length must equal popcount(output_pmask)

    Verifier checks:
      - len(src_ranks) == popcount(output_pmask)
      - All src_ranks in valid range and present in input pmask
      - Input/output inner types match (tensor shape, element type, encryption schema if applicable)

    Examples:
      // Shuffle plaintext tensor
      %0 = mpir.shfl %input {src_ranks = array<i64: 0, 0>}
           : !mpir.mp<tensor<10xf32>, 1> -> !mpir.mp<tensor<10xf32>, 6>
      // Input: party 0 has data (pmask=0b001)
      // Output: parties 1,2 receive data (pmask=0b110)
      // Both parties 1,2 pull from rank 0

      // Shuffle encrypted tensor (e.g., for homomorphic computation delegation)
      %1 = mpir.shfl %ciphertext {src_ranks = array<i64: 0>}
           : !mpir.mp<!mpir.encrypted<tensor<10xf32>, "paillier">, 1>
           -> !mpir.mp<!mpir.encrypted<tensor<10xf32>, "paillier">, 2>
      // Party 0 sends ciphertext to party 1 for computation
  }];

  let arguments = (ins
    MPIR_MPAnyTensor:$input,
    DenseI64ArrayAttr:$src_ranks
  );
  let results = (outs MPIR_MPAnyTensor:$result);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ShuffleDynOp - Dynamic shuffle
//===----------------------------------------------------------------------===//

def MPIR_ShuffleDynOp : Op<Mpir_Dialect, "shfl_dyn", [Pure]> {
  let summary = "SPMD shuffle with dynamic source ranks and metadata";
  let description = [{
    Dynamic version of shuffle where source routing is a runtime value.

    Key design (New Type System):
    - Input pmask: Encoded in input type via MP<T, pmask>
    - Output pmask: Use MPDynamic<T> type for runtime-determined pmask
    - src_ranks: Runtime value (1D tensor) specifying routing

    Note: This op requires metadata operands to describe dynamic shuffle behavior.
    The actual output pmask is determined at runtime based on which parties
    participate in the shuffle.

    Single-party semantics: If this party is in the runtime output mask, pull data
    from the corresponding source rank in src_ranks tensor; otherwise, no-op.

    Operands:
      - input: value to shuffle
      - src_ranks: 1D tensor<Nxi64> where N = number of output parties

    Verifier checks:
      - src_ranks is 1D tensor of i64
      - Input/output inner types match (tensor shape, element type, encryption if applicable)
      - Output type should be MPDynamic<T> to indicate runtime-determined pmask

    Example (plaintext tensor):
      // Dynamic shuffle from party 0 to runtime-determined parties
      %ranks = tensor.from_elements %c0, %c0 : tensor<2xi64>
      %0 = mpir.shfl_dyn %input, %ranks
           : !mpir.mp<tensor<10xf32>, 1>, tensor<2xi64>
           -> !mpir.mp_dynamic<tensor<10xf32>>
      // Output uses MPDynamic type for runtime-determined pmask

    Example (encrypted tensor):
      // Dynamic shuffle of encrypted data
      %1 = mpir.shfl_dyn %enc_input, %ranks
           : !mpir.mp<encrypted<tensor<10xf32>, "paillier">, 1>, tensor<2xi64>
           -> !mpir.mp_dynamic<encrypted<tensor<10xf32>, "paillier">>
      // Encryption schema preserved in output type
  }];

  let arguments = (ins
    MPIR_MPAnyTensor:$input,
    1DTensorOf<[I64]>:$src_ranks
  );
  let results = (outs MPIR_MPDynamicTensor:$result);

  let assemblyFormat = "$input `,` $src_ranks attr-dict `:` type($input) `,` type($src_ranks) `->` type($result)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ConvOp - Multi-party convergence (disjoint merge)
//===----------------------------------------------------------------------===//

def MPIR_ConvOp : Op<Mpir_Dialect, "conv", [Pure]> {
  let summary = "SPMD convergence: merge disjoint-masked values";
  let description = [{
    Logical merge of multiple values with disjoint pmasks into a single distributed
    value with union pmask. This is purely a compile-time/analysis-time construct.

    Key design (New Type System):
    - Input pmasks: Encoded in each input type via MP<T, pmask>
    - Output pmask: Encoded in result type = union of all input pmasks
    - No attributes needed! Type system handles everything

    Semantics (aligned with mplang/core/expr/ast.py ConvExpr):
    - Takes N inputs, all with identical dtype/shape (for tensors) or schema (for tables)
    - Input pmasks must be pairwise disjoint (no party holds multiple inputs)
    - Output pmask = union of all input pmasks (bitwise OR)
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
    - All input pmasks are pairwise disjoint (no overlap)
    - Output pmask equals union of input pmasks
    - At least one input

    Example (plaintext tensors):
      // Party 0 has %a (pmask=1), party 2 has %b (pmask=4)
      %merged = mpir.conv(%a, %b)
                : (!mpir.mp<tensor<10xf32>, 1>, !mpir.mp<tensor<10xf32>, 4>)
                -> !mpir.mp<tensor<10xf32>, 5>
      // Result has pmask=5 (0b101 = 0b001 | 0b100)
      // Party 0 sees %a, party 2 sees %b

    Example (three-way merge):
      %merged3 = mpir.conv(%x, %y, %z)
                : (!mpir.mp<tensor<10xf32>, 1>,
                   !mpir.mp<tensor<10xf32>, 2>,
                   !mpir.mp<tensor<10xf32>, 4>)
                -> !mpir.mp<tensor<10xf32>, 7>
      // Output pmask = 1 | 2 | 4 = 7 (parties 0,1,2)

    Example (table convergence):
      %table_merged = mpir.conv(%t0, %t1)
                      : (!mpir.mp<tuple<tensor<10xi32>, tensor<10xf32>>, 1>,
                         !mpir.mp<tuple<tensor<10xi32>, tensor<10xf32>>, 2>)
                      -> !mpir.mp<tuple<tensor<10xi32>, tensor<10xf32>>, 3>
      // Tables (represented as tuples) can also converge
  }];

  let arguments = (ins
    Variadic<MPIR_MPTensorOrTable>:$inputs
  );
  let results = (outs MPIR_MPTensorOrTable:$result);

  let assemblyFormat = "`(` $inputs `)` attr-dict `:` `(` type($inputs) `)` `->` type($result)";

  let hasVerifier = 1;
}
