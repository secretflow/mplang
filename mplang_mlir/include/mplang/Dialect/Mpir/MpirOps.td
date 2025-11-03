//===- MpirOps.td ---------------------------------------------*- tablegen -*-===//
// Mpir SPMD operations: call, shuffle, convergence.
// All ops describe multi-party behavior in a homogeneous/isomorphic IR.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
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

//===----------------------------------------------------------------------===//
// Terminator Operations (must be defined before control flow ops)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ReturnOp - Return from peval/cond regions
//===----------------------------------------------------------------------===//

def MPIR_ReturnOp : Op<Mpir_Dialect, "return", [
    Pure, Terminator
  ]> {
  let summary = "Return values from MPIR regions";
  let description = [{
    Terminates a region and returns zero or more values. Used in:
    - peval/peval_dyn regions: returns computed values
    - cond then/else branches: returns branch results

    The returned values must match the expected types of the parent operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>
  ];
}

//===----------------------------------------------------------------------===//
// ConditionOp - Return condition from while loop condition region
//===----------------------------------------------------------------------===//

def MPIR_ConditionOp : Op<Mpir_Dialect, "condition", [
    Pure, Terminator
  ]> {
  let summary = "Return condition from while loop condition region";
  let description = [{
    Terminates the condition region of a while loop and returns the
    boolean condition that determines whether to continue looping.

    The condition must be an MP<i1, pmask> value.
  }];

  let arguments = (ins AnyTypeOf<[MPIR_MPType, MPIR_MPDynamicType]>:$condition);

  let assemblyFormat = "$condition attr-dict `:` type($condition)";
}

//===----------------------------------------------------------------------===//
// YieldOp - Yield values from while loop body region
//===----------------------------------------------------------------------===//

def MPIR_YieldOp : Op<Mpir_Dialect, "yield", [
    Pure, Terminator
  ]> {
  let summary = "Yield values from while loop body region";
  let description = [{
    Terminates the body region of a while loop and returns the next
    iteration state values.

    The yielded values become the arguments to the next iteration's
    condition and body regions.
  }];

  let arguments = (ins Variadic<AnyType>:$results);

  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// UniformCondOp - Uniform conditional with short-circuit execution
//===----------------------------------------------------------------------===//

def MPIR_UniformCondOp : Op<Mpir_Dialect, "uniform_cond", [
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"ReturnOp">
  ]> {
  let summary = "Uniform SPMD conditional with short-circuit execution";
  let description = [{
    CRITICAL: Conditional execution where the predicate MUST be uniform across
    all parties (i.e., all parties must have the same boolean value).

    *** WHY UNIFORM? ***
    Non-uniform predicates cause DEADLOCK in multi-party protocols:
    - If P0/P2 have pred=true (execute then), P1 has pred=false (execute else)
    - Then branch calls smpc.seal() (needs all parties P0/P1/P2)
    - Else branch calls smpc.reveal() (needs all parties)
    - Result: P0/P2 wait for P1 in seal(), P1 waits for P0/P2 in reveal()
    - DEADLOCK: All parties hang forever!

    *** SHORT-CIRCUIT SEMANTICS ***
    Unlike data-parallel elementwise blend (jax.where):
    - Only the CHOSEN branch executes (if pred=true: then, else: else)
    - Unchosen branch has ZERO side-effects (not executed at all)
    - This allows skipping expensive multi-party protocols in unchosen branch
    - Performance: O(expensive_then) + O(cheap_else) becomes O(expensive_then)
      or O(cheap_else) depending on pred, not their sum

    *** RUNTIME VERIFICATION ***
    When verify_uniform=true (default):
    - Runtime checks predicate uniformity via O(P²) all-gather
    - If non-uniform detected: raises ValueError with diagnostic info
    - Can disable for trusted/performance-critical paths (use carefully!)

    *** FOR DIVERGENT PREDICATES ***
    If predicates can diverge (e.g., per-sample masks), use:
    - mpir.peval with jax.where/jax.select inside the region
    - Both values computed, elementwise blend based on mask
    - No structural control flow divergence (safe for multi-party ops)

    Key design:
    - Predicate: MP<i1, pmask> or MP<tensor<i1>, pmask> (scalar only!)
    - Uniformity: All parties MUST have identical boolean value
    - Execution: Short-circuit - only chosen branch executes
    - Verification: Runtime uniformity check (optional, default on)
    - Both branches must return same types (structural consistency)

    Semantics (aligned with mplang/core/primitive.py uniform_cond):
    - All parties evaluate predicate to same boolean
    - If pred=true: execute then branch, skip else branch
    - If pred=false: skip then branch, execute else branch
    - Non-uniform pred + verify_uniform=true: runtime error

    Example (uniform decision - all parties agree):
      %pred = ... : !mpir.mp<i1, 7>  // All parties have same value
      %result = mpir.uniform_cond %pred : !mpir.mp<i1, 7>
                -> !mpir.mp<tensor<10xf32>, 7> {
        // Then branch - expensive multi-party protocol
        %a = mpir.peval @expensive_smpc() {rmask = 7 : i64}
             : () -> !mpir.mp<tensor<10xf32>, 7>
        mpir.return %a : !mpir.mp<tensor<10xf32>, 7>
      } {
        // Else branch - cheap local computation
        %b = mpir.peval @cheap_local() {rmask = 7 : i64}
             : () -> !mpir.mp<tensor<10xf32>, 7>
        mpir.return %b : !mpir.mp<tensor<10xf32>, 7>
      }
      // If pred=true: only expensive_smpc runs (else branch skipped)
      // If pred=false: only cheap_local runs (then branch skipped)

    Example (disable verification for trusted path):
      %pred = ... : !mpir.mp<i1, 3>
      %result = mpir.uniform_cond %pred {verify_uniform = false}
                : !mpir.mp<i1, 3> -> !mpir.mp<tensor<10xf32>, 3> {
        %a = ... : !mpir.mp<tensor<10xf32>, 3>
        mpir.return %a : !mpir.mp<tensor<10xf32>, 3>
      } {
        %b = ... : !mpir.mp<tensor<10xf32>, 3>
        mpir.return %b : !mpir.mp<tensor<10xf32>, 3>
      }
  }];

  let arguments = (ins
    AnyTypeOf<[MPIR_MPType, MPIR_MPDynamicType]>:$condition,
    DefaultValuedAttr<BoolAttr, "true">:$verify_uniform
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion, SizedRegion<1>:$elseRegion);

  let assemblyFormat = "$condition attr-dict `:` type($condition) `->` type($results) $thenRegion $elseRegion";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// UniformWhileOp - Uniform while loop with synchronous iterations
//===----------------------------------------------------------------------===//

def MPIR_UniformWhileOp : Op<Mpir_Dialect, "uniform_while", [
    RecursiveMemoryEffects
  ]> {
  let summary = "Uniform SPMD while loop with synchronous iteration";
  let description = [{
    CRITICAL: While loop where the condition MUST be uniform across all parties
    (i.e., all parties must agree whether to continue or exit at each iteration).

    *** WHY UNIFORM? ***
    Non-uniform loop conditions cause similar DEADLOCK issues as uniform_cond:
    - If P0/P2 have cond=true (continue iteration), P1 has cond=false (exit)
    - P0/P2 enter loop body, call multi-party ops (e.g., smpc.matmul)
    - P1 exits loop, proceeds to next code (e.g., different multi-party op)
    - Result: Protocol mismatch, parties call different operations
    - DEADLOCK or PROTOCOL ERROR!

    *** SYNCHRONOUS ITERATION ***
    All parties must:
    - Enter/exit iterations together (lockstep execution)
    - Evaluate condition to same boolean at each iteration
    - Execute loop body together or exit together
    - Conservative: If ANY party wants to exit, ALL parties exit

    *** RUNTIME VERIFICATION ***
    When verify_uniform=true (default):
    - Runtime checks condition uniformity at each iteration via O(P²) all-gather
    - If non-uniform detected: raises ValueError with iteration info
    - Can disable for trusted/performance-critical loops (use carefully!)

    *** FOR DIVERGENT LOOPS ***
    If loop conditions can diverge (e.g., per-party convergence), use:
    - Fixed iteration count with mpir.peval and masking inside body
    - Or refactor to ensure uniform termination condition
    - Divergent loops are NOT safe for multi-party protocols!

    Key design:
    - Loop state: MP values passed to condition and body regions
    - Condition region: evaluates to MP<i1, pmask> (scalar only!)
    - Uniformity: All parties MUST agree on continue/exit decision
    - Execution: All parties execute iterations synchronously
    - Verification: Runtime uniformity check at each iteration (optional)

    Semantics (aligned with mplang/core/primitive.py uniform_while):
    - Initialize loop state with init_args
    - Condition region receives current state, returns MP<i1>
    - All parties check: if uniform true, execute body; if false, exit
    - Body region receives current state, returns next state
    - Repeat until condition becomes uniform false
    - Conservative: If non-uniform condition detected, exit (safe default)

    Example (uniform counter - all parties agree on max):
      %init = ... : !mpir.mp<tensor<i32>, 7>
      %max = ... : !mpir.mp<tensor<i32>, 7>  // All parties have same max
      %result = mpir.uniform_while (%arg = %init) : (!mpir.mp<tensor<i32>, 7>)
                -> !mpir.mp<tensor<i32>, 7> {
        // Condition region
        ^cond(%state: !mpir.mp<tensor<i32>, 7>):
          %cmp = mpir.peval(%state, %max) {
            ^bb0(%s: tensor<i32>, %m: tensor<i32>):
              %c = arith.cmpi slt, %s, %m : tensor<i32>
              mpir.return %c : i1
          } {rmask = 7 : i64} : (!mpir.mp<tensor<i32>, 7>, !mpir.mp<tensor<i32>, 7>)
                             -> !mpir.mp<i1, 7>
          mpir.condition %cmp : !mpir.mp<i1, 7>
      } do {
        // Body region - can safely call multi-party ops
        ^body(%state: !mpir.mp<tensor<i32>, 7>):
          %next = mpir.peval(%state) {
            ^bb0(%s: tensor<i32>):
              %c1 = arith.constant 1 : i32
              %inc = arith.addi %s, %c1 : tensor<i32>
              mpir.return %inc : tensor<i32>
          } {rmask = 7 : i64} : (!mpir.mp<tensor<i32>, 7>)
                             -> !mpir.mp<tensor<i32>, 7>
          mpir.yield %next : !mpir.mp<tensor<i32>, 7>
      }
      // All parties increment together, exit together when state >= max

    Example (disable verification for trusted fixed iterations):
      %init = ... : !mpir.mp<tensor<i32>, 3>
      %result = mpir.uniform_while (%i = %init) {verify_uniform = false}
                : (!mpir.mp<tensor<i32>, 3>) -> !mpir.mp<tensor<i32>, 3> {
        ^cond(%state: !mpir.mp<tensor<i32>, 3>):
          // Trusted: know condition is always uniform
          %cmp = ... : !mpir.mp<i1, 3>
          mpir.condition %cmp : !mpir.mp<i1, 3>
      } do {
        ^body(%state: !mpir.mp<tensor<i32>, 3>):
          %next = ... : !mpir.mp<tensor<i32>, 3>
          mpir.yield %next : !mpir.mp<tensor<i32>, 3>
      }
  }];

  let arguments = (ins
    Variadic<AnyType>:$initArgs,
    DefaultValuedAttr<BoolAttr, "true">:$verify_uniform
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$before, SizedRegion<1>:$after);

  let assemblyFormat = [{
    `(` $initArgs `)` attr-dict `:` functional-type($initArgs, $results)
    $before `do` $after
  }];

  let hasVerifier = 1;
}
