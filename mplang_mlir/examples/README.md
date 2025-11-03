# Mpir Dialect Examples

This directory contains example programs demonstrating the Mpir (Multi-Party Intermediate Representation) dialect for secure multi-party computation.

## Overview

The Mpir dialect is an MLIR-based IR designed for expressing multi-party computation programs. It provides:

- **Type system** for tracking data ownership across parties
- **Operations** for secure computation, data conversion, and control flow
- **Verification** to ensure correctness of multi-party programs

## Examples

### 1. `phe_example.mlir` - Homomorphic Encryption Workflow

Demonstrates a complete Paillier encryption workflow:
- Key generation on a single party
- Encryption of private data from multiple parties
- Homomorphic addition on encoded data
- Decryption of results

**Key concepts:**
- `peval` operation with `rmask` to control execution parties
- `conv` operation to share public keys and combine encoded values
- `MP<Encoded<T, schema>, pmask>` type for encoded data

**Run:**
```bash
mpir-opt phe_example.mlir
```

### 2. `mpc_example.mlir` - Secure Multi-Party Computation

Shows various MPC patterns:
- Secure sum of private values
- Secure comparison with conditional output
- Secure statistics (mean computation)
- Iterative secure gradient descent

**Key concepts:**
- Secret sharing via `conv` operation
- `pmask` semantics for party participation
- `uniform_cond` for conditional execution
- `uniform_while` for iterative computation
- Security guarantees of MPC protocols

**Run:**
```bash
mpir-opt mpc_example.mlir
```

### 3. `control_flow_patterns.mlir` - Control Flow Best Practices

Comprehensive guide to uniform control flow:
- Basic and nested conditionals
- While loops with iteration counters
- Convergence-based loops
- `verify_uniform` attribute usage
- Type consistency requirements
- Common pitfalls and how to avoid them

**Key concepts:**
- `uniform_cond` requirements and constraints
- `uniform_while` patterns
- Runtime verification with `verify_uniform`
- Performance and security considerations

**Run:**
```bash
mpir-opt control_flow_patterns.mlir
```

## Core Concepts

### Type System

The Mpir type system tracks data ownership and encoding:

```mlir
// Basic types
!mpir.mp<tensor<10xf32>, 7>                    // Plaintext tensor on parties 0,1,2
!mpir.table<["id", "age"], [i64, i32]>         // Table with named columns

// Encoded values (encryption, secret sharing, serialization)
!mpir.enc<tensor<10xf32>, "paillier">          // PHE-encrypted tensor
!mpir.enc<!mpir.table<["id"], [i64]>, "parquet">  // Parquet-encoded table

// Multi-party encoded values
!mpir.mp<!mpir.enc<tensor<10xf32>, "paillier">, 3>  // Parties 0,1 have encrypted data
!mpir.mp<!mpir.enc<!mpir.table<["data"], [f32]>, "csv">, 1>  // CSV-encoded table on party 0

// Dynamic party mask (runtime-determined)
!mpir.mp_dynamic<tensor<10xf32>>   // Party ownership determined at runtime
```

**TableType** (`!mpir.table<...>`): Represents structured data (dataframe/SQL table) with named columns.
- Used for database-style operations (SQL kernels, PSI, join, aggregation)
- Can be encoded for transmission/storage (e.g., `"parquet"`, `"csv"`, `"aes-gcm"`)
- Note: Computation typically doesn't work on encoded tables directly, but encoding is useful for I/O

### pmask (Party Mask)

The `pmask` is a bitmask indicating which parties have the data:
- `pmask=1` (binary: 001) → Party 0 only
- `pmask=2` (binary: 010) → Party 1 only
- `pmask=3` (binary: 011) → Parties 0 and 1
- `pmask=4` (binary: 100) → Party 2 only
- `pmask=7` (binary: 111) → All parties (0, 1, 2)

### Key Operations

#### 1. `peval` - Partial Evaluation

Executes a function on specific parties:

```mlir
%result = mpir.peval @my_func(%arg) {rmask = 3 : i64}
          : (!mpir.mp<tensor<10xf32>, 3>) -> !mpir.mp<tensor<10xf32>, 3>
```

- `rmask`: Execution mask (which parties run the computation)
- Must be subset of input `pmask` union
- For operations without MP inputs (e.g., keygen), `rmask` can be freely specified

#### 2. `conv` - Data Conversion

Combines data from different parties into secret shares:

```mlir
%shares = mpir.conv(%x0, %x1)
          : (!mpir.mp<tensor<10xf32>, 1>, !mpir.mp<tensor<10xf32>, 2>)
          -> !mpir.mp<tensor<10xf32>, 3>
```

- Input `pmask`s must be disjoint (non-overlapping)
- Output `pmask` = union of input `pmask`s
- Creates secret shares visible to result parties

#### 3. `uniform_cond` - Uniform Conditional

Conditional execution with uniform control flow:

```mlir
%result = mpir.uniform_cond %cond {verify_uniform = true}
          : !mpir.mp<i1, 7> -> !mpir.mp<tensor<10xf32>, 7> {
    mpir.return %then_value : !mpir.mp<tensor<10xf32>, 7>
  } {
    mpir.return %else_value : !mpir.mp<tensor<10xf32>, 7>
  }
```

- Condition must be scalar `MP<i1, pmask>`
- All parties must have same boolean value
- Both branches must return same types
- `verify_uniform=true` enables runtime verification

#### 4. `uniform_while` - Uniform While Loop

Iterative computation with uniform loop condition:

```mlir
%result = mpir.uniform_while (%init) : (!mpir.mp<tensor<10xf32>, 7>)
                                    -> (!mpir.mp<tensor<10xf32>, 7>) {
^bb0(%x: !mpir.mp<tensor<10xf32>, 7>):
  %cond = mpir.peval @check_condition(%x) {rmask = 7 : i64}
          : (!mpir.mp<tensor<10xf32>, 7>) -> !mpir.mp<i1, 7>
  mpir.condition %cond : !mpir.mp<i1, 7>
} do {
^bb0(%x: !mpir.mp<tensor<10xf32>, 7>):
  %x_next = mpir.peval @compute_next(%x) {rmask = 7 : i64}
            : (!mpir.mp<tensor<10xf32>, 7>) -> !mpir.mp<tensor<10xf32>, 7>
  mpir.yield %x_next : !mpir.mp<tensor<10xf32>, 7>
}
```

- Condition region returns `MP<i1, pmask>`
- Body region yields same types as init
- Loop condition must be uniform across parties

## Verification

The Mpir dialect includes comprehensive static verification:

### Type Checking
- `pmask` disjointness in `conv` operations
- `rmask` subset validation in `peval` operations
- Type consistency in control flow branches

### Control Flow Verification
- Scalar boolean conditions for `uniform_cond`/`uniform_while`
- Branch type matching in conditionals
- Yield type matching in loops
- Proper terminator presence

### Run Tests
```bash
cd mplang_mlir
./run_tests.sh
```

All test files are in `test/Dialect/Mpir/`:
- `type_parsing.mlir` - Type system tests
- `uniform_cond.mlir` - Conditional operation tests
- `uniform_cond_errors.mlir` - Conditional error detection
- `uniform_while_errors.mlir` - Loop error detection
- `conv_errors.mlir` - Conv operation validation
- `peval_errors.mlir` - rmask validation

## Building and Testing

### Prerequisites
- MLIR 18.1+
- CMake 3.20+
- Ninja (recommended)

### Build
```bash
cd mplang_mlir
mkdir -p built
cd built
cmake -G Ninja \
  -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  ..
ninja mpir-opt
```

### Run Examples
```bash
cd mplang_mlir
./built/tools/mpir-opt/mpir-opt examples/phe_example.mlir
./built/tools/mpir-opt/mpir-opt examples/mpc_example.mlir
./built/tools/mpir-opt/mpir-opt examples/control_flow_patterns.mlir
```

### Run Tests
```bash
cd mplang_mlir
./run_tests.sh              # Run all tests
./run_tests.sh type_parsing.mlir  # Run specific test
```

## Further Reading

- **Architecture**: `design/architecture_v2.md` - Overall system design
- **Control Flow**: `design/control_flow.md` - Detailed control flow semantics
- **Type System**: `include/mplang/Dialect/Mpir/MpirTypes.td` - Type definitions
- **Operations**: `include/mplang/Dialect/Mpir/MpirOps.td` - Operation definitions
- **Verifiers**: `lib/Dialect/Mpir/MpirOps.cpp` - Verification implementation

## Python Integration

The Mpir dialect is designed to work with the MPLang Python framework. Python-level code is traced into Mpir IR, which can then be:

1. **Optimized** - Apply IR-level transformations
2. **Verified** - Check correctness at compile time
3. **Executed** - Interpret or lower to target backends (SPU, FHE, etc.)

For Python integration examples, see the main MPLang repository `tutorials/` directory.

## Support

For questions or issues:
- Open an issue in the MPLang repository
- Refer to inline comments in example files
- Check test files for additional usage patterns
