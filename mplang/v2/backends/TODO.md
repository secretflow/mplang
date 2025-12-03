# Backend Values TODO

## Value Wrappers for Dialect Data

Design principle: Keep `edsl/serde.py` simple by wrapping dialect-specific data types in `Value`
subclasses within each `*_impl.py`. This provides:

- Clear separation of concerns (serde stays generic)
- Extensibility for future data formats
- Consistent pattern across all backends

### TableValue (table_impl.py)

- [x] **Create `TableValue` wrapper class**: Wrap `pa.Table` in a `Value` subclass with proper serde.
  - Use Arrow IPC format for serialization (efficient, preserves schema)
  - Add `_wrap(pa.Table | pd.DataFrame) -> TableValue` helper
  - Add `_unwrap(TableValue) -> pa.Table` helper
  - Future: extend to support other table backends (Polars, DuckDB relations, etc.)

- [x] **Update table_impl.py def_impl functions**: Use `_wrap`/`_unwrap` helpers at entry/exit
  points of each implementation function.

### TensorValue (tensor_impl.py)

- [x] **Create `TensorValue` wrapper class**: Wrap numpy and numpy-like arrays in a `Value` subclass.
  - Handle `np.ndarray` directly
  - Handle JAX arrays via `np.asarray()` conversion
  - Handle other numpy-like objects (e.g., `jnp.ndarray`, cupy arrays) with duck typing
  - Use base64-encoded bytes for serialization (consistent with current serde)

- [x] **Update tensor_impl.py def_impl functions**: Use `_wrap`/`_unwrap` pattern.

- [x] **Remove numpy special handling from serde.py**: ~After `TensorValue` is implemented, remove the special `np.ndarray` handling~. **Reverted**: We kept explicit `np.ndarray` handling in `serde.py` to support direct serialization of inputs (like JAX arrays) without forcing manual wrapping in `TensorValue` everywhere.

## Refactoring (WrapValue Pattern)

- [x] **Create `WrapValue[T]` base class**: Generic base class in `value.py` implementing the `_convert`/`wrap`/`unwrap` pattern.
- [x] **Refactor `TensorValue`**: Inherit from `WrapValue[np.ndarray]`.
- [x] **Refactor `TableValue`**: Inherit from `WrapValue[pa.Table]`.
- [x] **Refactor `SPUShare`**: Inherit from `WrapValue[libspu.Share]`.
- [x] **Refactor `ECPoint`**: Inherit from `WrapValue[bytes]`.
- [x] **Refactor `BFVPublicContext`/`BFVSecretContext`**: Inherit from `WrapValue[ts.Context]`.
- [x] **Create `BytesValue`**: Inherit from `WrapValue[bytes]` for crypto byte data (hashes, keys, ciphertexts).

## Stability & Fixes

- [x] **Fix SimpSimulator Hangs**: Implemented fail-fast logic in `SimpSimulator` to cancel pending futures and shutdown communicators immediately upon any worker exception.
- [x] **Fix Serde Recursion**: Fixed `RecursionError` in `serde.py` when serializing JAX arrays by adding explicit `np.ndarray` support.

## Naming Conventions

- [ ] **Unify Value subclass naming style**: Current naming is inconsistent:
  - `RuntimePrivateKey`, `RuntimePublicKey`, `RuntimeSymmetricKey` (Runtime prefix)
  - `ECPoint`, `SPUShare`, `MockQuote` (No prefix)
  - `BFVPublicContext`, `BFVSecretContext`, `BFVValue` (BFV prefix)
  - Consider standardizing on a consistent pattern.

## Type Annotations

- [x] **Strict type checking in crypto_impl.py**: Refactored all `def_impl` functions to use strict
  `isinstance` checks instead of `hasattr` duck typing. Supported types are explicitly listed:
  - `sym_encrypt/decrypt`: Accept `RuntimeSymmetricKey | BytesValue` for keys
  - `scalar_from_int`: Accept `TensorValue | int | bool | np.integer | np.bool_`
  - `mul_impl`: Accept `int | TensorValue | np.integer` for scalar
  - `select_impl`: Accept `TensorValue | int` for condition
  - Created `BytesValue(WrapValue[bytes])` for hash/key/ciphertext data

- [x] **Strict type checking in bfv_impl.py**: Replaced `hasattr(data, "data")` checks with
  proper `isinstance(data, TensorValue)` checks in `encode_impl` and `encrypt_impl`.

- [x] **Complete type annotations in remaining impl files**: Improved `def_impl` function
  signatures in:
  - `phe_impl.py` - Added strict types for PHEContext, PHEEncoder, WrappedCiphertext
  - `table_impl.py` - Added strict types for TableValue and table-like inputs
  - `spu_impl.py` - Added strict types for SPUShare, SPUConfig, and share handling

- [x] **Enforce Value-only parameters in def_impl**: All `def_impl` functions now only accept
  `Value` subclasses as parameters (not raw `np.ndarray`, `int`, etc.). This ensures type system
  consistency between abstract_eval and impl:
  - `bfv_impl.py`: `encode_impl`, `encrypt_impl`, `add/sub/mul_impl` now use `TensorValue`
  - `spu_impl.py`: `makeshares_impl` accepts `TensorValue`, `reconstruct_impl` returns `TensorValue`
  - `table_impl.py`: `tensor2table_impl` accepts `TensorValue`

## Performance

- [x] **Benchmark serde performance**: Identified major performance regression (75s vs 12s) caused
  by `use_serde=True` doing eager serialization in `SimpSimulator._run_party`. Fixed by implementing
  **lazy serde**: moved serialization to `ThreadCommunicator.send()` to only serialize data during
  actual inter-party communication, matching HTTP worker behavior. Test time: 75s → 14s.

## Completed

- [x] Create `Value` base class (`value.py`)
- [x] Migrate BFV types to Value (`BFVPublicContext`, `BFVSecretContext`, `BFVValue`)
- [x] Migrate Crypto RuntimeKey types to Value (`RuntimePrivateKey`, `RuntimePublicKey`,
  `RuntimeSymmetricKey`)
- [x] Migrate TEE MockQuote to Value
- [x] Create `ECPoint` wrapper (no monkey-patching for `coincurve.PublicKey`)
- [x] Create `SPUShare` wrapper (no monkey-patching for `libspu.Share`)
- [x] Refactor `SPUShare` to directly hold `libspu.Share` in memory (zero-copy)
- [x] Delete `backends/serde.py` - replaced by `@serde.register_class` pattern
- [x] Fix type annotations in `crypto_impl.py`
- [x] Remove `libspu.Visibility` from IR attrs - use string-based `Visibility` type instead
- [x] Move `SPUConfig.to_runtime_config()` to `spu_impl.py` (runtime-only)
- [x] Remove Enum serde code and `import_module` security risk from `serde.py`
- [x] Create `BytesValue` wrapper for cryptographic byte data (hash outputs, keys, ciphertexts)
- [x] Refactor `crypto_impl.py` to use strict `isinstance` type checks (no `hasattr` duck typing)
- [x] Refactor `bfv_impl.py` to use `isinstance(data, TensorValue)` checks
- [x] Remove invalid test `TestKEMWithRawTensorKey` that bypassed proper Value wrapping
