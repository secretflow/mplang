# Backend Values TODO

## Serde Support

- [ ] **Add PyArrow Table serde support**: `pa.Table` is used by `table_impl.py` but has no serde
  support in `mplang/v2/edsl/serde.py`. Consider using Arrow IPC format for serialization.

- [ ] **Review numpy special handling in serde**: Check if the special handling for `np.ndarray` in
  `serde.py` (lines ~217-248) can be removed now that we have `@serde.register_class` pattern.
  May be able to unify with Value pattern.

## Naming Conventions

- [ ] **Unify Value subclass naming style**: Current naming is inconsistent:
  - `RuntimePrivateKey`, `RuntimePublicKey`, `RuntimeSymmetricKey` (Runtime prefix)
  - `ECPoint`, `SPUShare`, `MockQuote` (No prefix)
  - `BFVPublicContext`, `BFVSecretContext`, `BFVValue` (BFV prefix)
  - Consider standardizing on a consistent pattern.

## Type Annotations

- [ ] **Complete type annotations in remaining impl files**: Continue improving `def_impl` function
  signatures in:
  - `bfv_impl.py` - Some `Any` types remain for BFV values
  - `phe_impl.py` - Some `Any` types remain for PHE values
  - `table_impl.py` - Table/DataFrame types could be more specific
  - `tensor_impl.py` - Return types could use `np.ndarray`

## Performance

- [ ] **Benchmark serde performance**: After all serde refactoring is complete, compare v2 test
  suite performance with the previous commit to measure any serialization overhead introduced
  by the new `@serde.register_class` pattern. Run `uv run pytest tests/v2/ --durations=0`.

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
