# PR #260: Init V2 - MPLang Architecture Rewrite

## 📋 Summary

This PR introduces **MPLang2**, a complete architectural rewrite of the MPLang framework focusing on:

1. **Dialect Extensibility**: All operations (tensor, table, crypto, BFV, PHE) are first-class dialects that can be traced and extended independently
2. **Enhanced Type System**: Explicit support for `ScalarType`, `SSType`, `VectorType`, `CustomType` beyond v1's `TensorType`/`TableType`
3. **Cleaner API Design**: Explicit frontends, simplified device placement, consistent naming

**Status**: ~95% feature parity with v1. Both versions coexist peacefully.

---

## 🎯 Motivation

### Problems with v1 Architecture

1. **Limited Dialect Extensibility**
   - Only `simp` dialect is first-class; others (tensor, table, crypto) must embed in `@primitive`
   - Cannot trace or extend non-simp operations independently
   - Hard to add new computational backends

2. **Type System Limitations**
   - Only `TensorType` and `TableType` natively supported
   - Other types (ciphertexts, keys, custom types) simulated via Tensor, losing compile-time checks
   - No way to represent secret-shared types (`SSType`) explicitly

3. **API Inconsistencies**
   - Implicit behaviors (e.g., auto JAX tracing on PPU)
   - Mixed paradigms (device() for both placement and execution)
   - Difficult to compose multiple computational paradigms

### v2 Solutions

- **All dialects equal**: `simp`, `tensor`, `table`, `spu`, `crypto`, `bfv`, `phe` all first-class
- **Extensible types**: `BaseType` → `ScalarType | TensorType | TableType | SSType | VectorType | CustomType`
- **Explicit frontends**: `device("P0").jax` makes JAX execution explicit
- **Simplified placement**: `put()` for constants, cleaner than `device()(lambda: val)()`

---

## 📦 Major Changes

### 1. Code Structure Reorganization

```
mplang/
├── v1/                    # Original codebase (preserved)
│   ├── core/              # Primitives, tracer, interpreter
│   ├── ops/               # Basic, JAX, SQL ops
│   ├── kernels/           # Backend implementations
│   ├── runtime/           # Simulator, Driver
│   └── simp/              # SIMP dialect
└── v2/                    # New architecture
    ├── edsl/              # typing.py, graph.py, primitive.py, tracer.py, interpreter.py, jit.py
    ├── dialects/          # simp, tensor, table, spu, tee, crypto, bfv, phe, dtypes
    ├── backends/          # simp_simulator, simp_http_driver, crypto_impl, bfv_impl, etc.
    └── libs/              # device.py (API), mpc/ (aggregation, groupby, permutation)
```

**Backward Compatibility**: v1 code untouched, continues to work.

### 2. New Dialects

| Dialect | Purpose | Status |
|---------|---------|--------|
| `crypto` | KEM, ECDH, AES-GCM encryption | ✅ Complete (23 tests) |
| `bfv` | Homomorphic encryption (SEAL) | ✅ Complete (8 tests) |
| `phe` | Paillier-like encryption | ✅ Complete |
| `tensor` | JAX operations (v1-compatible) | ✅ Complete |
| `table` | SQL/table operations (v1-compatible) | ✅ Complete |
| `simp` | Low-level party communication | ✅ Complete |
| `spu` | Secure multi-party computation | ✅ Complete |
| `tee` | Trusted execution environment | ✅ Complete |

### 3. Examples and Tutorials

**New Examples** (`examples/v2/`):

- `sgb.py` - SecureBoost with optimized BFV (SIMD batching)
- `bfv_sort_agg.py` - Homomorphic sort & aggregate
- `phe_sort_agg.py` - Paillier-based aggregation
- `pcall_jax.py` - Multi-party JAX computation

**Migrated Tutorials** (`tutorials/v2/`):

- `00_device_basics.py` - Device placement, masks, auto-device
- `01_function_decorator.py` - Compilation, auditability, performance
- `02_simulation_and_driver.py` - Local vs distributed execution
- `03_run_jax.py` - JAX on PPU/SPU
- `04_ir_dump_and_analysis.py` - Graph inspection
- `05_run_sql.py` - SQL on PPU/TEE
- `06_pipeline.py` - Hybrid JAX + Table I/O

All v1 examples updated: `import mplang` → `import mplang.v1 as mp`

### 4. New Dependencies

```toml
"cryptography>=43.0.0",  # Apache 2.0 License
"coincurve>=20.0.0",     # MIT/BSD License
```

**Purpose**: Support `crypto` dialect for:

- ECC operations (secp256k1) via `coincurve`
- AES-GCM, X25519 KEM via `cryptography`

**Usage**: Only loaded when importing `mplang.v2.dialects.crypto` (lazy import).

**License Compatibility**: Both Apache 2.0 / BSD compatible with project's Apache 2.0 license.

---

## 🧪 Testing

### Test Coverage

| Metric | v1 | v2 | Parity |
|--------|----|----|--------|
| Test Count | 748 | 551 | 73.7% |
| Core Tests | ✅ | ✅ | Full |
| Dialect Tests | ✅ | ✅ | Full |
| Integration Tests | ✅ | ⚠️ | Partial |

**v2 Test Breakdown**:

- `backends/` - 102 tests (crypto, BFV, HTTP, simulator)
- `dialects/` - 203 tests (all 8 dialects covered)
- `edsl/` - 89 tests (typing, tracing, interpretation, JIT)
- `libs/` - 157 tests (device API, MPC primitives)

**All Tests Passing**: ✅ `pytest tests/v2/` passes 551/551

### Quality Checks

- ✅ **mypy**: 0 errors (`uv run mypy mplang/`)
- ✅ **ruff**: 0 errors (`uv run ruff check .`)
- ✅ **license-eye**: All 279 files valid (`license-eye header check`)

---

## ⚠️ Breaking Changes

**TL;DR**: None for v1 users. v2 is opt-in via `import mplang.v2`.

See [BREAKING_CHANGES.md](./BREAKING_CHANGES.md) for detailed migration guide.

**Key API Differences**:

1. **Import**: `import mplang.v2 as mp` (explicit version)
2. **Device+JAX**: `device("P0").jax(fn)` (explicit frontend for PPU)
3. **Constants**: `put("P0", 42)` (replaces `device("P0")(lambda: 42)()`)
4. **SQL**: `table.run_sql(q, out_type=schema, tbl=t)` (explicit output type)

**Migration Timeline**:

- **Now - Q1 2026**: v1 and v2 coexist, `import mplang` defaults to v1
- **Q1 2026 - Q3 2026**: Deprecation warnings for `import mplang`
- **Q3 2026+**: v1 maintenance mode (bug fixes only)

---

## 🎯 Review Focus Areas

### 1. Architecture Design

- [ ] Dialect extensibility model (first-class vs embedded)
- [ ] Type system design (`BaseType` hierarchy)
- [ ] Graph IR representation (Op list + SSA)
- [ ] Interpreter/Tracer separation of concerns

### 2. API Ergonomics

- [ ] Device API clarity (`device()`, `put()`, frontends)
- [ ] Table I/O usability (`read()`, `write()`, `run_sql()`)
- [ ] Type inference/checking balance
- [ ] Error messages and debugging experience

### 3. Security & Correctness

- [ ] Crypto primitives usage (coincurve, cryptography)
- [ ] BFV implementation (SEAL bindings)
- [ ] Secret sharing semantics (`SSType`)
- [ ] TEE attestation placeholder (mock in tests)

### 4. Documentation & Examples

- [ ] Tutorial completeness (v1 → v2 parity)
- [ ] Example code quality (sgb.py, bfv_sort_agg.py)
- [ ] Migration guide clarity (BREAKING_CHANGES.md)
- [ ] API reference gaps (docstrings)

### 5. Performance

- [ ] SecureBoost v2 vs v1 benchmarks (design/sgb_v2.md)
- [ ] BFV SIMD optimizations effectiveness
- [ ] HTTP Driver latency overhead
- [ ] Memory usage (multi-CT support)

---

## 📚 Documentation

**New Files**:

- `AGENTS.md` - AI assistant guidance (architecture map, patterns, workflows)
- `BREAKING_CHANGES.md` - Migration guide and timeline
- `design/sgb_v2.md` - SecureBoost v2 design doc with benchmarks
- `tutorials/MIGRATION.md` - Step-by-step v1→v2 migration

**Updated**:

- All v1 examples: imports updated to `import mplang.v1 as mp`
- All v1 tutorials: imports updated to `import mplang.v1 as mp`

---

## 🚀 Future Work (Post-Merge)

### P1 (Next PR)

- [ ] CI/CD: Split v1/v2 test jobs, add tutorial validation
- [ ] Benchmarks: Formalize v1 vs v2 performance comparison suite
- [ ] Integration tests: Cross-device data flow, large-scale table I/O

### P2 (Q1 2026)

- [ ] Analysis module: Port v1's diagram generation to v2
- [ ] Optimization passes: Dead code elimination, constant folding
- [ ] Distributed runtime: Kubernetes deployment support

### P3 (Future)

- [ ] v2 default: Make `import mplang` point to v2 (with warnings)
- [ ] v1 deprecation: Clear EOL timeline
- [ ] MLIR backend: Explore lowering to MLIR for optimization

---

## 📝 Checklist

- [x] All tests passing (551/551)
- [x] Code quality checks pass (mypy, ruff, license-eye)
- [x] Breaking changes documented
- [x] Migration guide provided
- [x] Examples working
- [x] Tutorials complete
- [ ] Performance benchmarks run
- [ ] Security review (crypto usage)
- [ ] Documentation reviewed
- [ ] CI/CD updated

---

## 🤝 Acknowledgments

This rewrite was informed by:

- Production experience with v1 SecureBoost and vertical federated learning
- Community feedback on API ergonomics
- MLIR/LLVM dialect design patterns
- JAX/XLA tracing architecture

Special thanks to early v2 adopters for testing and feedback.

---

**Reviewers**: @core-team @security-team

**Related Issues**: #XXX (dialect extensibility), #YYY (type system)

**Related PRs**: None (first v2 PR)
