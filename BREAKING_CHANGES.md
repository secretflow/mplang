# Breaking Changes: MPLang v1 → MPLang2

> **Status**: MPLang2 is available alongside v1. Both versions are maintained until v2 reaches production stability (target: Q1 2026).

## Summary

MPLang2 is a **major architectural rewrite** introducing:

- **Dialect extensibility**: All operations (tensor, table, crypto, BFV) are first-class dialects
- **Type system**: Explicit `ScalarType`, `SSType`, `VectorType`, `CustomType` support
- **Cleaner API**: Explicit frontends, simplified device placement

**Migration impact**: Low for existing v1 users (v1 code continues to work). New projects should prefer v2.

---

## 1. Import Path Changes

### ✅ Recommended Migration Path

```python
# Before (MPLang v1)
import mplang as mp

# After: Explicit version import
import mplang.v1 as mp  # Continue using v1
# OR
import mplang.v2 as mp  # Migrate to v2
```

### ⚠️ Compatibility

- **v1 still works**: `import mplang` defaults to v1 for backward compatibility
- **Explicit is better**: Use `mplang.v1` or `mplang.v2` to avoid ambiguity
- **Future**: Default may change to v2 in a future release (with deprecation warning)

---

## 2. Device API Changes

### 2.1 PPU JAX Execution (Explicit Frontend)

```python
# v1: Implicit JAX tracing
result = mp.device("P0")(lambda a, b: jnp.add(a, b))(x, y)

# v2: Explicit frontend for PPU
result = mp.device("P0", "jax")(lambda a, b: jnp.add(a, b))(x, y)
#                        ^^^^^ Required for JAX on PPU

# SPU always uses JAX natively (no frontend needed)
result = mp.device("SP0")(lambda a, b: jnp.add(a, b))(x, y)  # Same in v1/v2
```

**Why?** v2 supports multiple frontends (JAX, SQL, custom). Explicit is clearer.

### 2.2 Constant Placement

```python
# v1: Using device + lambda
x = mp.device("P0")(lambda: 42)()

# v2: Direct placement with put()
x = mp.put("P0", 42)  # Cleaner and more explicit
```

**Migration tool**: Replace `mp.device(dev)(lambda: val)()` → `mp.put(dev, val)`

---

## 3. SQL/Table API Changes

### 3.1 Import and Execution

```python
# v1
from mplang.ops import sql_cc
from mplang.core.dtypes import INT64

result = mp.device("P0")(sql_cc.run_sql)(
    "SELECT * FROM tbl",
    input_table=tbl
)

# v2
from mplang2.dialects import table
from mplang2.edsl.typing import TableType, i64

schema = TableType({"id": i64, "value": i64})
result = table.run_sql(
    "SELECT * FROM tbl",
    out_type=schema,  # Explicit output schema required
    tbl=tbl
)
```

**Key differences**:

- v2 requires explicit `out_type` parameter (no auto-inference)
- Schema uses scalar types directly (`i64` not `TensorType([...], i64)`)
- Cleaner namespace: `table.read/write/run_sql`

---

## 4. Type System Changes

### 4.1 DType Definitions

```python
# v1
from mplang.core import dtypes
t = dtypes.TensorType([3], dtypes.INT64)

# v2
from mplang2.edsl import typing as elt
t = elt.TensorType(elt.i64, (3,))  # shape as tuple, not list
```

### 4.2 Table Schema

```python
# v1
from mplang.core.dtypes import INT64, FLOAT64
schema = mp.TableType.from_dict({"id": INT64, "value": FLOAT64})

# v2
from mplang2.edsl.typing import TableType, i64, f64
schema = TableType({"id": i64, "value": f64})  # Direct construction
```

---

## 5. Compilation/JIT Changes

### 5.1 Function Decorator

```python
# v1 & v2: Same API
@mp.function
def my_func(x, y):
    return x + y

# v1 & v2: Same compilation
traced = mp.compile(simulator, my_func)
```

**No breaking changes** in basic compilation API.

### 5.2 IR Inspection

```python
# v1
ir_text = traced.dump_ir()

# v2
ir_text = traced.compiler_ir()  # More explicit name
ir_verbose = traced.compiler_ir(verbose=True)  # With type annotations
```

---

## 6. Tutorial Mapping

| v1 Tutorial | v2 Equivalent | Notes |
|-------------|---------------|-------|
| `tutorials/v1/device/00_device_basics.py` | `tutorials/v2/00_device_basics.py` | ✅ Full parity |
| `tutorials/v1/device/01_function_decorator.py` | `tutorials/v2/01_function_decorator.py` | ✅ Full parity |
| `tutorials/v1/device/02_simulation_and_driver.py` | `tutorials/v2/02_simulation_and_driver.py` | ✅ + CLI docs |
| `tutorials/v1/device/03_run_jax.py` | `tutorials/v2/03_run_jax.py` | ⚠️ Explicit frontend |
| `tutorials/v1/device/04_run_sql.py` | `tutorials/v2/05_run_sql.py` | ⚠️ out_type required |
| `tutorials/v1/device/05_pipeline.py` | `tutorials/v2/06_pipeline.py` | ✅ Table I/O works |
| `tutorials/v1/device/06_ir_dump_and_analysis.py` | `tutorials/v2/04_ir_dump_and_analysis.py` | ⚠️ No diagrams yet |

---

## 7. New Dependencies

v2 introduces additional dependencies for cryptographic operations:

```toml
# pyproject.toml additions
"cryptography>=43.0.0",  # For AES-GCM, X25519 (KEM)
"coincurve>=20.0.0",     # For secp256k1 ECC operations
```

**Purpose**: Support `mplang2.dialects.crypto` (Key Encapsulation Mechanism, Digital Envelope).

**Impact**: ~5MB additional install size. Both are Apache 2.0 / BSD licensed.

**Optional**: If you don't use crypto dialect, these are runtime-optional (import-time dependency).

---

## 8. Migration Timeline

### Phase 1: Coexistence (Current - Q1 2026)

- ✅ v1 and v2 both available
- ✅ `import mplang` defaults to v1
- ✅ All v1 tests passing
- ✅ v2 feature-complete (~95% parity)

### Phase 2: v2 Default (Q1 2026 - Q3 2026)

- `import mplang` will show deprecation warning
- Recommend explicit `mplang.v1` or `mplang.v2`
- v1 still fully functional

### Phase 3: v1 Maintenance (Q3 2026+)

- v1 moves to maintenance mode (bug fixes only)
- v2 becomes default import
- v1 remains available for legacy code

**No forced breaking**: v1 will remain importable indefinitely for backward compatibility.

---

## 9. Automated Migration Script

```bash
# Replace import statements (regex-based)
find . -name "*.py" -type f -exec sed -i \
  's/^import mplang as mp$/import mplang.v2 as mp/' {} +

# Or use AST-based tool (coming soon)
# python scripts/migrate_to_v2.py --path src/
```

---

## 10. Decision Guide: Should I Migrate?

### ✅ Migrate to v2 if

- Starting a new project
- Need crypto/BFV/custom dialects
- Want better type safety
- Need extensible dialect system

### ⏸️ Stay on v1 if

- Existing production code
- No immediate need for new features
- Prefer stability over new features
- Limited migration resources

### ⚠️ Must migrate eventually if

- Using deprecated APIs (will be removed in v3)
- Need new features (only in v2)
- Want long-term support (v1 → maintenance mode Q3 2026)

---

## 11. Getting Help

- **Migration guide**: [tutorials/MIGRATION.md](tutorials/MIGRATION.md)
- **Examples**: [examples/v2/](examples/v2/)
- **Tutorials**: [tutorials/v2/](tutorials/v2/)
- **Issues**: <https://github.com/secretflow/mplang/issues>
- **Discussions**: <https://github.com/secretflow/mplang/discussions>

---

## Appendix: API Quick Reference

| Feature | v1 API | v2 API |
|---------|--------|--------|
| Import | `import mplang as mp` | `import mplang.v2 as mp` |
| PPU JAX | `device("P0")(fn)(...)` | `device("P0", "jax")(fn)(...)` |
| Constants | `device("P0")(lambda: 42)()` | `put("P0", 42)` |
| SQL | `sql_cc.run_sql(q, tbl=t)` | `table.run_sql(q, out_type=s, tbl=t)` |
| IR dump | `traced.dump_ir()` | `traced.compiler_ir()` |
| Types | `dtypes.INT64` | `elt.i64` |
