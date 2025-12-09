# MPLang v1 → MPLang2 Migration Guide

> Last updated: 2025-12-01 (File I/O + Pipeline + Distributed CLI verified)

## Overview

MPLang2 is a type-safe EDSL replacement for MPLang v1, supporting nested DSLs (e.g., MP + BFV).
Target: Replace `tutorials/device/*` with `tutorials/mplang2/*`.

**Goal is NOT 100% API compatibility**, but functional parity with cleaner design.

## Completion Status: ~95%

### ✅ Fully Implemented

| Category | APIs |
|----------|------|
| Device API | `device()`, `put()`, `get_dev_attr()`, `set_dev_attr()` |
| Runtime | `Simulator`, `Driver`, `evaluate()`, `fetch()`, `ClusterSpec` |
| CLI | `python -m mplang2.backends.cli up` (start workers) |
| Compilation | `@function`, `jit`, `compile()`, `trace()`, `TracedFunction.compiler_ir()` |
| Type System | `TensorType`, `ScalarType`, `TableType`, `MPType`, `SSType`, `VectorType` |
| Dialects | `simp`, `tensor`, `table`, `spu`, `tee`, `crypto`, `bfv`, `phe` |
| Table I/O | `table.read()`, `table.write()` (CSV, Parquet, JSON, Feather) |
| Type Conversion | `dtypes.to_jax()`, `dtypes.to_numpy()`, `dtypes.from_arrow()`, `dtypes.from_pandas()` |

### ❌ Missing

| Feature | Used In | Notes |
|---------|---------|-------|
| `mp.analysis.dump()` | 06_ir_dump_and_analysis.py | Mermaid diagrams, reports |

## Key API Differences

### Import

```python
# v1
import mplang as mp

# v2
import mplang2 as mp
```

### Device Execution with JAX

```python
# v1: implicit JAX tracing
mp.device("P0")(lambda a, b: a + b)(x, y)

# v2: explicit frontend via .jax property for PPU
mp.device("P0").jax(lambda a, b: a + b)(x, y)
# SPU always uses JAX natively, no frontend needed
mp.device("SP0")(lambda a, b: a + b)(x, y)
```

### Constants / Data Placement

```python
# v1
x = mp.device("P0")(lambda: 42)()

# v2
x = mp.put("P0", 42)
```

### SQL

```python
# v1
from mplang.ops import sql_cc
result = mp.device("P0")(sql_cc.run_sql)(query, input_table=tbl)

# v2
from mplang2.dialects import table
result = table.run_sql(query, out_type=schema, tbl=tbl)
```

## Tutorial Mapping

| v1 Tutorial | v2 Tutorial | Status |
|-------------|-------------|--------|
| 00_device_basics.py | ✅ 00_device_basics.py | Done |
| 01_function_decorator.py | ✅ 01_function_decorator.py | Done |
| 02_simulation_and_driver.py | ✅ 02_simulation_and_driver.py | Done (includes CLI + live driver) |
| 03_run_jax.py | ✅ 03_run_jax.py | Done |
| 04_run_sql.py | ✅ 05_run_sql.py | Done (PPU only, TEE needs work) |
| 05_pipeline.py | ✅ 06_pipeline.py | Done |
| 06_ir_dump_and_analysis.py | ✅ 04_ir_dump_and_analysis.py | Partial (no analysis module) |

## TODO

1. ~~Add `TableType.from_dict()` helper method~~ ❌ Not needed (use `TableType()` directly)
2. Port `mplang/analysis/diagram.py` → `mplang2/analysis/`
3. ~~Implement `table.read()` / `table.write()` for file I/O~~ ✅ Done
4. ~~Port remaining tutorials (04_run_sql, 05_pipeline)~~ ✅ Done
5. Fix TEE table operations in simulator (low priority)

---

## DType System Comparison

### Architecture Difference

| Aspect | mplang v1 | mplang2 |
|--------|-----------|---------|
| Design | Single `DType` dataclass | Type class hierarchy |
| Core | `DType(name, bitwidth, is_signed, ...)` | `ScalarType` → `IntegerType` / `FloatType` / `ComplexType` |
| Type Safety | Runtime checks | Compile-time type distinction |

### v1 DType (`mplang/core/dtypes.py`)

```python
@dataclass(frozen=True)
class DType:
    name: str
    bitwidth: int
    is_signed: bool | None
    is_floating: bool
    is_complex: bool
    is_table_only: bool  # STRING, DATE, etc.
```

### v2 Types (`mplang2/edsl/typing.py`)

```python
class ScalarType(BaseType): ...
class IntegerType(ScalarType):  # i8, i16, i32, i64, u8...
    bitwidth: int
    signed: bool
class FloatType(ScalarType):    # f16, f32, f64
    bitwidth: int
class ComplexType(ScalarType):  # c64, c128
    inner_type: FloatType
```

### Feature Gap

| Feature | v1 | v2 | Status |
|---------|----|----|--------|
| `from_numpy()` / `to_numpy()` | ✅ | ✅ `dtypes.py` | ✅ Fixed |
| `from_jax()` / `to_jax()` | ✅ | ✅ `dtypes.py` | ✅ Fixed |
| `from_pandas_dtype()` | ✅ | ❌ | Medium |
| `from_arrow_dtype()` | ✅ | ❌ | Medium |
| `from_any()` (universal) | ✅ | ✅ `dtypes.from_dtype()` | ✅ Fixed |
| Table-only types (STRING, DATE, etc.) | ✅ | ✅ `typing.py` | ✅ Fixed |
| Large integers (i128, i256) | ❌ | ✅ | - |
| Bool type constant | ✅ `BOOL` | ✅ `bool_`, `i1` | ✅ Fixed |

### Completed Improvements (2025-11-30)

**P0: Added `mplang2/dialects/dtypes.py`** (renamed from type_utils.py) with clean API:

```python
from mplang2.dialects import dtypes

dtypes.to_jax(scalar_types.f32)      # ScalarType → jax dtype
dtypes.to_numpy(scalar_types.i64)    # ScalarType → numpy dtype
dtypes.from_dtype(np.float32)        # any dtype → ScalarType
dtypes.from_dtype(jnp.int64)         # works with JAX too
dtypes.from_dtype("float32")         # works with strings too
```

**P3: Added `bool_` and Table-only types to `mplang2/edsl/typing.py`**:

```python
# Boolean
bool_ = IntegerType(bitwidth=1, signed=True)
i1 = bool_  # alias

# Table-only types (for SQL/TableType)
STRING = CustomType("string")
DATE = CustomType("date")
TIME = CustomType("time")
TIMESTAMP = CustomType("timestamp")
DECIMAL = CustomType("decimal")
BINARY = CustomType("binary")
JSON = CustomType("json")
UUID = CustomType("uuid")
INTERVAL = CustomType("interval")
```

### Remaining TODO

1. ~~Add `TableType.from_dict()` helper method~~ ❌ Not needed (use `TableType()` directly)
2. Port `mplang/analysis/diagram.py` → `mplang2/analysis/`
3. ~~Implement `table.read()` / `table.write()` for file I/O~~ ✅ Done
4. ~~Port 04_run_sql tutorial~~ ✅ Done (as 05_run_sql.py)
5. ~~Port remaining tutorial (05_pipeline)~~ ✅ Done (as 06_pipeline.py)
6. Fix TEE table operations in simulator (low priority)
7. ~~Add `from_pandas_dtype()` / `from_arrow_dtype()` to dtypes.py~~ ✅ Done
