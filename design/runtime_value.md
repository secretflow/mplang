# RuntimeValue Base Class Design (Optimized)

## Status: Draft

## Author: @jint

## Date: 2025-12-02

---

## 1. Motivation

### Current Problems

1. **No Type Safety**: `def_impl` functions use `Any` everywhere.

   ```python
   def add_impl(interpreter, op, lhs: Any, rhs: Any) -> BFVValue | np.ndarray
   ```

2. **Scattered Serde**: Serialization logic was spread across multiple files (`dialects/serde.py`, `backends/serde.py`). While we recently cleaned up `dialects/serde.py`, `backends/serde.py` still relies on monkey-patching.
3. **Mixed Return Types**: Impls return raw `np.ndarray`, `BFVValue`, `bytes`, etc. without a common interface.

### Goals

- Unified base class for all runtime values.
- Built-in serialization using the `@serde.register_class` pattern (no monkey-patching).
- Debug/profiling metadata.

---

## 2. Design

### 2.1 RuntimeValue Base Class

```python
# mplang/v2/edsl/runtime.py

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Any
import time

# Use top-level import for serde to avoid circular deps (proven safe in recent refactor)
from mplang.v2.edsl import serde


@dataclass
class RuntimeValue(ABC):
    """Base class for all runtime values in MPLang.

    Provides:
    - Unified serialization interface via `to_json`/`from_json`
    - Debug/profiling metadata via `_created_at`, `_source_op`
    """

    # Class-level type identifier for serde dispatch.
    # Subclasses must define this (or let @serde.register_class handle it).
    _serde_kind: ClassVar[str]

    # Optional metadata for debugging/profiling
    _created_at: float = field(default_factory=time.time, repr=False, compare=False)
    _source_op: str | None = field(default=None, repr=False, compare=False)

    # =========== Serialization Interface ===========

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict.

        Note: Do NOT include `_kind` in the returned dict; `serde.to_json` adds it automatically.
        """
        ...

    @classmethod
    @abstractmethod
    def from_json(cls, data: dict[str, Any]) -> "RuntimeValue":
        """Deserialize from JSON-compatible dict."""
        ...
```

### 2.2 Concrete Implementations

We use the `@serde.register_class` decorator to register types immediately.

```python
# ============= TensorValue =============

@serde.register_class
@dataclass
class TensorValue(RuntimeValue):
    """Wraps numpy/jax arrays as runtime values."""
    _serde_kind: ClassVar[str] = "runtime.TensorValue"

    data: np.ndarray

    def to_json(self) -> dict:
        return {
            "data": base64.b64encode(self.data.tobytes()).decode("ascii"),
            "dtype": str(self.data.dtype),
            "shape": list(self.data.shape),
        }

    @classmethod
    def from_json(cls, data: dict) -> "TensorValue":
        arr = np.frombuffer(
            base64.b64decode(data["data"]), dtype=data["dtype"]
        ).reshape(data["shape"])
        return cls(data=arr)


# ============= BFVCipherValue =============

@serde.register_class
@dataclass
class BFVCipherValue(RuntimeValue):
    """Wraps BFV ciphertext as runtime value."""
    _serde_kind: ClassVar[str] = "runtime.BFVCipherValue"

    ciphertext: Any  # sealapi.Ciphertext
    context: "BFVPublicContext"

    def to_json(self) -> dict:
        # Use /dev/shm on Linux for better performance
        fname = _get_seal_temp_path()
        try:
            self.ciphertext.save(fname)
            with open(fname, "rb") as f:
                data_bytes = f.read()
        finally:
            os.unlink(fname)

        return {
            "data_bytes": base64.b64encode(data_bytes).decode("ascii"),
            "ctx": self.context.to_json(),
        }

    @classmethod
    def from_json(cls, data: dict) -> "BFVCipherValue":
        # ... symmetric implementation
        ...


# ============= KeyValue =============

@serde.register_class
@dataclass
class KeyValue(RuntimeValue):
    """Wraps cryptographic keys as runtime values."""
    _serde_kind: ClassVar[str] = "runtime.KeyValue"

    key_bytes: bytes
    key_type: str  # "public", "private", "symmetric"
    suite: str     # "ec256", "aes128", etc.

    def to_json(self) -> dict:
        return {
            "key_bytes": base64.b64encode(self.key_bytes).decode("ascii"),
            "key_type": self.key_type,
            "suite": self.suite,
        }

    @classmethod
    def from_json(cls, data: dict) -> "KeyValue":
        return cls(
            key_bytes=base64.b64decode(data["key_bytes"]),
            key_type=data["key_type"],
            suite=data["suite"],
        )
```

### 2.3 Type Hierarchy

```
RuntimeValue (ABC)
├── TensorValue          # numpy/jax arrays
├── TableValue           # pyarrow tables
├── ScalarValue          # int/float/bool/str
├── BytesValue           # raw bytes
├── KeyValue             # crypto keys
├── BFVCipherValue       # BFV ciphertext
├── BFVPlainValue        # BFV plaintext
├── BFVContextValue      # BFV context (public/secret)
├── PHECipherValue       # PHE ciphertext
├── SPUShareValue        # SPU secret share
└── TEEQuoteValue        # TEE attestation quote
```

---

## 3. Serde Simplification

### Current State

- `edsl/serde.py`: Core logic + primitives.
- `dialects/serde.py`: **Deleted** (merged into dialects).
- `backends/serde.py`: Still exists, uses monkey-patching for runtime objects.

### After

- `edsl/serde.py`: Unchanged. It already supports any object with `_serde_kind` and `to_json`.
- `backends/serde.py`: **Delete**. All runtime values will define their own serialization via `RuntimeValue` subclasses and `@serde.register_class`.

### Serde Registration Logic

No changes needed to `edsl/serde.py`. The existing logic handles registered classes automatically:

```python
# Existing logic in edsl/serde.py handles this:
if hasattr(obj, "_serde_kind") and hasattr(obj, "to_json"):
    data = obj.to_json()
    data["_kind"] = obj._serde_kind
    return data
```

---

## 4. Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

1. Create `mplang/v2/edsl/runtime.py` with `RuntimeValue` base class.
2. Implement `TensorValue`, `ScalarValue`, `BytesValue` using `@serde.register_class`.
3. Verify `edsl/serde.py` handles them correctly without modification.
4. Add tests in `tests/v2/edsl/test_runtime_value.py`.

### Phase 2: Migrate BFV Types (Week 2)

1. Rename `BFVValue` -> `BFVCipherValue`/`BFVPlainValue` (extend `RuntimeValue`).
2. Rename `BFVPublicContext`/`BFVSecretContext` -> `BFVContextValue`.
3. Delete `backends/serde.py` (move logic inline).
4. Update `bfv_impl.py` to use new types.
5. Add tests.

### Phase 3: Migrate Crypto Types (Week 2)

1. Replace `RuntimePublicKey`/`RuntimePrivateKey`/`RuntimeSymmetricKey` with `KeyValue`.
2. Update `crypto_impl.py`.
3. Add tests.

### Phase 4: Migrate Other Types (Week 3)

1. `TableValue` for PyArrow tables.
2. `PHECipherValue` for PHE.
3. `SPUShareValue` for SPU.
4. `TEEQuoteValue` for TEE.

---

## 5. Migration Guide

### Before

```python
# bfv_impl.py
@dataclass
class BFVValue:
    data: sealapi.Ciphertext | sealapi.Plaintext
    ctx: BFVPublicContext
    is_cipher: bool = True

# backends/serde.py
BFVValue._serde_kind = "bfv_impl.BFVValue"
def bfv_value_to_json(self): ...
BFVValue.to_json = bfv_value_to_json  # Monkey-patch!
serde.register_class(BFVValue)
```

### After

```python
# bfv_impl.py
from mplang.v2.edsl import serde, runtime

@serde.register_class
@dataclass
class BFVCipherValue(runtime.RuntimeValue):
    _serde_kind: ClassVar[str] = "runtime.BFVCipherValue"

    ciphertext: sealapi.Ciphertext
    context: BFVContextValue

    def to_json(self) -> dict:
        # Built-in, no monkey-patching
        ...

    @classmethod
    def from_json(cls, data: dict) -> "BFVCipherValue":
        ...

# No backends/serde.py needed!
```
