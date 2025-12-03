# Runtime Value System

## Status: Implemented

## Author: @jint

## Date: 2025-12-03

---

## 1. Overview

MPLang v2 uses a unified `Value` base class for all runtime values. This provides:

- **Type Safety**: `def_impl` functions use typed `Value` subclasses instead of `Any`.
- **Unified Serde**: All values implement `to_json`/`from_json` via `@serde.register_class`.
- **Consistent Naming**: All subclasses use `XxxValue` suffix convention.

---

## 2. Value Hierarchy

```
Value (ABC)                          # mplang/v2/backends/value.py
├── WrapValue[T]                     # Generic wrapper for underlying data
│   ├── TensorValue                  # np.ndarray (tensor_impl.py)
│   ├── TableValue                   # pa.Table (table_impl.py)
│   ├── BytesValue                   # bytes (crypto_impl.py)
│   ├── ECPointValue                 # EC point bytes (crypto_impl.py)
│   ├── SPUShareValue                # libspu.Share (spu_impl.py)
│   ├── BFVPublicContextValue        # ts.Context without secret key (bfv_impl.py)
│   └── BFVSecretContextValue        # ts.Context with secret key (bfv_impl.py)
├── PrivateKeyValue                  # KEM private key (crypto_impl.py)
├── PublicKeyValue                   # KEM public key (crypto_impl.py)
├── SymmetricKeyValue                # Symmetric key (crypto_impl.py)
├── BFVValue                         # BFV ciphertext/plaintext vector (bfv_impl.py)
└── MockQuoteValue                   # TEE attestation quote (tee_impl.py)
```

---

## 3. Implementation Pattern

### 3.1 WrapValue Pattern

For values wrapping a single underlying object, use `WrapValue[T]`:

```python
@serde.register_class
@dataclass
class TensorValue(WrapValue[np.ndarray]):
    """Wraps numpy arrays as runtime values."""
    _serde_kind: ClassVar[str] = "tensor_impl.TensorValue"

    @classmethod
    def _convert(cls, data: Any) -> np.ndarray:
        """Convert input to canonical form."""
        if isinstance(data, TensorValue):
            return data.data
        if isinstance(data, np.ndarray):
            return data
        # Handle JAX arrays, etc.
        return np.asarray(data)

    def to_json(self) -> dict[str, Any]:
        return {
            "data": base64.b64encode(self.data.tobytes()).decode("ascii"),
            "dtype": str(self.data.dtype),
            "shape": list(self.data.shape),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TensorValue:
        arr = np.frombuffer(
            base64.b64decode(data["data"]), dtype=data["dtype"]
        ).reshape(data["shape"])
        return cls(arr)
```

### 3.2 Direct Value Pattern

For values with multiple fields, extend `Value` directly:

```python
@serde.register_class
@dataclass
class BFVValue(Value):
    """BFV encrypted/encoded vector."""
    _serde_kind: ClassVar[str] = "bfv_impl.BFVValue"

    ctx: BFVPublicContextValue
    vec: ts.BFVVector

    def to_json(self) -> dict[str, Any]:
        return {
            "ctx": self.ctx.to_json(),
            "vec": base64.b64encode(self.vec.serialize()).decode("ascii"),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BFVValue:
        ctx = BFVPublicContextValue.from_json(data["ctx"])
        vec = ts.bfv_vector_from(ctx.data, base64.b64decode(data["vec"]))
        return cls(ctx=ctx, vec=vec)
```

---

## 4. Serde Integration

Values are registered with `@serde.register_class` decorator. The `edsl/serde.py` module handles dispatch:

```python
# Serialization: automatic _kind injection
def to_json(obj: Any) -> Any:
    if hasattr(obj, "_serde_kind") and hasattr(obj, "to_json"):
        data = obj.to_json()
        data["_kind"] = obj._serde_kind
        return data
    # ... handle primitives, lists, dicts

# Deserialization: registry lookup
def from_json(data: Any) -> Any:
    if isinstance(data, dict) and "_kind" in data:
        cls = _CLASS_REGISTRY[data["_kind"]]
        return cls.from_json(data)
    # ... handle primitives, lists, dicts
```

---

## 5. Usage in def_impl

All `def_impl` functions use `Value` subclasses for type safety:

```python
@tensor_dialect.def_impl("tensor.add")
def add_impl(
    interpreter: Interpreter,
    op: Operation,
    lhs: TensorValue,
    rhs: TensorValue,
) -> TensorValue:
    result = lhs.data + rhs.data
    return TensorValue(result)
```

---

## 6. Extension Guide

### Adding a New Value Type

1. **Create the class** in the appropriate `*_impl.py`:

```python
@serde.register_class
@dataclass
class MyValue(WrapValue[MyUnderlyingType]):
    _serde_kind: ClassVar[str] = "my_impl.MyValue"

    @classmethod
    def _convert(cls, data: Any) -> MyUnderlyingType:
        # Convert input to canonical form
        ...

    def to_json(self) -> dict[str, Any]:
        # Serialize to JSON-compatible dict
        ...

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> MyValue:
        # Deserialize from dict
        ...
```

2. **Use in def_impl functions** with proper type hints.

3. **Add tests** for serde roundtrip and impl functions.

### Naming Convention

- Use `XxxValue` suffix for all Value subclasses
- The `_serde_kind` should be `"module.ClassName"` format

### Performance Considerations

- Use `WrapValue` for zero-copy wrapping when possible
- For large data (tensors, tables), use efficient binary formats (Arrow IPC, numpy bytes)
- Lazy serialization: serialize only during actual network transfer

---

## 7. Current Value Types

| Value Class | Wrapped Type | Location | Description |
|-------------|--------------|----------|-------------|
| `TensorValue` | `np.ndarray` | tensor_impl.py | NumPy/JAX arrays |
| `TableValue` | `pa.Table` | table_impl.py | PyArrow tables |
| `BytesValue` | `bytes` | crypto_impl.py | Raw byte data |
| `ECPointValue` | `bytes` | crypto_impl.py | EC curve points |
| `PrivateKeyValue` | - | crypto_impl.py | KEM private keys |
| `PublicKeyValue` | - | crypto_impl.py | KEM public keys |
| `SymmetricKeyValue` | - | crypto_impl.py | Symmetric keys |
| `SPUShareValue` | `libspu.Share` | spu_impl.py | SPU secret shares |
| `BFVPublicContextValue` | `ts.Context` | bfv_impl.py | BFV context (public) |
| `BFVSecretContextValue` | `ts.Context` | bfv_impl.py | BFV context (secret) |
| `BFVValue` | `ts.BFVVector` | bfv_impl.py | BFV vector (cipher/plain) |
| `MockQuoteValue` | - | tee_impl.py | TEE attestation quote |

---

## 8. Future Extensions

### Potential New Value Types

- `PHECipherValue` - Paillier homomorphic encryption ciphertext
- `PHEPublicKeyValue` / `PHEPrivateKeyValue` - PHE key pairs
- `CKKSValue` - CKKS scheme for approximate arithmetic
- `OTMessageValue` - Oblivious transfer messages

### Potential Improvements

- **Streaming serde**: For very large values, support chunked serialization
- **Compression**: Optional compression for network transfer
- **Versioning**: Add version field to `_serde_kind` for backward compatibility
- **Validation**: Add schema validation in `from_json`
