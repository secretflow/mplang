# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MPLang Core Typing System: Design and Rationale.

This module defines the production type system for MPLang, an EDSL for multi-party
privacy-preserving computation. This document explains the core principles and design
decisions that shape this system, intended for future maintainers and developers.

===========================
Tensor Shape System
===========================
MPLang supports a flexible shape system for tensors to handle various compilation and
runtime scenarios:

**Shape Representations:**
    - `None`: Fully dynamic/unranked tensor (shape unknown at compile time)
        Example: `Tensor[i32, None]`

    - `()`: Scalar (0-dimensional tensor)
        Example: `Tensor[i32, ()]`

    - `(dim1, dim2, ...)`: Ranked tensor with static or dynamic dimensions
        - Positive integers: Static dimension sizes
        - `-1`: Dynamic/unknown dimension size
        Examples:
            - `Tensor[i32, (3, 10)]` - Fully static 2D tensor
            - `Tensor[i32, (-1, 10)]` - Dynamic batch size, static feature size
            - `Tensor[i32, (-1, -1)]` - Fully dynamic 2D tensor

**Utility Properties:**
    - `.is_scalar`: Check if tensor is 0-dimensional
    - `.is_unranked`: Check if shape is None
    - `.is_fully_static`: Check if all dimensions are statically known
    - `.rank`: Get number of dimensions (None for unranked)
    - `.has_dynamic_dims()`: Check if any dimension is dynamic

===========================
Principle 1: Orthogonality and Composition
===========================
The type system is built on three orthogonal pillars. Each type represents a single,
well-defined concept. Complex ideas are expressed by composing these simple types,
rather than by creating a large, monolithic set of specific types.

1.  **Layout Types**: Describe the physical shape and structure of data.
    - `Scalar`: Atomic data types (f32, i64).
    - `Tensor`: A multi-dimensional array of a `ScalarType` element type.
    - `Table`: A dictionary-like structure with named columns of any type.

    2.  **Encryption Types**: Wrap other types to confer privacy properties by making them opaque.
    - `SS`: A single share of a secret-shared value.
    - Note: Element-wise HE types (like `phe.CiphertextType`) are defined in their respective dialects (e.g., `phe`).

3.  **Distribution Types**: Wrap other types to describe their physical location among parties.
    - `MP`: Represents a value logically held by multiple parties.

An example of composition: `MP[SS[Tensor[f32, (10,)]], (0, 1)]` represents a
10-element float tensor, which is secret-shared (`SS`), and whose shares are distributed
between parties 0 and 1 (`MP`).

===========================
Principle 2: The "Three Worlds" of Homomorphic Encryption
===========================
A critical design decision is the strict separation of HE-based computation into three
distinct, non-interacting "worlds." This avoids ambiguity in operator semantics (e.g., `transpose`),
clarifies the user's mental model, and aligns the type system with the practical realities of
underlying HE libraries.

-   **World 1: The Plaintext World**
    - **Core Type**: `Tensor[Scalar, ...]`
    - **API Standard**: Follows NumPy/JAX conventions. All layout and arithmetic operations are valid.

    - **Core Type**: `Tensor[EncryptedScalar, ...]` (e.g., `Tensor[phe.CiphertextType, ...]`)
    - **API Standard**: Follows TenSEAL-like (Tensor-level) conventions. Layout operations
      (`transpose`, `reshape`) are valid as they merely shuffle independent ciphertext objects.
      Arithmetic operations are overloaded for element-wise HE computation.

===========================
Principle 3: Contracts via Protocols
===========================
The system uses `typing.Protocol` to define behavioral contracts (similar to Traits in Rust).
This allows for writing generic functions that operate on any type satisfying a contract,
promoting extensibility and loose coupling via structural subtyping ("duck typing").

- `EncryptedTrait`: For types representing data in an obscured form.
- `Distributed`: For types describing data distribution.

===========================
Rationale for the `EncryptedTrait` Protocol
===========================
The name `EncryptedTrait` was deliberately chosen over the more general `PrivacyBearing` after
careful consideration.

1.  **Scope is Naturally Limited**: Other privacy techniques like Differential Privacy or
    Federated Learning are algorithmic or orchestration patterns that do not require new
    type wrappers for the data itself. A DP-protected tensor is still a `Tensor`.
    Therefore, the protocol only needs to cover technologies that transform data into an
    opaque representation.

2.  **Secret Sharing as a form of Encryption**: The key insight is to conceptualize
    Secret Sharing (`SS`) as a form of multi-key encryption. For a holder of a single
    share, the other parties' shares are analogous to the "key" needed to recover the
    secret. Both `HE` and `SS` render the data opaque and require external information
    (a key or other shares) for recovery. This powerful mental model allows both `HE`/`SIMD_HE`
    and `SS` to logically implement the `Encrypted` protocol.

This makes `Encrypted` a name that is both intuitive to engineers and conceptually
consistent within the practical scope of this library.
"""

from __future__ import annotations

from typing import Any, ClassVar, Generic, TypeVar

from mplang.v2.edsl import serde

# ==============================================================================
# --- Base Type & Type Aliases
# ==============================================================================

T = TypeVar("T")


class BaseType:
    """Base class for all MPLang types."""

    def __repr__(self) -> str:
        return str(self)


# ==============================================================================
# --- Type Protocols (Contracts)
# ==============================================================================


class EncryptedTrait:
    """A contract for types that represent data in an encrypted or obscured form."""

    _pt_type: BaseType
    _enc_schema: str

    @property
    def pt_type(self) -> BaseType:
        return self._pt_type

    @property
    def enc_schema(self) -> str:
        return self._enc_schema


# ==============================================================================
# --- Pillar 1: Layout Types
# ==============================================================================


class ScalarType(BaseType):
    """Base class for all scalar types (integers, floats, complex).

    This serves as the common parent for IntegerType, FloatType, and ComplexType,
    allowing code to accept any scalar type without needing union types.
    """


@serde.register_class
class IntegerType(ScalarType):
    """Represents a variable-length integer type.

    This is a standard integer type with configurable bit width, used for
    arbitrary-precision arithmetic. It can represent integers that exceed
    the range of fixed-width types like i64.

    Examples:
        >>> i128 = IntegerType(bitwidth=128, signed=True)  # i128
        >>> u256 = IntegerType(bitwidth=256, signed=False)  # u256

    Note:
        Encoding-specific metadata (e.g., fixed-point scale, semantic type)
        should be maintained as attributes on operations/objects that use
        IntegerType, not on the type itself.
    """

    def __init__(self, *, bitwidth: int = 32, signed: bool = True):
        """Initialize an IntegerType.

        Args:
            bitwidth: Number of bits for the integer representation.
                     Common values: 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096.
            signed: Whether the integer is signed (True) or unsigned (False).
        """
        if bitwidth <= 0 or (bitwidth & (bitwidth - 1)) != 0:
            raise ValueError(f"bitwidth must be a positive power of 2, got {bitwidth}")
        self.bitwidth = bitwidth
        self.signed = signed

    def __str__(self) -> str:
        sign_prefix = "i" if self.signed else "u"
        return f"{sign_prefix}{self.bitwidth}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntegerType):
            return False
        return self.bitwidth == other.bitwidth and self.signed == other.signed

    def __hash__(self) -> int:
        return hash(("IntegerType", self.bitwidth, self.signed))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "mplang.IntegerType"

    def to_json(self) -> dict[str, Any]:
        return {"bitwidth": self.bitwidth, "signed": self.signed}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> IntegerType:
        return cls(bitwidth=data["bitwidth"], signed=data["signed"])


@serde.register_class
class FloatType(ScalarType):
    """Represents a floating-point type.

    This supports standard IEEE 754 floating-point types with configurable
    precision (bitwidth).

    Examples:
        >>> f16 = FloatType(bitwidth=16)  # half precision
        >>> f32 = FloatType(bitwidth=32)  # single precision
        >>> f64 = FloatType(bitwidth=64)  # double precision
    """

    def __init__(self, *, bitwidth: int = 32):
        """Initialize a FloatType.

        Args:
            bitwidth: Number of bits for the float representation.
                     Standard values: 16 (half), 32 (single), 64 (double).
        """
        if bitwidth not in (16, 32, 64, 128):
            raise ValueError(f"bitwidth must be 16, 32, 64, or 128, got {bitwidth}")
        self.bitwidth = bitwidth

    def __str__(self) -> str:
        return f"f{self.bitwidth}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FloatType):
            return False
        return self.bitwidth == other.bitwidth

    def __hash__(self) -> int:
        return hash(("FloatType", self.bitwidth))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "mplang.FloatType"

    def to_json(self) -> dict[str, Any]:
        return {"bitwidth": self.bitwidth}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FloatType:
        return cls(bitwidth=data["bitwidth"])


@serde.register_class
class ComplexType(ScalarType):
    """Represents a complex number type.

    Complex numbers are represented as pairs of floating-point values.
    Both real and imaginary parts use the same floating-point type.

    Examples:
        >>> c64 = ComplexType(inner_type=f32)  # complex64 (2x float32)
        >>> c128 = ComplexType(inner_type=f64)  # complex128 (2x float64)
    """

    def __init__(self, *, inner_type: FloatType):
        """Initialize a ComplexType.

        Args:
            inner_type: The floating-point type for real and imaginary parts.
                       Common values: f16, f32, f64, f128.
        """
        if not isinstance(inner_type, FloatType):
            raise TypeError(
                f"inner_type must be a FloatType, got {type(inner_type).__name__}"
            )
        self.inner_type = inner_type

    def __str__(self) -> str:
        return f"c{self.inner_type.bitwidth * 2}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ComplexType):
            return False
        return self.inner_type == other.inner_type

    def __hash__(self) -> int:
        return hash(("ComplexType", self.inner_type))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "mplang.ComplexType"

    def to_json(self) -> dict[str, Any]:
        return {"inner_type": serde.to_json(self.inner_type)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ComplexType:
        inner = serde.from_json(data["inner_type"])
        if not isinstance(inner, FloatType):
            raise TypeError(f"ComplexType inner must be FloatType, got {type(inner)}")
        return cls(inner_type=inner)


# ==============================================================================
# --- Predefined Scalar Type Instances
# ==============================================================================

# Numeric scalar types - comprehensive set aligned with common dtypes
# Integer types (signed)
i8 = IntegerType(bitwidth=8, signed=True)
i16 = IntegerType(bitwidth=16, signed=True)
i32 = IntegerType(bitwidth=32, signed=True)
i64 = IntegerType(bitwidth=64, signed=True)

# Fixed-width integer types (unsigned)
u8 = IntegerType(bitwidth=8, signed=False)
u16 = IntegerType(bitwidth=16, signed=False)
u32 = IntegerType(bitwidth=32, signed=False)
u64 = IntegerType(bitwidth=64, signed=False)

# Floating point types
f16 = FloatType(bitwidth=16)
f32 = FloatType(bitwidth=32)
f64 = FloatType(bitwidth=64)

# Complex types
c64 = ComplexType(inner_type=f32)  # 2x float32 = 64 bits total
c128 = ComplexType(inner_type=f64)  # 2x float64 = 128 bits total

# Boolean type (1-bit integer, commonly used)
bool_ = IntegerType(bitwidth=1, signed=True)
i1 = bool_  # Alias for MLIR convention

# Variable-length integer types (common sizes)
i128 = IntegerType(bitwidth=128, signed=True)
i256 = IntegerType(bitwidth=256, signed=True)
u128 = IntegerType(bitwidth=128, signed=False)
u256 = IntegerType(bitwidth=256, signed=False)


@serde.register_class
class TensorType(BaseType, Generic[T]):
    """Represents a ranked tensor of a given element type and shape.

    Following MLIR's RankedTensorType design - all tensors must have a known rank.
    This simplifies type inference and reduces complexity compared to supporting
    fully unranked tensors.

    Shape must be a tuple where each dimension can be:
        - Positive integer: Static dimension size
        - -1: Dynamic/unknown dimension size

    Examples:
        Tensor[i32, ()]         # Scalar (0-dim tensor)
        Tensor[i32, (-1, 10)]   # Partially dynamic shape (rank=2)
        Tensor[i32, (3, 10)]    # Fully static shape (rank=2)
        Tensor[i32, (-1,)]      # 1D tensor with dynamic size
    """

    def __init__(self, element_type: BaseType, shape: tuple[int, ...]):
        # Allow any BaseType to support custom types like PointType, EncryptedScalar
        if not isinstance(element_type, BaseType):
            raise TypeError(
                f"Tensor element type must be a BaseType, but got {type(element_type).__name__}."
            )
        self.element_type = element_type
        self.shape = shape

        # Validate shape is a tuple
        if not isinstance(shape, tuple):
            raise TypeError(f"Shape must be a tuple, got {type(shape).__name__}")

        # Validate each dimension
        for dim in shape:
            if not isinstance(dim, int):
                raise TypeError(
                    f"Shape dimensions must be integers, got {type(dim).__name__}"
                )
            if dim < -1 or dim == 0:
                raise ValueError(
                    f"Invalid dimension {dim}: must be positive or -1 for dynamic"
                )

    def __class_getitem__(cls, params: tuple | Any) -> Any:
        """Enables the syntax `Tensor[element_type, shape]`.

        Args:
            params: Either a single element_type or (element_type, shape) tuple

        Returns:
            TensorType instance or GenericAlias
        """
        # Check if we are doing type specialization (Generic[T]) or instance creation
        # Heuristic: If params contains a Type (class), it's a type spec.
        args = params if isinstance(params, tuple) else (params,)
        if any(isinstance(a, type) for a in args):
            return super().__class_getitem__(params)  # type: ignore[misc]

        if not isinstance(params, tuple):
            raise TypeError(
                "Tensor requires shape parameter. Use Tensor[element_type, shape] "
                "where shape is (), or a tuple of integers."
            )

        if len(params) != 2:
            raise TypeError(
                f"Tensor expects 2 parameters (element_type, shape), got {len(params)}"
            )

        element_type, shape = params
        return cls(element_type, shape)

    def __str__(self) -> str:
        shape_str = ", ".join(str(d) for d in self.shape)
        return f"Tensor[{self.element_type}, ({shape_str})]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorType):
            return False
        return self.element_type == other.element_type and self.shape == other.shape

    def __hash__(self) -> int:
        return hash((self.element_type, self.shape))

    @property
    def is_scalar(self) -> bool:
        """Check if this is a scalar (0-dimensional) tensor."""
        return self.shape == ()

    @property
    def is_fully_static(self) -> bool:
        """Check if all dimensions are statically known."""
        return all(dim > 0 for dim in self.shape)

    @property
    def rank(self) -> int:
        """Get the rank (number of dimensions) of the tensor.

        Returns:
            int: Number of dimensions (always available for ranked tensors)
        """
        return len(self.shape)

    def has_dynamic_dims(self) -> bool:
        """Check if tensor has any dynamic dimensions (-1)."""
        return any(dim == -1 for dim in self.shape)

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "mplang.TensorType"

    def to_json(self) -> dict[str, Any]:
        return {
            "element_type": serde.to_json(self.element_type),
            "shape": list(self.shape),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TensorType[Any]:
        element_type = serde.from_json(data["element_type"])
        shape = tuple(data["shape"])
        return cls(element_type, shape)


Tensor = TensorType


@serde.register_class
class VectorType(BaseType):
    """Represents a packed SIMD vector of a given element type and size.

    Unlike Tensor, which represents a logical multi-dimensional array,
    Vector represents a physical packed layout (SIMD).
    This is the underlying payload for SIMD_HE schemes (BFV, CKKS).

    Args:
        element_type: The type of elements in the vector (must be ScalarType).
        size: The number of elements (slots) in the vector.
    """

    def __init__(self, element_type: ScalarType, size: int):
        if not isinstance(element_type, ScalarType):
            raise TypeError(
                f"Vector element type must be a ScalarType, got {type(element_type).__name__}"
            )
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"Vector size must be a positive integer, got {size}")

        self.element_type = element_type
        self.size = size

    def __class_getitem__(cls, params: tuple) -> VectorType:
        """Enables the syntax `Vector[element_type, size]`."""
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("Vector expects 2 parameters (element_type, size)")

        element_type, size = params
        return cls(element_type, size)

    def __str__(self) -> str:
        return f"Vector[{self.element_type}, {self.size}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorType):
            return False
        return self.element_type == other.element_type and self.size == other.size

    def __hash__(self) -> int:
        return hash(("VectorType", self.element_type, self.size))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "mplang.VectorType"

    def to_json(self) -> dict[str, Any]:
        return {
            "element_type": serde.to_json(self.element_type),
            "size": self.size,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> VectorType:
        element_type = serde.from_json(data["element_type"])
        if not isinstance(element_type, ScalarType):
            raise TypeError(
                f"VectorType element must be ScalarType, got {type(element_type)}"
            )
        return cls(element_type, data["size"])


Vector = VectorType


@serde.register_class
class TableType(BaseType):
    """Represents a table with a named schema of types.

    Examples:
        >>> TableType({"id": i64, "name": STRING})
        Table[{'id': i64, 'name': Custom[string]}]

        >>> Table[{"col_a": i32, "col_b": f64}]
        Table[{'col_a': i32, 'col_b': f64}]
    """

    def __init__(self, schema: dict[str, BaseType]):
        self.schema = schema

    def __class_getitem__(cls, schema: dict[str, BaseType]) -> TableType:
        """Enables the syntax `Table[{'col_a': i32, ...}]`."""
        return cls(schema)

    def __str__(self) -> str:
        schema_str = ", ".join(f"'{k}': {v}" for k, v in self.schema.items())
        return f"Table[{{{schema_str}}}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TableType):
            return NotImplemented
        return self.schema == other.schema

    def __hash__(self) -> int:
        return hash(("TableType", tuple(self.schema.items())))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "mplang.TableType"

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": {name: serde.to_json(t) for name, t in self.schema.items()},
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TableType:
        schema = {name: serde.from_json(t) for name, t in data["schema"].items()}
        return cls(schema)


Table = TableType


@serde.register_class
class CustomType(BaseType):
    """Opaque/custom type identified by a string kind.

    Used for types that don't have explicit structure (like encryption keys,
    database handles, or other opaque objects) but need to be tracked in the
    type system.

    Examples::

        >>> key_type = CustomType("EncryptionKey")
        >>> handle_type = CustomType("DatabaseHandle")
        >>> token_type = CustomType("AuthToken")

    The kind string serves as the identifier for equality and hashing.
    Two CustomTypes are equal if and only if their kinds are equal.

    Attributes:
        kind: String identifier for this custom type.
    """

    def __init__(self, kind: str):
        """Initialize a custom type.

        Args:
            kind: String identifier for this custom type.
                  Should be descriptive (e.g., "EncryptionKey", "Handle").

        Raises:
            TypeError: If kind is not a string.
            ValueError: If kind is empty or whitespace-only.
        """
        if not isinstance(kind, str):
            raise TypeError(f"kind must be str, got {type(kind).__name__}")
        if not kind or kind.strip() == "":
            raise ValueError("kind must be a non-empty string")

        self._kind = kind

    @property
    def kind(self) -> str:
        """Return the string identifier for this custom type."""
        return self._kind

    def __eq__(self, other: object) -> bool:
        """Two CustomTypes are equal if their kinds match."""
        if not isinstance(other, CustomType):
            return False
        return self._kind == other._kind

    def __hash__(self) -> int:
        """Hash based on kind for use in sets and dicts."""
        return hash(("CustomType", self._kind))

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"CustomType({self._kind!r})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        return f"Custom[{self._kind}]"

    def __class_getitem__(cls, kind: str) -> CustomType:
        """Enable Custom["TypeName"] syntax sugar.

        Examples::

            >>> EncryptionKey = Custom["EncryptionKey"]
            >>> # Equivalent to:
            >>> EncryptionKey = CustomType("EncryptionKey")
        """
        return cls(kind)

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "mplang.CustomType"

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CustomType:
        return cls(data["kind"])


# Shorthand alias
Custom = CustomType

# ==============================================================================
# --- Table-only Types (for SQL/DataFrame operations)
# ==============================================================================
# These types are used in TableType schemas but don't have direct tensor
# equivalents. They use CustomType for flexibility.

STRING = CustomType("string")
DATE = CustomType("date")
TIME = CustomType("time")
TIMESTAMP = CustomType("timestamp")
DECIMAL = CustomType("decimal")
BINARY = CustomType("binary")
JSON = CustomType("json")
UUID = CustomType("uuid")
INTERVAL = CustomType("interval")

# ==============================================================================
# --- Pillar 2: Encryption Types
# ==============================================================================


@serde.register_class
class SSType(BaseType, EncryptedTrait, Generic[T]):
    """Represents a single share of a secret value `T`."""

    def __init__(self, secret_type: BaseType, enc_schema: str = "ss"):
        self._pt_type = secret_type
        self._enc_schema = enc_schema

    def __class_getitem__(cls, secret_type: BaseType | Any) -> Any:
        """Enables the syntax `SS[Tensor[...]]`."""
        # Check if we are doing type specialization (Generic[T]) or instance creation
        if isinstance(secret_type, type):
            return super().__class_getitem__(secret_type)  # type: ignore[misc]
        return cls(secret_type)

    def __str__(self) -> str:
        return f"SS[{self.pt_type}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SSType):
            return False
        return self.pt_type == other.pt_type and self.enc_schema == other.enc_schema

    def __hash__(self) -> int:
        return hash(("SSType", self.pt_type, self.enc_schema))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "mplang.SSType"

    def to_json(self) -> dict[str, Any]:
        return {
            "secret_type": serde.to_json(self._pt_type),
            "enc_schema": self._enc_schema,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SSType[Any]:
        secret_type = serde.from_json(data["secret_type"])
        return cls(secret_type, enc_schema=data.get("enc_schema", "ss"))


SS = SSType

# ==============================================================================
# --- Pillar 3: Distribution Types
# ==============================================================================


@serde.register_class
class MPType(BaseType, Generic[T]):
    """Represents a logical value distributed among multiple parties.

    Args:
        value_type: The type of the value held by parties
        parties: Tuple of party IDs (static mask) or None (dynamic mask)
    """

    def __init__(self, value_type: BaseType, parties: tuple[int, ...] | None):
        self._value_type = value_type
        self._parties = parties

    @property
    def value_type(self) -> BaseType:
        return self._value_type

    @property
    def parties(self) -> tuple[int, ...] | None:
        return self._parties

    def __class_getitem__(
        cls, params: tuple[BaseType, tuple[int, ...] | None] | Any
    ) -> Any:
        """Enables the syntax `MP[Tensor[...], (0, 1)]` or `MP[Tensor[...], None]`."""
        # Check if we are doing type specialization (Generic[T]) or instance creation
        # Heuristic: If params contains a Type (class), it's a type spec.
        args = params if isinstance(params, tuple) else (params,)
        if any(isinstance(a, type) for a in args):
            return super().__class_getitem__(params)  # type: ignore[misc]

        value_type, parties = params
        return cls(value_type, parties)

    def __str__(self) -> str:
        return f"MP[{self.value_type}, parties={self.parties}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MPType):
            return False
        return self.value_type == other.value_type and self.parties == other.parties

    def __hash__(self) -> int:
        return hash(("MPType", self.value_type, self.parties))

    # --- Serde methods ---
    _serde_kind: ClassVar[str] = "mplang.MPType"

    def to_json(self) -> dict[str, Any]:
        return {
            "value_type": serde.to_json(self._value_type),
            "parties": list(self._parties) if self._parties is not None else None,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> MPType[Any]:
        value_type = serde.from_json(data["value_type"])
        parties = tuple(data["parties"]) if data["parties"] is not None else None
        return cls(value_type, parties)
