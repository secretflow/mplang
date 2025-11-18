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
    - `HE`: Element-wise Homomorphic Encryption of a single scalar.
    - `SIMD_HE`: Packed (SIMD) Homomorphic Encryption of a vector of scalars into a single ciphertext.
    - `SS`: A single share of a secret-shared value.

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

-   **World 2: The Element-wise HE World**
    - **Core Type**: `Tensor[HE[Scalar], ...]`
    - **API Standard**: Follows TenSEAL-like (Tensor-level) conventions. Layout operations
      (`transpose`, `reshape`) are valid as they merely shuffle independent ciphertext objects.
      Arithmetic operations are overloaded for element-wise HE computation.

-   **World 3: The Packed (SIMD) HE World**
    - **Core Type**: `SIMD_HE[Scalar, PackingShape, ...]`
    - **API Standard**: Follows Microsoft SEAL-like (Ciphertext-level) conventions. This is
      an opaque, non-tensor type. It only supports specialized vector operations like `simd_add`
      and `simd_rotate`. It is explicitly *not* a `ScalarType` and cannot be an element of a `Tensor`.

This separation is programmatically enforced. Attempting to create a `Tensor[SIMD_HE[...]]`
will raise a `TypeError`.

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

from typing import Any, TypeVar

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


# ==============================================================================
# --- Predefined Scalar Type Instances
# ==============================================================================

# Numeric scalar types - comprehensive set aligned with common dtypes
# Integer types (signed)pes (signed)
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

# Variable-length integer types (common sizes)
i128 = IntegerType(bitwidth=128, signed=True)
i256 = IntegerType(bitwidth=256, signed=True)
u128 = IntegerType(bitwidth=128, signed=False)
u256 = IntegerType(bitwidth=256, signed=False)


class TensorType(BaseType):
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
        # Only ScalarType and ScalarHEType can be tensor elements
        # SIMD_HE cannot be a tensor element (it's already a packed vector)
        # ScalarHEType inherits from ScalarType, so we only need to check ScalarType
        if not isinstance(element_type, ScalarType):
            raise TypeError(
                f"Tensor element type must be a ScalarType (including ScalarHEType), but got {type(element_type).__name__}. "
                "Note: SIMD_HE cannot be an element of a Tensor."
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

    def __class_getitem__(cls, params: tuple) -> TensorType:  # type: ignore[misc]
        """Enables the syntax `Tensor[element_type, shape]`.

        Args:
            params: Either a single element_type or (element_type, shape) tuple

        Returns:
            TensorType instance
        """
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


Tensor = TensorType


class TableType(BaseType):
    """Represents a table with a named schema of types."""

    def __init__(self, schema: dict[str, BaseType]):
        self.schema = schema

    def __class_getitem__(cls, schema: dict[str, BaseType]) -> TableType:
        """Enables the syntax `Table[{'col_a': i32, ...}]`."""
        return cls(schema)

    def __str__(self) -> str:
        schema_str = ", ".join(f"'{k}': {v}" for k, v in self.schema.items())
        return f"Table[{{{schema_str}}}]"


Table = TableType


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


# Shorthand alias
Custom = CustomType

# ==============================================================================
# --- Pillar 2: Encryption Types
# ==============================================================================


class ScalarHEType(ScalarType, EncryptedTrait):
    """Represents a single scalar value encrypted with an HE scheme.

    Inherits from ScalarType, so it can be used as a tensor element type.

    Note:
        Encoding details (e.g., integer encoding, fixed-point scale) should
        be tracked as attributes on encrypt/decrypt operations, not in the
        type itself. This keeps the type system clean and focused on the
        logical plaintext type.
    """

    def __init__(self, scalar_type: ScalarType, scheme: str = "ckks"):
        if not isinstance(scalar_type, ScalarType):
            raise TypeError(
                f"HE encryption requires a ScalarType, got {type(scalar_type).__name__}."
            )
        self._pt_type = scalar_type
        self._scheme = scheme

    def __class_getitem__(cls, params: Any) -> ScalarHEType:
        if isinstance(params, tuple):
            return cls(params[0], params[1])
        return cls(params)

    def __str__(self) -> str:
        return f"HE[{self.pt_type}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScalarHEType):
            return False
        return self._pt_type == other._pt_type and self._scheme == other._scheme

    def __hash__(self) -> int:
        return hash(("ScalarHEType", self._pt_type, self._scheme))


HE = ScalarHEType


class SIMDHEType(BaseType, EncryptedTrait):
    """Represents a SINGLE ciphertext packing a vector of scalars using HE SIMD mode."""

    def __init__(
        self, scalar_type: ScalarType, packing_shape: tuple[int,], scheme: str = "ckks"
    ):
        if not isinstance(scalar_type, ScalarType):
            raise TypeError(
                f"SIMD_HE requires a ScalarType, got {type(scalar_type).__name__}."
            )
        self._pt_type = scalar_type
        self.packing_shape = packing_shape
        self._scheme = scheme

    @property
    def pt_type(self) -> BaseType:
        # Represent packed ciphertext's logical plaintext as a Tensor type.
        return TensorType(self._pt_type, self.packing_shape)

    @property
    def scalar_type(self) -> BaseType:
        return self._pt_type

    def __class_getitem__(cls, params: tuple[ScalarType, tuple[int,]]) -> SIMDHEType:
        scalar_type, packing_shape = params
        return cls(scalar_type, packing_shape)

    def __str__(self) -> str:
        return f"SIMD_HE[{self.scalar_type}, {self.packing_shape}]"


SIMD_HE = SIMDHEType


class SSType(BaseType, EncryptedTrait):
    """Represents a single share of a secret value `T`."""

    def __init__(self, secret_type: BaseType, enc_schema: str = "ss"):
        self._pt_type = secret_type
        self._enc_schema = enc_schema

    def __class_getitem__(cls, secret_type: BaseType) -> SSType:
        """Enables the syntax `SS[Tensor[...]]`."""
        return cls(secret_type)

    def __str__(self) -> str:
        return f"SS[{self.pt_type}]"


SS = SSType

# ==============================================================================
# --- Pillar 3: Distribution Types
# ==============================================================================


class MPType(BaseType):
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
        cls, params: tuple[BaseType, tuple[int, ...] | None]
    ) -> MPType:
        """Enables the syntax `MP[Tensor[...], (0, 1)]` or `MP[Tensor[...], None]`."""
        value_type, parties = params
        return cls(value_type, parties)

    def __str__(self) -> str:
        return f"MP[{self.value_type}, parties={self.parties}]"


MP = MPType
