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
    - `Tensor`: A multi-dimensional array of a `ScalarTrait` element type.
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
      and `simd_rotate`. It is explicitly *not* `ScalarTrait` and cannot be an element of a `Tensor`.

This separation is programmatically enforced. Attempting to create a `Tensor[SIMD_HE[...]]`
will raise a `TypeError`.

===========================
Principle 3: Contracts via Protocols
===========================
The system uses `typing.Protocol` to define behavioral contracts (similar to Traits in Rust).
This allows for writing generic functions that operate on any type satisfying a contract,
promoting extensibility and loose coupling via structural subtyping ("duck typing").

- `ScalarTrait`: For types that can be an element of a `Tensor`.
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

from typing import TypeVar

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


class ScalarTrait:
    """A contract for types that can be treated as an element of a Tensor."""

    # Marker protocol; no members required.
    # (No body needed)


# ---------------------------- Encrypted Trait -------------------------------


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


class ScalarType(BaseType, ScalarTrait):
    """Represents a single scalar value type (e.g., f32, i64)."""

    def __init__(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return self._name


f32 = ScalarType("f32")
f64 = ScalarType("f64")
i32 = ScalarType("i32")
i64 = ScalarType("i64")


class TensorType(BaseType):
    """Represents a tensor of a given element type and shape.

    Shape can be:
        - None: Fully dynamic/unranked tensor (runtime shape)
        - (): Scalar (0-dimensional tensor)
        - (-1, 10): Partially dynamic (first dim unknown, second dim is 10)
        - (3, 10): Fully ranked tensor with static shape

    A dimension can be:
        - Positive integer: Static dimension size
        - -1: Dynamic/unknown dimension size

    Examples:
        Tensor[i32, None]       # Fully dynamic/unranked
        Tensor[i32, ()]         # Scalar (0-dim tensor)
        Tensor[i32, (-1, 10)]   # Partially dynamic shape
        Tensor[i32, (3, 10)]    # Fully static shape
    """

    def __init__(self, element_type: BaseType, shape: tuple[int, ...] | None):
        if not isinstance(element_type, ScalarTrait):
            raise TypeError(
                f"Tensor element type must be ScalarTrait, but got {type(element_type).__name__}. "
                "Note: SIMD_HE is not ScalarTrait and cannot be an element of a Tensor."
            )
        self.element_type = element_type
        self.shape = shape

        # Validate shape dimensions if shape is provided
        if shape is not None:
            for dim in shape:
                if not isinstance(dim, int):
                    raise TypeError(
                        f"Shape dimensions must be integers, got {type(dim).__name__}"
                    )
                if dim < -1 or dim == 0:
                    raise ValueError(
                        f"Invalid dimension {dim}: must be positive or -1 for dynamic"
                    )

    def __class_getitem__(cls, params: tuple) -> "TensorType":  # type: ignore[misc]
        """Enables the syntax `Tensor[element_type, shape]`.

        Args:
            params: Either a single element_type or (element_type, shape) tuple

        Returns:
            TensorType instance
        """
        if not isinstance(params, tuple):
            raise TypeError(
                "Tensor requires shape parameter. Use Tensor[element_type, shape] "
                "where shape is None, (), or a tuple of integers."
            )

        if len(params) != 2:
            raise TypeError(
                f"Tensor expects 2 parameters (element_type, shape), got {len(params)}"
            )

        element_type, shape = params
        return cls(element_type, shape)

    def __str__(self) -> str:
        if self.shape is None:
            return f"Tensor[{self.element_type}, None]"
        shape_str = ", ".join(str(d) for d in self.shape)
        return f"Tensor[{self.element_type}, ({shape_str})]"

    def __eq__(self, other) -> bool:
        if not isinstance(other, TensorType):
            return False
        return self.element_type == other.element_type and self.shape == other.shape

    def __hash__(self) -> int:
        return hash((self.element_type, self.shape))

    @property
    def is_scalar(self) -> bool:
        """Check if this is a scalar (0-dimensional) tensor."""
        return self.shape is not None and self.shape == ()

    @property
    def is_unranked(self) -> bool:
        """Check if this tensor has fully dynamic/unranked shape."""
        return self.shape is None

    @property
    def is_fully_static(self) -> bool:
        """Check if all dimensions are statically known."""
        if self.shape is None:
            return False
        return all(dim > 0 for dim in self.shape)

    @property
    def rank(self) -> int | None:
        """Get the rank (number of dimensions) of the tensor.

        Returns:
            int: Number of dimensions if shape is known
            None: If tensor is unranked (shape is None)
        """
        if self.shape is None:
            return None
        return len(self.shape)

    def has_dynamic_dims(self) -> bool:
        """Check if tensor has any dynamic dimensions (-1)."""
        if self.shape is None:
            return True
        return any(dim == -1 for dim in self.shape)


Tensor = TensorType


class TableType(BaseType):
    """Represents a table with a named schema of types."""

    def __init__(self, schema: dict[str, BaseType]):
        self.schema = schema

    def __class_getitem__(cls, schema: dict[str, BaseType]):
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

    def __eq__(self, other) -> bool:
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

    def __class_getitem__(cls, kind: str):
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


class ScalarHEType(BaseType, ScalarTrait, EncryptedTrait):
    """Represents a single scalar value encrypted with an HE scheme."""

    def __init__(self, scalar_type: ScalarType, scheme: str = "ckks"):
        if not isinstance(scalar_type, ScalarType):
            raise TypeError(
                f"HE encryption is defined only over ScalarType, got {type(scalar_type).__name__}."
            )
        self._pt_type = scalar_type
        self._scheme = scheme

    def __class_getitem__(cls, params):
        if isinstance(params, tuple):
            return cls(params[0], params[1])
        return cls(params)

    def __str__(self) -> str:
        return f"HE[{self.pt_type}]"


HE = ScalarHEType


class SIMDHEType(BaseType, EncryptedTrait):
    """Represents a SINGLE ciphertext packing a vector of scalars using HE SIMD mode."""

    def __init__(
        self, scalar_type: ScalarType, packing_shape: tuple[int,], scheme: str = "ckks"
    ):
        if not isinstance(scalar_type, ScalarType):
            raise TypeError(
                f"SIMD_HE is defined only over ScalarType, got {type(scalar_type).__name__}."
            )
        self._pt_type = scalar_type
        self.packing_shape = packing_shape
        self._scheme = scheme

    @property
    def pt_type(self) -> BaseType:
        # Represent packed ciphertext's logical plaintext as a Tensor type.
        return TensorType(self._pt_type, self.packing_shape)

    @property
    def scalar_type(self) -> ScalarType:
        assert isinstance(self._pt_type, ScalarType)
        return self._pt_type

    def __class_getitem__(cls, params: tuple[ScalarType, tuple[int,]]):
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

    def __class_getitem__(cls, secret_type: BaseType):
        """Enables the syntax `SS[Tensor[...]]`."""
        return cls(secret_type)

    def __str__(self) -> str:
        return f"SS[{self.pt_type}]"


SS = SSType

# ==============================================================================
# --- Pillar 3: Distribution Types
# ==============================================================================


class MPType(BaseType):
    """Represents a logical value distributed among multiple parties."""

    def __init__(self, value_type: BaseType, parties: tuple[int, ...]):
        self._value_type = value_type
        self._parties = parties

    @property
    def value_type(self) -> BaseType:
        return self._value_type

    @property
    def parties(self) -> tuple[int, ...]:
        return self._parties

    def __class_getitem__(cls, params: tuple[BaseType, tuple[int, ...]]):
        """Enables the syntax `MP[Tensor[...], (0, 1)]`."""
        value_type, parties = params
        return cls(value_type, parties)

    def __str__(self) -> str:
        return f"MP[{self.value_type}, parties={self.parties}]"


MP = MPType


# ==============================================================================
# --- Example Usage and Verification of Design
# ==============================================================================

if __name__ == "__main__":
    print("--- MPLang Typing System Demonstration & Verification ---")

    print("\n[Tensor Shape Variations]")
    unranked = Tensor[i32, None]
    print(f"  - Unranked/fully dynamic: {unranked}")

    scalar = Tensor[i32, ()]
    print(f"  - Scalar (0-dim tensor): {scalar}")

    partial_dynamic = Tensor[i32, (-1, 10)]
    print(f"  - Partial dynamic (batch, features): {partial_dynamic}")

    fully_ranked = Tensor[i32, (3, 10)]
    print(f"  - Fully ranked/static: {fully_ranked}")

    print("\n[World 1: Plaintext]")
    plain_tensor = Tensor[f32, (10, 20)]
    print(f"  - Plaintext Tensor: {plain_tensor}")

    print("\n[World 2: Element-wise HE]")
    elementwise_he_tensor = Tensor[HE[f64], (100,)]
    print(f"  - Element-wise HE Tensor: {elementwise_he_tensor}")

    print("\n[World 3: Packed (SIMD) HE]")
    simd_he_vector = SIMD_HE[f32, (4096,)]
    print(f"  - SIMD HE Vector: {simd_he_vector}")

    print("\n[Verifying Design Constraint]")
    try:
        invalid_tensor = Tensor[simd_he_vector, (4,)]
    except TypeError as e:
        print("  - SUCCESS: Caught invalid type construction as expected.")
        print(f"    Error: {e}")

    print("\n[Secret Sharing]")
    ss_tensor_share = SS[Tensor[i32, (5, 5)]]
    print(f"  - A single share of a secret tensor: {ss_tensor_share}")

    print("\n[Composition with Distribution]")
    mp_ss_tensor = MP[ss_tensor_share, (0, 1)]
    print(f"  - Logical secret-shared tensor: {mp_ss_tensor}")

    mp_simd_he_vector = MP[simd_he_vector, (2,)]
    print(f"  - SIMD HE vector held by Party 2: {mp_simd_he_vector}")

    print("\n[Dynamic Shape in Distributed Settings]")
    dynamic_ss = SS[Tensor[f32, (-1, 128)]]
    mp_dynamic = MP[dynamic_ss, (0, 1, 2)]
    print(f"  - Dynamic batch size, distributed: {mp_dynamic}")

    print("\n[Using 'EncryptedTrait']")

    def check_encryption(t):  # runtime helper; skip strict typing for demo
        if isinstance(t, EncryptedTrait):
            # For SS types, show enc_schema; for HE/SIMD_HE, just show it's encrypted
            if hasattr(t, "enc_schema"):
                print(
                    f"  - '{t}' is Encrypted. Plaintext type: {t.pt_type}, schema: {t.enc_schema}"
                )
            else:
                print(f"  - '{t}' is Encrypted. Plaintext type: {t.pt_type}")
        else:
            print(f"  - '{t}' is NOT Encrypted.")

    check_encryption(simd_he_vector)
    check_encryption(ss_tensor_share)
    check_encryption(plain_tensor)
    # Check HE[f64] itself
    check_encryption(HE[f64])
