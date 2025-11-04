# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MPLang Type System: Class-based AST for type expressions.

This module implements a composable, extensible type system where types form
an immutable AST (expression tree). Pure Expr AST design - all types directly
inherit from Type ABC:
- Base value types: TensorType, TableType
- Representation types: EncodedType, EncryptedType, SecretSharedType
- Multi-party type: MpType (describes multi-party distribution with pmask)

All types inherit from the abstract base class `Type`, and can compose
recursively (e.g., MpType(Encrypted(Encoded(Tensor(...))))).

Minimal predicates/utilities (is_tensor, unwrap_repr, same_dtype, etc.) enable
practical verification at the Python level for ops/primitives.

Design principles:
- Representation types directly inherit Type (standard Expr AST pattern)
- Party count encoded in MpType.pmask, not in SecretSharedType
- Callers explicitly unwrap type trees when needed
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mplang.core.dtypes import DType

if TYPE_CHECKING:
    from mplang.core.mask import Mask
else:
    # Runtime: import Mask to avoid AttributeError
    from mplang.core.mask import Mask

__all__ = [
    "EncodedType",
    "EncryptedType",
    "MpType",
    "SecretSharedType",
    "TableType",
    "TensorType",
    "Type",
    "is_mp",
    "is_table",
    "is_tensor",
    "same_dtype",
    "same_schema",
    "same_security",
    "same_shape",
    "unwrap_repr",
]


# ============================================================================
# Abstract base class
# ============================================================================


class Type(ABC):
    """Abstract base class for all type expressions.

    All types are immutable and hashable, forming nodes in a type-level AST.
    Concrete types include base value types (TensorType, TableType) and
    wrapper types (EncodedType, EncryptedType, SecretSharedType, etc.).
    """

    @abstractmethod
    def __repr__(self) -> str:
        """Return a readable string representation of the type."""
        ...

    @abstractmethod
    def __hash__(self) -> int:
        """Return a stable hash for use in dicts/sets."""
        ...

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Check structural equality."""
        ...


# ============================================================================
# Base value types (materialized types)
# ============================================================================


@dataclass(frozen=True)
class TensorType(Type):
    """Tensor type with dtype and shape.

    Shape supports dynamic dimensions (e.g., -1 for unknown size).
    All dimensions must be integers.

    Example:
        >>> TensorType(DType.f32(), (3, 4))
        f32[3, 4]
    """

    dtype: DType
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, DType):
            raise TypeError(f"TensorType.dtype must be DType, got {type(self.dtype)}")
        if not all(isinstance(d, int) for d in self.shape):
            raise TypeError("TensorType.shape dims must all be ints")

    def __repr__(self) -> str:
        shape_str = ", ".join(str(d) for d in self.shape)
        return (
            f"{self.dtype.short_name()}[{shape_str}]"
            if self.shape
            else self.dtype.short_name()
        )


@dataclass(frozen=True)
class TableType(Type):
    """Table type with ordered schema (column name-type pairs).

    Schema must be non-empty, column names must be unique and non-empty,
    and all column types must be DType.

    Example:
        >>> TableType((("id", DType.i64()), ("name", DType.string())))
        Tbl(id:i64, name:str)
    """

    columns: tuple[tuple[str, DType], ...]

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("TableType schema cannot be empty")
        names = [n for n, _ in self.columns]
        if not all(n for n in names):
            raise ValueError("TableType column names cannot be empty")
        if len(names) != len(set(names)):
            raise ValueError("TableType column names must be unique")
        for n, t in self.columns:
            if not isinstance(t, DType):
                raise TypeError(
                    f"TableType column {n!r} type must be DType, got {type(t)}"
                )

    def __repr__(self) -> str:
        cols_str = ", ".join(f"{n}:{t.short_name()}" for n, t in self.columns)
        return f"Tbl({cols_str})"


# ============================================================================
# Multi-party type (describes distribution with pmask)
# ============================================================================


@dataclass(frozen=True)
class MpType(Type):
    """Multi-party type describing distributed data held by multiple parties.

    MpType wraps an inner type and adds party mask (pmask) to describe which
    parties hold/execute the data. The inner type can be a base type (Tensor/Table)
    or a representation wrapper (Enc/HE/SS).

    Attributes:
        inner: The inner type (can be base or wrapped).
        pmask: Party mask indicating which parties hold the data (None = runtime-decided).
        quals: Optional qualifiers (e.g., device placement, execution hints).

    Example:
        >>> MpType(TensorType(DType.i64(), (10,)), pmask=Mask.from_int(0b111))
        Mp<0x7>(i64[10])
    """

    inner: Type
    pmask: Mask | None = None
    quals: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.inner, Type):
            raise TypeError(
                f"MpType.inner must be a Type, got {type(self.inner).__name__}"
            )
        # Normalize quals to None if empty
        if self.quals is not None and not self.quals:
            object.__setattr__(self, "quals", None)

    def __repr__(self) -> str:
        pmask_str = f"<{self.pmask:X}>" if self.pmask is not None else ""
        quals_str = ""
        if self.quals:
            items = [f"{k}={v!r}" for k, v in sorted(self.quals.items())]
            quals_str = "{" + ", ".join(items) + "}"
        return f"Mp{pmask_str}({self.inner!r}){quals_str}"


# ============================================================================
# Representation types (encoded, encrypted, secret-shared)
# ============================================================================


def _freeze_params(
    params: Mapping[str, object] | None,
) -> tuple[tuple[str, object], ...]:
    """Normalize params dict to a sorted tuple for immutability and stable hashing."""
    if not params:
        return ()
    return tuple(sorted(params.items(), key=lambda kv: kv[0]))


@dataclass(frozen=True)
class EncodedType(Type):
    """Type expression for encoded values (e.g., fixed-point encoding).

    Attributes:
        inner: The underlying type being encoded.
        codec: Short string describing the encoding scheme (e.g., 'fixed', 'zkp').
        params: Optional parameters (e.g., scale, precision), frozen as sorted tuple.

    Example:
        >>> EncodedType(
        ...     TensorType(DType.i64(), (1024,)), codec="fixed", params={"scale": 16}
        ... )
        Enc{scale=16}(i64[1024]; fixed)
    """

    inner: Type
    codec: str
    params: tuple[tuple[str, object], ...] = ()

    def __init__(
        self,
        inner: Type,
        codec: str,
        *,
        params: Mapping[str, object] | None = None,
    ):
        if not isinstance(inner, Type):
            raise TypeError(
                f"EncodedType.inner must be a Type, got {type(inner).__name__}"
            )
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "codec", codec)
        object.__setattr__(self, "params", _freeze_params(params))

    def __repr__(self) -> str:
        p = (
            "{" + ", ".join(f"{k}={v!r}" for k, v in self.params) + "}"
            if self.params
            else ""
        )
        return f"Enc{p}({self.inner!r}; {self.codec})"


@dataclass(frozen=True)
class EncryptedType(Type):
    """Type expression for homomorphically encrypted values.

    Attributes:
        inner: The underlying type.
        scheme: HE scheme (e.g., 'bfv', 'ckks').
        key_ref: Opaque reference to key material (optional).
        params: Optional parameters (e.g., n=16384 for poly modulus degree).

    Example:
        >>> EncryptedType(
        ...     TensorType(DType.f64(), (2048,)), scheme="ckks", params={"n": 16384}
        ... )
        HE{n=16384}(f64[2048]; ckks)
    """

    inner: Type
    scheme: str
    key_ref: str | None = None
    params: tuple[tuple[str, object], ...] = ()

    def __init__(
        self,
        inner: Type,
        scheme: str,
        *,
        key_ref: str | None = None,
        params: Mapping[str, object] | None = None,
    ):
        if not isinstance(inner, Type):
            raise TypeError(
                f"EncryptedType.inner must be a Type, got {type(inner).__name__}"
            )
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "key_ref", key_ref)
        object.__setattr__(self, "params", _freeze_params(params))

    def __repr__(self) -> str:
        parts = []
        if self.params:
            parts.append("{" + ", ".join(f"{k}={v!r}" for k, v in self.params) + "}")
        parts_str = "".join(parts)
        key_str = f", key={self.key_ref}" if self.key_ref else ""
        return f"HE{parts_str}({self.inner!r}; {self.scheme}{key_str})"


@dataclass(frozen=True)
class SecretSharedType(Type):
    """Type expression for MPC secret-shared values.

    Party count is encoded in the outer MpType's pmask.
    Use MpType(SecretSharedType(...), pmask) to represent which parties hold shares.

    Attributes:
        inner: The underlying type.
        scheme: MPC scheme (e.g., '3pc', '2pc', 'aby3').
        field_bits: Optional field size in bits (e.g., 64, 128).
        params: Additional scheme parameters.

    Example:
        >>> SecretSharedType(
        ...     TensorType(DType.i64(), (10, 10)),
        ...     scheme="aby3",
        ...     field_bits=64,
        ... )
        SS{field_bits=64}(i64[10, 10]; aby3)
    """

    inner: Type
    scheme: str
    field_bits: int | None = None
    params: tuple[tuple[str, object], ...] = ()

    def __init__(
        self,
        inner: Type,
        scheme: str,
        *,
        field_bits: int | None = None,
        params: Mapping[str, object] | None = None,
    ):
        if not isinstance(inner, Type):
            raise TypeError(
                f"SecretSharedType.inner must be a Type, got {type(inner).__name__}"
            )
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "field_bits", field_bits)
        object.__setattr__(self, "params", _freeze_params(params))

    def __repr__(self) -> str:
        attrs = []
        if self.field_bits is not None:
            attrs.append(f"field_bits={self.field_bits}")
        if self.params:
            for k, v in self.params:
                attrs.append(f"{k}={v!r}")
        attr_str = "{" + ", ".join(attrs) + "}" if attrs else ""
        return f"SS{attr_str}({self.inner!r}; {self.scheme})"


# ============================================================================
# Minimal predicates and utilities
# ============================================================================


def is_tensor(ty: Type) -> bool:
    """Check if type is exactly TensorType (no unwrapping)."""
    return isinstance(ty, TensorType)


def is_table(ty: Type) -> bool:
    """Check if type is exactly TableType (no unwrapping)."""
    return isinstance(ty, TableType)


def is_mp(ty: Type) -> bool:
    """Check if type is exactly MpType (no unwrapping)."""
    return isinstance(ty, MpType)


def unwrap_repr(ty: Type) -> Type:
    """Remove all representation wrappers (Enc/HE/SS).

    Returns the inner type after removing all outermost representation wrappers.
    If no repr wrappers, returns the type unchanged.
    """
    cur = ty
    while isinstance(cur, (EncodedType, EncryptedType, SecretSharedType)):
        cur = cur.inner
    return cur


def same_dtype(a: Type, b: Type) -> bool:
    """Check if two types have the same base Tensor dtype.

    Unwraps MpType and representation wrappers to reach TensorType.
    """
    # Unwrap MpType if present
    ca = a.inner if isinstance(a, MpType) else a
    cb = b.inner if isinstance(b, MpType) else b
    # Unwrap representation wrappers
    ca = unwrap_repr(ca)
    cb = unwrap_repr(cb)
    # Check if both are TensorType with same dtype
    if not (isinstance(ca, TensorType) and isinstance(cb, TensorType)):
        return False
    return ca.dtype == cb.dtype


def same_shape(a: Type, b: Type) -> bool:
    """Check if two types have the same base Tensor shape.

    Unwraps MpType and representation wrappers to reach TensorType.
    """
    # Unwrap MpType if present
    ca = a.inner if isinstance(a, MpType) else a
    cb = b.inner if isinstance(b, MpType) else b
    # Unwrap representation wrappers
    ca = unwrap_repr(ca)
    cb = unwrap_repr(cb)
    # Check if both are TensorType with same shape
    if not (isinstance(ca, TensorType) and isinstance(cb, TensorType)):
        return False
    return ca.shape == cb.shape


def same_schema(a: Type, b: Type) -> bool:
    """Check if two types have the same base Table schema.

    Unwraps MpType and representation wrappers to reach TableType.
    """
    # Unwrap MpType if present
    ca = a.inner if isinstance(a, MpType) else a
    cb = b.inner if isinstance(b, MpType) else b
    # Unwrap representation wrappers
    ca = unwrap_repr(ca)
    cb = unwrap_repr(cb)
    # Check if both are TableType with same columns
    if not (isinstance(ca, TableType) and isinstance(cb, TableType)):
        return False
    return ca.columns == cb.columns


def _security_chain(
    ty: Type,
) -> tuple[tuple[str, str, tuple[tuple[str, object], ...]], ...]:
    """Extract the security/representation wrapper chain.

    Returns a tuple of (wrapper_class_name, scheme/codec, params) from outermost to innermost.
    Skips MpType if present, then collects all representation wrapper instances.
    """
    chain: list[tuple[str, str, tuple[tuple[str, object], ...]]] = []
    cur = ty
    # Skip MpType if present
    if isinstance(cur, MpType):
        cur = cur.inner
    # Now collect repr wrappers
    while isinstance(cur, (EncodedType, EncryptedType, SecretSharedType)):
        cls_name = cur.__class__.__name__
        if isinstance(cur, EncodedType):
            tag = cur.codec
            params = cur.params
        elif isinstance(cur, EncryptedType):
            tag = cur.scheme
            params = cur.params
        else:  # SecretSharedType
            tag = cur.scheme
            params = cur.params
        chain.append((cls_name, tag, params))
        cur = cur.inner
    return tuple(chain)


def same_security(a: Type, b: Type) -> bool:
    """Check if two types have identical security/representation wrapper chains.

    Compares the sequence of representation wrappers (Enc/HE/SS) by class name,
    scheme/codec, and params.
    """
    return _security_chain(a) == _security_chain(b)
