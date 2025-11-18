"""PHE (Partially Homomorphic Encryption) dialect for the EDSL.

Design principles:
- Keep the dialect purely element-level; primitives talk about encrypted scalars.
- Reuse `tensor.elementwise` to lift those primitives across tensors when needed.
- Provide ergonomic wrappers that work for both scalar and tensor operands.

Example:
```python
from mplang2.dialects import tensor, phe
import numpy as np

pk, sk = phe.keygen()
x = tensor.constant(np.array([1.0, 2.0, 3.0]))
y = tensor.constant(np.array([4.0, 5.0, 6.0]))
scale = tensor.constant(2.0)

ct_x = phe.encrypt(x, pk)  # Tensor[HE[f64], (3,)]
ct_y = phe.encrypt(y, pk)  # Tensor[HE[f64], (3,)]
ct_sum = phe.add(ct_x, ct_y)  # Tensor[HE[f64], (3,)]
ct_scaled = phe.mul_scalar(ct_sum, scale)  # Tensor[HE[f64], (3,)]
result = phe.decrypt(ct_scaled, sk)  # Tensor[f64, (3,)]
```

Wrappers such as `phe.encrypt` are polymorphic: if any argument is a tensor,
they call `tensor.elementwise` under the hood; otherwise they bind the element
primitive directly for scalar workflows.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import mplang2.edsl as el
import mplang2.edsl.typing as elt
from mplang2.dialects import tensor

# ==============================================================================
# --- Type Definitions
# ==============================================================================

# Opaque key types for PHE (singleton instances)
PHEPublicKeyType: elt.CustomType = elt.CustomType("PHEPublicKey")
PHEPrivateKeyType: elt.CustomType = elt.CustomType("PHEPrivateKey")

# ==============================================================================
# --- Key Management Operations
# ==============================================================================

keygen_p = el.Primitive("phe.keygen")
encrypt_p = el.Primitive("phe.encrypt")
decrypt_p = el.Primitive("phe.decrypt")


@keygen_p.def_abstract_eval
def _keygen_ae(
    *,
    scheme: str = "paillier",
    key_size: int = 2048,
    max_value: int | None = None,
    fxp_bits: int | None = None,
) -> tuple[elt.CustomType, elt.CustomType]:
    """Generate PHE key pair.

    Args:
        scheme: PHE scheme name
        key_size: Key size in bits
        max_value: Optional range-encoding bound
        fxp_bits: Optional fixed-point fractional bits

    Returns:
        Tuple of (PHEPublicKeyType, PHEPrivateKeyType) singleton instances
    """
    return (PHEPublicKeyType, PHEPrivateKeyType)


@encrypt_p.def_abstract_eval
def _encrypt_ae(pt: elt.ScalarType, pkey: elt.CustomType) -> elt.ScalarHEType:
    """Encrypt plaintext scalar using the PHE public key."""
    if not isinstance(pt, elt.ScalarType):
        raise TypeError(f"encrypt expects ScalarType plaintext, got {pt}")
    if pkey != PHEPublicKeyType:
        raise TypeError(f"encrypt expects PHEPublicKey, got {pkey}")
    return elt.ScalarHEType(pt)


@decrypt_p.def_abstract_eval
def _decrypt_ae(ct: elt.ScalarHEType, sk: elt.CustomType) -> elt.ScalarType:
    """Decrypt ciphertext scalar using PHE private key."""
    if not isinstance(ct, elt.ScalarHEType):
        raise TypeError(f"decrypt expects HE[...] input, got {ct}")
    if sk != PHEPrivateKeyType:
        raise TypeError(f"decrypt expects PHEPrivateKey, got {sk}")
    pt_type = ct.pt_type
    if not isinstance(pt_type, elt.ScalarType):
        raise TypeError(f"HE ciphertext carries non-Scalar plaintext type {pt_type}")
    return pt_type


# ==============================================================================
# --- Element-level Homomorphic Operations
# ==============================================================================

add_cc_p = el.Primitive("phe.add_cc")
add_cp_p = el.Primitive("phe.add_cp")
mul_cp_p = el.Primitive("phe.mul_cp")


@add_cc_p.def_abstract_eval
def _add_cc_ae(
    operand1: elt.ScalarHEType, operand2: elt.ScalarHEType
) -> elt.ScalarHEType:
    """Ciphertext + ciphertext → ciphertext."""
    if operand1.pt_type != operand2.pt_type:
        raise TypeError(
            f"Type mismatch: HE[{operand1.pt_type}] vs HE[{operand2.pt_type}]"
        )
    return operand1


@add_cp_p.def_abstract_eval
def _add_cp_ae(
    ciphertext: elt.ScalarHEType, plaintext: elt.ScalarType
) -> elt.ScalarHEType:
    """Ciphertext + plaintext → ciphertext."""
    if ciphertext.pt_type != plaintext:
        raise TypeError(
            f"Type mismatch: HE[{ciphertext.pt_type}] vs plaintext {plaintext}"
        )
    return ciphertext


@mul_cp_p.def_abstract_eval
def _mul_cp_ae(
    ciphertext: elt.ScalarHEType, plaintext: elt.ScalarType
) -> elt.ScalarHEType:
    """Element-level homomorphic scalar multiplication.

    Args:
        ciphertext: Encrypted scalar (HE[T])
        plaintext: Plaintext scalar (T)

    Returns:
        Encrypted result (HE[T])

    Raises:
        TypeError: If types are incompatible
    """
    if ciphertext.pt_type != plaintext:
        raise TypeError(
            f"Type mismatch: HE[{ciphertext.pt_type}] vs plaintext {plaintext}"
        )
    return ciphertext


# ==============================================================================
# --- User-facing API
# ==============================================================================


def keygen(
    scheme: str = "paillier",
    key_size: int = 2048,
    max_value: int | None = None,
    fxp_bits: int | None = None,
) -> tuple[el.Object, el.Object]:
    """Generate PHE key pair.

    Args:
        scheme: PHE scheme name (default: "paillier")
                Supported schemes: "paillier", "elgamal", "okamoto-uchiyama"
        key_size: Key size in bits (default: 2048)
                  Larger keys provide more security but slower computation
        max_value: Optional range-encoding bound B for integers/floats.
                  If provided, values are encoded in range [-B, B].
                  Choose B larger than expected computation results.
        fxp_bits: Optional fixed-point fractional bits for float encoding.
                  Higher values provide more precision but reduce range.

    Returns:
        Tuple of (public_key, private_key)

    Example:
        >>> # Basic usage with defaults
        >>> pk, sk = keygen()
        >>>
        >>> # Custom key size for higher security
        >>> pk, sk = keygen(key_size=4096)
        >>>
        >>> # With range encoding for large computations
        >>> pk, sk = keygen(max_value=2**64, fxp_bits=16)
    """
    # Build attrs dict
    attrs: dict[str, Any] = {
        "scheme": scheme,
        "key_size": key_size,
    }
    if max_value is not None:
        attrs["max_value"] = max_value
    if fxp_bits is not None:
        attrs["fxp_bits"] = fxp_bits

    return keygen_p.bind(**attrs)


def _has_tensor_args(*objs: el.Object) -> bool:
    """Check whether any argument carries a TensorType."""
    return any(isinstance(obj.type, elt.TensorType) for obj in objs)


class OperandInfo(NamedTuple):
    """Classification of operand for PHE operation dispatch."""

    is_tensor: bool
    is_encrypted: bool
    scalar_type: elt.BaseType


def _inspect_operand(obj: el.Object) -> OperandInfo:
    """Classify operand layout/security for dispatch."""
    obj_type = obj.type
    if isinstance(obj_type, elt.TensorType):
        elem = obj_type.element_type
        if isinstance(elem, elt.ScalarHEType):
            return OperandInfo(True, True, elem.pt_type)
        if isinstance(elem, elt.ScalarType):
            return OperandInfo(True, False, elem)
        raise TypeError(
            f"PHE operations support Tensor[ScalarType] or Tensor[HE[...]], got Tensor[{elem}]"
        )
    if isinstance(obj_type, elt.ScalarHEType):
        return OperandInfo(False, True, obj_type.pt_type)
    if isinstance(obj_type, elt.ScalarType):
        return OperandInfo(False, False, obj_type)
    raise TypeError(f"PHE operations expect Scalar or Tensor operands, got {obj_type}")


BinaryFn = Callable[[el.Object, el.Object], Any]


def _apply_binary(fn: BinaryFn, lhs: el.Object, rhs: el.Object) -> el.Object:
    """Apply scalar primitive, lifting to tensor.elementwise when needed."""
    if _has_tensor_args(lhs, rhs):
        return tensor.elementwise(fn, lhs, rhs)
    return fn(lhs, rhs)


def _add_cp(ciphertext: el.Object, plaintext: el.Object) -> el.Object:
    """Ciphertext ⊕ plaintext helper (order enforced)."""
    return _apply_binary(add_cp_p.bind, ciphertext, plaintext)


def _mul_cp(ciphertext: el.Object, plaintext: el.Object) -> el.Object:
    """Ciphertext ⊗ plaintext helper (order enforced)."""
    return _apply_binary(mul_cp_p.bind, ciphertext, plaintext)


def _encrypt_scalar_element(plaintext: el.Object, public_key: el.Object) -> el.Object:
    """Encrypt a single scalar element."""
    return encrypt_p.bind(plaintext, public_key)


def encrypt(plaintext: Any, public_key: Any) -> el.Object:
    """Encrypt plaintext using the PHE public key.

    Type transformations:
        Scalar[T] → HE[T]
        Tensor[T, S] → Tensor[HE[T], S]

    Args:
        plaintext: Scalar or tensor to encrypt
        public_key: PHE public key

    Returns:
        Encrypted result with same structure as input
    """
    if _has_tensor_args(plaintext):
        return tensor.elementwise(_encrypt_scalar_element, plaintext, public_key)
    return encrypt_p.bind(plaintext, public_key)


def decrypt(ciphertext: Any, private_key: Any) -> el.Object:
    """Decrypt ciphertext using the PHE private key."""
    if _has_tensor_args(ciphertext):
        return tensor.elementwise(decrypt_p.bind, ciphertext, private_key)
    return decrypt_p.bind(ciphertext, private_key)


def add(lhs: el.Object, rhs: el.Object) -> el.Object:
    """Homomorphic addition.

    Supports:
        HE[T] + HE[T] → HE[T]  (ciphertext + ciphertext)
        HE[T] + T → HE[T]      (ciphertext + plaintext)
        T + HE[T] → HE[T]      (plaintext + ciphertext)

    Args:
        lhs: Left operand (encrypted or plaintext)
        rhs: Right operand (encrypted or plaintext)

    Returns:
        Encrypted sum

    Raises:
        TypeError: If no operand is encrypted or types mismatch
    """
    lhs_info = _inspect_operand(lhs)
    rhs_info = _inspect_operand(rhs)

    if not (lhs_info.is_encrypted or rhs_info.is_encrypted):
        raise TypeError("phe.add requires at least one ciphertext operand")

    # CT + CT
    if lhs_info.is_encrypted and rhs_info.is_encrypted:
        return _apply_binary(add_cc_p.bind, lhs, rhs)

    # CT + PT or PT + CT
    if lhs_info.is_encrypted:
        return _add_cp(lhs, rhs)
    return _add_cp(rhs, lhs)


def mul(lhs: el.Object, rhs: el.Object) -> el.Object:
    """Homomorphic multiplication (ciphertext × plaintext only).

    Supports:
        HE[T] × T → HE[T]  (ciphertext × plaintext)
        T × HE[T] → HE[T]  (plaintext × ciphertext)

    Args:
        lhs: Left operand
        rhs: Right operand

    Returns:
        Encrypted product

    Raises:
        TypeError: If both operands are encrypted or both are plaintext

    Note:
        Ciphertext × ciphertext is not supported (would require FHE).
    """
    lhs_info = _inspect_operand(lhs)
    rhs_info = _inspect_operand(rhs)

    # CT * PT
    if lhs_info.is_encrypted and not rhs_info.is_encrypted:
        return _mul_cp(lhs, rhs)
    # PT * CT
    if rhs_info.is_encrypted and not lhs_info.is_encrypted:
        return _mul_cp(rhs, lhs)
    # CT * CT (not supported)
    if lhs_info.is_encrypted and rhs_info.is_encrypted:
        raise TypeError("phe.mul supports ciphertext * plaintext only, not CT * CT")
    # PT * PT (invalid)
    raise TypeError("phe.mul requires at least one ciphertext operand")


def mul_scalar(ciphertext: el.Object, plaintext: el.Object) -> el.Object:
    """Backward-compatible wrapper for ciphertext × plaintext multiplication."""
    return mul(ciphertext, plaintext)


__all__ = [
    "PHEPrivateKeyType",
    "PHEPublicKeyType",
    "add",
    "add_cc_p",
    "add_cp_p",
    "decrypt",
    "decrypt_p",
    "encrypt",
    "encrypt_p",
    "keygen",
    "keygen_p",
    "mul",
    "mul_cp_p",
    "mul_scalar",
]
