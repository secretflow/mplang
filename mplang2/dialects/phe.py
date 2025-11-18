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

from typing import Any

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
    """Encrypt plaintext scalar using PHE public key."""
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
    return ct.pt_type


# ==============================================================================
# --- Element-level Homomorphic Operations
# ==============================================================================

add_p = el.Primitive("phe.add")
mul_scalar_p = el.Primitive("phe.mul_scalar")


@add_p.def_abstract_eval
def _add_ae(operand1: elt.ScalarHEType, operand2: elt.ScalarHEType) -> elt.ScalarHEType:
    """Element-level homomorphic addition.

    Args:
        operand1: First encrypted scalar (HE[T])
        operand2: Second encrypted scalar (HE[T])

    Returns:
        Encrypted result (HE[T])

    Raises:
        TypeError: If operands are not HE-encrypted or have mismatched types
    """
    if not isinstance(operand1, elt.ScalarHEType):
        raise TypeError(f"phe.add expects HE[...] operands, got {operand1}")
    if not isinstance(operand2, elt.ScalarHEType):
        raise TypeError(f"phe.add expects HE[...] operands, got {operand2}")

    # Verify underlying scalar types match
    if operand1.pt_type != operand2.pt_type:
        raise TypeError(
            f"Type mismatch: HE[{operand1.pt_type}] vs HE[{operand2.pt_type}]"
        )

    # Return same type as input
    return operand1


@mul_scalar_p.def_abstract_eval
def _mul_scalar_ae(
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
    if not isinstance(ciphertext, elt.ScalarHEType):
        raise TypeError(f"phe.mul_scalar expects HE[...] ciphertext, got {ciphertext}")
    if not isinstance(plaintext, elt.ScalarType):
        raise TypeError(f"phe.mul_scalar expects ScalarType plaintext, got {plaintext}")

    # Verify scalar types match
    if ciphertext.pt_type != plaintext:
        raise TypeError(
            f"Type mismatch: HE[{ciphertext.pt_type}] vs plaintext {plaintext}"
        )

    # Return ciphertext type
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

    return keygen_p.bind(**attrs)  # type: ignore[return-value]


def _has_tensor_args(*objs: el.Object) -> bool:
    """Check whether any argument carries a TensorType."""
    return any(isinstance(obj.type, elt.TensorType) for obj in objs)


def encrypt(plaintext: el.Object, public_key: el.Object) -> el.Object:
    """Encrypt plaintext using the PHE public key.

    Tensor inputs trigger `tensor.elementwise`; scalar inputs bind `encrypt_p`
    directly.
    """
    if _has_tensor_args(plaintext):
        return tensor.elementwise(encrypt_p.bind, plaintext, public_key)
    return encrypt_p.bind(plaintext, public_key)


def decrypt(ciphertext: el.Object, private_key: el.Object) -> el.Object:
    """Decrypt ciphertext using the PHE private key."""
    if _has_tensor_args(ciphertext):
        return tensor.elementwise(decrypt_p.bind, ciphertext, private_key)
    return decrypt_p.bind(ciphertext, private_key)


def add(lhs: el.Object, rhs: el.Object) -> el.Object:
    """Homomorphic addition (tensor-aware wrapper)."""
    if _has_tensor_args(lhs, rhs):
        return tensor.elementwise(add_p.bind, lhs, rhs)
    return add_p.bind(lhs, rhs)


def mul_scalar(ciphertext: el.Object, plaintext: el.Object) -> el.Object:
    """Homomorphic scalar multiplication (tensor-aware wrapper)."""
    if _has_tensor_args(ciphertext, plaintext):
        return tensor.elementwise(mul_scalar_p.bind, ciphertext, plaintext)
    return mul_scalar_p.bind(ciphertext, plaintext)


__all__ = [
    "PHEPrivateKeyType",
    "PHEPublicKeyType",
    "add",
    "add_p",
    "decrypt",
    "decrypt_p",
    "encrypt",
    "encrypt_p",
    "keygen",
    "keygen_p",
    "mul_scalar",
    "mul_scalar_p",
]
