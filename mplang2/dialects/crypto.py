"""Crypto dialect for the EDSL.

Provides cryptographic primitives including ECC, Hashing, and Symmetric Encryption.
"""

from __future__ import annotations

import mplang2.edsl as el
import mplang2.edsl.typing as elt

# ==============================================================================
# --- Type Definitions
# ==============================================================================


class PointType(elt.BaseType):
    """Type for an ECC Point."""

    def __init__(self, curve: str = "secp256k1"):
        self.curve = curve

    def __str__(self) -> str:
        return f"Point[{self.curve}]"


class ScalarType(elt.BaseType):
    """Type for an ECC Scalar (integer modulo curve order)."""

    def __init__(self, curve: str = "secp256k1"):
        self.curve = curve

    def __str__(self) -> str:
        return f"Scalar[{self.curve}]"


# ==============================================================================
# --- Primitives
# ==============================================================================

# ECC
generator_p = el.Primitive("crypto.ec_generator")
mul_p = el.Primitive("crypto.ec_mul")
add_p = el.Primitive("crypto.ec_add")
sub_p = el.Primitive("crypto.ec_sub")
point_to_bytes_p = el.Primitive("crypto.ec_point_to_bytes")
random_scalar_p = el.Primitive("crypto.ec_random_scalar")
scalar_from_int_p = el.Primitive("crypto.ec_scalar_from_int")

# Symmetric / Hash
hash_p = el.Primitive("crypto.hash")
sym_encrypt_p = el.Primitive("crypto.sym_encrypt")
sym_decrypt_p = el.Primitive("crypto.sym_decrypt")
select_p = el.Primitive("crypto.select")


# ==============================================================================
# --- Abstract Evaluation (Type Inference)
# ==============================================================================


@generator_p.def_abstract_eval
def _generator_ae(curve: str = "secp256k1") -> PointType:
    return PointType(curve)


@mul_p.def_abstract_eval
def _mul_ae(point: PointType, scalar: ScalarType) -> PointType:
    return PointType(point.curve)


@add_p.def_abstract_eval
def _add_ae(p1: PointType, p2: PointType) -> PointType:
    return PointType(p1.curve)


@sub_p.def_abstract_eval
def _sub_ae(p1: PointType, p2: PointType) -> PointType:
    return PointType(p1.curve)


@point_to_bytes_p.def_abstract_eval
def _pt_to_bytes_ae(point: PointType) -> elt.TensorType:
    return elt.TensorType(elt.u8, (64,))


@random_scalar_p.def_abstract_eval
def _random_scalar_ae(curve: str = "secp256k1") -> ScalarType:
    return ScalarType(curve)


@scalar_from_int_p.def_abstract_eval
def _scalar_from_int_ae(val: elt.BaseType, curve: str = "secp256k1") -> ScalarType:
    return ScalarType(curve)


@hash_p.def_abstract_eval
def _hash_ae(data: elt.BaseType) -> elt.TensorType:
    return elt.TensorType(elt.u8, (32,))


@sym_encrypt_p.def_abstract_eval
def _sym_encrypt_ae(key: elt.BaseType, plaintext: elt.BaseType) -> elt.TensorType:
    # Dynamic shape for ciphertext
    return elt.TensorType(elt.u8, (-1,))


@sym_decrypt_p.def_abstract_eval
def _sym_decrypt_ae(
    key: elt.BaseType, ciphertext: elt.BaseType, target_type: elt.BaseType
) -> elt.BaseType:
    return target_type


@select_p.def_abstract_eval
def _select_ae(
    cond: elt.BaseType, true_val: elt.BaseType, false_val: elt.BaseType
) -> elt.BaseType:
    return true_val


# ==============================================================================
# --- Helper Functions (Ops)
# ==============================================================================


def ec_generator(curve: str = "secp256k1") -> el.graph.Value:
    """Get the generator point G for the curve."""
    return generator_p.bind(curve=curve)


def ec_mul(point: el.graph.Value, scalar: el.graph.Value) -> el.graph.Value:
    """Scalar multiplication: point * scalar."""
    return mul_p.bind(point, scalar)


def ec_add(p1: el.graph.Value, p2: el.graph.Value) -> el.graph.Value:
    """Point addition: p1 + p2."""
    return add_p.bind(p1, p2)


def ec_sub(p1: el.graph.Value, p2: el.graph.Value) -> el.graph.Value:
    """Point subtraction: p1 - p2."""
    return sub_p.bind(p1, p2)


def ec_point_to_bytes(point: el.graph.Value) -> el.graph.Value:
    """Serialize point to bytes."""
    return point_to_bytes_p.bind(point)


def ec_random_scalar(curve: str = "secp256k1") -> el.graph.Value:
    """Generate a random scalar."""
    return random_scalar_p.bind(curve=curve)


def ec_scalar_from_int(val: el.graph.Value, curve: str = "secp256k1") -> el.graph.Value:
    """Convert an integer tensor to a scalar."""
    return scalar_from_int_p.bind(val, curve=curve)


def hash_bytes(data: el.graph.Value) -> el.graph.Value:
    """Hash bytes (SHA256). Returns 32-byte tensor."""
    return hash_p.bind(data)


def sym_encrypt(key: el.graph.Value, plaintext: el.graph.Value) -> el.graph.Value:
    """Symmetric encrypt (XOR stream or AES-GCM)."""
    return sym_encrypt_p.bind(key, plaintext)


def sym_decrypt(
    key: el.graph.Value, ciphertext: el.graph.Value, target_type: elt.BaseType
) -> el.graph.Value:
    """Symmetric decrypt."""
    return sym_decrypt_p.bind(key, ciphertext, target_type=target_type)


def select(
    cond: el.graph.Value, true_val: el.graph.Value, false_val: el.graph.Value
) -> el.graph.Value:
    """Select between two values based on condition."""
    return select_p.bind(cond, true_val, false_val)
