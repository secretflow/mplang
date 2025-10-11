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

"""FHE (Fully Homomorphic Encryption) frontend operations."""

from mplang.core.dtype import UINT8
from mplang.core.tensor import TensorType
from mplang.ops.base import stateless_mod

_FHE_MOD = stateless_mod("fhe")


@_FHE_MOD.simple_op()
def keygen(
    *,
    scheme: str = "CKKS",
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: tuple[int, ...] | None = None,
    global_scale: int | None = None,
    plain_modulus: int | None = None,
) -> tuple[TensorType, TensorType]:
    """Generate an FHE key pair: returns (private_context, public_context).

    Args:
        scheme: FHE scheme to use ("CKKS" for approximate, "BFV" for exact integer)
        poly_modulus_degree: Polynomial modulus degree (default: 8192)
        coeff_mod_bit_sizes: Coefficient modulus bit sizes for CKKS (optional)
        global_scale: Global scale for CKKS (optional)
        plain_modulus: Plain modulus for BFV (optional)

    Returns:
        Tuple of (private_context, public_context) represented as UINT8[(-1, 0)]

    Contexts are represented with a sentinel TensorType UINT8[(-1, 0)] to indicate
    non-structural, backend-only handles.
    """
    if scheme not in ("CKKS", "BFV"):
        raise ValueError("Unsupported scheme. Choose either 'CKKS' or 'BFV'.")
    if scheme == "CKKS":
        assert plain_modulus is None, "plain_modulus is not used in CKKS scheme."
    context_spec = TensorType(UINT8, (-1, 0))
    return context_spec, context_spec


@_FHE_MOD.simple_op()
def encrypt(plaintext: TensorType, context: TensorType) -> TensorType:
    """Encrypt plaintext using FHE context: returns ciphertext with same semantic type as plaintext.

    Args:
        plaintext: Data to encrypt (scalar, vector, or matrix)
        context: FHE context (private or public)

    Returns:
        Ciphertext with same semantic type as plaintext
    """
    _ = context
    return plaintext


@_FHE_MOD.simple_op()
def decrypt(ciphertext: TensorType, context: TensorType) -> TensorType:
    """Decrypt ciphertext using FHE context: returns plaintext with same semantic type as ciphertext.

    Args:
        ciphertext: Encrypted data to decrypt
        context: FHE context (must be private context with secret key)

    Returns:
        Plaintext with same semantic type as ciphertext
    """
    _ = context
    return ciphertext


@_FHE_MOD.simple_op()
def add(operand1: TensorType, operand2: TensorType) -> TensorType:
    """Add two FHE operands (ciphertext + ciphertext or ciphertext + plaintext).

    Args:
        operand1: First operand (ciphertext or plaintext)
        operand2: Second operand (ciphertext or plaintext)

    Returns:
        Result of homomorphic addition

    Note: At least one operand must be ciphertext.
    """
    assert (
        operand1.dtype == operand2.dtype
    ), f"Operand dtypes must match, got {operand1.dtype} and {operand2.dtype}."
    # TODO(zjj): it is indeed possible to add different shapes with broadcasting
    assert (
        operand1.shape == operand2.shape
    ), f"Operand shapes must match, got {operand1.shape} and {operand2.shape}."
    return operand1


@_FHE_MOD.simple_op()
def mul(operand1: TensorType, operand2: TensorType) -> TensorType:
    """Multiply two FHE operands (ciphertext * ciphertext or ciphertext * plaintext).

    Args:
        operand1: First operand (ciphertext or plaintext)
        operand2: Second operand (ciphertext or plaintext)

    Returns:
        Result of homomorphic multiplication

    Note: At least one operand must be ciphertext.
    For BFV scheme, plaintext operands must be integers.
    """
    assert (
        operand1.dtype == operand2.dtype
    ), f"Operand dtypes must match, got {operand1.dtype} and {operand2.dtype}."
    # TODO(zjj): it is indeed possible to add different shapes with broadcasting
    assert (
        operand1.shape == operand2.shape
    ), f"Operand shapes must match, got {operand1.shape} and {operand2.shape}."
    return operand1


@_FHE_MOD.simple_op()
def dot(operand1: TensorType, operand2: TensorType) -> TensorType:
    """Compute dot product of FHE operands (ciphertext · ciphertext or ciphertext · plaintext).

    Args:
        operand1: First operand (ciphertext or plaintext)
        operand2: Second operand (ciphertext or plaintext)

    Returns:
        Result of homomorphic dot product with computed shape following numpy rules

    Note: At least one operand must be ciphertext.
    TenSEAL supports dot product for tensors up to 2D×2D.
    """
    # Calculate result shape using numpy dot product rules
    import numpy as np

    # Create dummy arrays to determine result shape
    dummy_op1 = np.zeros(operand1.shape)
    dummy_op2 = np.zeros(operand2.shape)
    dummy_result = np.dot(dummy_op1, dummy_op2)

    return TensorType(operand1.dtype, dummy_result.shape)


@_FHE_MOD.simple_op()
def polyval(ciphertext: TensorType, coeffs: TensorType) -> TensorType:
    """Evaluate polynomial on encrypted data with plaintext coefficients.

    Args:
        ciphertext: Encrypted data (scalar, vector, or matrix)
        coeffs: Plaintext polynomial coefficients as 1D array [c0, c1, c2, ...]
                representing c0 + c1*x + c2*x^2 + ...

    Returns:
        Result of polynomial evaluation with same shape and dtype as ciphertext

    Note: Polynomial must have degree >= 1 (at least 2 coefficients required).
    For BFV scheme, coefficients must be integers.
    """
    _ = coeffs
    return ciphertext
