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

from mplang.v1.core import UINT8, TensorType
from mplang.v1.ops.base import stateless_mod

_fhe_MOD = stateless_mod("fhe")


@_fhe_MOD.simple_op()
def keygen(
    *,
    scheme: str = "CKKS",
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: tuple[int, ...] | None = None,
    global_scale: int | None = None,
    plain_modulus: int | None = None,
) -> tuple[TensorType, TensorType, TensorType]:
    """Generate an FHE key pair for Vector backend: returns (private_context, public_context, evaluation_context).

    Args:
        scheme: FHE scheme to use ("CKKS" for approximate, "BFV" for exact integer)
        poly_modulus_degree: Polynomial modulus degree (default: 8192)
        coeff_mod_bit_sizes: Coefficient modulus bit sizes for CKKS (optional)
        global_scale: Global scale for CKKS (optional)
        plain_modulus: Plain modulus for BFV (optional)

    Returns:
        Tuple of (private_context, public_context, evaluation_context) represented as UINT8[(-1, 0)]

    Contexts are represented with a sentinel TensorType UINT8[(-1, 0)] to indicate
    non-structural, backend-only handles.

    Note: Vector backend only supports 1D data. For multi-dimensional tensors,
    use mplang.ops.fhe instead.
    """
    if scheme not in ("CKKS", "BFV"):
        raise ValueError("Unsupported scheme. Choose either 'CKKS' or 'BFV'.")
    if scheme == "CKKS":
        assert plain_modulus is None, "plain_modulus is not used in CKKS scheme."
    context_spec = TensorType(UINT8, (-1, 0))
    return context_spec, context_spec, context_spec


@_fhe_MOD.simple_op()
def encrypt(plaintext: TensorType, context: TensorType) -> TensorType:
    """Encrypt plaintext using FHE Vector backend: returns ciphertext with same semantic type.

    Args:
        plaintext: Data to encrypt (scalar or 1D vector only)
        context: FHE context (private or public)

    Returns:
        Ciphertext with same semantic type as plaintext

    Raises:
        ValueError: If plaintext has more than 1 dimension

    Note: Vector backend only supports scalars (shape=()) and 1D vectors (shape=(n,)).
    For multi-dimensional data, use mplang.ops.fhe.encrypt instead.
    """
    _ = context
    if len(plaintext.shape) > 1:
        raise ValueError(
            f"FHE Vector backend only supports 1D data. Got shape {plaintext.shape}. "
            "Use mplang.ops.fhe for multi-dimensional tensors."
        )
    return plaintext


@_fhe_MOD.simple_op()
def decrypt(ciphertext: TensorType, context: TensorType) -> TensorType:
    """Decrypt ciphertext using FHE Vector backend: returns plaintext with same semantic type.

    Args:
        ciphertext: Encrypted data to decrypt (scalar or 1D vector)
        context: FHE context (must be private context with secret key)

    Returns:
        Plaintext with same semantic type as ciphertext

    Note: Ciphertext encrypted with public context can be decrypted with
    the corresponding private context.
    """
    _ = context
    return ciphertext


@_fhe_MOD.simple_op()
def add(operand1: TensorType, operand2: TensorType) -> TensorType:
    """Add two FHE operands (ciphertext + ciphertext or ciphertext + plaintext).

    Args:
        operand1: First operand (ciphertext or plaintext, scalar or 1D vector)
        operand2: Second operand (ciphertext or plaintext, scalar or 1D vector)

    Returns:
        Result of homomorphic addition

    Raises:
        ValueError: If operands have incompatible shapes or dtypes

    Note: At least one operand must be ciphertext. Both operands must have
    the same shape (no broadcasting in Vector backend).
    """
    assert operand1.dtype == operand2.dtype, (
        f"Operand dtypes must match, got {operand1.dtype} and {operand2.dtype}."
    )
    assert operand1.shape == operand2.shape, (
        f"Operand shapes must match, got {operand1.shape} and {operand2.shape}."
    )
    return operand1


@_fhe_MOD.simple_op()
def sub(operand1: TensorType, operand2: TensorType) -> TensorType:
    """Subtract two FHE operands (ciphertext - ciphertext or ciphertext - plaintext).

    Args:
        operand1: First operand (ciphertext or plaintext, scalar or 1D vector)
        operand2: Second operand (ciphertext or plaintext, scalar or 1D vector)

    Returns:
        Result of homomorphic subtraction

    Raises:
        ValueError: If operands have incompatible shapes or dtypes

    Note: At least one operand must be ciphertext. Both operands must have
    the same shape (no broadcasting in Vector backend).
    """
    assert operand1.dtype == operand2.dtype, (
        f"Operand dtypes must match, got {operand1.dtype} and {operand2.dtype}."
    )
    assert operand1.shape == operand2.shape, (
        f"Operand shapes must match, got {operand1.shape} and {operand2.shape}."
    )
    return operand1


@_fhe_MOD.simple_op()
def mul(operand1: TensorType, operand2: TensorType) -> TensorType:
    """Multiply two FHE operands (ciphertext * ciphertext or ciphertext * plaintext).

    Args:
        operand1: First operand (ciphertext or plaintext, scalar or 1D vector)
        operand2: Second operand (ciphertext or plaintext, scalar or 1D vector)

    Returns:
        Result of homomorphic multiplication

    Raises:
        ValueError: If operands have incompatible shapes or dtypes

    Note: At least one operand must be ciphertext. Both operands must have
    the same shape (no broadcasting in Vector backend).
    For BFV scheme, plaintext operands must be integers.
    """
    assert operand1.dtype == operand2.dtype, (
        f"Operand dtypes must match, got {operand1.dtype} and {operand2.dtype}."
    )
    assert operand1.shape == operand2.shape, (
        f"Operand shapes must match, got {operand1.shape} and {operand2.shape}."
    )
    return operand1


@_fhe_MOD.simple_op()
def dot(operand1: TensorType, operand2: TensorType) -> TensorType:
    """Compute dot product of FHE operands (ciphertext · ciphertext or ciphertext · plaintext).

    Args:
        operand1: First operand (ciphertext or plaintext, must be 1D vector)
        operand2: Second operand (ciphertext or plaintext, must be 1D vector)

    Returns:
        Scalar result of homomorphic dot product (shape=())

    Raises:
        ValueError: If operands are not 1D vectors or have different lengths

    Note: Both operands must be 1D vectors (not scalars). For scalar multiplication,
    use mul() instead. This operation always returns a scalar.
    """
    if len(operand1.shape) != 1:
        raise ValueError(
            f"Dot product requires 1D vectors, got shape {operand1.shape} for operand1"
        )
    if len(operand2.shape) != 1:
        raise ValueError(
            f"Dot product requires 1D vectors, got shape {operand2.shape} for operand2"
        )
    if operand1.shape[0] != operand2.shape[0]:
        raise ValueError(
            f"Dot product dimension mismatch: {operand1.shape[0]} vs {operand2.shape[0]}"
        )

    # Dot product of 1D vectors returns a scalar
    return TensorType(operand1.dtype, ())


@_fhe_MOD.simple_op()
def polyval(ciphertext: TensorType, coeffs: TensorType) -> TensorType:
    """Evaluate polynomial on encrypted data with plaintext coefficients.

    Args:
        ciphertext: Encrypted data (scalar or 1D vector)
        coeffs: Plaintext polynomial coefficients as 1D array [c0, c1, c2, ...]
                representing c0 + c1*x + c2*x^2 + ...

    Returns:
        Result of polynomial evaluation with same shape and dtype as ciphertext

    Raises:
        ValueError: If coefficients array is not 1D or has fewer than 2 elements

    Note: Polynomial must have degree >= 1 (at least 2 coefficients required).
    Constant polynomials (degree 0, single coefficient) are NOT supported due to
    TenSEAL limitation. For constant values, use: ct * 0 + constant instead.
    For BFV scheme, coefficients must be integers.

    Common use case - Sigmoid approximation:
        sigmoid_coeffs = [0.5, 0.15012, 0.0, -0.0018027]
        result = polyval(ciphertext, sigmoid_coeffs)
    """
    if len(coeffs.shape) != 1:
        raise ValueError(
            f"Polynomial coefficients must be 1D array, got shape {coeffs.shape}"
        )
    _ = coeffs
    return ciphertext


@_fhe_MOD.simple_op()
def negate(ciphertext: TensorType) -> TensorType:
    """Negate encrypted data (unary minus).

    Args:
        ciphertext: Encrypted data (scalar or 1D vector)

    Returns:
        Negated ciphertext with same shape and dtype

    Note: Equivalent to multiplying by -1.
    """
    return ciphertext


@_fhe_MOD.simple_op()
def square(ciphertext: TensorType) -> TensorType:
    """Square encrypted data (element-wise).

    Args:
        ciphertext: Encrypted data (scalar or 1D vector)

    Returns:
        Squared ciphertext with same shape and dtype

    Note: More efficient than mul(ciphertext, ciphertext) in some FHE schemes.
    """
    return ciphertext
