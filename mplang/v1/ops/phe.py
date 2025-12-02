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

"""PHE (Partially Homomorphic Encryption) frontend operations."""

from mplang.v1.core import UINT8, TensorType
from mplang.v1.ops.base import stateless_mod

_PHE_MOD = stateless_mod("phe")


@_PHE_MOD.simple_op()
def keygen(
    *,
    scheme: str = "paillier",
    key_size: int = 2048,
    max_value: int | None = None,
    fxp_bits: int | None = None,
) -> tuple[TensorType, TensorType]:
    """Generate a PHE key pair: returns (public_key, private_key).

    Keys are represented with a sentinel TensorType UINT8[(-1, 0)] to indicate
    non-structural, backend-only handles. Runtime validation will treat this
    shape as an opaque placeholder and skip dtype/shape checks.

    Attributes (forwarded to backend):
        scheme: PHE scheme (default: 'paillier')
        key_size: Modulus size in bits (default: 2048)
        max_value: Optional range-encoding bound B. If provided, the backend will
            encode/decode integers/floats within [-B, B] and treat (B, N-B) as overflow.
            Pick B to exceed the largest intermediate magnitude you expect in homomorphic
            combinations. If omitted, backend default is used (currently 2**32).
        fxp_bits: Optional fixed-point fractional bits for float encoding (default backend value).
    """
    key_spec = TensorType(UINT8, (-1, 0))
    return key_spec, key_spec


@_PHE_MOD.simple_op()
def encrypt(plaintext: TensorType, public_key: TensorType) -> TensorType:
    """Encrypt plaintext using PHE public key: returns ciphertext with same semantic type as plaintext."""
    _ = public_key
    return plaintext


@_PHE_MOD.simple_op()
def add(operand1: TensorType, operand2: TensorType) -> TensorType:
    """Add two PHE operands (semantics depend on backend representation)."""
    _ = operand2
    return operand1


@_PHE_MOD.simple_op()
def mul(ciphertext: TensorType, plaintext: TensorType) -> TensorType:
    """Multiply a PHE ciphertext with a plaintext value (ciphertext dtype preserved)."""
    if plaintext.dtype.is_floating:
        raise ValueError(
            "PHE multiplication does not support floating-point plaintext."
        )
    return ciphertext


@_PHE_MOD.simple_op()
def decrypt(ciphertext: TensorType, private_key: TensorType) -> TensorType:
    """Decrypt ciphertext using PHE private key: returns plaintext with same semantic type as ciphertext."""
    _ = private_key
    return ciphertext


@_PHE_MOD.simple_op()
def dot(ciphertext: TensorType, plaintext: TensorType) -> TensorType:
    """Compute dot product of ciphertext with plaintext.

    Args:
        ciphertext: The ciphertext operand (first argument)
        plaintext: The plaintext operand (second argument)

    Returns:
        TensorType: Result tensor type with computed shape following numpy dot product rules
    """
    # For dot product, we need to calculate the result shape
    # This follows numpy dot product rules
    import numpy as np

    # Create dummy arrays to determine result shape
    dummy_ct = np.zeros(ciphertext.shape)
    dummy_pt = np.zeros(plaintext.shape)
    dummy_result = np.dot(dummy_ct, dummy_pt)

    return TensorType(ciphertext.dtype, dummy_result.shape)


@_PHE_MOD.simple_op()
def gather(ciphertext: TensorType, indices: TensorType, *, axis: int = 0) -> TensorType:
    """Gather elements from ciphertext using indices.

    Args:
        ciphertext: The ciphertext to gather from
        indices: The indices to gather
        axis: The axis along which to gather (default: 0)
    """
    # Calculate result shape based on axis parameter
    ct_shape = list(ciphertext.shape)
    indices_shape = list(indices.shape)

    # Normalize negative axis
    normalized_axis = axis if axis >= 0 else len(ct_shape) + axis

    # Result shape: replace the axis dimension with indices shape
    result_shape = (
        ct_shape[:normalized_axis] + indices_shape + ct_shape[normalized_axis + 1 :]
    )
    return TensorType(ciphertext.dtype, tuple(result_shape))


@_PHE_MOD.simple_op()
def scatter(
    ciphertext: TensorType,
    indices: TensorType,
    updates: TensorType,
    *,
    axis: int = 0,
) -> TensorType:
    """Scatter updates into ciphertext at specified indices.

    Args:
        ciphertext: The ciphertext to scatter into
        indices: The indices to scatter at
        updates: The ciphertext updates to scatter
        axis: The axis along which to scatter (default: 0)

    Returns:
        TensorType: Result tensor type with same shape and dtype as original ciphertext
    """
    return ciphertext


@_PHE_MOD.simple_op()
def concat(operand0: TensorType, operand1: TensorType, *, axis: int = 0) -> TensorType:
    """Concatenate ciphertext tensors along specified axis.

    Args:
        operand0: The first ciphertext operand to concatenate
        operand1: The second ciphertext operand to concatenate
        axis: Axis along which to concatenate

    Returns:
        TensorType: Result tensor type with computed shape following numpy concatenation rules
    """
    # All operands should have same dtype
    first_dtype = operand0.dtype
    if operand1.dtype != first_dtype:
        raise ValueError("All operands must have the same dtype for concatenation")

    # Calculate result shape using numpy concatenation logic
    import numpy as np

    dummy_arrays = [np.zeros(operand0.shape), np.zeros(operand1.shape)]
    dummy_result = np.concatenate(dummy_arrays, axis=axis)

    return TensorType(first_dtype, dummy_result.shape)


@_PHE_MOD.simple_op()
def reshape(ciphertext: TensorType, *, new_shape: tuple[int, ...]) -> TensorType:
    """Reshape ciphertext to new shape.

    Args:
        ciphertext: The ciphertext to reshape
        new_shape: The target shape (can contain -1 for inferred dimension)

    Returns:
        TensorType: Result tensor type with computed shape following numpy reshape rules
    """
    # Calculate the actual result shape (handling -1 inference)
    import numpy as np

    dummy_array = np.zeros(ciphertext.shape)
    # use this to check the correctness of new_shape
    dummy_result = dummy_array.reshape(new_shape)
    actual_shape = dummy_result.shape

    return TensorType(ciphertext.dtype, actual_shape)


@_PHE_MOD.simple_op()
def transpose(
    ciphertext: TensorType, *, axes: tuple[int, ...] | None = None
) -> TensorType:
    """Transpose ciphertext by permuting axes.

    Args:
        ciphertext: The ciphertext to transpose
        axes: Permutation of axes (None for default reverse order)

    Returns:
        TensorType: Result tensor type with computed shape following numpy transpose rules
    """
    # Calculate result shape using numpy transpose logic
    import numpy as np

    dummy_array = np.zeros(ciphertext.shape)
    dummy_result = np.transpose(dummy_array, axes)

    return TensorType(ciphertext.dtype, dummy_result.shape)
