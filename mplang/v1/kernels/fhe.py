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

"""FHE Vector backend implementation using TenSEAL CKKSVector/BFVVector.

This module provides FHE operations using TenSEAL's vector-based encryption,
which only supports 1D data. All operations enforce 1D shape constraints.
"""

from typing import Any

import numpy as np
import tenseal as ts

from mplang.v1.core import DType, PFunction, TensorLike
from mplang.v1.kernels.base import kernel_def
from mplang.v1.kernels.value import TensorValue


class FHEContext:
    """FHE context manager for TenSEAL vector operations.

    Note: This context is optimized for vector-based encryption (CKKSVector/BFVVector),
    which only supports 1D data.
    """

    def __init__(self, context: Any, scheme: str = "CKKS"):
        self.context = context
        assert scheme in (
            "CKKS",
            "BFV",
        ), f"Unsupported scheme type for TenSEAL: {scheme}"
        self._scheme = scheme

    @property
    def dtype(self) -> Any:
        return np.dtype("O")  # Use object dtype for binary data

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    @property
    def scheme(self) -> str:
        return self._scheme

    @property
    def global_scale(self) -> Any:
        if self._scheme != "CKKS":
            raise ValueError("global_scale is only applicable for CKKS scheme.")
        return self.context.global_scale

    @property
    def is_private(self) -> Any:
        return self.context.is_private()

    @property
    def is_public(self) -> Any:
        return self.context.is_public()

    def make_context_public(self) -> None:
        """Remove secret key from context to make it public."""
        self.context.make_context_public()

    def serialize(self, save_secret_key: bool = True) -> Any:
        """Serialize the context."""
        return self.context.serialize(
            save_public_key=True,
            save_secret_key=save_secret_key,
            save_galois_keys=True,
            save_relin_keys=True,
        )

    def drop_secret_key(self) -> "FHEContext":
        """Create a public-only copy of this context."""
        proto = self.serialize(save_secret_key=False)
        new_ctx = ts.context_from(proto)
        return FHEContext(new_ctx, self._scheme)

    def __repr__(self) -> str:
        return f"FHEContext(scheme={self.scheme}, is_private={self.is_private}, is_public={self.is_public})"


class CipherText:
    """Ciphertext wrapper for TenSEAL vector operations.

    Note: Only supports 1D shapes (scalars represented as shape=(1,) or shape=()).
    """

    def __init__(
        self,
        ct_data: Any,
        semantic_dtype: DType,
        semantic_shape: tuple[int, ...],
        scheme: str,
        context: FHEContext | None = None,
    ):
        # Validate shape constraints for vector backend
        if len(semantic_shape) > 1:
            raise ValueError(
                f"FHE Vector backend only supports 1D data (scalars or vectors). "
                f"Got shape {semantic_shape}. Use fhe.py (tensor backend) for multi-dimensional data."
            )

        self.ct_data = ct_data
        self.semantic_dtype = semantic_dtype
        self.semantic_shape = semantic_shape
        self._scheme = scheme
        self._context = context

    @property
    def dtype(self) -> Any:
        return self.semantic_dtype.to_numpy()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.semantic_shape

    @property
    def scheme(self) -> str:
        return self._scheme

    @property
    def context(self) -> FHEContext | None:
        return self._context

    def __repr__(self) -> str:
        return f"CipherText(dtype={self.semantic_dtype}, shape={self.semantic_shape}, scheme={self.scheme})"


def _convert_to_numpy(obj: TensorLike) -> np.ndarray:
    """Convert a TensorLike object to numpy array."""
    if isinstance(obj, np.ndarray):
        return obj

    # Try to use .numpy() method if available
    if hasattr(obj, "numpy"):
        numpy_method = getattr(obj, "numpy", None)
        if callable(numpy_method):
            try:
                return np.asarray(numpy_method())
            except Exception:
                pass

    return np.asarray(obj)


def _validate_1d_shape(shape: tuple[int, ...], operation: str) -> None:
    """Validate that shape is 1D (scalar or vector) for vector backend operations."""
    if len(shape) > 1:
        raise ValueError(
            f"FHE Vector backend operation '{operation}' only supports 1D data. "
            f"Got shape {shape}. Use fhe.py (tensor backend) for multi-dimensional data."
        )


@kernel_def("fhe.keygen")
def _fhe_keygen(pfunc: PFunction) -> Any:
    """Generate FHE context for vector operations.

    Returns:
        A tuple containing three FHEContext objects:
        - [0]: Private context with secret key
        - [1]: Public context without secret key (for distribution to other parties)
        - [2]: Evaluation context (same as public context for TenSEAL)
    """
    scheme = pfunc.attrs.get("scheme", "CKKS")
    poly_modulus_degree = pfunc.attrs.get("poly_modulus_degree", 8192)

    if scheme == "CKKS":
        # CKKS parameters for floating point operations
        coeff_mod_bit_sizes = pfunc.attrs.get("coeff_mod_bit_sizes", [60, 40, 40, 60])
        global_scale = pfunc.attrs.get("global_scale", 2**40)

        try:
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            )
            context.generate_galois_keys()
            context.generate_relin_keys()
            context.global_scale = global_scale

            private_context = FHEContext(context, scheme="CKKS")
            public_context = private_context.drop_secret_key()
            eval_context = public_context  # For TenSEAL, eval context is same as public

            return (private_context, public_context, eval_context)

        except Exception as e:
            raise RuntimeError(f"Failed to create CKKS context: {e}") from e

    elif scheme == "BFV":
        # BFV parameters for integer operations
        plain_modulus = pfunc.attrs.get("plain_modulus", 1032193)
        coeff_mod_bit_sizes = pfunc.attrs.get("coeff_mod_bit_sizes", [60, 40, 40, 60])

        try:
            context = ts.context(
                ts.SCHEME_TYPE.BFV,
                poly_modulus_degree=poly_modulus_degree,
                plain_modulus=plain_modulus,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            )
            context.generate_galois_keys()
            context.generate_relin_keys()

            private_context = FHEContext(context, scheme="BFV")
            public_context = private_context.drop_secret_key()
            eval_context = public_context

            return (private_context, public_context, eval_context)

        except Exception as e:
            raise RuntimeError(f"Failed to create BFV context: {e}") from e
    else:
        raise ValueError(f"Unsupported FHE scheme: {scheme}. Use 'CKKS' or 'BFV'.")


@kernel_def("fhe.encrypt")
def _fhe_encrypt(pfunc: PFunction, plaintext: Any, context: FHEContext) -> Any:
    """Encrypt plaintext data using FHE vector context.

    Only supports 1D data (scalars or vectors).
    """
    if not isinstance(context, FHEContext):
        raise TypeError(f"Expected FHEContext, got {type(context)}")

    try:
        plaintext_np = _convert_to_numpy(plaintext)

        # Validate shape
        _validate_1d_shape(plaintext_np.shape, "encrypt")

        # Determine semantic dtype based on input data type
        if context.scheme == "CKKS":
            # Preserve the input floating-point dtype (float32 or float64)
            if np.issubdtype(plaintext_np.dtype, np.floating):
                semantic_dtype = DType.from_numpy(plaintext_np.dtype)
            else:
                # Default to float32 for non-floating types
                semantic_dtype = DType.from_numpy(np.dtype("float32"))
        else:  # BFV
            if not np.issubdtype(plaintext_np.dtype, np.integer):
                raise RuntimeError("BFV scheme requires integer semantic_dtype")
            semantic_dtype = DType.from_numpy(np.dtype("int64"))

        # Handle scalar (convert to 1-element vector)
        if plaintext_np.shape == ():
            plaintext_np = np.array([plaintext_np.item()])
            semantic_shape: tuple = ()
        else:
            semantic_shape = plaintext_np.shape

        # Encrypt based on scheme
        if context.scheme == "CKKS":
            plaintext_data = plaintext_np.astype(np.float64).tolist()
            ct_data = ts.ckks_vector(context.context, plaintext_data)
        elif context.scheme == "BFV":
            plaintext_data = plaintext_np.astype(np.int64).tolist()
            ct_data = ts.bfv_vector(context.context, plaintext_data)
        else:
            raise ValueError(f"Unsupported scheme: {context.scheme}")

        # Create CipherText wrapper
        ciphertext = CipherText(
            ct_data=ct_data,
            semantic_dtype=semantic_dtype,
            semantic_shape=semantic_shape,
            scheme=context.scheme,
            context=context,
        )

        return (ciphertext,)

    except Exception as e:
        raise RuntimeError(f"FHE vector encryption failed: {e}") from e


@kernel_def("fhe.decrypt")
def _fhe_decrypt(pfunc: PFunction, ciphertext: CipherText, context: FHEContext) -> Any:
    """Decrypt ciphertext using FHE vector context."""
    if not isinstance(ciphertext, CipherText):
        raise TypeError(f"Expected CipherText, got {type(ciphertext)}")
    if not isinstance(context, FHEContext):
        raise TypeError(f"Expected FHEContext, got {type(context)}")

    # Validate scheme compatibility
    if ciphertext.scheme != context.scheme:
        raise ValueError(
            f"Scheme mismatch: ciphertext uses {ciphertext.scheme}, context uses {context.scheme}"
        )

    # Check if context has secret key
    if not context.is_private:
        raise ValueError("Context must have secret key for decryption")

    try:
        # If the ciphertext was encrypted with a public context,
        # we need to link it to the private context for decryption
        ct_to_decrypt = ciphertext.ct_data

        # Check if the ciphertext's context is missing or public. If so, link it to the
        # private context provided for decryption.
        try:
            # A ciphertext might not have a context if deserialized, or it might have a public one.
            if (
                not ct_to_decrypt.context()
                or not ct_to_decrypt.context().has_secret_key()
            ):
                ct_to_decrypt.link_context(context.context)
        except Exception:
            # Fallback for cases where .context() might fail. Linking is the safe action.
            ct_to_decrypt.link_context(context.context)

        # Decrypt
        decrypted_list = ct_to_decrypt.decrypt()

        # Convert to numpy array using the semantic dtype from ciphertext
        if context.scheme == "CKKS":
            # Use the dtype stored in the ciphertext's semantic_dtype
            target_dtype = ciphertext.semantic_dtype.to_numpy()
            decrypted_np = np.array(decrypted_list, dtype=target_dtype)
        else:  # BFV
            decrypted_np = np.array(decrypted_list, dtype=np.int64)

        # Restore original shape
        if ciphertext.semantic_shape == ():
            # Scalar: shape ()
            result_np = decrypted_np[0:1].reshape(())
        else:
            # Vector: keep 1D array
            result_np = decrypted_np

        # Return TensorValue to adhere to kernel Value I/O convention
        return (TensorValue(np.asarray(result_np)),)

    except Exception as e:
        raise RuntimeError(f"FHE vector decryption failed: {e}") from e


@kernel_def("fhe.add")
def _fhe_add(pfunc: PFunction, lhs: Any, rhs: Any) -> Any:
    """Perform homomorphic addition between ciphertexts or ciphertext and plaintext."""
    try:
        if isinstance(lhs, CipherText) and isinstance(rhs, CipherText):
            result = _fhe_add_ct2ct(lhs, rhs)
        elif isinstance(lhs, CipherText):
            result = _fhe_add_ct2pt(lhs, rhs)
        elif isinstance(rhs, CipherText):
            result = _fhe_add_ct2pt(rhs, lhs)
        else:
            raise ValueError("At least one operand must be CipherText")
        return (result,)
    except Exception as e:
        raise RuntimeError(f"FHE vector addition failed: {e}") from e


def _fhe_add_ct2ct(ct1: CipherText, ct2: CipherText) -> CipherText:
    """Add two ciphertexts (vector backend)."""
    # Validate compatibility
    if ct1.scheme != ct2.scheme:
        raise ValueError("CipherText operands must use same scheme")

    # Validate shapes
    if ct1.semantic_shape != ct2.semantic_shape:
        raise ValueError(
            f"CipherText operands must have same shape for vector addition. "
            f"Got {ct1.semantic_shape} and {ct2.semantic_shape}"
        )

    # Perform addition
    result_ct_data = ct1.ct_data + ct2.ct_data

    # Create result CipherText
    return CipherText(
        ct_data=result_ct_data,
        semantic_dtype=ct1.semantic_dtype,
        semantic_shape=ct1.semantic_shape,
        scheme=ct1.scheme,
        context=ct1.context,
    )


def _fhe_add_ct2pt(ciphertext: CipherText, plaintext: TensorLike) -> CipherText:
    """Add ciphertext and plaintext (vector backend)."""
    # Convert plaintext to numpy
    plaintext_np = _convert_to_numpy(plaintext)

    # Validate shape
    _validate_1d_shape(plaintext_np.shape, "add_plain")

    # Handle scalar plaintext
    if plaintext_np.shape == ():
        plaintext_np = np.array([plaintext_np.item()])
        is_scalar_pt = True
    else:
        is_scalar_pt = False

    # For ciphertext scalar + plaintext vector or vice versa, need matching shapes
    if ciphertext.semantic_shape == () and not is_scalar_pt:
        raise ValueError(
            f"Shape mismatch: cannot add scalar ciphertext with vector plaintext {plaintext_np.shape}"
        )
    if ciphertext.semantic_shape != () and is_scalar_pt:
        # Broadcast scalar plaintext to match ciphertext shape
        plaintext_np = np.full(ciphertext.semantic_shape, plaintext_np[0])

    # Validate final shape match (unless both scalars)
    if (
        ciphertext.semantic_shape != ()
        and plaintext_np.shape != ciphertext.semantic_shape
    ):
        raise ValueError(
            f"Shape mismatch: ciphertext shape {ciphertext.semantic_shape} vs plaintext shape {plaintext_np.shape}"
        )

    # Perform addition based on scheme
    if ciphertext.scheme == "CKKS":
        plaintext_list = plaintext_np.astype(np.float64).tolist()
        result_ct_data = ciphertext.ct_data + plaintext_list
    elif ciphertext.scheme == "BFV":
        if not np.issubdtype(plaintext_np.dtype, np.integer):
            raise RuntimeError("BFV scheme requires integer plaintext")
        plaintext_list = plaintext_np.astype(np.int64).tolist()
        result_ct_data = ciphertext.ct_data + plaintext_list
    else:
        raise ValueError(f"Unsupported scheme: {ciphertext.scheme}")

    # Create result CipherText
    return CipherText(
        ct_data=result_ct_data,
        semantic_dtype=ciphertext.semantic_dtype,
        semantic_shape=ciphertext.semantic_shape,
        scheme=ciphertext.scheme,
        context=ciphertext.context,
    )


@kernel_def("fhe.sub")
def _fhe_sub(pfunc: PFunction, lhs: Any, rhs: Any) -> Any:
    """Perform homomorphic subtraction between ciphertexts or ciphertext and plaintext."""
    try:
        if isinstance(lhs, CipherText) and isinstance(rhs, CipherText):
            result = _fhe_sub_ct2ct(lhs, rhs)
        elif isinstance(lhs, CipherText):
            result = _fhe_sub_ct2pt(lhs, rhs)
        else:
            raise ValueError("Left operand must be CipherText for subtraction")
        return (result,)
    except Exception as e:
        raise RuntimeError(f"FHE vector subtraction failed: {e}") from e


def _fhe_sub_ct2ct(ct1: CipherText, ct2: CipherText) -> CipherText:
    """Subtract two ciphertexts (vector backend)."""
    # Validate compatibility
    if ct1.scheme != ct2.scheme:
        raise ValueError("CipherText operands must use same scheme")

    # Validate shapes
    if ct1.semantic_shape != ct2.semantic_shape:
        raise ValueError(
            f"CipherText operands must have same shape for vector subtraction. "
            f"Got {ct1.semantic_shape} and {ct2.semantic_shape}"
        )

    # Perform subtraction
    result_ct_data = ct1.ct_data - ct2.ct_data

    # Create result CipherText
    return CipherText(
        ct_data=result_ct_data,
        semantic_dtype=ct1.semantic_dtype,
        semantic_shape=ct1.semantic_shape,
        scheme=ct1.scheme,
        context=ct1.context,
    )


def _fhe_sub_ct2pt(ciphertext: CipherText, plaintext: TensorLike) -> CipherText:
    """Subtract plaintext from ciphertext (vector backend)."""
    # Convert plaintext to numpy
    plaintext_np = _convert_to_numpy(plaintext)

    # Validate shape
    _validate_1d_shape(plaintext_np.shape, "sub_plain")

    # Handle scalar plaintext
    if plaintext_np.shape == ():
        plaintext_np = np.array([plaintext_np.item()])
        is_scalar_pt = True
    else:
        is_scalar_pt = False

    # Shape validation and broadcasting
    if ciphertext.semantic_shape == () and not is_scalar_pt:
        raise ValueError(
            "Shape mismatch: cannot subtract vector plaintext from scalar ciphertext"
        )
    if ciphertext.semantic_shape != () and is_scalar_pt:
        # Broadcast scalar plaintext
        plaintext_np = np.full(ciphertext.semantic_shape, plaintext_np[0])

    if (
        ciphertext.semantic_shape != ()
        and plaintext_np.shape != ciphertext.semantic_shape
    ):
        raise ValueError(
            f"Shape mismatch: ciphertext shape {ciphertext.semantic_shape} vs plaintext shape {plaintext_np.shape}"
        )

    # Perform subtraction based on scheme
    if ciphertext.scheme == "CKKS":
        plaintext_list = plaintext_np.astype(np.float64).tolist()
        result_ct_data = ciphertext.ct_data - plaintext_list
    elif ciphertext.scheme == "BFV":
        if not np.issubdtype(plaintext_np.dtype, np.integer):
            raise RuntimeError("BFV scheme requires integer plaintext")
        plaintext_list = plaintext_np.astype(np.int64).tolist()
        result_ct_data = ciphertext.ct_data - plaintext_list
    else:
        raise ValueError(f"Unsupported scheme: {ciphertext.scheme}")

    # Create result CipherText
    return CipherText(
        ct_data=result_ct_data,
        semantic_dtype=ciphertext.semantic_dtype,
        semantic_shape=ciphertext.semantic_shape,
        scheme=ciphertext.scheme,
        context=ciphertext.context,
    )


@kernel_def("fhe.mul")
def _fhe_mul(pfunc: PFunction, lhs: Any, rhs: Any) -> Any:
    """Perform homomorphic multiplication between ciphertexts or ciphertext and plaintext."""
    try:
        if isinstance(lhs, CipherText) and isinstance(rhs, CipherText):
            result = _fhe_mul_ct2ct(lhs, rhs)
        elif isinstance(lhs, CipherText):
            result = _fhe_mul_ct2pt(lhs, rhs)
        elif isinstance(rhs, CipherText):
            result = _fhe_mul_ct2pt(rhs, lhs)
        else:
            raise ValueError("At least one operand must be CipherText")
        return (result,)
    except Exception as e:
        raise RuntimeError(f"FHE vector multiplication failed: {e}") from e


def _fhe_mul_ct2ct(ct1: CipherText, ct2: CipherText) -> CipherText:
    """Multiply two ciphertexts (vector backend)."""
    # Validate compatibility
    if ct1.scheme != ct2.scheme:
        raise ValueError("CipherText operands must use same scheme")

    # Validate shapes
    if ct1.semantic_shape != ct2.semantic_shape:
        raise ValueError(
            f"CipherText operands must have same shape for vector multiplication. "
            f"Got {ct1.semantic_shape} and {ct2.semantic_shape}"
        )

    # Perform multiplication
    result_ct_data = ct1.ct_data * ct2.ct_data

    # Create result CipherText
    return CipherText(
        ct_data=result_ct_data,
        semantic_dtype=ct1.semantic_dtype,
        semantic_shape=ct1.semantic_shape,
        scheme=ct1.scheme,
        context=ct1.context,
    )


def _fhe_mul_ct2pt(ciphertext: CipherText, plaintext: TensorLike) -> CipherText:
    """Multiply ciphertext and plaintext (vector backend)."""
    # Convert plaintext to numpy
    plaintext_np = _convert_to_numpy(plaintext)

    # Validate shape
    _validate_1d_shape(plaintext_np.shape, "mul_plain")

    # Handle scalar plaintext
    if plaintext_np.shape == ():
        plaintext_np = np.array([plaintext_np.item()])
        is_scalar_pt = True
    else:
        is_scalar_pt = False

    # Shape validation and broadcasting
    if ciphertext.semantic_shape == () and not is_scalar_pt:
        raise ValueError(
            "Shape mismatch: cannot multiply scalar ciphertext with vector plaintext"
        )
    if ciphertext.semantic_shape != () and is_scalar_pt:
        # Broadcast scalar plaintext
        plaintext_np = np.full(ciphertext.semantic_shape, plaintext_np[0])

    if (
        ciphertext.semantic_shape != ()
        and plaintext_np.shape != ciphertext.semantic_shape
    ):
        raise ValueError(
            f"Shape mismatch: ciphertext shape {ciphertext.semantic_shape} vs plaintext shape {plaintext_np.shape}"
        )

    # Perform multiplication based on scheme
    if ciphertext.scheme == "CKKS":
        plaintext_list = plaintext_np.astype(np.float64).tolist()
        result_ct_data = ciphertext.ct_data * plaintext_list
    elif ciphertext.scheme == "BFV":
        if not np.issubdtype(plaintext_np.dtype, np.integer):
            raise RuntimeError("BFV scheme requires integer plaintext")
        plaintext_list = plaintext_np.astype(np.int64).tolist()
        result_ct_data = ciphertext.ct_data * plaintext_list
    else:
        raise ValueError(f"Unsupported scheme: {ciphertext.scheme}")

    # Create result CipherText
    return CipherText(
        ct_data=result_ct_data,
        semantic_dtype=ciphertext.semantic_dtype,
        semantic_shape=ciphertext.semantic_shape,
        scheme=ciphertext.scheme,
        context=ciphertext.context,
    )


@kernel_def("fhe.dot")
def _fhe_dot(pfunc: PFunction, lhs: Any, rhs: Any) -> Any:
    """Perform homomorphic dot product (only supports 1D × 1D vectors).

    Result is a scalar (shape=()).
    """
    try:
        if isinstance(lhs, CipherText) and isinstance(rhs, CipherText):
            result = _fhe_dot_ct2ct(lhs, rhs)
        elif isinstance(lhs, CipherText):
            result = _fhe_dot_ct2pt(lhs, rhs)
        elif isinstance(rhs, CipherText):
            result = _fhe_dot_ct2pt(rhs, lhs)
        else:
            raise ValueError("At least one operand must be CipherText")
        return (result,)
    except Exception as e:
        raise RuntimeError(f"FHE vector dot product failed: {e}") from e


def _fhe_dot_ct2ct(ct1: CipherText, ct2: CipherText) -> CipherText:
    """Dot product of two ciphertexts (vector backend, 1D only)."""
    # Validate compatibility
    if ct1.scheme != ct2.scheme:
        raise ValueError("CipherText operands must use same scheme")

    # Validate 1D vector shapes (no scalars allowed in dot product)
    if ct1.semantic_shape == () or ct2.semantic_shape == ():
        raise ValueError("Dot product requires 1D vectors, not scalars")

    if len(ct1.semantic_shape) != 1 or len(ct2.semantic_shape) != 1:
        raise ValueError(
            f"Vector backend dot product only supports 1D X 1D. "
            f"Got shapes {ct1.semantic_shape} and {ct2.semantic_shape}"
        )

    if ct1.semantic_shape[0] != ct2.semantic_shape[0]:
        raise ValueError(
            f"Dot product dimension mismatch: {ct1.semantic_shape[0]} vs {ct2.semantic_shape[0]}"
        )

    # Perform dot product
    result_ct_data = ct1.ct_data.dot(ct2.ct_data)

    # Result is scalar
    return CipherText(
        ct_data=result_ct_data,
        semantic_dtype=ct1.semantic_dtype,
        semantic_shape=(),  # Dot product result is scalar
        scheme=ct1.scheme,
        context=ct1.context,
    )


def _fhe_dot_ct2pt(ciphertext: CipherText, plaintext: TensorLike) -> CipherText:
    """Dot product of ciphertext and plaintext (vector backend, 1D only)."""
    # Convert plaintext to numpy
    plaintext_np = _convert_to_numpy(plaintext)

    # Validate 1D shapes
    if ciphertext.semantic_shape == ():
        raise ValueError("Dot product requires 1D vector ciphertext, not scalar")

    _validate_1d_shape(plaintext_np.shape, "dot_plain")

    if plaintext_np.shape == ():
        raise ValueError("Dot product requires 1D vector plaintext, not scalar")

    if len(ciphertext.semantic_shape) != 1 or len(plaintext_np.shape) != 1:
        raise ValueError(
            f"Vector backend dot product only supports 1D X 1D. "
            f"Got shapes {ciphertext.semantic_shape} and {plaintext_np.shape}"
        )

    if ciphertext.semantic_shape[0] != plaintext_np.shape[0]:
        raise ValueError(
            f"Dot product dimension mismatch: {ciphertext.semantic_shape[0]} vs {plaintext_np.shape[0]}"
        )

    # Perform dot product based on scheme
    if ciphertext.scheme == "CKKS":
        plaintext_list = plaintext_np.astype(np.float64).tolist()
        result_ct_data = ciphertext.ct_data.dot(plaintext_list)
    elif ciphertext.scheme == "BFV":
        if not np.issubdtype(plaintext_np.dtype, np.integer):
            raise RuntimeError("BFV scheme requires integer plaintext")
        plaintext_list = plaintext_np.astype(np.int64).tolist()
        result_ct_data = ciphertext.ct_data.dot(plaintext_list)
    else:
        raise ValueError(f"Unsupported scheme: {ciphertext.scheme}")

    # Result is scalar
    return CipherText(
        ct_data=result_ct_data,
        semantic_dtype=ciphertext.semantic_dtype,
        semantic_shape=(),  # Dot product result is scalar
        scheme=ciphertext.scheme,
        context=ciphertext.context,
    )


@kernel_def("fhe.polyval")
def _fhe_polyval(pfunc: PFunction, ciphertext: CipherText, coeffs: TensorLike) -> Any:
    """Evaluate polynomial on encrypted vector data with plaintext coefficients.

    Args:
        ciphertext: Encrypted data (CipherText, scalar or 1D vector)
        coeffs: Plaintext polynomial coefficients as 1D array [c0, c1, c2, ...]
                representing c0 + c1*x + c2*x^2 + ...

    Returns:
        CipherText with polynomial evaluation result (same shape as input)

    Note:
        TenSEAL has a known issue with constant polynomials (degree 0, single coefficient).
        For constants, consider using scalar multiplication instead: ct * 0 + constant.
    """
    if not isinstance(ciphertext, CipherText):
        raise TypeError(f"Expected CipherText, got {type(ciphertext)}")

    try:
        # Convert and validate coefficients
        coeffs_np = _convert_to_numpy(coeffs)

        if coeffs_np.ndim != 1:
            raise ValueError(
                f"Polynomial coefficients must be 1D array, got shape {coeffs_np.shape}"
            )

        if len(coeffs_np) == 0:
            raise ValueError("Polynomial coefficients cannot be empty")

        # Check for constant polynomial (TenSEAL limitation)
        if len(coeffs_np) == 1:
            raise ValueError(
                "TenSEAL does not support constant polynomials (degree 0). "
                "For constant values, use scalar multiplication instead: ct * 0 + constant"
            )

        # Validate scheme-specific requirements
        if ciphertext.scheme == "BFV":
            if not np.issubdtype(coeffs_np.dtype, np.integer):
                raise RuntimeError(
                    "BFV scheme requires integer polynomial coefficients"
                )
            coeffs_list = coeffs_np.astype(np.int64).tolist()
        else:  # CKKS
            coeffs_list = coeffs_np.astype(np.float64).tolist()

        # Perform polynomial evaluation
        result_ct_data = ciphertext.ct_data.polyval(coeffs_list)

        # Create result CipherText (same shape as input)
        return (
            CipherText(
                ct_data=result_ct_data,
                semantic_dtype=ciphertext.semantic_dtype,
                semantic_shape=ciphertext.semantic_shape,
                scheme=ciphertext.scheme,
                context=ciphertext.context,
            ),
        )

    except Exception as e:
        raise RuntimeError(f"FHE vector polyval failed: {e}") from e


@kernel_def("fhe.negate")
def _fhe_negate(pfunc: PFunction, ciphertext: CipherText) -> Any:
    """Negate encrypted data (compute -x)."""
    if not isinstance(ciphertext, CipherText):
        raise TypeError(f"Expected CipherText, got {type(ciphertext)}")

    try:
        # Perform negation
        result_ct_data = -ciphertext.ct_data

        # Create result CipherText
        return (
            CipherText(
                ct_data=result_ct_data,
                semantic_dtype=ciphertext.semantic_dtype,
                semantic_shape=ciphertext.semantic_shape,
                scheme=ciphertext.scheme,
                context=ciphertext.context,
            ),
        )

    except Exception as e:
        raise RuntimeError(f"FHE vector negation failed: {e}") from e


@kernel_def("fhe.square")
def _fhe_square(pfunc: PFunction, ciphertext: CipherText) -> Any:
    """Square encrypted data (compute x²)."""
    if not isinstance(ciphertext, CipherText):
        raise TypeError(f"Expected CipherText, got {type(ciphertext)}")

    try:
        # Perform squaring (x * x)
        result_ct_data = ciphertext.ct_data**2

        # Create result CipherText
        return (
            CipherText(
                ct_data=result_ct_data,
                semantic_dtype=ciphertext.semantic_dtype,
                semantic_shape=ciphertext.semantic_shape,
                scheme=ciphertext.scheme,
                context=ciphertext.context,
            ),
        )

    except Exception as e:
        raise RuntimeError(f"FHE vector square failed: {e}") from e
