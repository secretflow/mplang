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

"""FHE (Fully Homomorphic Encryption) backend implementation using TenSEAL."""

import numpy as np
import tenseal as ts
from typing import Any, Union

from mplang.core.dtype import DType
from mplang.core.mptype import TensorLike
from mplang.core.pfunc import PFunction
from mplang.kernels.base import kernel_def


# TODO: (TenSEAL context can not access the underlying scheme type, so we pass it around manually)
class FHEContext:
    """FHE context manager for TenSEAL operations."""

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
    def global_scale(self) -> float:
        if self._scheme != "CKKS":
            raise ValueError("global_scale is only applicable for CKKS scheme.")
        return self.context.global_scale

    @property
    def is_private(self) -> bool:
        return self.context.is_private()

    @property
    def is_public(self) -> bool:
        return self.context.is_public()

    def make_context_public(self) -> None:
        """Remove secret key from context to make it public."""
        self.context.make_context_public()

    def serialize(self, save_secret_key: bool = True) -> bytes:
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
    """Ciphertext wrapper for TenSEAL operations."""

    def __init__(
        self,
        ct_data: Any,
        semantic_dtype: DType,
        semantic_shape: tuple[int, ...],
        scheme: str,
        context: FHEContext | None = None,
    ):
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


@kernel_def("fhe.keygen")
def _fhe_keygen(pfunc: PFunction) -> Any:
    """Generate FHE context.

    Returns:
        A list containing two FHEContext objects:
        - [0]: Private context with secret key
        - [1]: Public context without secret key (for distribution to other parties)
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

            # Create private context (with secret key)
            private_context = FHEContext(context, scheme)
            # Create public context (without secret key)
            public_context = private_context.drop_secret_key()

            return [private_context, public_context]

        except Exception as e:
            raise RuntimeError(f"Failed to generate CKKS context: {e}") from e

    elif scheme == "BFV":
        # BFV parameters for integer operations
        plain_modulus = pfunc.attrs.get("plain_modulus", 1032193)

        try:
            context = ts.context(
                ts.SCHEME_TYPE.BFV,
                poly_modulus_degree=poly_modulus_degree,
                plain_modulus=plain_modulus,
            )
            context.generate_galois_keys()
            context.generate_relin_keys()

            # Create private context (with secret key)
            private_context = FHEContext(context, scheme)
            # Create public context (without secret key)
            public_context = private_context.drop_secret_key()

            return [private_context, public_context]

        except Exception as e:
            raise RuntimeError(f"Failed to generate BFV context: {e}") from e
    else:
        raise ValueError(f"Unsupported FHE scheme: {scheme}")


@kernel_def("fhe.encrypt")
def _fhe_encrypt(pfunc: PFunction, plaintext: Any, context: FHEContext) -> Any:
    """Encrypt plaintext data using FHE context."""
    if not isinstance(context, FHEContext):
        raise ValueError("Second argument must be an FHEContext instance")

    try:
        # Convert plaintext to numpy to get semantic type info
        plaintext_np = _convert_to_numpy(plaintext)
        semantic_dtype = DType.from_numpy(plaintext_np.dtype)
        semantic_shape = plaintext_np.shape

        if context.scheme == "CKKS":
            # CKKS supports floating point operations
            if semantic_shape == ():
                # Scalar - use 1D tensor with single element
                plaintext_data = np.array([float(plaintext_np.item())])
                ct_data = ts.ckks_tensor(context.context, plaintext_data)
            else:
                # Vector/Matrix - use tensor directly
                plaintext_data = plaintext_np.astype(np.float64)
                ct_data = ts.ckks_tensor(context.context, plaintext_data)

        elif context.scheme == "BFV":
            # BFV supports integer operations only
            # Check semantic dtype is integer by checking is_signed is not None and is_floating is False
            if semantic_dtype.is_signed is None or semantic_dtype.is_floating:
                raise ValueError(
                    f"BFV scheme requires integer semantic_dtype, got {semantic_dtype}"
                )

            if not np.issubdtype(plaintext_np.dtype, np.integer):
                raise ValueError("BFV scheme only supports integer data types")

            if semantic_shape == ():
                # Scalar - use 1D tensor with single element
                plaintext_data = np.array([int(plaintext_np.item())])
                ct_data = ts.bfv_tensor(context.context, plaintext_data)
            else:
                # Vector/Matrix - use tensor directly
                plaintext_data = plaintext_np.astype(np.int64)
                ct_data = ts.bfv_tensor(context.context, plaintext_data)
        else:
            raise ValueError(f"Unsupported scheme: {context.scheme}")

        ciphertext = CipherText(
            ct_data=ct_data,
            semantic_dtype=semantic_dtype,
            semantic_shape=semantic_shape,
            scheme=context.scheme,
            context=context,
        )

        return [ciphertext]

    except Exception as e:
        raise RuntimeError(f"Failed to encrypt data: {e}") from e


@kernel_def("fhe.decrypt")
def _fhe_decrypt(pfunc: PFunction, ciphertext: CipherText, context: FHEContext) -> Any:
    """Decrypt ciphertext using FHE context."""
    if not isinstance(ciphertext, CipherText):
        raise ValueError("First argument must be a CipherText instance")
    if not isinstance(context, FHEContext):
        raise ValueError("Second argument must be an FHEContext instance")

    # Validate scheme compatibility
    if ciphertext.scheme != context.scheme:
        raise ValueError(
            f"Scheme mismatch: ciphertext uses {ciphertext.scheme}, context uses {context.scheme}"
        )

    # Check if context has secret key
    if not context.is_private:
        raise ValueError("Context must have secret key for decryption")

    try:
        # Link the ciphertext to the private context before decryption
        # This is necessary when the ciphertext was created with a public context
        ciphertext.ct_data.link_context(context.context)

        # Decrypt the data - returns PlainTensor
        plain_tensor = ciphertext.ct_data.decrypt()

        # Convert to numpy array
        decrypted_list = plain_tensor.tolist()
        decrypted_array = np.array(decrypted_list)

        if ciphertext.semantic_shape == ():
            # Scalar case - extract single element
            result = decrypted_array.flat[0]
            if context.scheme == "CKKS":
                result = float(result)
            else:  # BFV
                result = int(result)
            return [np.array(result, dtype=ciphertext.dtype)]
        else:
            # Vector/Matrix case - reshape to original shape
            result_array = decrypted_array.reshape(ciphertext.semantic_shape)
            return [result_array.astype(ciphertext.dtype)]

    except Exception as e:
        raise RuntimeError(f"Failed to decrypt data: {e}") from e


@kernel_def("fhe.add")
def _fhe_add(pfunc: PFunction, lhs: Any, rhs: Any) -> Any:
    """Perform homomorphic addition between ciphertexts or ciphertext and plaintext."""
    try:
        if isinstance(lhs, CipherText) and isinstance(rhs, CipherText):
            return [_fhe_add_ct2ct(lhs, rhs)]
        elif isinstance(lhs, CipherText):
            return [_fhe_add_ct2pt(lhs, rhs)]
        elif isinstance(rhs, CipherText):
            return [_fhe_add_ct2pt(rhs, lhs)]
        else:
            raise ValueError("At least one operand must be a CipherText")
    except Exception as e:
        raise RuntimeError(f"Failed to perform homomorphic addition: {e}") from e


def _fhe_add_ct2ct(ct1: CipherText, ct2: CipherText) -> CipherText:
    """Add two ciphertexts."""
    # Validate compatibility
    if ct1.scheme != ct2.scheme:
        raise ValueError("CipherText operands must use same scheme")

    # For simplicity, assume same shape for now
    if ct1.semantic_shape != ct2.semantic_shape:
        raise ValueError("CipherText operands must have same shape")

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
    """Add ciphertext and plaintext."""
    # Convert plaintext to numpy
    plaintext_np = _convert_to_numpy(plaintext)

    # For simplicity, assume same shape for now
    if ciphertext.semantic_shape != plaintext_np.shape:
        raise ValueError("CipherText and plaintext must have same shape")

    # Perform addition based on scheme
    if ciphertext.scheme == "CKKS":
        if plaintext_np.shape == ():
            # Scalar - convert to 1D array
            plaintext_data = np.array([float(plaintext_np.item())])
        else:
            # Vector/Matrix - use directly
            plaintext_data = plaintext_np.astype(np.float64)
        result_ct_data = ciphertext.ct_data + plaintext_data
    elif ciphertext.scheme == "BFV":
        if not np.issubdtype(plaintext_np.dtype, np.integer):
            raise ValueError("BFV scheme only supports integer plaintext")
        if plaintext_np.shape == ():
            # Scalar - convert to 1D array
            plaintext_data = np.array([int(plaintext_np.item())])
        else:
            # Vector/Matrix - use directly
            plaintext_data = plaintext_np.astype(np.int64)
        result_ct_data = ciphertext.ct_data + plaintext_data
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
            return [_fhe_mul_ct2ct(lhs, rhs)]
        elif isinstance(lhs, CipherText):
            return [_fhe_mul_ct2pt(lhs, rhs)]
        elif isinstance(rhs, CipherText):
            return [_fhe_mul_ct2pt(rhs, lhs)]
        else:
            raise ValueError("At least one operand must be a CipherText")
    except Exception as e:
        raise RuntimeError(f"Failed to perform homomorphic multiplication: {e}") from e


def _fhe_mul_ct2ct(ct1: CipherText, ct2: CipherText) -> CipherText:
    """Multiply two ciphertexts."""
    # Validate compatibility
    if ct1.scheme != ct2.scheme:
        raise ValueError("CipherText operands must use same scheme")

    # For simplicity, assume same shape for now
    if ct1.semantic_shape != ct2.semantic_shape:
        raise ValueError("CipherText operands must have same shape")

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
    """Multiply ciphertext and plaintext."""
    # Convert plaintext to numpy
    plaintext_np = _convert_to_numpy(plaintext)

    # For simplicity, assume same shape for now
    if ciphertext.semantic_shape != plaintext_np.shape:
        raise ValueError("CipherText and plaintext must have same shape")

    # Perform multiplication based on scheme
    if ciphertext.scheme == "CKKS":
        if plaintext_np.shape == ():
            # Scalar - convert to 1D array
            plaintext_data = np.array([float(plaintext_np.item())])
        else:
            # Vector/Matrix - use directly
            plaintext_data = plaintext_np.astype(np.float64)
        result_ct_data = ciphertext.ct_data * plaintext_data
    elif ciphertext.scheme == "BFV":
        if not np.issubdtype(plaintext_np.dtype, np.integer):
            raise ValueError("BFV scheme only supports integer plaintext")
        if plaintext_np.shape == ():
            # Scalar - convert to 1D array
            plaintext_data = np.array([int(plaintext_np.item())])
        else:
            # Vector/Matrix - use directly
            plaintext_data = plaintext_np.astype(np.int64)
        result_ct_data = ciphertext.ct_data * plaintext_data
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
    """Perform homomorphic dot product between ciphertexts or ciphertext and plaintext.

    TenSEAL supports dot product for tensors up to 2D×2D.
    """
    try:
        if isinstance(lhs, CipherText) and isinstance(rhs, CipherText):
            return [_fhe_dot_ct2ct(lhs, rhs)]
        elif isinstance(lhs, CipherText):
            return [_fhe_dot_ct2pt(lhs, rhs)]
        elif isinstance(rhs, CipherText):
            return [_fhe_dot_ct2pt(rhs, lhs)]
        else:
            raise ValueError("At least one operand must be a CipherText")
    except Exception as e:
        raise RuntimeError(f"Failed to perform homomorphic dot product: {e}") from e


def _fhe_dot_ct2ct(ct1: CipherText, ct2: CipherText) -> CipherText:
    """Dot product of two ciphertexts."""
    # Validate compatibility
    if ct1.scheme != ct2.scheme:
        raise ValueError("CipherText operands must use same scheme")

    # Check dimension limits (TenSEAL supports up to 2D×2D)
    if len(ct1.semantic_shape) > 2 or len(ct2.semantic_shape) > 2:
        raise ValueError(
            f"TenSEAL only supports dot product for tensors up to 2D×2D, "
            f"got shapes {ct1.semantic_shape} and {ct2.semantic_shape}"
        )

    # Perform dot product
    result_ct_data = ct1.ct_data.dot(ct2.ct_data)

    # Infer result shape based on input shapes
    result_shape = _infer_dot_shape(ct1.semantic_shape, ct2.semantic_shape)

    # Create result CipherText
    return CipherText(
        ct_data=result_ct_data,
        semantic_dtype=ct1.semantic_dtype,
        semantic_shape=result_shape,
        scheme=ct1.scheme,
        context=ct1.context,
    )


def _fhe_dot_ct2pt(ciphertext: CipherText, plaintext: TensorLike) -> CipherText:
    """Dot product of ciphertext and plaintext."""
    # Convert plaintext to numpy
    plaintext_np = _convert_to_numpy(plaintext)

    # Check dimension limits
    if len(ciphertext.semantic_shape) > 2 or len(plaintext_np.shape) > 2:
        raise ValueError(
            f"TenSEAL only supports dot product for tensors up to 2D×2D, "
            f"got shapes {ciphertext.semantic_shape} and {plaintext_np.shape}"
        )

    # Perform dot product based on scheme
    if ciphertext.scheme == "CKKS":
        plaintext_data = plaintext_np.astype(np.float64)
        result_ct_data = ciphertext.ct_data.dot(plaintext_data)
    elif ciphertext.scheme == "BFV":
        if not np.issubdtype(plaintext_np.dtype, np.integer):
            raise ValueError("BFV scheme only supports integer plaintext")
        plaintext_data = plaintext_np.astype(np.int64)
        result_ct_data = ciphertext.ct_data.dot(plaintext_data)
    else:
        raise ValueError(f"Unsupported scheme: {ciphertext.scheme}")

    # Infer result shape
    result_shape = _infer_dot_shape(ciphertext.semantic_shape, plaintext_np.shape)

    # Create result CipherText
    return CipherText(
        ct_data=result_ct_data,
        semantic_dtype=ciphertext.semantic_dtype,
        semantic_shape=result_shape,
        scheme=ciphertext.scheme,
        context=ciphertext.context,
    )


def _infer_dot_shape(
    shape1: tuple[int, ...], shape2: tuple[int, ...]
) -> tuple[int, ...]:
    """Infer the result shape of dot product."""
    # Scalar result for 1D × 1D
    if len(shape1) == 1 and len(shape2) == 1:
        if shape1[0] != shape2[0]:
            raise ValueError(
                f"Incompatible shapes for dot product: {shape1} and {shape2}"
            )
        return ()

    # Vector result for 2D × 1D
    if len(shape1) == 2 and len(shape2) == 1:
        if shape1[1] != shape2[0]:
            raise ValueError(
                f"Incompatible shapes for dot product: {shape1} and {shape2}"
            )
        return (shape1[0],)

    # Matrix result for 2D × 2D
    if len(shape1) == 2 and len(shape2) == 2:
        if shape1[1] != shape2[0]:
            raise ValueError(
                f"Incompatible shapes for dot product: {shape1} and {shape2}"
            )
        return (shape1[0], shape2[1])

    raise ValueError(f"Unsupported dot product shapes: {shape1} and {shape2}")


@kernel_def("fhe.polyval")
def _fhe_polyval(pfunc: PFunction, ciphertext: CipherText, coeffs: TensorLike) -> Any:
    """Evaluate polynomial on encrypted data with plaintext coefficients.

    Args:
        ciphertext: Encrypted data (CipherText)
        coeffs: Plaintext polynomial coefficients as numpy array [c0, c1, c2, ...]
                representing c0 + c1*x + c2*x^2 + ...

    Returns:
        CipherText with polynomial evaluation result
    """
    if not isinstance(ciphertext, CipherText):
        raise ValueError("First argument must be a CipherText instance")

    try:
        # Convert coeffs to numpy
        coeffs_np = _convert_to_numpy(coeffs)

        if coeffs_np.ndim != 1:
            raise ValueError(
                f"Coefficients must be 1D array, got shape {coeffs_np.shape}"
            )

        if len(coeffs_np) == 0:
            raise ValueError("Coefficients array cannot be empty")

        if len(coeffs_np) == 1:
            raise ValueError(
                "Polynomial must have degree >= 1 (at least 2 coefficients required)"
            )

        # Convert coefficients based on scheme
        if ciphertext.scheme == "CKKS":
            coeffs_data = coeffs_np.astype(np.float64).tolist()
        elif ciphertext.scheme == "BFV":
            if not np.issubdtype(coeffs_np.dtype, np.integer):
                raise ValueError("BFV scheme only supports integer coefficients")
            coeffs_data = coeffs_np.astype(np.int64).tolist()
        else:
            raise ValueError(f"Unsupported scheme: {ciphertext.scheme}")

        # Evaluate polynomial using TenSEAL's polyval
        result_ct_data = ciphertext.ct_data.polyval(coeffs_data)

        # Create result CipherText with same shape and dtype
        return [
            CipherText(
                ct_data=result_ct_data,
                semantic_dtype=ciphertext.semantic_dtype,
                semantic_shape=ciphertext.semantic_shape,
                scheme=ciphertext.scheme,
                context=ciphertext.context,
            )
        ]

    except Exception as e:
        raise RuntimeError(f"Failed to evaluate polynomial: {e}") from e
