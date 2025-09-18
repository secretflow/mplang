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

"""PHE (Partially Homomorphic Encryption) backend implementation using lightPHE."""

from typing import Any

import numpy as np
from lightphe import LightPHE

from mplang.core.dtype import DType
from mplang.core.mptype import TensorLike
from mplang.core.pfunc import PFunction, TensorHandler

# This controls the decimal precision used in lightPHE for float operations
PRECISION = 12


class PublicKey:
    """PHE Public Key that implements TensorLike protocol."""

    def __init__(self, key_data: Any, scheme: str, key_size: int):
        self.key_data = key_data
        self.scheme = scheme
        self.key_size = key_size

    @property
    def dtype(self) -> Any:
        return np.dtype("O")  # Use object dtype for binary data

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    def __repr__(self) -> str:
        return f"PublicKey(scheme={self.scheme}, key_size={self.key_size})"


class PrivateKey:
    """PHE Private Key that implements TensorLike protocol."""

    def __init__(self, sk_data: Any, pk_data: Any, scheme: str, key_size: int):
        self.sk_data = sk_data  # Store private key data
        self.pk_data = pk_data  # Store public key data as well
        self.scheme = scheme
        self.key_size = key_size

    @property
    def dtype(self) -> Any:
        return np.dtype("O")  # Use object dtype for binary data

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    def __repr__(self) -> str:
        return f"PrivateKey(scheme={self.scheme}, key_size={self.key_size})"


class CipherText:
    """PHE CipherText that implements TensorLike protocol."""

    def __init__(
        self,
        ct_data: Any,
        semantic_dtype: DType,
        semantic_shape: tuple[int, ...],
        scheme: str,
        key_size: int,
        pk_data: Any = None,  # Store public key for operations
    ):
        self.ct_data = ct_data
        self.semantic_dtype = semantic_dtype
        self.semantic_shape = semantic_shape
        self.scheme = scheme
        self.key_size = key_size
        self.pk_data = pk_data

    @property
    def dtype(self) -> Any:
        return self.semantic_dtype.to_numpy()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.semantic_shape

    def __repr__(self) -> str:
        return f"CipherText(dtype={self.semantic_dtype}, shape={self.semantic_shape}, scheme={self.scheme})"


class PHEHandler(TensorHandler):
    """Handler for PHE operations using lightPHE library."""

    def __init__(self) -> None:
        super().__init__()

    def setup(self, rank: int) -> None: ...
    def teardown(self) -> None: ...
    def list_fn_names(self) -> list[str]:
        return [
            "phe.keygen",
            "phe.encrypt",
            "phe.add",
            "phe.mul",
            "phe.decrypt",
            # our extensions
            "phe.dot",
            "phe.gather",
            "phe.scatter",
            "phe.concat",
            "phe.reshape",
            "phe.transpose",
        ]

    def execute(
        self,
        pfunc: PFunction,
        args: list[TensorLike],
    ) -> list[TensorLike]:
        """Execute PHE operations."""
        if pfunc.fn_type == "phe.keygen":
            return self._execute_keygen(pfunc, args)
        elif pfunc.fn_type == "phe.encrypt":
            return self._execute_encrypt(pfunc, args)
        elif pfunc.fn_type == "phe.add":
            return self._execute_add(pfunc, args)
        elif pfunc.fn_type == "phe.mul":
            return self._execute_mul(pfunc, args)
        elif pfunc.fn_type == "phe.decrypt":
            return self._execute_decrypt(pfunc, args)
        elif pfunc.fn_type == "phe.dot":
            return self._execute_dot(pfunc, args)
        elif pfunc.fn_type == "phe.gather":
            return self._execute_gather(pfunc, args)
        elif pfunc.fn_type == "phe.scatter":
            return self._execute_scatter(pfunc, args)
        elif pfunc.fn_type == "phe.concat":
            return self._execute_concat(pfunc, args)
        elif pfunc.fn_type == "phe.reshape":
            return self._execute_reshape(pfunc, args)
        elif pfunc.fn_type == "phe.transpose":
            return self._execute_transpose(pfunc, args)
        else:
            raise ValueError(f"Unsupported PHE function type: {pfunc.fn_type}")

    def _convert_to_numpy(self, obj: TensorLike) -> np.ndarray:
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

    def _execute_keygen(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:
        if len(args) != 0:
            raise ValueError("Key generation expects no arguments")

        scheme = pfunc.attrs.get("scheme", "paillier")
        key_size = pfunc.attrs.get("key_size", 2048)

        # Validate scheme
        if scheme.lower() not in ["paillier", "elgamal"]:
            raise ValueError(f"Unsupported PHE scheme: {scheme}")

        scheme = scheme.capitalize()

        try:
            # Set higher precision for better accuracy with floats
            phe = LightPHE(
                algorithm_name=scheme,
                key_size=key_size,
                precision=PRECISION,
            )

            pk_data = phe.cs.keys["public_key"]
            sk_data = phe.cs.keys["private_key"]

            public_key = PublicKey(key_data=pk_data, scheme=scheme, key_size=key_size)
            private_key = PrivateKey(
                sk_data=sk_data,
                pk_data=pk_data,
                scheme=scheme,
                key_size=key_size,
            )

            return [public_key, private_key]

        except Exception as e:
            raise RuntimeError(f"Failed to generate PHE keys: {e}") from e

    def _execute_encrypt(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:
        """Execute encryption.

        Args:
            pfunc: PFunction for encryption
            args: [plaintext, public_key] where plaintext is TensorLike and public_key is PublicKey

        Returns:
            list[TensorLike]: [CipherText] with same semantic type as plaintext
        """
        if len(args) != 2:
            raise ValueError(
                "Encryption expects exactly two arguments: plaintext and public_key"
            )

        plaintext, public_key = args

        # Validate public_key type
        if not isinstance(public_key, PublicKey):
            raise ValueError("Second argument must be a PublicKey instance")

        try:
            # Convert plaintext to numpy to get semantic type info
            plaintext_np = self._convert_to_numpy(plaintext)
            semantic_dtype = DType.from_numpy(plaintext_np.dtype)
            semantic_shape = plaintext_np.shape

            # Create lightPHE instance with the same scheme/key_size as the key
            # Use higher precision for better float accuracy
            phe = LightPHE(
                algorithm_name=public_key.scheme,
                key_size=public_key.key_size,
                precision=PRECISION,
            )
            # Set the public key
            phe.cs.keys["public_key"] = public_key.key_data

            # Always use list encryption for consistent behavior and to handle negative numbers
            flat_data = plaintext_np.flatten()

            if semantic_dtype.is_floating:
                data_list = [float(x) for x in flat_data]
            else:  # integer types
                data_list = [int(x) for x in flat_data]

            lightphe_ciphertext = [phe.encrypt([val]) for val in data_list]

            # Create CipherText object
            ciphertext = CipherText(
                ct_data=lightphe_ciphertext,
                semantic_dtype=semantic_dtype,
                semantic_shape=semantic_shape,
                scheme=public_key.scheme,
                key_size=public_key.key_size,
                pk_data=public_key.key_data,  # Store public key for later operations
            )

            return [ciphertext]

        except Exception as e:
            raise RuntimeError(f"Failed to encrypt data: {e}") from e

    def _execute_mul(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:
        """Execute homomorphic multiplication of ciphertext with plaintext.

        Supports broadcasting between ciphertext and plaintext tensors of different shapes.

        Args:
            pfunc: PFunction for multiplication
            args: Two operands - ciphertext and plaintext

        Returns:
            list[TensorLike]: [result] where result is a CipherText
        """
        if len(args) != 2:
            raise ValueError("Multiplication expects exactly two arguments")

        ciphertext, plaintext = args

        # Validate that first argument is a CipherText
        if not isinstance(ciphertext, CipherText):
            raise ValueError("First argument must be a CipherText instance")

        try:
            # Convert plaintext to numpy
            plaintext_np = self._convert_to_numpy(plaintext)

            # Use numpy broadcasting to determine result shape and broadcast operands
            # Create dummy arrays with the same shapes to test broadcasting
            try:
                dummy_ct = np.zeros(ciphertext.semantic_shape)
                dummy_pt = np.zeros(plaintext_np.shape)
                broadcasted_dummy = dummy_ct * dummy_pt
                result_shape = broadcasted_dummy.shape
            except ValueError as e:
                raise ValueError(
                    f"Operands cannot be broadcast together: CipherText shape {ciphertext.semantic_shape} "
                    f"vs plaintext shape {plaintext_np.shape}: {e}"
                )

            # Broadcast plaintext to match result shape if needed
            if plaintext_np.shape != result_shape:
                plaintext_broadcasted = np.broadcast_to(plaintext_np, result_shape)
            else:
                plaintext_broadcasted = plaintext_np

            # If ciphertext needs broadcasting, we need to replicate its encrypted values
            if ciphertext.semantic_shape != result_shape:
                # Use numpy to create a properly broadcasted index mapping
                # Create a dummy array with same shape as ciphertext, fill with indices
                dummy_ct = np.arange(np.prod(ciphertext.semantic_shape)).reshape(
                    ciphertext.semantic_shape
                )
                # Broadcast this to the result shape
                broadcasted_indices = np.broadcast_to(dummy_ct, result_shape).flatten()

                # Replicate ciphertext data according to the broadcasted indices
                raw_ct: list[Any] = ciphertext.ct_data
                broadcasted_ct_data = [raw_ct[int(idx)] for idx in broadcasted_indices]
            else:
                # No broadcasting needed for ciphertext
                broadcasted_ct_data = ciphertext.ct_data

            # Flatten the broadcasted plaintext data for element-wise multiplication
            target_dtype = ciphertext.semantic_dtype
            flat_data = plaintext_broadcasted.flatten()

            if target_dtype.is_floating:
                multiplier = [float(x) for x in flat_data]
            else:  # integer types
                multiplier = [int(x) for x in flat_data]

            # Perform homomorphic multiplication
            # In Paillier, ciphertext * plaintext is supported
            result_ciphertext = [
                broadcasted_ct_data[i] * [multiplier[i]] for i in range(len(multiplier))
            ]

            # Create result CipherText with the broadcasted shape
            return [
                CipherText(
                    ct_data=result_ciphertext,
                    semantic_dtype=ciphertext.semantic_dtype,
                    semantic_shape=result_shape,
                    scheme=ciphertext.scheme,
                    key_size=ciphertext.key_size,
                    pk_data=ciphertext.pk_data,
                )
            ]

        except ValueError:
            # Re-raise ValueError directly (validation errors)
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to perform multiplication: {e}") from e

    def _execute_add(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:
        """Execute homomorphic addition.

        Args:
            pfunc: PFunction for addition
            args: Two operands - can be CipherText + CipherText, CipherText + plaintext, etc.

        Returns:
            list[TensorLike]: [result] where result type depends on operand types
        """
        if len(args) != 2:
            raise ValueError("Addition expects exactly two arguments")

        lhs, rhs = args

        try:
            # Handle CipherText + CipherText
            if isinstance(lhs, CipherText) and isinstance(rhs, CipherText):
                return [self._execute_add_ct2ct(lhs, rhs)]

            # Handle CipherText + plaintext
            elif isinstance(lhs, CipherText):
                return [self._execute_add_ct2pt(lhs, rhs)]

            # Handle plaintext + CipherText (use commutativity)
            elif isinstance(rhs, CipherText):
                return [self._execute_add_ct2pt(rhs, lhs)]

            else:
                # Both are plaintext - regular addition
                result_np = self._convert_to_numpy(lhs) + self._convert_to_numpy(rhs)
                return [result_np]

        except ValueError:
            # Re-raise ValueError directly (validation errors)
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to perform addition: {e}") from e

    def _execute_add_ct2ct(self, ct1: CipherText, ct2: CipherText) -> CipherText:
        """Execute CipherText + CipherText addition.

        Supports broadcasting between ciphertext tensors of different shapes.

        Args:
            ct1: First CipherText operand
            ct2: Second CipherText operand

        Returns:
            CipherText: Result of homomorphic addition
        """
        # Validate compatibility
        if ct1.scheme != ct2.scheme or ct1.key_size != ct2.key_size:
            raise ValueError("CipherText operands must use same scheme and key size")

        if ct1.pk_data != ct2.pk_data:
            raise ValueError("CipherText operands must be encrypted with same key")

        # Use numpy broadcasting to determine result shape and broadcast operands
        try:
            dummy_ct1 = np.zeros(ct1.semantic_shape)
            dummy_ct2 = np.zeros(ct2.semantic_shape)
            broadcasted_dummy = dummy_ct1 + dummy_ct2
            result_shape = broadcasted_dummy.shape
        except ValueError as e:
            raise ValueError(
                f"CipherText operands cannot be broadcast together: shape {ct1.semantic_shape} "
                f"vs shape {ct2.semantic_shape}: {e}"
            )

        # Broadcast ct1 if needed
        if ct1.semantic_shape != result_shape:
            dummy_ct1 = np.arange(np.prod(ct1.semantic_shape)).reshape(
                ct1.semantic_shape
            )
            broadcasted_indices1 = np.broadcast_to(dummy_ct1, result_shape).flatten()
            raw_ct1: list[Any] = ct1.ct_data
            broadcasted_ct1_data = [raw_ct1[int(idx)] for idx in broadcasted_indices1]
        else:
            broadcasted_ct1_data = ct1.ct_data

        # Broadcast ct2 if needed
        if ct2.semantic_shape != result_shape:
            dummy_ct2 = np.arange(np.prod(ct2.semantic_shape)).reshape(
                ct2.semantic_shape
            )
            broadcasted_indices2 = np.broadcast_to(dummy_ct2, result_shape).flatten()
            raw_ct2: list[Any] = ct2.ct_data
            broadcasted_ct2_data = [raw_ct2[int(idx)] for idx in broadcasted_indices2]
        else:
            broadcasted_ct2_data = ct2.ct_data

        # Perform homomorphic addition
        result_ciphertext = [
            broadcasted_ct1_data[i] + broadcasted_ct2_data[i]
            for i in range(len(broadcasted_ct1_data))
        ]

        # Create result CipherText with broadcasted shape
        return CipherText(
            ct_data=result_ciphertext,
            semantic_dtype=ct1.semantic_dtype,
            semantic_shape=result_shape,
            scheme=ct1.scheme,
            key_size=ct1.key_size,
            pk_data=ct1.pk_data,
        )

    def _execute_add_ct2pt(
        self, ciphertext: CipherText, plaintext: TensorLike
    ) -> CipherText:
        """Execute CipherText + plaintext addition.

        Supports broadcasting between ciphertext and plaintext tensors of different shapes.

        Args:
            ciphertext: CipherText operand
            plaintext: Plaintext operand

        Returns:
            CipherText: Result of homomorphic addition
        """
        # Convert plaintext to numpy
        plaintext_np = self._convert_to_numpy(plaintext)

        # Use numpy broadcasting to determine result shape and broadcast operands
        try:
            dummy_ct = np.zeros(ciphertext.semantic_shape)
            dummy_pt = np.zeros(plaintext_np.shape)
            broadcasted_dummy = dummy_ct + dummy_pt
            result_shape = broadcasted_dummy.shape
        except ValueError as e:
            raise ValueError(
                f"Operands cannot be broadcast together: CipherText shape {ciphertext.semantic_shape} "
                f"vs plaintext shape {plaintext_np.shape}: {e}"
            )

        # Broadcast plaintext to match result shape if needed
        if plaintext_np.shape != result_shape:
            plaintext_broadcasted = np.broadcast_to(plaintext_np, result_shape)
        else:
            plaintext_broadcasted = plaintext_np

        # Broadcast ciphertext if needed
        if ciphertext.semantic_shape != result_shape:
            dummy_ct = np.arange(np.prod(ciphertext.semantic_shape)).reshape(
                ciphertext.semantic_shape
            )
            broadcasted_indices = np.broadcast_to(dummy_ct, result_shape).flatten()
            raw_ct: list[Any] = ciphertext.ct_data
            broadcasted_ct_data = [raw_ct[int(idx)] for idx in broadcasted_indices]
        else:
            broadcasted_ct_data = ciphertext.ct_data

        # For ciphertext + plaintext addition, we encrypt the plaintext first
        # and then do ciphertext + ciphertext addition
        if ciphertext.pk_data is None:
            raise ValueError(
                "CipherText must contain public key data for plaintext addition"
            )

        # Create lightPHE instance to encrypt the plaintext
        phe = LightPHE(
            algorithm_name=ciphertext.scheme,
            key_size=ciphertext.key_size,
            precision=PRECISION,
        )
        phe.cs.keys["public_key"] = ciphertext.pk_data

        # Encrypt the broadcasted plaintext using same method as original encryption
        target_dtype = ciphertext.semantic_dtype
        flat_data = plaintext_broadcasted.flatten()

        if target_dtype.is_floating:
            data_list = [float(x) for x in flat_data]
        else:  # integer types
            data_list = [int(x) for x in flat_data]

        encrypted_plaintext = [phe.encrypt([val]) for val in data_list]

        # Perform addition
        result_ciphertext = [
            encrypted_plaintext[i] + broadcasted_ct_data[i]
            for i in range(len(encrypted_plaintext))
        ]

        # Create result CipherText with broadcasted shape
        return CipherText(
            ct_data=result_ciphertext,
            semantic_dtype=ciphertext.semantic_dtype,
            semantic_shape=result_shape,
            scheme=ciphertext.scheme,
            key_size=ciphertext.key_size,
            pk_data=ciphertext.pk_data,
        )

    def _create_encrypted_zero(self, ciphertext: CipherText) -> Any:
        """Create an encrypted zero value with the same scheme and key as the given ciphertext.

        Args:
            ciphertext: Reference CipherText to get scheme and key information

        Returns:
            Encrypted zero value compatible with the ciphertext
        """
        # Create lightPHE instance with the same configuration
        phe = LightPHE(
            algorithm_name=ciphertext.scheme,
            key_size=ciphertext.key_size,
            precision=PRECISION,
        )
        phe.cs.keys["public_key"] = ciphertext.pk_data

        # Encrypt zero value of appropriate type
        if ciphertext.semantic_dtype.is_floating:
            zero_val = 0.0
        else:
            zero_val = 0

        return phe.encrypt([zero_val])

    def _execute_decrypt(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:
        """Execute decryption.

        Args:
            pfunc: PFunction for decryption
            args: [ciphertext, private_key] where ciphertext is CipherText and private_key is PrivateKey

        Returns:
            list[TensorLike]: [plaintext] with same semantic type as original plaintext
        """
        if len(args) != 2:
            raise ValueError(
                "Decryption expects exactly two arguments: ciphertext and private_key"
            )

        ciphertext, private_key = args

        # Validate argument types
        if not isinstance(ciphertext, CipherText):
            raise ValueError("First argument must be a CipherText instance")
        if not isinstance(private_key, PrivateKey):
            raise ValueError("Second argument must be a PrivateKey instance")

        # Validate key compatibility
        if (
            ciphertext.scheme != private_key.scheme
            or ciphertext.key_size != private_key.key_size
        ):
            raise ValueError(
                "CipherText and PrivateKey must use same scheme and key size"
            )

        try:
            # Create lightPHE instance with the same scheme/key_size
            phe = LightPHE(
                algorithm_name=private_key.scheme,
                key_size=private_key.key_size,
                precision=PRECISION,
            )
            # Set both public and private keys (lightPHE needs both for proper decryption)
            phe.cs.keys["private_key"] = private_key.sk_data
            phe.cs.keys["public_key"] = private_key.pk_data

            # Decrypt the data
            target_dtype = ciphertext.semantic_dtype.to_numpy()
            decrypted_raw = [phe.decrypt(ct) for ct in ciphertext.ct_data]

            # Since we always use list encryption, decrypted_raw is always a list
            if not isinstance(decrypted_raw, list):
                raise RuntimeError(
                    f"Expected list from decryption, got {type(decrypted_raw)}"
                )

            # Validate expected number of elements
            expected_size = (
                int(np.prod(ciphertext.semantic_shape))
                if ciphertext.semantic_shape
                else 1
            )
            if len(decrypted_raw) != expected_size:
                raise RuntimeError(
                    f"Expected {expected_size} values, got {len(decrypted_raw)} values"
                )

            # Handle overflow for smaller integer types
            if target_dtype.kind in "iu":  # integer types
                info = np.iinfo(target_dtype)
                processed_data = [
                    max(info.min, min(info.max, val[0])) for val in decrypted_raw
                ]
            else:
                processed_data = decrypted_raw

            # Unified approach: create array from list, then reshape to target shape
            plaintext_np = np.array(processed_data, dtype=target_dtype).reshape(
                ciphertext.semantic_shape
            )

            return [plaintext_np]

        except Exception as e:
            raise RuntimeError(f"Failed to decrypt data: {e}") from e

    def _execute_dot(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:
        """Execute homomorphic dot product with zero-value optimization.

        Supports various dot product operations:
        - Scalar * Scalar -> Scalar
        - Vector * Vector -> Scalar (inner product)
        - Matrix * Vector -> Vector
        - N-D tensor * M-D tensor -> result based on numpy.dot semantics

        Optimization: Skip multiplication when plaintext value is 0, and handle
        the special case where all plaintext values are 0.

        Args:
            pfunc: PFunction for dot product
            args: [ciphertext, plaintext] where ciphertext is CipherText and plaintext is TensorLike

        Returns:
            list[TensorLike]: [result] where result is a CipherText
        """
        if len(args) != 2:
            raise ValueError("Dot product expects exactly two arguments")

        ciphertext, plaintext = args

        # Validate that first argument is a CipherText
        if not isinstance(ciphertext, CipherText):
            raise ValueError("First argument must be a CipherText instance")
        if isinstance(plaintext, CipherText):
            raise ValueError("Second argument must be a plaintext TensorLike")

        try:
            # Convert plaintext to numpy
            plaintext_np = self._convert_to_numpy(plaintext)

            # Use numpy.dot to determine result shape and validate compatibility
            # Create dummy arrays with same shapes to test dot product compatibility
            try:
                dummy_ct = np.zeros(ciphertext.semantic_shape)
                dummy_pt = np.zeros(plaintext_np.shape)
                dummy_result = np.dot(dummy_ct, dummy_pt)
                result_shape = dummy_result.shape
            except ValueError as e:
                raise ValueError(
                    f"Shapes are not compatible for dot product: CipherText shape {ciphertext.semantic_shape} "
                    f"vs plaintext shape {plaintext_np.shape}: {e}"
                )

            # Perform dot product based on input dimensions
            ct_shape = ciphertext.semantic_shape
            pt_shape = plaintext_np.shape
            target_dtype = ciphertext.semantic_dtype

            if target_dtype.is_floating:
                pt_data = plaintext_np.astype(float)
                # Use a small epsilon for floating point zero comparison
                epsilon = 1e-15
                is_zero_func = lambda x: abs(x) < epsilon
            else:  # integer types
                pt_data = plaintext_np.astype(int)
                is_zero_func = lambda x: x == 0

            # Helper function to create encrypted zero when needed
            def get_encrypted_zero():
                return self._create_encrypted_zero(ciphertext)

            if len(ct_shape) == 0 and len(pt_shape) == 0:
                # Scalar * Scalar
                pt_val = pt_data.item()
                if is_zero_func(pt_val):
                    result_ciphertext = get_encrypted_zero()
                else:
                    result_ciphertext = ciphertext.ct_data[0] * [
                        float(pt_val) if target_dtype.is_floating else int(pt_val)
                    ]
                result_ct_data = [result_ciphertext]

            elif len(ct_shape) == 1 and len(pt_shape) == 1:
                # Vector * Vector -> Scalar (inner product)
                if ct_shape[0] != pt_shape[0]:
                    raise ValueError(
                        f"Vector size mismatch: CipherText size {ct_shape[0]} "
                        f"vs plaintext size {pt_shape[0]}"
                    )

                # Compute element-wise products, skipping zeros
                non_zero_products = []
                for i in range(ct_shape[0]):
                    pt_val = pt_data[i]
                    if not is_zero_func(pt_val):
                        product = ciphertext.ct_data[i] * [
                            float(pt_val) if target_dtype.is_floating else int(pt_val)
                        ]
                        non_zero_products.append(product)

                # Handle result
                if not non_zero_products:
                    # All plaintext values are zero
                    result_ciphertext = get_encrypted_zero()
                else:
                    # Sum all non-zero products
                    result_ciphertext = non_zero_products[0]
                    for i in range(1, len(non_zero_products)):
                        result_ciphertext = result_ciphertext + non_zero_products[i]

                result_ct_data = [result_ciphertext]

            elif len(ct_shape) == 2 and len(pt_shape) == 1:
                # Matrix * Vector -> Vector
                if ct_shape[1] != pt_shape[0]:
                    raise ValueError(
                        f"Matrix-vector dimension mismatch: Matrix shape {ct_shape} "
                        f"vs vector shape {pt_shape}"
                    )

                result_ct_data = []
                for i in range(ct_shape[0]):  # For each row of the matrix
                    # Compute dot product of row i with the vector, skipping zeros
                    row_products = []
                    for j in range(ct_shape[1]):  # For each column in the row
                        pt_val = pt_data[j]
                        if not is_zero_func(pt_val):
                            ct_idx = i * ct_shape[1] + j
                            product = ciphertext.ct_data[ct_idx] * [
                                (
                                    float(pt_val)
                                    if target_dtype.is_floating
                                    else int(pt_val)
                                )
                            ]
                            row_products.append(product)

                    # Handle row result
                    if not row_products:
                        # All plaintext values in this row are zero
                        row_result = get_encrypted_zero()
                    else:
                        # Sum non-zero products for this row
                        row_result = row_products[0]
                        for k in range(1, len(row_products)):
                            row_result = row_result + row_products[k]

                    result_ct_data.append(row_result)

            elif len(ct_shape) == 1 and len(pt_shape) == 2:
                # Vector * Matrix -> Vector
                if ct_shape[0] != pt_shape[0]:
                    raise ValueError(
                        f"Vector-matrix dimension mismatch: Vector shape {ct_shape} "
                        f"vs matrix shape {pt_shape}"
                    )

                result_ct_data = []
                for j in range(pt_shape[1]):  # For each column of the matrix
                    # Compute dot product of vector with column j, skipping zeros
                    col_products = []
                    for i in range(pt_shape[0]):  # For each row in the column
                        pt_val = pt_data[i, j]
                        if not is_zero_func(pt_val):
                            product = ciphertext.ct_data[i] * [
                                (
                                    float(pt_val)
                                    if target_dtype.is_floating
                                    else int(pt_val)
                                )
                            ]
                            col_products.append(product)

                    # Handle column result
                    if not col_products:
                        # All plaintext values in this column are zero
                        col_result = get_encrypted_zero()
                    else:
                        # Sum non-zero products for this column
                        col_result = col_products[0]
                        for k in range(1, len(col_products)):
                            col_result = col_result + col_products[k]

                    result_ct_data.append(col_result)

            elif len(ct_shape) == 2 and len(pt_shape) == 2:
                # Matrix * Matrix -> Matrix
                if ct_shape[1] != pt_shape[0]:
                    raise ValueError(
                        f"Matrix dimension mismatch: First matrix shape {ct_shape} "
                        f"vs second matrix shape {pt_shape}"
                    )

                result_ct_data = []
                for i in range(ct_shape[0]):  # For each row of first matrix
                    for j in range(pt_shape[1]):  # For each column of second matrix
                        # Compute dot product of row i with column j, skipping zeros
                        products = []
                        for k in range(ct_shape[1]):  # Sum over common dimension
                            pt_val = pt_data[k, j]
                            if not is_zero_func(pt_val):
                                ct_idx = i * ct_shape[1] + k
                                product = ciphertext.ct_data[ct_idx] * [
                                    (
                                        float(pt_val)
                                        if target_dtype.is_floating
                                        else int(pt_val)
                                    )
                                ]
                                products.append(product)

                        # Handle element result
                        if not products:
                            # All plaintext values for this element are zero
                            element_result = get_encrypted_zero()
                        else:
                            # Sum non-zero products for this element
                            element_result = products[0]
                            for p in range(1, len(products)):
                                element_result = element_result + products[p]

                        result_ct_data.append(element_result)

            else:
                # General N-D tensor dot product
                # Flatten both tensors and perform generalized dot product
                ct_flat = ciphertext.ct_data
                pt_flat = pt_data.flatten()

                # For general case, we implement numpy.dot semantics
                # This is a simplified implementation for common cases
                if len(ct_shape) >= 2 and len(pt_shape) >= 1:
                    # Treat as matrix multiplication on the last axis of ct and first axis of pt
                    last_dim_ct = ct_shape[-1]
                    first_dim_pt = pt_shape[0]

                    if last_dim_ct != first_dim_pt:
                        raise ValueError(
                            f"Tensor dimension mismatch: CipherText last dimension {last_dim_ct} "
                            f"vs plaintext first dimension {first_dim_pt}"
                        )

                    # Reshape for matrix multiplication
                    ct_reshaped_size = int(np.prod(ct_shape[:-1]))
                    pt_reshaped_size = int(np.prod(pt_shape[1:]))

                    result_ct_data = []
                    for i in range(ct_reshaped_size):
                        for j in range(pt_reshaped_size):
                            # Compute dot product for element (i, j), skipping zeros
                            products = []
                            for k in range(last_dim_ct):
                                pt_idx = k * pt_reshaped_size + j
                                pt_val = pt_flat[pt_idx]
                                if not is_zero_func(pt_val):
                                    ct_idx = i * last_dim_ct + k
                                    product = ct_flat[ct_idx] * [
                                        (
                                            float(pt_val)
                                            if target_dtype.is_floating
                                            else int(pt_val)
                                        )
                                    ]
                                    products.append(product)

                            # Handle element result
                            if not products:
                                # All plaintext values for this element are zero
                                element_result = get_encrypted_zero()
                            else:
                                # Sum non-zero products
                                element_result = products[0]
                                for p in range(1, len(products)):
                                    element_result = element_result + products[p]
                            result_ct_data.append(element_result)
                else:
                    raise ValueError(
                        f"Unsupported tensor shapes for dot product: "
                        f"CipherText shape {ct_shape}, plaintext shape {pt_shape}"
                    )

            # Create result CipherText with computed shape
            return [
                CipherText(
                    ct_data=result_ct_data,
                    semantic_dtype=ciphertext.semantic_dtype,
                    semantic_shape=result_shape,
                    scheme=ciphertext.scheme,
                    key_size=ciphertext.key_size,
                    pk_data=ciphertext.pk_data,
                )
            ]

        except ValueError:
            # Re-raise ValueError directly (validation errors)
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to perform dot product: {e}") from e

    def _execute_gather(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:
        """Execute gather operation on CipherText.

        Supports gathering from multidimensional CipherText using multidimensional indices.
        The operation follows numpy.take semantics:
        - result.shape = indices.shape + ciphertext.shape[:axis] + ciphertext.shape[axis+1:]
        - Gathering is performed along the specified axis of ciphertext

        Args:
            pfunc: PFunction for gather (with axis parameter in attrs)
            args: [ciphertext, indices] where ciphertext is CipherText and indices is TensorLike of integers

        Returns:
            list[TensorLike]: [result] where result is a CipherText
        """
        if len(args) != 2:
            raise ValueError("Gather expects exactly two arguments")

        ciphertext, indices = args

        # Validate that first argument is a CipherText
        if not isinstance(ciphertext, CipherText):
            raise ValueError("First argument must be a CipherText instance")

        # Get axis parameter from pfunc.attrs, default to 0
        axis = pfunc.attrs.get("axis", 0)

        try:
            # Convert indices to numpy
            indices_np = self._convert_to_numpy(indices)

            if not np.issubdtype(indices_np.dtype, np.integer):
                raise ValueError("Indices must be of integer type")

            # Validate that ciphertext has at least 1 dimension for indexing
            if len(ciphertext.semantic_shape) == 0:
                raise ValueError("Cannot gather from scalar CipherText")

            # Normalize axis to positive value
            ndim = len(ciphertext.semantic_shape)
            if axis < 0:
                axis = ndim + axis
            if axis < 0 or axis >= ndim:
                raise ValueError(
                    f"Axis {pfunc.attrs.get('axis', 0)} is out of bounds for array of dimension {ndim}"
                )

            # Validate indices are within bounds for the specified axis
            axis_size = ciphertext.semantic_shape[axis]
            if np.any(indices_np < 0) or np.any(indices_np >= axis_size):
                raise ValueError(
                    f"Indices are out of bounds for axis {axis} with size {axis_size}. "
                    f"Got indices in range [{np.min(indices_np)}, {np.max(indices_np)}]"
                )

            # Calculate result shape: indices.shape + ciphertext.shape[:axis] + ciphertext.shape[axis+1:]
            result_shape = (
                indices_np.shape
                + ciphertext.semantic_shape[:axis]
                + ciphertext.semantic_shape[axis + 1 :]
            )

            # Calculate strides for multi-axis gathering
            ct_shape = ciphertext.semantic_shape

            # Stride calculations for arbitrary axis
            # Elements before axis contribute to outer stride
            outer_stride = int(np.prod(ct_shape[:axis])) if axis > 0 else 1
            # Elements after axis contribute to inner stride
            inner_stride = int(np.prod(ct_shape[axis + 1 :])) if axis < ndim - 1 else 1
            # Total stride for one step along the specified axis
            axis_stride = inner_stride

            # Perform gather operation
            gathered_ct_data = []

            # Iterate through all possible combinations of indices before the gather axis
            if axis == 0:
                # Special case: gathering along axis 0 (existing behavior)
                for idx in indices_np.flatten():
                    start_pos = int(idx) * axis_stride
                    end_pos = start_pos + axis_stride
                    slice_data = ciphertext.ct_data[start_pos:end_pos]
                    gathered_ct_data.extend(slice_data)
            else:
                # General case: gathering along arbitrary axis
                for outer_idx in range(outer_stride):
                    for gather_idx in indices_np.flatten():
                        # Calculate position in flattened ciphertext data
                        pos = (
                            outer_idx * (ct_shape[axis] * inner_stride)
                            + int(gather_idx) * inner_stride
                        )
                        slice_data = ciphertext.ct_data[pos : pos + inner_stride]
                        gathered_ct_data.extend(slice_data)

            # Validate we got the expected number of elements
            expected_size = int(np.prod(result_shape)) if result_shape else 1
            if len(gathered_ct_data) != expected_size:
                raise RuntimeError(
                    f"Internal error: Expected {expected_size} elements, got {len(gathered_ct_data)}"
                )

            # Create result CipherText
            return [
                CipherText(
                    ct_data=gathered_ct_data,
                    semantic_dtype=ciphertext.semantic_dtype,
                    semantic_shape=result_shape,
                    scheme=ciphertext.scheme,
                    key_size=ciphertext.key_size,
                    pk_data=ciphertext.pk_data,
                )
            ]

        except ValueError:
            # Re-raise ValueError directly (validation errors)
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to perform gather: {e}") from e
            if len(gathered_ct_data) != expected_size:
                raise RuntimeError(
                    f"Internal error: Expected {expected_size} elements, got {len(gathered_ct_data)}"
                )

            # Create result CipherText
            return [
                CipherText(
                    ct_data=gathered_ct_data,
                    semantic_dtype=ciphertext.semantic_dtype,
                    semantic_shape=result_shape,
                    scheme=ciphertext.scheme,
                    key_size=ciphertext.key_size,
                    pk_data=ciphertext.pk_data,
                )
            ]

        except ValueError:
            # Re-raise ValueError directly (validation errors)
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to perform gather: {e}") from e

    def _execute_scatter(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:
        """Execute scatter operation on CipherText.

        Supports scattering into multidimensional CipherText using multidimensional indices.
        The operation follows numpy scatter semantics:
        - Scattering is performed along the specified axis of ciphertext
        - indices.shape must equal updated.shape[:len(indices.shape)]
        - updated.shape must be indices.shape + ciphertext.shape[:axis] + ciphertext.shape[axis+1:]
        - Result shape is same as original ciphertext.shape

        Args:
            pfunc: PFunction for scatter (with axis parameter in attrs)
            args: [ciphertext, indices, updated] where ciphertext is CipherText, indices is TensorLike of integers,
                  and updated is the updated values as a CipherText

        Returns:
            list[TensorLike]: [result] where result is a CipherText with same shape as original ciphertext
        """
        if len(args) != 3:
            raise ValueError("Scatter expects exactly three arguments")

        ciphertext, indices, updated = args

        # Validate that first and third arguments are CipherTexts
        if not isinstance(ciphertext, CipherText) or not isinstance(
            updated, CipherText
        ):
            raise ValueError("First and third arguments must be CipherText instances")

        # Validate that both ciphertexts use same scheme/key_size
        if (
            ciphertext.scheme != updated.scheme
            or ciphertext.key_size != updated.key_size
        ):
            raise ValueError("Both CipherTexts must use same scheme and key size")

        if ciphertext.pk_data != updated.pk_data:
            raise ValueError("Both CipherTexts must be encrypted with same key")

        # Get axis parameter from pfunc.attrs, default to 0
        axis = pfunc.attrs.get("axis", 0)

        try:
            # Convert indices to numpy
            indices_np = self._convert_to_numpy(indices)

            if not np.issubdtype(indices_np.dtype, np.integer):
                raise ValueError("Indices must be of integer type")

            # Validate that ciphertext has at least 1 dimension for indexing
            if len(ciphertext.semantic_shape) == 0:
                raise ValueError("Cannot scatter into scalar CipherText")

            # Normalize axis to positive value
            ndim = len(ciphertext.semantic_shape)
            if axis < 0:
                axis = ndim + axis
            if axis < 0 or axis >= ndim:
                raise ValueError(
                    f"Axis {pfunc.attrs.get('axis', 0)} is out of bounds for array of dimension {ndim}"
                )

            # Validate indices are within bounds for the specified axis
            axis_size = ciphertext.semantic_shape[axis]
            if np.any(indices_np < 0) or np.any(indices_np >= axis_size):
                raise ValueError(
                    f"Indices are out of bounds for axis {axis} with size {axis_size}. "
                    f"Got indices in range [{np.min(indices_np)}, {np.max(indices_np)}]"
                )

            # Validate shape compatibility
            # Expected updated shape: indices.shape + ciphertext.shape[:axis] + ciphertext.shape[axis+1:]
            expected_updated_shape = (
                indices_np.shape
                + ciphertext.semantic_shape[:axis]
                + ciphertext.semantic_shape[axis + 1 :]
            )
            if updated.semantic_shape != expected_updated_shape:
                raise ValueError(
                    f"Updated CipherText shape mismatch. Expected {expected_updated_shape}, "
                    f"got {updated.semantic_shape}. "
                    f"Updated shape must be indices.shape + ciphertext.shape[:axis] + ciphertext.shape[axis+1:]"
                )

            # Calculate strides for multi-axis scattering
            ct_shape = ciphertext.semantic_shape

            # Stride calculations for arbitrary axis
            # Elements before axis contribute to outer stride
            outer_stride = int(np.prod(ct_shape[:axis])) if axis > 0 else 1
            # Elements after axis contribute to inner stride
            inner_stride = int(np.prod(ct_shape[axis + 1 :])) if axis < ndim - 1 else 1

            # Create a copy of the original ciphertext data for scattering
            scattered_ct_data = ciphertext.ct_data.copy()

            # Perform scatter operation
            indices_flat = indices_np.flatten()
            updated_ct_data = updated.ct_data

            if axis == 0:
                # Special case: scattering along axis 0 (existing behavior)
                axis_stride = inner_stride
                for i, idx in enumerate(indices_flat):
                    start_pos_updated = i * axis_stride
                    start_pos_original = int(idx) * axis_stride

                    for j in range(axis_stride):
                        if start_pos_updated + j < len(updated_ct_data):
                            scattered_ct_data[start_pos_original + j] = updated_ct_data[
                                start_pos_updated + j
                            ]
            else:
                # General case: scattering along arbitrary axis
                for outer_idx in range(outer_stride):
                    for i, scatter_idx in enumerate(indices_flat):
                        # Calculate position in flattened ciphertext data
                        start_pos_original = (
                            outer_idx * (ct_shape[axis] * inner_stride)
                            + int(scatter_idx) * inner_stride
                        )
                        start_pos_updated = (
                            outer_idx * len(indices_flat) + i
                        ) * inner_stride

                        # Update the ciphertext data
                        for j in range(inner_stride):
                            if start_pos_updated + j < len(updated_ct_data):
                                scattered_ct_data[start_pos_original + j] = (
                                    updated_ct_data[start_pos_updated + j]
                                )

            # Create result CipherText with same shape as original
            return [
                CipherText(
                    ct_data=scattered_ct_data,
                    semantic_dtype=ciphertext.semantic_dtype,
                    semantic_shape=ciphertext.semantic_shape,
                    scheme=ciphertext.scheme,
                    key_size=ciphertext.key_size,
                    pk_data=ciphertext.pk_data,
                )
            ]
        except ValueError:
            # Re-raise ValueError directly (validation errors)
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to perform scatter: {e}") from e

    def _execute_concat(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:
        """Execute concat operation on multiple CipherTexts.

        Supports concatenation along any axis of multidimensional CipherTexts.
        The axis parameter is obtained from pfunc.attrs.

        Args:
            pfunc: PFunction for concat (with axis parameter in attrs)
            args: List of CipherText operands to concatenate

        Returns:
            list[TensorLike]: [result] where result is a CipherText
        """
        if len(args) != 2:
            raise ValueError("Concat expects exactly two arguments")

        c1, c2 = args

        # Get axis parameter from pfunc.attrs, default to 0
        axis = pfunc.attrs.get("axis", 0)

        # Validate that all arguments are CipherText
        if not isinstance(c1, CipherText) or not isinstance(c2, CipherText):
            raise ValueError("All arguments must be CipherText instances")

        # Validate that all ciphertexts have the same key & scheme
        if c1.scheme != c2.scheme or c1.key_size != c2.key_size:
            raise ValueError("All CipherTexts must use same scheme and key size")
        if c1.pk_data != c2.pk_data:
            raise ValueError("All CipherTexts must be encrypted with same key")
        if c1.semantic_dtype != c2.semantic_dtype:
            raise ValueError(
                f"All CipherTexts must have same semantic dtype, got {c1.semantic_dtype} vs {c2.semantic_dtype}"
            )

        # Validate dimensions and axis
        if len(c1.semantic_shape) != len(c2.semantic_shape):
            raise ValueError(
                f"All CipherTexts must have same number of dimensions for concat, got {len(c1.semantic_shape)} vs {len(c2.semantic_shape)}"
            )

        # Handle scalar case
        if len(c1.semantic_shape) == 0:
            raise ValueError("Cannot concatenate scalar CipherTexts")

        # Normalize axis (handle negative axis)
        ndim = len(c1.semantic_shape)
        if axis < 0:
            axis = ndim + axis
        if axis < 0 or axis >= ndim:
            raise ValueError(
                f"axis {pfunc.attrs.get('axis', 0)} is out of bounds for array of dimension {ndim}"
            )

        # Validate that all dimensions except the concat axis are the same
        for i in range(ndim):
            if i != axis and c1.semantic_shape[i] != c2.semantic_shape[i]:
                raise ValueError(
                    f"All CipherTexts must have same shape except along concatenation axis {axis}. "
                    f"Shape mismatch at dimension {i}: {c1.semantic_shape[i]} vs {c2.semantic_shape[i]}"
                )

        try:
            # Calculate result shape
            result_shape = list(c1.semantic_shape)
            result_shape[axis] = c1.semantic_shape[axis] + c2.semantic_shape[axis]
            result_shape = tuple(result_shape)

            # For multidimensional concatenation, we need to interleave the data properly
            # Calculate strides for efficient indexing
            def calculate_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
                strides = [1]
                for dim in reversed(shape[1:]):
                    strides.append(strides[-1] * dim)
                return tuple(reversed(strides))

            # Calculate the number of slices before the concatenation axis
            pre_axis_size = int(np.prod(c1.semantic_shape[:axis])) if axis > 0 else 1
            # Calculate the size of data along and after the concatenation axis
            c1_post_axis_size = int(np.prod(c1.semantic_shape[axis:]))
            c2_post_axis_size = int(np.prod(c2.semantic_shape[axis:]))

            # Initialize result data
            concatenated_ct_data = []

            # Perform concatenation
            for pre_idx in range(pre_axis_size):
                # For each slice before the concatenation axis

                # Add data from c1 along the concatenation axis
                c1_start = pre_idx * c1_post_axis_size
                c1_end = c1_start + c1_post_axis_size
                concatenated_ct_data.extend(c1.ct_data[c1_start:c1_end])

                # Add data from c2 along the concatenation axis
                c2_start = pre_idx * c2_post_axis_size
                c2_end = c2_start + c2_post_axis_size
                concatenated_ct_data.extend(c2.ct_data[c2_start:c2_end])

            # Validate we got the expected number of elements
            expected_size = int(np.prod(result_shape))
            if len(concatenated_ct_data) != expected_size:
                raise RuntimeError(
                    f"Internal error: Expected {expected_size} elements, got {len(concatenated_ct_data)}"
                )

            # Create result CipherText
            return [
                CipherText(
                    ct_data=concatenated_ct_data,
                    semantic_dtype=c1.semantic_dtype,
                    semantic_shape=result_shape,
                    scheme=c1.scheme,
                    key_size=c1.key_size,
                    pk_data=c1.pk_data,
                )
            ]

        except ValueError:
            # Re-raise ValueError directly (validation errors)
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to perform concat: {e}") from e

    def _execute_reshape(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:
        """Execute reshape operation on CipherText.

        Changes the shape of a CipherText without changing its encrypted data.
        The new_shape parameter is obtained from pfunc.attrs.

        Args:
            pfunc: PFunction for reshape (with new_shape parameter in attrs)
            args: [ciphertext] where ciphertext is CipherText to reshape

        Returns:
            list[TensorLike]: [result] where result is a CipherText with new shape
        """
        if len(args) != 1:
            raise ValueError("Reshape expects exactly one argument")

        ciphertext = args[0]

        # Validate that argument is a CipherText
        if not isinstance(ciphertext, CipherText):
            raise ValueError("Argument must be a CipherText instance")

        # Get new_shape parameter from pfunc.attrs
        new_shape = pfunc.attrs.get("new_shape")
        if new_shape is None:
            raise ValueError("new_shape parameter is required for reshape operation")

        # Convert new_shape to tuple if it's a list
        if isinstance(new_shape, list):
            new_shape = tuple(new_shape)
        elif not isinstance(new_shape, tuple):
            raise ValueError("new_shape must be a tuple or list of integers")

        try:
            # Handle -1 dimension inference
            old_size = (
                int(np.prod(ciphertext.semantic_shape))
                if ciphertext.semantic_shape
                else 1
            )

            # Process new_shape to infer -1 dimensions
            inferred_shape = list(new_shape)
            negative_ones = [i for i, dim in enumerate(new_shape) if dim == -1]

            if len(negative_ones) > 1:
                raise ValueError("can only specify one unknown dimension")
            elif len(negative_ones) == 1:
                # Calculate the inferred dimension
                known_size = 1
                for dim in new_shape:
                    if dim != -1:
                        if dim <= 0:
                            raise ValueError(
                                f"negative dimensions not allowed (except -1): {dim}"
                            )
                        known_size *= dim

                if old_size % known_size != 0:
                    raise ValueError(
                        f"cannot reshape array of size {old_size} into shape {new_shape}"
                    )

                inferred_dim = old_size // known_size
                inferred_shape[negative_ones[0]] = inferred_dim
            else:
                # No -1 dimensions, validate that all dimensions are positive
                for dim in new_shape:
                    if dim <= 0:
                        raise ValueError(f"negative dimensions not allowed: {dim}")

            # Convert back to tuple
            inferred_shape = tuple(inferred_shape)

            # Validate that new shape has the same number of elements
            new_size = int(np.prod(inferred_shape)) if inferred_shape else 1

            if old_size != new_size:
                raise ValueError(
                    f"Cannot reshape CipherText with {old_size} elements to shape {inferred_shape} "
                    f"with {new_size} elements"
                )

            # Create result CipherText with new shape (ct_data remains the same)
            return [
                CipherText(
                    ct_data=ciphertext.ct_data,  # Same encrypted data
                    semantic_dtype=ciphertext.semantic_dtype,
                    semantic_shape=inferred_shape,  # Use the inferred shape
                    scheme=ciphertext.scheme,
                    key_size=ciphertext.key_size,
                    pk_data=ciphertext.pk_data,
                )
            ]

        except ValueError:
            # Re-raise ValueError directly (validation errors)
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to perform reshape: {e}") from e

    def _execute_transpose(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:
        """Execute transpose operation on CipherText.

        Permutes the dimensions of a CipherText according to the given axes.
        The axes parameter is obtained from pfunc.attrs.

        Args:
            pfunc: PFunction for transpose (with axes parameter in attrs)
            args: [ciphertext] where ciphertext is CipherText to transpose

        Returns:
            list[TensorLike]: [result] where result is a CipherText with transposed shape
        """
        if len(args) != 1:
            raise ValueError("Transpose expects exactly one argument")

        ciphertext = args[0]

        # Validate that argument is a CipherText
        if not isinstance(ciphertext, CipherText):
            raise ValueError("Argument must be a CipherText instance")

        # Handle scalar case
        if len(ciphertext.semantic_shape) == 0:
            # Transposing a scalar returns the same scalar
            return [ciphertext]

        # Get axes parameter from pfunc.attrs
        axes = pfunc.attrs.get("axes")

        # If axes is None, reverse all dimensions (default transpose behavior)
        if axes is None:
            axes = tuple(reversed(range(len(ciphertext.semantic_shape))))
        elif isinstance(axes, list):
            axes = tuple(axes)
        elif not isinstance(axes, tuple):
            raise ValueError("axes must be a tuple or list of integers, or None")

        try:
            # Validate axes
            ndim = len(ciphertext.semantic_shape)
            if len(axes) != ndim:
                raise ValueError(
                    f"axes length {len(axes)} does not match tensor dimensions {ndim}"
                )

            # Normalize negative axes and validate range
            normalized_axes = []
            for axis in axes:
                if axis < 0:
                    axis = ndim + axis
                if axis < 0 or axis >= ndim:
                    raise ValueError(
                        f"axis {axis} is out of bounds for array of dimension {ndim}"
                    )
                normalized_axes.append(axis)
            axes = tuple(normalized_axes)

            # Check for duplicate axes
            if len(set(axes)) != len(axes):
                raise ValueError("axes cannot contain duplicate values")

            # Calculate new shape
            old_shape = ciphertext.semantic_shape
            new_shape = tuple(old_shape[axis] for axis in axes)

            # For multidimensional transpose, we need to rearrange the encrypted data
            # Create mapping from old flat index to new flat index
            def transpose_data(ct_data: list, old_shape: tuple, axes: tuple):
                if len(old_shape) <= 1:
                    # 1D or scalar case - no actual transposition needed
                    return ct_data

                # Create numpy array to help with index calculations
                dummy_array = np.arange(len(ct_data)).reshape(old_shape)
                transposed_dummy = np.transpose(dummy_array, axes)

                # The new data should be arranged in the order that numpy.transpose would produce
                new_ct_data = [ct_data[idx] for idx in transposed_dummy.flatten()]

                return new_ct_data

            # Rearrange the encrypted data according to transpose
            transposed_ct_data = transpose_data(ciphertext.ct_data, old_shape, axes)

            # Create result CipherText with transposed shape and rearranged data
            return [
                CipherText(
                    ct_data=transposed_ct_data,
                    semantic_dtype=ciphertext.semantic_dtype,
                    semantic_shape=new_shape,
                    scheme=ciphertext.scheme,
                    key_size=ciphertext.key_size,
                    pk_data=ciphertext.pk_data,
                )
            ]

        except ValueError:
            # Re-raise ValueError directly (validation errors)
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to perform transpose: {e}") from e
