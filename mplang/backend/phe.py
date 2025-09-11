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

        if ct1.semantic_shape != ct2.semantic_shape:
            raise ValueError("CipherText operands must have same shape")

        # Perform homomorphic addition
        # lightPHE handles both scalar and tensor addition automatically
        raw_ct1: list[Any] = ct1.ct_data
        raw_ct2: list[Any] = ct2.ct_data
        result_ciphertext = [raw_ct1[i] + raw_ct2[i] for i in range(len(raw_ct1))]

        # Create result CipherText
        return CipherText(
            ct_data=result_ciphertext,
            semantic_dtype=ct1.semantic_dtype,
            semantic_shape=ct1.semantic_shape,
            scheme=ct1.scheme,
            key_size=ct1.key_size,
            pk_data=ct1.pk_data,
        )

    def _execute_add_ct2pt(
        self, ciphertext: CipherText, plaintext: TensorLike
    ) -> CipherText:
        """Execute CipherText + plaintext addition.

        Args:
            ciphertext: CipherText operand
            plaintext: Plaintext operand

        Returns:
            CipherText: Result of homomorphic addition
        """
        # Convert plaintext to numpy
        plaintext_np = self._convert_to_numpy(plaintext)

        # Validate shape compatibility
        if plaintext_np.shape != ciphertext.semantic_shape:
            raise ValueError(
                f"operands must have same shape: CipherText shape {ciphertext.semantic_shape} "
                f"vs plaintext shape {plaintext_np.shape}"
            )

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

        # Encrypt the plaintext using same method as original encryption
        target_dtype = ciphertext.semantic_dtype
        flat_data = plaintext_np.flatten()

        if target_dtype.is_floating:
            data_list = [float(x) for x in flat_data]
        else:  # integer types
            data_list = [int(x) for x in flat_data]

        encrypted_plaintext = [phe.encrypt([val]) for val in data_list]

        # Perform addition
        result_ciphertext = [
            encrypted_plaintext[i] + ciphertext.ct_data[i]
            for i in range(len(encrypted_plaintext))
        ]

        # Create result CipherText
        return CipherText(
            ct_data=result_ciphertext,
            semantic_dtype=ciphertext.semantic_dtype,
            semantic_shape=ciphertext.semantic_shape,
            scheme=ciphertext.scheme,
            key_size=ciphertext.key_size,
            pk_data=ciphertext.pk_data,
        )

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
        """Execute homomorphic dot product.

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

        try:
            # Convert plaintext to numpy
            plaintext_np = self._convert_to_numpy(plaintext)

            # Validate shape compatibility for dot product
            if len(ciphertext.semantic_shape) != 1 or len(plaintext_np.shape) != 1:
                raise ValueError("Both operands must be 1-D vectors for dot product")

            if ciphertext.semantic_shape[0] != plaintext_np.shape[0]:
                raise ValueError(
                    f"Vector size mismatch: CipherText size {ciphertext.semantic_shape[0]} "
                    f"vs plaintext size {plaintext_np.shape[0]}"
                )

            # Flatten the plaintext data
            target_dtype = ciphertext.semantic_dtype
            flat_data = plaintext_np.flatten()

            if target_dtype.is_floating:
                multiplier = [float(x) for x in flat_data]
            else:  # integer types
                multiplier = [int(x) for x in flat_data]

            # Perform homomorphic dot product
            # This is done by multiplying each element and summing up
            intermediate_products = [
                ciphertext.ct_data[i] * [multiplier[i]] for i in range(len(multiplier))
            ]

            # Sum up the intermediate products using lightPHE addition
            # Start with the first product and add the rest
            result_ciphertext = intermediate_products[0]
            for i in range(1, len(intermediate_products)):
                result_ciphertext = result_ciphertext + intermediate_products[i]

            # Create result CipherText with scalar shape
            return [
                CipherText(
                    ct_data=[
                        result_ciphertext
                    ],  # Wrap single result in list for consistency
                    semantic_dtype=ciphertext.semantic_dtype,
                    semantic_shape=(),  # Scalar result
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

        Args:
            pfunc: PFunction for gather
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

        try:
            # Convert indices to numpy
            indices_np = self._convert_to_numpy(indices)

            # indices should be 1-d or scalar
            assert indices_np.ndim in (0, 1), "Indices must be a scalar or 1-D array"

            if not np.issubdtype(indices_np.dtype, np.integer):
                raise ValueError("Indices must be of integer type")

            # Validate indices are within bounds
            if np.any(indices_np < 0) or np.any(
                indices_np >= ciphertext.semantic_shape[0]
            ):
                raise ValueError("Indices are out of bounds")

            # Perform gather operation
            gathered_ct_data = [ciphertext.ct_data[i] for i in indices_np.flatten()]

            # Determine new shape after gather
            new_shape = indices_np.shape

            # Create result CipherText
            return [
                CipherText(
                    ct_data=gathered_ct_data,
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
            raise RuntimeError(f"Failed to perform gather: {e}") from e

    def _execute_scatter(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:
        """Execute scatter operation on CipherText.

        Args:
            pfunc: PFunction for scatter
            args: [ciphertext, indices, updated] where ciphertext is CipherText, indices is TensorLike of integers,
                  and updated is the updated values as a CipherText

        Returns:
            list[TensorLike]: [result] where result is a CipherText with same shape as original ciphertext
        """
        if len(args) != 3:
            raise ValueError("Scatter expects exactly three arguments")

        ciphertext, indices, updated = args

        # Validate that first argument is a CipherText
        if not isinstance(ciphertext, CipherText) or not isinstance(
            updated, CipherText
        ):
            raise ValueError("First and third argument must be a CipherText instance")

        # Validate that both ciphertexts use same scheme/key_size
        if (
            ciphertext.scheme != updated.scheme
            or ciphertext.key_size != updated.key_size
        ):
            raise ValueError("Both CipherTexts must use same scheme and key size")

        assert (
            ciphertext.pk_data == updated.pk_data
        ), "Both CipherTexts must be encrypted with same key"

        try:
            # Convert indices to numpy
            indices_np = self._convert_to_numpy(indices)

            if not np.issubdtype(indices_np.dtype, np.integer):
                raise ValueError("Indices must be of integer type")

            # Validate shape
            if len(indices_np.shape) != 1:
                raise ValueError(
                    f"Indices must be a 1-D array, got shape {indices_np.shape}"
                )
            assert (
                len(updated.semantic_shape) == 1
            ), f"Updated values must be 1-D CipherText, got shape {updated.semantic_shape}"
            if len(indices_np) != updated.semantic_shape[0]:
                raise ValueError(
                    f"Number of indices must match number of updated values, got {len(indices_np)} vs {updated.semantic_shape[0]}"
                )

            # Create an empty list for scattered data
            scattered_ct_data = ciphertext.ct_data.copy()
            updated_ct_data = updated.ct_data

            # Scatter the ciphertext data according to indices
            for i, idx in enumerate(indices_np.flatten()):
                if idx < 0 or idx >= len(ciphertext.ct_data):
                    raise ValueError(f"Index {idx} is out of bounds")
                scattered_ct_data[idx] = updated_ct_data[i]

            # Create result CipherText
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

        Args:
            pfunc: PFunction for concat
            args: List of CipherText operands to concatenate

        Returns:
            list[TensorLike]: [result] where result is a CipherText
        """
        if len(args) != 2:
            raise ValueError("Concat expects exactly two arguments")

        c1, c2 = args

        # Validate that all arguments are CipherText
        assert isinstance(c1, CipherText) and isinstance(
            c2, CipherText
        ), "All arguments must be CipherText instances"

        # validate that all ciphertexts has the same key & scheme
        for arg in (c2,):
            if arg.scheme != c1.scheme or arg.key_size != c1.key_size:
                raise ValueError("All CipherTexts must use same scheme and key size")
            if arg.pk_data != c1.pk_data:
                raise ValueError("All CipherTexts must be encrypted with same key")
            if arg.semantic_dtype != c1.semantic_dtype:
                raise ValueError(
                    f"All CipherTexts must have same semantic dtype, got {c1.semantic_dtype} vs {arg.semantic_dtype}"
                )
            # Note: shapes don't need to be the same for concat, just dimension count should be the same
            if len(arg.semantic_shape) != len(c1.semantic_shape):
                raise ValueError(
                    f"All CipherTexts must have same number of dimensions for concat, got {len(c1.semantic_shape)} vs {len(arg.semantic_shape)}"
                )

        assert (
            len(c1.semantic_shape) == 1
        ), f"Only 1-D CipherTexts are supported for concat, got shape {c1.semantic_shape}"

        try:
            # Validate compatibility and collect data
            total_length = c1.semantic_shape[0] + c2.semantic_shape[0]
            semantic_dtype = c1.semantic_dtype
            scheme = c1.scheme
            key_size = c1.key_size
            pk_data = c1.pk_data

            # Concatenate the ciphertext data
            concatenated_ct_data = []
            for ct in [c1, c2]:
                concatenated_ct_data.extend(ct.ct_data)

            # Create result CipherText with new shape
            return [
                CipherText(
                    ct_data=concatenated_ct_data,
                    semantic_dtype=semantic_dtype,
                    semantic_shape=(total_length,),
                    scheme=scheme,
                    key_size=key_size,
                    pk_data=pk_data,
                )
            ]

        except ValueError:
            # Re-raise ValueError directly (validation errors)
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to perform concat: {e}") from e
