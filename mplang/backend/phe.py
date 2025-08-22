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

    def __init__(self, key_data: Any, public_key_data: Any, scheme: str, key_size: int):
        self.key_data = key_data
        self.public_key_data = public_key_data  # Store public key data as well
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
        public_key_data: Any = None,  # Store public key for operations
    ):
        self.ct_data = ct_data
        self.semantic_dtype = semantic_dtype
        self.semantic_shape = semantic_shape
        self.scheme = scheme
        self.key_size = key_size
        self.public_key_data = public_key_data

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

    def __init__(self):
        super().__init__()

    def setup(self) -> None: ...
    def teardown(self) -> None: ...
    def list_fn_names(self) -> list[str]:
        return [
            "phe.keygen",
            "phe.encrypt",
            "phe.add",
            "phe.decrypt",
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
        elif pfunc.fn_type == "phe.decrypt":
            return self._execute_decrypt(pfunc, args)
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
            phe = LightPHE(algorithm_name=scheme, key_size=key_size)

            pk_data = phe.cs.keys["public_key"]
            sk_data = phe.cs.keys["private_key"]

            pk_data = PublicKey(key_data=pk_data, scheme=scheme, key_size=key_size)
            sk_data = PrivateKey(key_data=sk_data, public_key_data=pk_data.key_data, scheme=scheme, key_size=key_size)

            return [pk_data, sk_data]

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
            phe = LightPHE(
                algorithm_name=public_key.scheme, key_size=public_key.key_size
            )
            # Set the public key
            phe.cs.keys["public_key"] = public_key.key_data

            # Encrypt the data
            if plaintext_np.ndim == 0:
                # Scalar case
                encrypted = phe.encrypt(plaintext_np.item())
                lightphe_ciphertext = encrypted
            else:
                # Array case - encrypt element-wise
                flat_data = plaintext_np.flatten()
                encrypted_data = [phe.encrypt(float(x)) for x in flat_data]
                lightphe_ciphertext = encrypted_data

            # Create CipherText object
            ciphertext = CipherText(
                ct_data=lightphe_ciphertext,
                semantic_dtype=semantic_dtype,
                semantic_shape=semantic_shape,
                scheme=public_key.scheme,
                key_size=public_key.key_size,
                public_key_data=public_key.key_data,  # Store public key for later operations
            )

            return [ciphertext]

        except Exception as e:
            raise RuntimeError(f"Failed to encrypt data: {e}") from e

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

        if ct1.semantic_shape != ct2.semantic_shape:
            raise ValueError("CipherText operands must have same shape")

        # Perform homomorphic addition
        if len(ct1.semantic_shape) == 0:
            # Scalar case
            result_ciphertext = ct1.ct_data + ct2.ct_data
        else:
            # Array case
            ct1_list = ct1.ct_data
            ct2_list = ct2.ct_data
            result_ciphertext = [
                c1 + c2 for c1, c2 in zip(ct1_list, ct2_list, strict=True)
            ]

        # Create result CipherText
        return CipherText(
            ct_data=result_ciphertext,
            semantic_dtype=ct1.semantic_dtype,
            semantic_shape=ct1.semantic_shape,
            scheme=ct1.scheme,
            key_size=ct1.key_size,
            public_key_data=ct1.public_key_data,
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
                f"Shape mismatch: CipherText shape {ciphertext.semantic_shape} "
                f"vs plaintext shape {plaintext_np.shape}"
            )

        # For ciphertext + plaintext addition, we encrypt the plaintext first
        # and then do ciphertext + ciphertext addition
        if ciphertext.public_key_data is None:
            raise ValueError("CipherText must contain public key data for plaintext addition")
        
        # Create lightPHE instance to encrypt the plaintext
        phe = LightPHE(algorithm_name=ciphertext.scheme, key_size=ciphertext.key_size)
        phe.cs.keys["public_key"] = ciphertext.public_key_data
        
        # Encrypt the plaintext
        if len(ciphertext.semantic_shape) == 0:
            # Scalar case
            plaintext_value = plaintext_np.item()
            encrypted_plaintext = phe.encrypt(plaintext_value)
            
            # Add the two ciphertexts
            result_ciphertext = ciphertext.ct_data + encrypted_plaintext
        else:
            # Array case
            pt_flat = plaintext_np.flatten()
            encrypted_data = [phe.encrypt(float(pt)) for pt in pt_flat]
            
            # Add element-wise
            ct_list = ciphertext.ct_data
            result_ciphertext = [
                ct + pt_ct for ct, pt_ct in zip(ct_list, encrypted_data, strict=True)
            ]

        # Create result CipherText
        return CipherText(
            ct_data=result_ciphertext,
            semantic_dtype=ciphertext.semantic_dtype,
            semantic_shape=ciphertext.semantic_shape,
            scheme=ciphertext.scheme,
            key_size=ciphertext.key_size,
            public_key_data=ciphertext.public_key_data,
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
                algorithm_name=private_key.scheme, key_size=private_key.key_size
            )
            # Set both public and private keys (lightPHE needs both for proper decryption)
            phe.cs.keys["private_key"] = private_key.key_data
            phe.cs.keys["public_key"] = private_key.public_key_data

            # Decrypt the data
            if len(ciphertext.semantic_shape) == 0:
                # Scalar case
                decrypted = phe.decrypt(ciphertext.ct_data)
                # Convert to target dtype with proper handling of large integers
                target_dtype = ciphertext.semantic_dtype.to_numpy()
                
                # Handle overflow for smaller integer types
                if target_dtype.kind in 'iu':  # integer types
                    # Check if the value fits in the target type
                    info = np.iinfo(target_dtype)
                    if decrypted < info.min or decrypted > info.max:
                        # Clamp to valid range for the target type
                        decrypted = max(info.min, min(info.max, decrypted))
                
                plaintext_np = np.array(decrypted, dtype=target_dtype)
            else:
                # Array case
                encrypted_data = ciphertext.ct_data
                decrypted_data = [phe.decrypt(ct) for ct in encrypted_data]
                target_dtype = ciphertext.semantic_dtype.to_numpy()
                
                # Handle overflow for arrays
                if target_dtype.kind in 'iu':  # integer types
                    info = np.iinfo(target_dtype)
                    decrypted_data = [
                        max(info.min, min(info.max, val)) for val in decrypted_data
                    ]
                
                plaintext_np = np.array(
                    decrypted_data, dtype=target_dtype
                ).reshape(ciphertext.semantic_shape)

            return [plaintext_np]

        except Exception as e:
            raise RuntimeError(f"Failed to decrypt data: {e}") from e
