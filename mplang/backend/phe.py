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

from mplang.backend.base import kernel_def
from mplang.core.dtype import DType
from mplang.core.mptype import TensorLike
from mplang.core.pfunc import PFunction

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


def _to_numpy(obj: TensorLike) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "numpy"):
        try:
            return np.asarray(obj.numpy())  # type: ignore
        except Exception:
            pass
    return np.asarray(obj)


@kernel_def("phe.keygen")
def _phe_keygen(pfunc: PFunction) -> Any:
    scheme = pfunc.attrs.get("scheme", "paillier")
    key_size = pfunc.attrs.get("key_size", 2048)
    if scheme.lower() not in ["paillier", "elgamal"]:
        raise ValueError(f"Unsupported PHE scheme: {scheme}")
    scheme_cap = scheme.capitalize()
    try:
        phe = LightPHE(
            algorithm_name=scheme_cap, key_size=key_size, precision=PRECISION
        )
        pk_data = phe.cs.keys["public_key"]
        sk_data = phe.cs.keys["private_key"]
        public_key = PublicKey(key_data=pk_data, scheme=scheme_cap, key_size=key_size)
        private_key = PrivateKey(
            sk_data=sk_data, pk_data=pk_data, scheme=scheme_cap, key_size=key_size
        )
        return (public_key, private_key)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to generate PHE keys: {e}") from e


@kernel_def("phe.encrypt")
def _phe_encrypt(pfunc: PFunction, plaintext: Any, public_key: Any) -> Any:
    if not isinstance(public_key, PublicKey):
        raise ValueError("second arg must be PublicKey")
    try:
        pt_np = _to_numpy(plaintext)
        semantic_dtype = DType.from_numpy(pt_np.dtype)
        semantic_shape = pt_np.shape
        phe = LightPHE(
            algorithm_name=public_key.scheme,
            key_size=public_key.key_size,
            precision=PRECISION,
        )
        phe.cs.keys["public_key"] = public_key.key_data
        flat = pt_np.flatten()
        if semantic_dtype.is_floating:
            data_list = [float(x) for x in flat]
        else:
            data_list = [int(x) for x in flat]
        ct_data = phe.encrypt(data_list)
        ciphertext = CipherText(
            ct_data=ct_data,
            semantic_dtype=semantic_dtype,
            semantic_shape=semantic_shape,
            scheme=public_key.scheme,
            key_size=public_key.key_size,
            pk_data=public_key.key_data,
        )
        return ciphertext
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to encrypt data: {e}") from e


@kernel_def("phe.mul")
def _phe_mul(pfunc: PFunction, ciphertext: Any, plaintext: Any) -> Any:
    if not isinstance(ciphertext, CipherText):
        raise ValueError("first arg must be CipherText")
    try:
        pt_np = _to_numpy(plaintext)
        if pt_np.shape != ciphertext.semantic_shape:
            raise ValueError("shape mismatch for phe.mul")
        target_dtype = ciphertext.semantic_dtype
        flat = pt_np.flatten()
        if target_dtype.is_floating:
            mult = [float(x) for x in flat]
        else:
            mult = [int(x) for x in flat]
        res_ct = ciphertext.ct_data * mult
        return CipherText(
            ct_data=res_ct,
            semantic_dtype=ciphertext.semantic_dtype,
            semantic_shape=ciphertext.semantic_shape,
            scheme=ciphertext.scheme,
            key_size=ciphertext.key_size,
            pk_data=ciphertext.pk_data,
        )
    except ValueError:
        raise
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to perform multiplication: {e}") from e


@kernel_def("phe.add")
def _phe_add(pfunc: PFunction, lhs: Any, rhs: Any) -> Any:
    try:
        if isinstance(lhs, CipherText) and isinstance(rhs, CipherText):
            return _phe_add_ct2ct(lhs, rhs)
        elif isinstance(lhs, CipherText):
            return _phe_add_ct2pt(lhs, rhs)
        elif isinstance(rhs, CipherText):
            return _phe_add_ct2pt(rhs, lhs)
        else:
            return _to_numpy(lhs) + _to_numpy(rhs)
    except ValueError:
        raise
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to perform addition: {e}") from e


def _phe_add_ct2ct(ct1: CipherText, ct2: CipherText) -> CipherText:
    if ct1.scheme != ct2.scheme or ct1.key_size != ct2.key_size:
        raise ValueError("CipherText operands must use same scheme/key size")
    if ct1.pk_data != ct2.pk_data:
        raise ValueError("CipherText operands must share public key")
    if ct1.semantic_shape != ct2.semantic_shape:
        raise ValueError("CipherText operands must have same shape")
    res_ct = ct1.ct_data + ct2.ct_data
    return CipherText(
        ct_data=res_ct,
        semantic_dtype=ct1.semantic_dtype,
        semantic_shape=ct1.semantic_shape,
        scheme=ct1.scheme,
        key_size=ct1.key_size,
        pk_data=ct1.pk_data,
    )


def _phe_add_ct2pt(ciphertext: CipherText, plaintext: TensorLike) -> CipherText:
    pt_np = _to_numpy(plaintext)
    if pt_np.shape != ciphertext.semantic_shape:
        raise ValueError("operands must have same shape")
    if ciphertext.pk_data is None:
        raise ValueError("CipherText missing public key data")
    phe = LightPHE(
        algorithm_name=ciphertext.scheme,
        key_size=ciphertext.key_size,
        precision=PRECISION,
    )
    phe.cs.keys["public_key"] = ciphertext.pk_data
    target_dtype = ciphertext.semantic_dtype
    flat = pt_np.flatten()
    if target_dtype.is_floating:
        data_list = [float(x) for x in flat]
    else:
        data_list = [int(x) for x in flat]
    enc_pt = phe.encrypt(data_list)
    res_ct = ciphertext.ct_data + enc_pt
    return CipherText(
        ct_data=res_ct,
        semantic_dtype=ciphertext.semantic_dtype,
        semantic_shape=ciphertext.semantic_shape,
        scheme=ciphertext.scheme,
        key_size=ciphertext.key_size,
        pk_data=ciphertext.pk_data,
    )


@kernel_def("phe.decrypt")
def _phe_decrypt(pfunc: PFunction, ciphertext: Any, private_key: Any) -> Any:
    if not isinstance(ciphertext, CipherText):
        raise ValueError("first arg must be CipherText")
    if not isinstance(private_key, PrivateKey):
        raise ValueError("second arg must be PrivateKey")
    if (
        ciphertext.scheme != private_key.scheme
        or ciphertext.key_size != private_key.key_size
    ):
        raise ValueError("CipherText and PrivateKey must use same scheme/key size")
    try:
        phe = LightPHE(
            algorithm_name=private_key.scheme,
            key_size=private_key.key_size,
            precision=PRECISION,
        )
        phe.cs.keys["private_key"] = private_key.sk_data
        phe.cs.keys["public_key"] = private_key.pk_data
        target_dtype = ciphertext.semantic_dtype.to_numpy()
        decrypted_raw = phe.decrypt(ciphertext.ct_data)
        if not isinstance(decrypted_raw, list):
            raise RuntimeError("Expected list from decryption")
        expected_size = (
            int(np.prod(ciphertext.semantic_shape)) if ciphertext.semantic_shape else 1
        )
        if len(decrypted_raw) != expected_size:
            raise RuntimeError("Unexpected decrypted length")
        if target_dtype.kind in "iu":
            info = np.iinfo(target_dtype)
            processed = [max(info.min, min(info.max, v)) for v in decrypted_raw]
        else:
            processed = decrypted_raw
        plaintext_np = np.array(processed, dtype=target_dtype).reshape(
            ciphertext.semantic_shape
        )
        return plaintext_np
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to decrypt data: {e}") from e
