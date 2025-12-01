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

from __future__ import annotations

import json
from typing import Any, ClassVar

import numpy as np
from lightphe import LightPHE
from lightphe.models.Ciphertext import Ciphertext

from mplang.v1.core import DType, PFunction
from mplang.v1.kernels.base import kernel_def
from mplang.v1.kernels.value import (
    TensorValue,
    Value,
    ValueDecodeError,
    ValueProtoBuilder,
    ValueProtoReader,
    register_value,
)
from mplang.v1.protos.v1alpha1 import value_pb2 as _value_pb2

# This controls the decimal precision used in lightPHE for float operations
# we force it to 0 to only support integer operations
# we will support negative and floating-point with our own encoding/decoding
PRECISION = 0


@register_value
class PublicKey(Value):
    """PHE Public Key Value type."""

    KIND: ClassVar[str] = "mplang.phe.PublicKey"
    WIRE_VERSION: ClassVar[int] = 1

    def __init__(
        self,
        key_data: Any,
        scheme: str,
        key_size: int,
        max_value: int = 2**100,
        fxp_bits: int = 12,
        modulus: int | None = None,
    ):
        self.key_data = key_data
        self.scheme = scheme
        self.key_size = key_size
        self.max_value = max_value  # Maximum absolute value B for range encoding
        self.fxp_bits = fxp_bits  # Fixed-point precision bits for float encoding
        self.modulus = modulus  # Paillier modulus N for range encoding

    @property
    def dtype(self) -> Any:
        return np.dtype("O")  # Use object dtype for binary data

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    @property
    def max_float_value(self) -> float:
        """Maximum float value that can be encoded."""
        return float(self.max_value / (2**self.fxp_bits))

    def to_proto(self) -> _value_pb2.ValueProto:
        """Serialize PublicKey to wire format."""
        return (
            ValueProtoBuilder(self.KIND, self.WIRE_VERSION)
            .set_attr("scheme", self.scheme)
            .set_attr("key_size", self.key_size)
            .set_attr("max_value", self.max_value)
            .set_attr("fxp_bits", self.fxp_bits)
            .set_attr("modulus", str(self.modulus) if self.modulus is not None else "")
            .set_payload(json.dumps(self.key_data).encode("utf-8"))
            .build()
        )

    @classmethod
    def from_proto(cls, proto: _value_pb2.ValueProto) -> PublicKey:
        """Deserialize PublicKey from wire format."""
        reader = ValueProtoReader(proto)
        if reader.version != cls.WIRE_VERSION:
            raise ValueDecodeError(f"Unsupported PublicKey version {reader.version}")

        # Read metadata from runtime_attrs
        scheme = reader.get_attr("scheme")
        key_size = reader.get_attr("key_size")
        max_value = reader.get_attr("max_value")
        fxp_bits = reader.get_attr("fxp_bits")
        modulus_str = reader.get_attr("modulus")
        modulus = None if modulus_str == "" else int(modulus_str)

        # JSON deserialize the public key dict
        key_data = json.loads(reader.payload.decode("utf-8"))

        return cls(
            key_data=key_data,
            scheme=scheme,
            key_size=key_size,
            max_value=max_value,
            fxp_bits=fxp_bits,
            modulus=modulus,
        )

    def __repr__(self) -> str:
        return f"PublicKey(scheme={self.scheme}, key_size={self.key_size}, max_value={self.max_value}, fxp_bits={self.fxp_bits})"


@register_value
class PrivateKey(Value):
    """PHE Private Key Value type."""

    KIND: ClassVar[str] = "mplang.phe.PrivateKey"
    WIRE_VERSION: ClassVar[int] = 1

    def __init__(
        self,
        sk_data: Any,
        pk_data: Any,
        scheme: str,
        key_size: int,
        max_value: int = 2**100,
        fxp_bits: int = 12,
        modulus: int | None = None,
    ):
        self.sk_data = sk_data  # Store private key data
        self.pk_data = pk_data  # Store public key data as well
        self.scheme = scheme
        self.key_size = key_size
        self.max_value = max_value  # Maximum absolute value B for range encoding
        self.fxp_bits = fxp_bits  # Fixed-point precision bits for float encoding
        self.modulus = modulus  # Paillier modulus N for range encoding

    @property
    def dtype(self) -> Any:
        return np.dtype("O")  # Use object dtype for binary data

    @property
    def shape(self) -> tuple[int, ...]:
        return ()

    @property
    def max_float_value(self) -> float:
        """Maximum float value that can be encoded."""
        return float(self.max_value / (2**self.fxp_bits))

    def to_proto(self) -> _value_pb2.ValueProto:
        """Serialize PrivateKey to wire format."""
        # JSON serialize both key dicts (contain int values)
        # Store both keys in a single dict to avoid needing length metadata
        keys_dict = {"sk": self.sk_data, "pk": self.pk_data}

        return (
            ValueProtoBuilder(self.KIND, self.WIRE_VERSION)
            .set_attr("scheme", self.scheme)
            .set_attr("key_size", self.key_size)
            .set_attr("max_value", self.max_value)
            .set_attr("fxp_bits", self.fxp_bits)
            .set_attr("modulus", str(self.modulus) if self.modulus is not None else "")
            .set_payload(json.dumps(keys_dict).encode("utf-8"))
            .build()
        )

    @classmethod
    def from_proto(cls, proto: _value_pb2.ValueProto) -> PrivateKey:
        """Deserialize PrivateKey from wire format."""
        reader = ValueProtoReader(proto)
        if reader.version != cls.WIRE_VERSION:
            raise ValueDecodeError(f"Unsupported PrivateKey version {reader.version}")

        # Read metadata from runtime_attrs
        scheme = reader.get_attr("scheme")
        key_size = reader.get_attr("key_size")
        max_value = reader.get_attr("max_value")
        fxp_bits = reader.get_attr("fxp_bits")
        modulus_str = reader.get_attr("modulus")
        modulus = None if modulus_str == "" else int(modulus_str)

        # JSON deserialize both key dicts
        keys_dict = json.loads(reader.payload.decode("utf-8"))
        sk_data = keys_dict["sk"]
        pk_data = keys_dict["pk"]

        return cls(
            sk_data=sk_data,
            pk_data=pk_data,
            scheme=scheme,
            key_size=key_size,
            max_value=max_value,
            fxp_bits=fxp_bits,
            modulus=modulus,
        )

    def __repr__(self) -> str:
        return f"PrivateKey(scheme={self.scheme}, key_size={self.key_size}, max_value={self.max_value}, fxp_bits={self.fxp_bits})"


@register_value
class CipherText(Value):
    """PHE CipherText Value type."""

    KIND: ClassVar[str] = "mplang.phe.CipherText"
    WIRE_VERSION: ClassVar[int] = 1

    def __init__(
        self,
        ct_data: Any,
        semantic_dtype: DType,
        semantic_shape: tuple[int, ...],
        scheme: str,
        key_size: int,
        pk_data: Any = None,  # Store public key for operations
        max_value: int = 2**100,
        fxp_bits: int = 12,
        modulus: int | None = None,
    ):
        self.ct_data = ct_data
        self.semantic_dtype = semantic_dtype
        self.semantic_shape = semantic_shape
        self.scheme = scheme
        self.key_size = key_size
        self.pk_data = pk_data
        self.max_value = max_value
        self.fxp_bits = fxp_bits
        self.modulus = modulus

    @property
    def dtype(self) -> Any:
        return self.semantic_dtype.to_numpy()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.semantic_shape

    @property
    def max_float_value(self) -> float:
        """Maximum float value that can be encoded."""
        return float(self.max_value / (2**self.fxp_bits))

    def to_proto(self) -> _value_pb2.ValueProto:
        """Serialize CipherText to wire format.

        WARNING: This serialization is tightly coupled to lightphe.Ciphertext
        internal attributes (value, algorithm_name, keys). Any changes to these
        attributes in future lightphe versions will break serialization.

        TODO: Check if lightphe provides official serialization methods and
        migrate to them if available. Consider adding version compatibility checks.
        """
        # JSON serialize ciphertext components
        # ct_data is a list of lightPHE Ciphertext objects
        # Each Ciphertext has: value, algorithm_name, keys
        # We need to serialize the list of ciphertexts
        if not isinstance(self.ct_data, list):
            raise ValueError(f"ct_data should be a list, got {type(self.ct_data)}")

        ct_list = []
        for ct in self.ct_data:
            if not isinstance(ct, Ciphertext):
                raise TypeError(
                    f"ct_data must contain lightphe.Ciphertext objects, got {type(ct).__name__}"
                )
            ct_list.append({
                "value": ct.value,
                "algorithm_name": ct.algorithm_name,
                "keys": ct.keys,
            })

        # Combine ct_data and pk_data into single dict
        payload_dict = {
            "ct_list": ct_list,
            "pk": self.pk_data if self.pk_data is not None else None,
        }

        return (
            ValueProtoBuilder(self.KIND, self.WIRE_VERSION)
            .set_attr("semantic_dtype", str(self.semantic_dtype))
            .set_attr("semantic_shape", list(self.semantic_shape))
            .set_attr("scheme", self.scheme)
            .set_attr("key_size", self.key_size)
            .set_attr("max_value", self.max_value)
            .set_attr("fxp_bits", self.fxp_bits)
            .set_attr("modulus", str(self.modulus) if self.modulus is not None else "")
            .set_payload(json.dumps(payload_dict).encode("utf-8"))
            .build()
        )

    @classmethod
    def from_proto(cls, proto: _value_pb2.ValueProto) -> CipherText:
        """Deserialize CipherText from wire format."""
        reader = ValueProtoReader(proto)
        if reader.version != cls.WIRE_VERSION:
            raise ValueDecodeError(f"Unsupported CipherText version {reader.version}")

        # Read metadata from runtime_attrs
        semantic_dtype_str = reader.get_attr("semantic_dtype")
        semantic_shape = reader.get_attr("semantic_shape")
        scheme = reader.get_attr("scheme")
        key_size = reader.get_attr("key_size")
        max_value = reader.get_attr("max_value")
        fxp_bits = reader.get_attr("fxp_bits")
        modulus_str = reader.get_attr("modulus")
        modulus = None if modulus_str == "" else int(modulus_str)

        # JSON deserialize ciphertext and public key
        payload_dict = json.loads(reader.payload.decode("utf-8"))
        ct_list = payload_dict["ct_list"]
        pk_data = payload_dict["pk"]

        # Reconstruct ct_data: list of Ciphertext objects
        ct_data = []
        for ct_dict in ct_list:
            if ct_dict["keys"] is None or ct_dict["algorithm_name"] is None:
                raise ValueDecodeError(
                    "Invalid CipherText: missing keys or algorithm_name in serialized data"
                )
            ct_data.append(
                Ciphertext(
                    algorithm_name=ct_dict["algorithm_name"],
                    keys=ct_dict["keys"],
                    value=ct_dict["value"],
                )
            )

        # Parse dtype string back to DType
        dtype = DType.from_any(semantic_dtype_str)

        return cls(
            ct_data=ct_data,
            semantic_dtype=dtype,
            semantic_shape=tuple(semantic_shape),
            scheme=scheme,
            key_size=key_size,
            pk_data=pk_data,
            max_value=max_value,
            fxp_bits=fxp_bits,
            modulus=modulus,
        )

    def __repr__(self) -> str:
        return f"CipherText(dtype={self.semantic_dtype}, shape={self.semantic_shape}, scheme={self.scheme})"


# Range-based encoding functions for negative numbers and floats
def _range_encode_integer(value: int, max_value: int, modulus: int) -> int:
    """
    Range encoding function for integers.
    - Positive numbers: encode(m) = m
    - Negative numbers: encode(m) = N + m
    """
    if not (-max_value <= value <= max_value):
        raise ValueError(
            f"Integer value {value} out of range [-{max_value}, {max_value}]"
        )

    if value >= 0:
        encoded = value
    else:
        encoded = modulus + value

    return encoded


def _range_encode_float(
    value: float, max_value: int, fxp_bits: int, modulus: int
) -> int:
    """
    Range encoding function for floats.
    1. Fixed-point conversion: scaled_int = round(value * 2^fxp_bits)
    2. Integer encoding rules
    """
    max_float = max_value / (2**fxp_bits)
    if not (-max_float <= value <= max_float):
        raise ValueError(
            f"Float value {value} out of range [-{max_float}, {max_float}]"
        )

    # Fixed-point encoding: float → scaled integer
    scaled_int = round(value * (2**fxp_bits))

    # Use integer encoding rules
    return _range_encode_integer(scaled_int, max_value, modulus)


def _range_encode_mixed(
    value: Any, max_value: int, fxp_bits: int, modulus: int, semantic_dtype: DType
) -> int:
    """
    Mixed encoding function - automatically handle integers and floats based on semantic type.
    Use semantic_dtype to choose between integer and float encoding.
    """
    if semantic_dtype.is_floating:
        # For floating semantic types, always use float encoding
        return _range_encode_float(float(value), max_value, fxp_bits, modulus)
    else:
        # For integer semantic types, use integer encoding
        return _range_encode_integer(int(value), max_value, modulus)


def _range_decode_integer(encoded_value: int, max_value: int, modulus: int) -> int:
    """
    Range decoding function for integers.
    - If r <= max_value: decode(r) = r
    - If r >= N - max_value: decode(r) = r - N
    - If max_value < r < N - max_value: overflow error
    """

    # Ensure handling integer
    if isinstance(encoded_value, (list, tuple)):
        encoded_value = encoded_value[0]
    encoded_value = int(encoded_value) % modulus

    if encoded_value <= max_value:
        return encoded_value
    elif encoded_value >= modulus - max_value:
        return encoded_value - modulus
    else:
        raise ValueError(f"Decoded value {encoded_value} is in overflow region")


def _range_decode_float(
    encoded_value: int, max_value: int, fxp_bits: int, modulus: int
) -> float:
    """
    Range decoding function for floats.
    1. Integer decoding: decoded_int = range_decode_integer(encoded_value)
    2. Fixed-point conversion: value = decoded_int / 2^fxp_bits
    """
    # First decode as integer
    decoded_int = _range_decode_integer(encoded_value, max_value, modulus)

    # Fixed-point decoding: scaled integer → float
    return float(decoded_int / (2**fxp_bits))


def _range_decode_mixed(
    encoded_value: int,
    max_value: int,
    fxp_bits: int,
    modulus: int,
    semantic_dtype: DType,
) -> Any:
    """
    Mixed decoding function - decode based on semantic type.
    Use semantic_dtype to choose between integer and float decoding.
    """
    if semantic_dtype.is_floating:
        # For floating semantic types, decode as float
        return _range_decode_float(encoded_value, max_value, fxp_bits, modulus)
    else:
        # For integer semantic types, decode as integer
        return _range_decode_integer(encoded_value, max_value, modulus)


@kernel_def("phe.keygen")
def _phe_keygen(pfunc: PFunction) -> Any:
    scheme = pfunc.attrs.get("scheme", "paillier")
    # use small key_size to speed up tests
    # in production use at least 2048 bits or 3072 bits for better security
    key_size = pfunc.attrs.get("key_size", 2048)
    # Accept very large max_value; allow decimal string input, kept simple like other attrs
    max_value = int(pfunc.attrs.get("max_value", 2**32))
    fxp_bits = int(pfunc.attrs.get("fxp_bits", 12))

    # Validate scheme
    if scheme.lower() not in ["paillier"]:
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
        modulus = phe.cs.plaintext_modulo  # Get Paillier modulus N

        # Validate safety: N should be much larger than 3*max_value
        if modulus <= 3 * max_value:
            raise ValueError(
                f"Modulus {modulus} is too small for max_value {max_value}. Require N >> 3*B"
            )

        public_key = PublicKey(
            key_data=pk_data,
            scheme=scheme,
            key_size=key_size,
            max_value=max_value,
            fxp_bits=fxp_bits,
            modulus=modulus,
        )
        private_key = PrivateKey(
            sk_data=sk_data,
            pk_data=pk_data,
            scheme=scheme,
            key_size=key_size,
            max_value=max_value,
            fxp_bits=fxp_bits,
            modulus=modulus,
        )

        return [public_key, private_key]

    except Exception as e:
        raise RuntimeError(f"Failed to generate PHE keys: {e}") from e


@kernel_def("phe.encrypt")
def _phe_encrypt(
    pfunc: PFunction, plaintext: TensorValue, public_key: PublicKey
) -> Any:
    # Validate public_key type
    if not isinstance(public_key, PublicKey):
        raise ValueError("Second argument must be a PublicKey instance")

    try:
        # Convert plaintext to numpy to get semantic type info
        plaintext_np = plaintext.to_numpy()
        semantic_dtype = DType.from_numpy(plaintext_np.dtype)
        semantic_shape = plaintext_np.shape

        # Create lightPHE instance with the same scheme/key_size as the key
        phe = LightPHE(
            algorithm_name=public_key.scheme,
            key_size=public_key.key_size,
            precision=PRECISION,
        )

        # CRITICAL: Set the same modulus as the key to ensure consistency
        if public_key.modulus is not None:
            phe.cs.plaintext_modulo = public_key.modulus
            phe.cs.ciphertext_modulo = public_key.modulus * public_key.modulus

        # Set the public key
        phe.cs.keys["public_key"] = public_key.key_data

        # Prepare data for encryption using range encoding
        flat_data = plaintext_np.flatten()

        # Use mixed encoding for consistent handling of integers and floats
        encoded_data_list = []
        for val in flat_data:
            # Use mixed encoding to handle both integers and floats uniformly
            if public_key.modulus is None:
                raise ValueError(
                    "Public key modulus is None, key generation may have failed"
                )
            encoded_val = _range_encode_mixed(
                val,
                public_key.max_value,
                public_key.fxp_bits,
                public_key.modulus,
                semantic_dtype,
            )
            encoded_data_list.append(encoded_val)

        # Encrypt the encoded values (note: not passing as list, just the value)
        lightphe_ciphertext = [phe.encrypt(val) for val in encoded_data_list]

        # Create CipherText object with encoding parameters
        ciphertext = CipherText(
            ct_data=lightphe_ciphertext,
            semantic_dtype=semantic_dtype,
            semantic_shape=semantic_shape,
            scheme=public_key.scheme,
            key_size=public_key.key_size,
            pk_data=public_key.key_data,
            max_value=public_key.max_value,
            fxp_bits=public_key.fxp_bits,
            modulus=public_key.modulus,
        )

        return [ciphertext]

    except Exception as e:
        raise RuntimeError(f"Failed to encrypt data: {e}") from e


@kernel_def("phe.mul")
def _phe_mul(pfunc: PFunction, ciphertext: CipherText, plaintext: TensorValue) -> Any:
    # Validate that first argument is a CipherText
    if not isinstance(ciphertext, CipherText):
        raise ValueError("First argument must be a CipherText instance")

    try:
        # Convert plaintext to numpy
        plaintext_np = plaintext.to_numpy()

        # Check if plaintext is floating point type - multiplication not supported
        if np.issubdtype(plaintext_np.dtype, np.floating):
            raise ValueError(
                f"Homomorphic multiplication with floating point plaintext is not supported. "
                f"Got plaintext dtype: {plaintext_np.dtype}"
            )

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
            ) from e

        # Broadcast plaintext to match result shape if needed
        if plaintext_np.shape != result_shape:
            plaintext_broadcasted = np.broadcast_to(plaintext_np, result_shape)
        else:
            plaintext_broadcasted = plaintext_np

        # If ciphertext needs broadcasting, we need to replicate its encrypted values
        if ciphertext.semantic_shape != result_shape:
            # Use numpy to create a properly broadcasted index mapping
            # Create a dummy array with same shape as ciphertext, fill with indices
            dummy_ct = (
                np.arange(np.prod(ciphertext.semantic_shape))
                .reshape(ciphertext.semantic_shape)
                .astype(np.int64)
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

        # For multiplication, plaintext multipliers should NOT be encoded
        # The ciphertext already contains the encoded value, multiplying by raw plaintext preserves semantics
        raw_multipliers = []
        for val in flat_data:
            # Convert to appropriate numeric type but don't apply any encoding
            if target_dtype.is_floating:
                raw_val = float(val)
            else:
                raw_val = int(val)
            raw_multipliers.append(raw_val)

        # Perform homomorphic multiplication
        # In Paillier, ciphertext * plaintext is supported
        result_ciphertext = [
            broadcasted_ct_data[i] * raw_multipliers[i]
            for i in range(len(raw_multipliers))
        ]

        # Create result CipherText with the broadcasted shape and encoding parameters
        return [
            CipherText(
                ct_data=result_ciphertext,
                semantic_dtype=ciphertext.semantic_dtype,
                semantic_shape=result_shape,
                scheme=ciphertext.scheme,
                key_size=ciphertext.key_size,
                pk_data=ciphertext.pk_data,
                max_value=ciphertext.max_value,
                fxp_bits=ciphertext.fxp_bits,
                modulus=ciphertext.modulus,
            )
        ]

    except ValueError:
        # Re-raise ValueError directly (validation errors)
        raise
    except Exception as e:
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
            return TensorValue(lhs.to_numpy() + rhs.to_numpy())
    except ValueError:
        raise
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to perform addition: {e}") from e


def _phe_add_ct2ct(ct1: CipherText, ct2: CipherText) -> CipherText:
    # Validate compatibility
    if ct1.scheme != ct2.scheme or ct1.key_size != ct2.key_size:
        raise ValueError("CipherText operands must use same scheme and key size")

    if ct1.pk_data != ct2.pk_data:
        raise ValueError("CipherText operands must be encrypted with same key")

    # Check for mixed precision issue: floating point ciphertext + integer ciphertext
    # This would cause decode failures due to different fixed-point encoding scales
    if ct1.semantic_dtype.is_floating != ct2.semantic_dtype.is_floating:
        raise ValueError(
            f"Cannot add ciphertexts with different numeric types due to fixed-point encoding. "
            f"First CipherText dtype: {ct1.semantic_dtype}, second CipherText dtype: {ct2.semantic_dtype}. "
            f"Both operands must have the same numeric type (both floating or both integer)."
        )

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
        ) from e

    # Broadcast ct1 if needed
    if ct1.semantic_shape != result_shape:
        dummy_ct1 = (
            np.arange(np.prod(ct1.semantic_shape))
            .reshape(ct1.semantic_shape)
            .astype(np.int64)
        )
        broadcasted_indices1 = np.broadcast_to(dummy_ct1, result_shape).flatten()
        raw_ct1: list[Any] = ct1.ct_data
        broadcasted_ct1_data = [raw_ct1[int(idx)] for idx in broadcasted_indices1]
    else:
        broadcasted_ct1_data = ct1.ct_data

    # Broadcast ct2 if needed
    if ct2.semantic_shape != result_shape:
        dummy_ct2 = (
            np.arange(np.prod(ct2.semantic_shape))
            .reshape(ct2.semantic_shape)
            .astype(np.int64)
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

    # Create result CipherText with broadcasted shape and encoding parameters
    return CipherText(
        ct_data=result_ciphertext,
        semantic_dtype=ct1.semantic_dtype,
        semantic_shape=result_shape,
        scheme=ct1.scheme,
        key_size=ct1.key_size,
        pk_data=ct1.pk_data,
        max_value=ct1.max_value,
        fxp_bits=ct1.fxp_bits,
        modulus=ct1.modulus,
    )


def _phe_add_ct2pt(ciphertext: CipherText, plaintext: TensorValue) -> CipherText:
    # Convert plaintext to numpy
    plaintext_np = plaintext.to_numpy()
    plaintext_dtype = DType.from_numpy(plaintext_np.dtype)

    # Check for mixed precision issue: floating point ciphertext + integer plaintext
    # This would cause decode failures due to 2**fxp * f + i scaling mismatch
    if ciphertext.semantic_dtype.is_floating and not plaintext_dtype.is_floating:
        raise ValueError(
            f"Cannot add integer plaintext to floating point ciphertext due to fixed-point encoding. "
            f"CipherText dtype: {ciphertext.semantic_dtype}, plaintext dtype: {plaintext_dtype}. "
            f"Both operands must have the same numeric type (both floating or both integer)."
        )

    # Check for mixed precision issue: integer ciphertext + floating point plaintext
    if not ciphertext.semantic_dtype.is_floating and plaintext_dtype.is_floating:
        raise ValueError(
            f"Cannot add floating point plaintext to integer ciphertext due to fixed-point encoding. "
            f"CipherText dtype: {ciphertext.semantic_dtype}, plaintext dtype: {plaintext_dtype}. "
            f"Both operands must have the same numeric type (both floating or both integer)."
        )

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
        ) from e

    # Broadcast plaintext to match result shape if needed
    if plaintext_np.shape != result_shape:
        plaintext_broadcasted = np.broadcast_to(plaintext_np, result_shape)
    else:
        plaintext_broadcasted = plaintext_np

    # Broadcast ciphertext if needed
    if ciphertext.semantic_shape != result_shape:
        dummy_ct = (
            np.arange(np.prod(ciphertext.semantic_shape))
            .reshape(ciphertext.semantic_shape)
            .astype(np.int64)
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

    # Use range encoding for consistency with encryption
    encoded_data_list = []
    for val in flat_data:
        if ciphertext.modulus is None:
            raise ValueError("Ciphertext modulus is None, encryption may have failed")
        encoded_val = _range_encode_mixed(
            val,
            ciphertext.max_value,
            ciphertext.fxp_bits,
            ciphertext.modulus,
            target_dtype,
        )
        encoded_data_list.append(encoded_val)

    encrypted_plaintext = [phe.encrypt(val) for val in encoded_data_list]

    # Perform addition
    result_ciphertext = [
        encrypted_plaintext[i] + broadcasted_ct_data[i]
        for i in range(len(encrypted_plaintext))
    ]

    # Create result CipherText with broadcasted shape and encoding parameters
    return CipherText(
        ct_data=result_ciphertext,
        semantic_dtype=ciphertext.semantic_dtype,
        semantic_shape=result_shape,
        scheme=ciphertext.scheme,
        key_size=ciphertext.key_size,
        pk_data=ciphertext.pk_data,
        max_value=ciphertext.max_value,
        fxp_bits=ciphertext.fxp_bits,
        modulus=ciphertext.modulus,
    )


def _create_encrypted_zero(ciphertext: CipherText) -> Any:
    # Create lightPHE instance with the same configuration
    phe = LightPHE(
        algorithm_name=ciphertext.scheme,
        key_size=ciphertext.key_size,
        precision=PRECISION,
    )

    # CRITICAL: Set the same modulus as the original ciphertext
    if ciphertext.modulus is not None:
        phe.cs.plaintext_modulo = ciphertext.modulus
        phe.cs.ciphertext_modulo = ciphertext.modulus * ciphertext.modulus

    phe.cs.keys["public_key"] = ciphertext.pk_data

    # Encrypt zero value using range encoding for consistency
    if ciphertext.modulus is None:
        raise ValueError("Ciphertext modulus is None, encryption may have failed")

    zero_encoded = _range_encode_mixed(
        0,
        ciphertext.max_value,
        ciphertext.fxp_bits,
        ciphertext.modulus,
        ciphertext.semantic_dtype,
    )

    return phe.encrypt(zero_encoded)


@kernel_def("phe.decrypt")
def _phe_decrypt(
    pfunc: PFunction, ciphertext: CipherText, private_key: PrivateKey
) -> Any:
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
        raise ValueError("CipherText and PrivateKey must use same scheme and key size")

    try:
        # Create lightPHE instance with the same scheme/key_size
        phe = LightPHE(
            algorithm_name=private_key.scheme,
            key_size=private_key.key_size,
            precision=PRECISION,
        )

        # CRITICAL FIX: Manually set the moduli to match the original encryption
        # This ensures the decryption uses the same mathematical structure
        if ciphertext.modulus is not None:
            # Force the lightPHE instance to use the same modulus as during encryption
            phe.cs.plaintext_modulo = ciphertext.modulus
            # For Paillier: ciphertext_modulo = N^2
            phe.cs.ciphertext_modulo = ciphertext.modulus * ciphertext.modulus

        # Set both public and private keys (lightPHE needs both for proper decryption)
        phe.cs.keys["private_key"] = private_key.sk_data
        phe.cs.keys["public_key"] = private_key.pk_data

        # Decrypt the data
        target_dtype = ciphertext.semantic_dtype.to_numpy()
        decrypted_raw = [phe.decrypt(ct) for ct in ciphertext.ct_data]

        # Decode using range decoding
        if ciphertext.modulus is None:
            raise ValueError("Ciphertext modulus is None, encryption may have failed")

        decoded_data = []
        for encrypted_val in decrypted_raw:
            # Extract numeric value from lightPHE result
            if isinstance(encrypted_val, (int, float)):
                raw_val = encrypted_val
            elif hasattr(encrypted_val, "__getitem__") and len(encrypted_val) > 0:
                raw_val = encrypted_val[0]
            else:
                raise ValueError(f"Cannot extract numeric value from {encrypted_val}")

            # Convert to int for decoding
            int_val = int(
                raw_val
            )  # Use mixed decoding which returns values based on semantic type
            decoded_val = _range_decode_mixed(
                int_val,
                ciphertext.max_value,
                ciphertext.fxp_bits,
                ciphertext.modulus,
                ciphertext.semantic_dtype,
            )
            decoded_data.append(decoded_val)

        # Convert to target dtype
        if target_dtype.kind in "iu":  # integer types
            # Convert floats back to integers for integer semantic types
            # decoded_data are numeric (ints or floats); normalize to Python int
            ints = [round(v) if isinstance(v, float) else v for v in decoded_data]
            if np.issubdtype(target_dtype, np.unsignedinteger):
                # Reduce modulo 2^k for unsigned to preserve ring semantics
                width = np.iinfo(target_dtype).bits
                mod = 1 << width
                processed_data = [v % mod for v in ints]
            else:
                # Signed integers: clamp to dtype range
                info = np.iinfo(target_dtype)
                processed_data = [max(info.min, min(info.max, v)) for v in ints]
        else:  # float types
            processed_data = decoded_data

        # Create array and reshape to target shape
        plaintext_np = np.array(processed_data, dtype=target_dtype).reshape(
            ciphertext.semantic_shape
        )

        return [TensorValue(plaintext_np)]

    except Exception as e:
        raise RuntimeError(f"Failed to decrypt data: {e}") from e


@kernel_def("phe.dot")
def _phe_dot(pfunc: PFunction, ciphertext: CipherText, plaintext: TensorValue) -> Any:
    """Execute homomorphic dot product with zero-value optimization.

    Supports various dot product operations:
    - Scalar * Scalar -> Scalar
    - Vector * Vector -> Scalar (inner product)
    - Matrix * Vector -> Vector
    - N-D tensor * M-D tensor -> result based on numpy.dot semantics

    Optimization: Skip multiplication when plaintext value is 0, and handle
    the special case where all plaintext values are 0.

    """
    # Validate that first argument is a CipherText
    if not isinstance(ciphertext, CipherText):
        raise ValueError("First argument must be a CipherText instance")
    if isinstance(plaintext, CipherText):
        raise ValueError("Second argument must be a plaintext TensorLike")

    try:
        # Convert plaintext to numpy
        plaintext_np = plaintext.to_numpy()

        # Check if plaintext is floating point type - dot product not supported
        if np.issubdtype(plaintext_np.dtype, np.floating):
            raise ValueError(
                f"Homomorphic dot product with floating point plaintext is not supported. "
                f"Got plaintext dtype: {plaintext_np.dtype}"
            )

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
            ) from e

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
        def get_encrypted_zero() -> Any:
            return _create_encrypted_zero(ciphertext)

        if len(ct_shape) == 0 and len(pt_shape) == 0:
            # Scalar * Scalar
            pt_val = pt_data.item()
            if is_zero_func(pt_val):
                result_ciphertext = get_encrypted_zero()
            else:
                # Use single value (not list) for multiplication
                val = float(pt_val) if target_dtype.is_floating else int(pt_val)
                result_ciphertext = ciphertext.ct_data[0] * val
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
                    # Convert to appropriate type and use single value (not list)
                    val = float(pt_val) if target_dtype.is_floating else int(pt_val)
                    product = ciphertext.ct_data[i] * val
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
                        # Use single value (not list) for multiplication
                        val = float(pt_val) if target_dtype.is_floating else int(pt_val)
                        product = ciphertext.ct_data[ct_idx] * val
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
                        # Use single value (not list) for multiplication
                        val = float(pt_val) if target_dtype.is_floating else int(pt_val)
                        product = ciphertext.ct_data[i] * val
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
                            # Use single value (not list) for multiplication
                            val = (
                                float(pt_val)
                                if target_dtype.is_floating
                                else int(pt_val)
                            )
                            product = ciphertext.ct_data[ct_idx] * val
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
                                # Use single value (not list) for multiplication
                                val = (
                                    float(pt_val)
                                    if target_dtype.is_floating
                                    else int(pt_val)
                                )
                                product = ct_flat[ct_idx] * val
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

        # Create result CipherText with computed shape and encoding parameters
        return [
            CipherText(
                ct_data=result_ct_data,
                semantic_dtype=ciphertext.semantic_dtype,
                semantic_shape=result_shape,
                scheme=ciphertext.scheme,
                key_size=ciphertext.key_size,
                pk_data=ciphertext.pk_data,
                max_value=ciphertext.max_value,
                fxp_bits=ciphertext.fxp_bits,
                modulus=ciphertext.modulus,
            )
        ]

    except ValueError:
        # Re-raise ValueError directly (validation errors)
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to perform dot product: {e}") from e


@kernel_def("phe.gather")
def _phe_gather(pfunc: PFunction, ciphertext: CipherText, indices: TensorValue) -> Any:
    """Execute gather operation on CipherText.

    Supports gathering from multidimensional CipherText using multidimensional indices.
    The operation follows numpy.take semantics:
    - result.shape = indices.shape + ciphertext.shape[:axis] + ciphertext.shape[axis+1:]
    - Gathering is performed along the specified axis of ciphertext
    """
    # Validate that first argument is a CipherText
    if not isinstance(ciphertext, CipherText):
        raise ValueError("First argument must be a CipherText instance")

    # Get axis parameter from pfunc.attrs, default to 0
    axis = pfunc.attrs.get("axis", 0)

    try:
        # Convert indices to numpy
        indices_np = indices.to_numpy()

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
                max_value=ciphertext.max_value,
                fxp_bits=ciphertext.fxp_bits,
                modulus=ciphertext.modulus,
            )
        ]

    except ValueError:
        # Re-raise ValueError directly (validation errors)
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to perform gather: {e}") from e


@kernel_def("phe.scatter")
def _phe_scatter(
    pfunc: PFunction,
    ciphertext: CipherText,
    indices: TensorValue,
    updated: CipherText,
) -> Any:
    """Execute scatter operation on CipherText.

    Supports scattering into multidimensional CipherText using multidimensional indices.
    The operation follows numpy scatter semantics:
    - Scattering is performed along the specified axis of ciphertext
    - indices.shape must equal updated.shape[:len(indices.shape)]
    - updated.shape must be indices.shape + ciphertext.shape[:axis] + ciphertext.shape[axis+1:]
    - Result shape is same as original ciphertext.shape

    """
    # Validate that first and third arguments are CipherTexts
    if not isinstance(ciphertext, CipherText) or not isinstance(updated, CipherText):
        raise ValueError("First and third arguments must be CipherText instances")

    # Validate that both ciphertexts use same scheme/key_size
    if ciphertext.scheme != updated.scheme or ciphertext.key_size != updated.key_size:
        raise ValueError("Both CipherTexts must use same scheme and key size")

    if ciphertext.pk_data != updated.pk_data:
        raise ValueError("Both CipherTexts must be encrypted with same key")

    # Get axis parameter from pfunc.attrs, default to 0
    axis = pfunc.attrs.get("axis", 0)

    try:
        # Convert indices to numpy
        indices_np = indices.to_numpy()

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
                            scattered_ct_data[start_pos_original + j] = updated_ct_data[
                                start_pos_updated + j
                            ]

        # Create result CipherText with same shape as original
        return [
            CipherText(
                ct_data=scattered_ct_data,
                semantic_dtype=ciphertext.semantic_dtype,
                semantic_shape=ciphertext.semantic_shape,
                scheme=ciphertext.scheme,
                key_size=ciphertext.key_size,
                pk_data=ciphertext.pk_data,
                max_value=ciphertext.max_value,
                fxp_bits=ciphertext.fxp_bits,
                modulus=ciphertext.modulus,
            )
        ]
    except ValueError:
        # Re-raise ValueError directly (validation errors)
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to perform scatter: {e}") from e


@kernel_def("phe.concat")
def _phe_concat(pfunc: PFunction, c1: CipherText, c2: CipherText) -> Any:
    """Execute concat operation on multiple CipherTexts.

    Supports concatenation along any axis of multidimensional CipherTexts.
    The axis parameter is obtained from pfunc.attrs.
    """
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
        result_shape_list = list(c1.semantic_shape)
        result_shape_list[axis] = c1.semantic_shape[axis] + c2.semantic_shape[axis]
        result_shape = tuple(result_shape_list)

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
                max_value=c1.max_value,
                fxp_bits=c1.fxp_bits,
                modulus=c1.modulus,
            )
        ]

    except ValueError:
        # Re-raise ValueError directly (validation errors)
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to perform concat: {e}") from e


@kernel_def("phe.reshape")
def _phe_reshape(pfunc: PFunction, ciphertext: CipherText) -> Any:
    """Execute reshape operation on CipherText.

    Changes the shape of a CipherText without changing its encrypted data.
    The new_shape parameter is obtained from pfunc.attrs.
    """
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
            int(np.prod(ciphertext.semantic_shape)) if ciphertext.semantic_shape else 1
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
        final_shape = tuple(inferred_shape)

        # Validate that new shape has the same number of elements
        new_size = int(np.prod(final_shape)) if final_shape else 1

        if old_size != new_size:
            raise ValueError(
                f"Cannot reshape CipherText with {old_size} elements to shape {final_shape} "
                f"with {new_size} elements"
            )

        # Create result CipherText with new shape and encoding parameters (ct_data remains the same)
        return [
            CipherText(
                ct_data=ciphertext.ct_data,  # Same encrypted data
                semantic_dtype=ciphertext.semantic_dtype,
                semantic_shape=final_shape,  # Use the final shape
                scheme=ciphertext.scheme,
                key_size=ciphertext.key_size,
                pk_data=ciphertext.pk_data,
                max_value=ciphertext.max_value,
                fxp_bits=ciphertext.fxp_bits,
                modulus=ciphertext.modulus,
            )
        ]

    except ValueError:
        # Re-raise ValueError directly (validation errors)
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to perform reshape: {e}") from e


@kernel_def("phe.transpose")
def _phe_transpose(pfunc: PFunction, ciphertext: CipherText) -> Any:
    """Execute transpose operation on CipherText.

    Permutes the dimensions of a CipherText according to the given axes.
    The axes parameter is obtained from pfunc.attrs.
    """
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
        def transpose_data(ct_data: list, old_shape: tuple, axes: tuple) -> list:
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
                max_value=ciphertext.max_value,
                fxp_bits=ciphertext.fxp_bits,
                modulus=ciphertext.modulus,
            )
        ]

    except ValueError:
        # Re-raise ValueError directly (validation errors)
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to perform transpose: {e}") from e
