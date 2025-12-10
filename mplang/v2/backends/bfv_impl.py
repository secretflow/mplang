# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BFV Runtime Implementation.

Implements execution logic for BFV primitives using TenSEAL low-level API (sealapi).
"""

from __future__ import annotations

import base64
import os
import uuid
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
import tenseal as ts
import tenseal.sealapi as sealapi

from mplang.v2.backends.tensor_impl import TensorValue
from mplang.v2.dialects import bfv
from mplang.v2.edsl import serde
from mplang.v2.edsl.graph import Operation
from mplang.v2.runtime.interpreter import Interpreter
from mplang.v2.runtime.value import Value, WrapValue

# =============================================================================
# Helper for SEAL serialization
# =============================================================================


def _get_seal_temp_path() -> str:
    """Get a temp file path for SEAL serialization.

    Uses /dev/shm on Linux for better performance (RAM-based tmpfs),
    falls back to regular tempfile on other platforms.
    """
    # Try /dev/shm first (Linux RAM-based tmpfs, ~30% faster)
    shm_dir = "/dev/shm"
    if os.path.isdir(shm_dir) and os.access(shm_dir, os.W_OK):
        return os.path.join(shm_dir, f"seal_{uuid.uuid4().hex}.bin")

    # Fallback to regular temp directory
    import tempfile

    return os.path.join(tempfile.gettempdir(), f"seal_{uuid.uuid4().hex}.bin")


@serde.register_class
class BFVParamContextValue(WrapValue[ts.Context]):
    """Wraps TenSEAL context with parameters only (no keys)."""

    _serde_kind: ClassVar[str] = "bfv_impl.BFVParamContextValue"

    def __init__(self, data: Any):
        super().__init__(data)
        self.ts_ctx = self._data

        # Extract underlying C++ objects
        self.seal_ctx = self.ts_ctx.seal_context()
        self.cpp_ctx = self.seal_ctx.data

        self.evaluator = sealapi.Evaluator(self.cpp_ctx)
        self.batch_encoder = sealapi.BatchEncoder(self.cpp_ctx)

    def _convert(self, data: Any) -> ts.Context:
        if isinstance(data, BFVParamContextValue):
            return data.unwrap()
        if isinstance(data, ts.Context):
            return data
        raise TypeError(f"Expected ts.Context, got {type(data)}")

    def to_json(self) -> dict[str, Any]:
        # Serialize TenSEAL context (parameters only)
        serialized = self.ts_ctx.serialize(
            save_public_key=False,
            save_secret_key=False,
            save_galois_keys=False,
            save_relin_keys=False,
        )
        return {"ctx_bytes": base64.b64encode(serialized).decode("ascii")}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BFVParamContextValue:
        ctx_bytes = base64.b64decode(data["ctx_bytes"])
        ts_ctx = ts.context_from(ctx_bytes)
        return cls(ts_ctx)


@serde.register_class
class BFVPublicContextValue(WrapValue[ts.Context]):
    """Wraps TenSEAL context and exposes low-level SEAL objects (Public only)."""

    _serde_kind: ClassVar[str] = "bfv_impl.BFVPublicContextValue"

    def __init__(self, data: Any):
        super().__init__(data)
        self.ts_ctx = self._data

        # Extract underlying C++ objects
        self.seal_ctx = self.ts_ctx.seal_context()
        self.cpp_ctx = self.seal_ctx.data

        self.evaluator = sealapi.Evaluator(self.cpp_ctx)
        self.batch_encoder = sealapi.BatchEncoder(self.cpp_ctx)

        # Extract keys
        self.public_key = self.ts_ctx.public_key().data
        self.relin_keys = self.ts_ctx.relin_keys().data
        self.galois_keys = self.ts_ctx.galois_keys().data

        self.encryptor = sealapi.Encryptor(self.cpp_ctx, self.public_key)

    def _convert(self, data: Any) -> ts.Context:
        if isinstance(data, BFVPublicContextValue):
            return data.unwrap()
        if isinstance(data, ts.Context):
            return data
        raise TypeError(f"Expected ts.Context, got {type(data)}")

    def to_json(self) -> dict[str, Any]:
        # Serialize TenSEAL context (without secret key)
        serialized = self.ts_ctx.serialize(save_secret_key=False)
        return {"ctx_bytes": base64.b64encode(serialized).decode("ascii")}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BFVPublicContextValue:
        ctx_bytes = base64.b64decode(data["ctx_bytes"])
        ts_ctx = ts.context_from(ctx_bytes)
        return cls(ts_ctx)


@serde.register_class
class BFVSecretContextValue(BFVPublicContextValue):
    """Wraps TenSEAL context and exposes low-level SEAL objects (including Secret)."""

    _serde_kind: ClassVar[str] = "bfv_impl.BFVSecretContextValue"

    def __init__(self, data: Any):
        # BFVPublicContextValue.__init__ calls WrapValue.__init__ which calls _convert
        # We need to ensure _convert is called and validation happens
        super().__init__(data)

        if not self.ts_ctx.has_secret_key():
            raise ValueError("Context does not have a secret key")

        self.secret_key = self.ts_ctx.secret_key().data
        self.decryptor = sealapi.Decryptor(self.cpp_ctx, self.secret_key)

    def make_public(self) -> BFVPublicContextValue:
        """Create a public-only version of this context."""
        # Serialize without secret key
        serialized = self.ts_ctx.serialize(save_secret_key=False)
        # Deserialize to create a new context
        new_ts_ctx = ts.context_from(serialized)
        return BFVPublicContextValue(new_ts_ctx)

    def to_json(self) -> dict[str, Any]:
        # Serialize TenSEAL context (with secret key)
        serialized = self.ts_ctx.serialize(save_secret_key=True)
        return {"ctx_bytes": base64.b64encode(serialized).decode("ascii")}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BFVSecretContextValue:
        ctx_bytes = base64.b64decode(data["ctx_bytes"])
        ts_ctx = ts.context_from(ctx_bytes)
        return cls(ts_ctx)


@serde.register_class
@dataclass
class BFVValue(Value):
    """Runtime value holding a SEAL Ciphertext or Plaintext."""

    _serde_kind: ClassVar[str] = "bfv_impl.BFVValue"

    data: Any  # sealapi.Ciphertext | sealapi.Plaintext
    ctx: BFVPublicContextValue
    is_cipher: bool = True

    def to_json(self) -> dict[str, Any]:
        # Serialize the ciphertext/plaintext via temp file (SEAL API requirement)
        # Use /dev/shm on Linux for better performance (no disk I/O)
        fname = _get_seal_temp_path()
        try:
            self.data.save(fname)
            with open(fname, "rb") as f:
                data_bytes = f.read()
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

        # Serialize context as parameters only (to save bandwidth)
        # We create a temporary BFVParamContextValue wrapper
        param_ctx = BFVParamContextValue(self.ctx.ts_ctx)
        ctx_json = serde.to_json(param_ctx)

        return {
            "data_bytes": base64.b64encode(data_bytes).decode("ascii"),
            "is_cipher": self.is_cipher,
            "ctx": ctx_json,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BFVValue:
        ctx = serde.from_json(data["ctx"])
        data_bytes = base64.b64decode(data["data_bytes"])
        is_cipher = data["is_cipher"]

        # Load via temp file (SEAL API requirement)
        # Use /dev/shm on Linux for better performance (no disk I/O)
        fname = _get_seal_temp_path()
        try:
            with open(fname, "wb") as f:
                f.write(data_bytes)

            if is_cipher:
                ct = sealapi.Ciphertext()
                ct.load(ctx.cpp_ctx, fname)
                return cls(data=ct, ctx=ctx, is_cipher=True)
            else:
                pt = sealapi.Plaintext()
                pt.load(ctx.cpp_ctx, fname)
                return cls(data=pt, ctx=ctx, is_cipher=False)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)


# =============================================================================
# Keygen Cache (Optimization: avoid regenerating keys for same parameters)
# =============================================================================
_KEYGEN_CACHE: dict[
    tuple[int, int], tuple[BFVPublicContextValue, BFVSecretContextValue]
] = {}


def clear_keygen_cache() -> None:
    """Clear the keygen cache."""
    _KEYGEN_CACHE.clear()


@bfv.keygen_p.def_impl
def keygen_impl(
    interpreter: Interpreter, op: Operation, *args: Any
) -> tuple[BFVPublicContextValue, BFVSecretContextValue]:
    poly_modulus_degree = op.attrs.get("poly_modulus_degree", 4096)
    # Use a default plain_modulus if not provided.
    plain_modulus = op.attrs.get("plain_modulus", 1032193)

    # Check cache first
    cache_key = (poly_modulus_degree, plain_modulus)
    if cache_key in _KEYGEN_CACHE:
        return _KEYGEN_CACHE[cache_key]

    # Generate context with secret key
    ts_ctx = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=poly_modulus_degree,
        plain_modulus=plain_modulus,
    )
    ts_ctx.generate_galois_keys()
    ts_ctx.generate_relin_keys()

    full_context = BFVSecretContextValue(ts_ctx)
    public_context = full_context.make_public()

    # Cache the result
    result = (public_context, full_context)
    _KEYGEN_CACHE[cache_key] = result

    # Return (PK, SK)
    return result


@bfv.make_relin_keys_p.def_impl
def make_relin_keys_impl(
    interpreter: Interpreter, op: Operation, sk: BFVSecretContextValue
) -> BFVSecretContextValue:
    return sk


@bfv.make_galois_keys_p.def_impl
def make_galois_keys_impl(
    interpreter: Interpreter, op: Operation, sk: BFVSecretContextValue
) -> BFVSecretContextValue:
    return sk


@bfv.create_encoder_p.def_impl
def create_encoder_impl(interpreter: Interpreter, op: Operation) -> dict[str, Any]:
    return {"poly_modulus_degree": op.attrs.get("poly_modulus_degree", 4096)}


@bfv.encode_p.def_impl
def encode_impl(
    interpreter: Interpreter,
    op: Operation,
    data: TensorValue,
    encoder: dict[str, Any],
) -> TensorValue:
    # Return raw data as "Logical Plaintext" wrapped in TensorValue
    return TensorValue.wrap(np.asarray(data.unwrap()))


@bfv.batch_encode_p.def_impl
def batch_encode_impl(
    interpreter: Interpreter,
    op: Operation,
    *args: Value,
) -> tuple[BFVValue | TensorValue, ...]:
    # args will be (tensor, encoder, key)
    key = args[-1]
    _encoder = args[-2]
    tensor_val = args[0]

    # Eager encoding using key.ctx
    # key is BFVPublicContextValue (or BFVSecretContextValue)
    ctx = cast(BFVPublicContextValue, key)

    results = []
    # Optimization: Convert to numpy array first to avoid JAX dispatch overhead
    # during iteration. This also ensures a single device-to-host transfer if on GPU.
    arr = np.asarray(cast(TensorValue, tensor_val).unwrap())

    # Iterate rows
    for i in range(arr.shape[0]):
        pt = sealapi.Plaintext()
        # Use tolist() for speed
        vec = arr[i].tolist()
        ctx.batch_encoder.encode(vec, pt)
        results.append(BFVValue(pt, ctx, is_cipher=False))

    return tuple(results)


@bfv.encrypt_p.def_impl
def encrypt_impl(
    interpreter: Interpreter,
    op: Operation,
    plaintext: TensorValue,
    pk: BFVPublicContextValue,
) -> BFVValue:
    # plaintext is TensorValue (from encode_impl)
    # pk is BFVPublicContextValue
    plaintext_arr = plaintext.unwrap().flatten()

    # 1. Create Plaintext
    pt = sealapi.Plaintext()

    # 2. Encode
    # We need to handle types. Assuming int64 vector.
    # Optimization: Use tolist() instead of list comprehension
    vec = plaintext_arr.tolist()
    pk.batch_encoder.encode(vec, pt)

    # 3. Encrypt
    ct = sealapi.Ciphertext()
    pk.encryptor.encrypt(pt, ct)

    return BFVValue(ct, pk, is_cipher=True)


@bfv.decrypt_p.def_impl
def decrypt_impl(
    interpreter: Interpreter,
    op: Operation,
    ciphertext: BFVValue,
    sk: BFVSecretContextValue,
) -> BFVValue:
    # ciphertext is BFVValue
    # sk is BFVSecretContextValue

    pt = sealapi.Plaintext()
    sk.decryptor.decrypt(ciphertext.data, pt)

    return BFVValue(pt, sk, is_cipher=False)


@bfv.decode_p.def_impl
def decode_impl(
    interpreter: Interpreter, op: Operation, plaintext: BFVValue, encoder: Any
) -> TensorValue:
    # plaintext is BFVValue(Plaintext)
    # encoder is dummy config

    vec = plaintext.ctx.batch_encoder.decode_int64(plaintext.data)
    return TensorValue.wrap(np.array(vec))


def _ensure_plaintext(ctx: BFVPublicContextValue, data: BFVValue | TensorValue) -> Any:
    """Convert data to sealapi.Plaintext using the given context."""
    if isinstance(data, BFVValue):
        if data.is_cipher:
            raise TypeError("Expected Plaintext, got Ciphertext")
        return data.data

    # data is TensorValue
    if not isinstance(data, TensorValue):
        raise TypeError(f"Expected BFVValue or TensorValue, got {type(data)}")
    pt = sealapi.Plaintext()
    arr = data.unwrap()
    vec = arr.flatten().tolist()
    ctx.batch_encoder.encode(vec, pt)
    return pt


@bfv.add_p.def_impl
def add_impl(
    interpreter: Interpreter,
    op: Operation,
    lhs: BFVValue | TensorValue,
    rhs: BFVValue | TensorValue,
) -> BFVValue | TensorValue:
    # Case 1: Both are BFVValues
    if isinstance(lhs, BFVValue) and isinstance(rhs, BFVValue):
        result_ct = sealapi.Ciphertext()

        if lhs.is_cipher and rhs.is_cipher:
            # Optimization: Handle transparent ciphertexts (zero)
            if lhs.data.is_transparent():
                return rhs
            if rhs.data.is_transparent():
                return lhs

            lhs.ctx.evaluator.add(lhs.data, rhs.data, result_ct)
            return BFVValue(result_ct, lhs.ctx, is_cipher=True)
        elif lhs.is_cipher and not rhs.is_cipher:
            # Optimization: Handle transparent ciphertext
            if lhs.data.is_transparent():
                # 0 + Plaintext -> Encrypt(Plaintext)
                # This is expensive, but necessary for correctness if we want to return a Ciphertext
                # Alternatively, if we allow returning Plaintext, we could just return rhs.
                # But BFV add usually expects to return Ciphertext if one input is Ciphertext.
                # For now, let's encrypt it.
                new_ct = sealapi.Ciphertext()
                lhs.ctx.encryptor.encrypt(rhs.data, new_ct)
                return BFVValue(new_ct, lhs.ctx, is_cipher=True)

            lhs.ctx.evaluator.add_plain(lhs.data, rhs.data, result_ct)
            return BFVValue(result_ct, lhs.ctx, is_cipher=True)
        elif not lhs.is_cipher and rhs.is_cipher:
            # Optimization: Handle transparent ciphertext
            if rhs.data.is_transparent():
                new_ct = sealapi.Ciphertext()
                rhs.ctx.encryptor.encrypt(lhs.data, new_ct)
                return BFVValue(new_ct, rhs.ctx, is_cipher=True)

            rhs.ctx.evaluator.add_plain(rhs.data, lhs.data, result_ct)
            return BFVValue(result_ct, rhs.ctx, is_cipher=True)
        else:
            raise NotImplementedError(
                "BFV Plaintext + Plaintext addition not implemented yet"
            )

    # Case 2: One is BFVValue (Ciphertext), other is Raw
    if isinstance(lhs, BFVValue) and lhs.is_cipher:
        # Optimization: Handle transparent ciphertext
        if lhs.data.is_transparent():
            # 0 + Raw -> Encrypt(Raw)
            pt = _ensure_plaintext(lhs.ctx, rhs)
            new_ct = sealapi.Ciphertext()
            lhs.ctx.encryptor.encrypt(pt, new_ct)
            return BFVValue(new_ct, lhs.ctx, is_cipher=True)

        pt = _ensure_plaintext(lhs.ctx, rhs)
        result_ct = sealapi.Ciphertext()
        lhs.ctx.evaluator.add_plain(lhs.data, pt, result_ct)
        return BFVValue(result_ct, lhs.ctx, is_cipher=True)

    if isinstance(rhs, BFVValue) and rhs.is_cipher:
        # Optimization: Handle transparent ciphertext
        if rhs.data.is_transparent():
            pt = _ensure_plaintext(rhs.ctx, lhs)
            new_ct = sealapi.Ciphertext()
            rhs.ctx.encryptor.encrypt(pt, new_ct)
            return BFVValue(new_ct, rhs.ctx, is_cipher=True)

        pt = _ensure_plaintext(rhs.ctx, lhs)
        result_ct = sealapi.Ciphertext()
        rhs.ctx.evaluator.add_plain(rhs.data, pt, result_ct)
        return BFVValue(result_ct, rhs.ctx, is_cipher=True)

    # Handle Plaintext + Plaintext (TensorValue + TensorValue)
    if isinstance(lhs, TensorValue) and isinstance(rhs, TensorValue):
        return TensorValue.wrap(lhs.unwrap() + rhs.unwrap())
    raise TypeError(f"Unsupported types for add: {type(lhs)}, {type(rhs)}")


@bfv.sub_p.def_impl
def sub_impl(
    interpreter: Interpreter,
    op: Operation,
    lhs: BFVValue | TensorValue,
    rhs: BFVValue | TensorValue,
) -> BFVValue | TensorValue:
    # Case 1: Both are BFVValues
    if isinstance(lhs, BFVValue) and isinstance(rhs, BFVValue):
        result_ct = sealapi.Ciphertext()

        if lhs.is_cipher and rhs.is_cipher:
            lhs.ctx.evaluator.sub(lhs.data, rhs.data, result_ct)
            return BFVValue(result_ct, lhs.ctx, is_cipher=True)
        elif lhs.is_cipher and not rhs.is_cipher:
            lhs.ctx.evaluator.sub_plain(lhs.data, rhs.data, result_ct)
            return BFVValue(result_ct, lhs.ctx, is_cipher=True)
        elif not lhs.is_cipher and rhs.is_cipher:
            neg_ct = sealapi.Ciphertext()
            rhs.ctx.evaluator.negate(rhs.data, neg_ct)
            rhs.ctx.evaluator.add_plain(neg_ct, lhs.data, result_ct)
            return BFVValue(result_ct, rhs.ctx, is_cipher=True)
        else:
            raise NotImplementedError(
                "BFV Plaintext - Plaintext subtraction not implemented yet"
            )

    # Case 2: One is BFVValue (Ciphertext), other is Raw
    if isinstance(lhs, BFVValue) and lhs.is_cipher:
        pt = _ensure_plaintext(lhs.ctx, rhs)
        result_ct = sealapi.Ciphertext()
        lhs.ctx.evaluator.sub_plain(lhs.data, pt, result_ct)
        return BFVValue(result_ct, lhs.ctx, is_cipher=True)

    if isinstance(rhs, BFVValue) and rhs.is_cipher:
        # Raw - CT
        pt = _ensure_plaintext(rhs.ctx, lhs)
        result_ct = sealapi.Ciphertext()
        neg_ct = sealapi.Ciphertext()
        rhs.ctx.evaluator.negate(rhs.data, neg_ct)
        rhs.ctx.evaluator.add_plain(neg_ct, pt, result_ct)
        return BFVValue(result_ct, rhs.ctx, is_cipher=True)

    # Handle Plaintext - Plaintext (TensorValue - TensorValue)
    if isinstance(lhs, TensorValue) and isinstance(rhs, TensorValue):
        return TensorValue.wrap(lhs.unwrap() - rhs.unwrap())
    raise TypeError(f"Unsupported types for sub: {type(lhs)}, {type(rhs)}")


@bfv.mul_p.def_impl
def mul_impl(
    interpreter: Interpreter,
    op: Operation,
    lhs: BFVValue | TensorValue,
    rhs: BFVValue | TensorValue,
) -> BFVValue | TensorValue:
    # Case 1: Both are BFVValues
    if isinstance(lhs, BFVValue) and isinstance(rhs, BFVValue):
        result_ct = sealapi.Ciphertext()

        if lhs.is_cipher and rhs.is_cipher:
            lhs.ctx.evaluator.multiply(lhs.data, rhs.data, result_ct)
            return BFVValue(result_ct, lhs.ctx, is_cipher=True)
        elif lhs.is_cipher and not rhs.is_cipher:
            # Optimization: Check for zero plaintext to avoid expensive exception handling
            if rhs.data.is_zero():
                # Return transparent zero ciphertext (no noise, size 0)
                # SEAL arithmetic ops handle transparent ciphertexts as zero.
                # We must ensure relinearize/rotate also handle it.
                return BFVValue(sealapi.Ciphertext(), lhs.ctx, is_cipher=True)

            try:
                lhs.ctx.evaluator.multiply_plain(lhs.data, rhs.data, result_ct)
                return BFVValue(result_ct, lhs.ctx, is_cipher=True)
            except RuntimeError as e:
                if "transparent" in str(e):
                    return BFVValue(sealapi.Ciphertext(), lhs.ctx, is_cipher=True)
                raise e
        elif not lhs.is_cipher and rhs.is_cipher:
            # Optimization: Check for zero plaintext
            if lhs.data.is_zero():
                return BFVValue(sealapi.Ciphertext(), rhs.ctx, is_cipher=True)

            try:
                rhs.ctx.evaluator.multiply_plain(rhs.data, lhs.data, result_ct)
                return BFVValue(result_ct, rhs.ctx, is_cipher=True)
            except RuntimeError as e:
                if "transparent" in str(e):
                    return BFVValue(sealapi.Ciphertext(), rhs.ctx, is_cipher=True)
                raise e
        else:
            raise NotImplementedError(
                "BFV Plaintext * Plaintext multiplication not implemented yet"
            )

    # Case 2: One is BFVValue (Ciphertext), other is TensorValue
    if isinstance(lhs, BFVValue) and lhs.is_cipher:
        # Check for zero plaintext to avoid "transparent ciphertext" error
        # Also check if plaintext is BFVValue(Plaintext)
        if isinstance(rhs, TensorValue) and np.all(rhs.unwrap() == 0):
            result_ct = sealapi.Ciphertext()
            lhs.ctx.encryptor.encrypt_zero(result_ct)
            return BFVValue(result_ct, lhs.ctx, is_cipher=True)

        try:
            pt = _ensure_plaintext(lhs.ctx, rhs)
            result_ct = sealapi.Ciphertext()
            lhs.ctx.evaluator.multiply_plain(lhs.data, pt, result_ct)
            return BFVValue(result_ct, lhs.ctx, is_cipher=True)
        except RuntimeError as e:
            # SEAL throws "result ciphertext is transparent" when multiplying by a zero plaintext.
            # This is mathematically valid (Enc(x) * 0 = Enc(0)), but SEAL enforces explicit zero encryption.
            # We catch this error and return a valid zero ciphertext to maintain operator semantics.
            if "transparent" in str(e):
                # Fallback for zero plaintext
                result_ct = sealapi.Ciphertext()
                lhs.ctx.encryptor.encrypt_zero(result_ct)
                return BFVValue(result_ct, lhs.ctx, is_cipher=True)
            raise e

    if isinstance(rhs, BFVValue) and rhs.is_cipher:
        # Check for zero plaintext to avoid "transparent ciphertext" error
        if isinstance(lhs, TensorValue) and np.all(lhs.unwrap() == 0):
            result_ct = sealapi.Ciphertext()
            rhs.ctx.encryptor.encrypt_zero(result_ct)
            return BFVValue(result_ct, rhs.ctx, is_cipher=True)

        try:
            pt = _ensure_plaintext(rhs.ctx, lhs)
            result_ct = sealapi.Ciphertext()
            rhs.ctx.evaluator.multiply_plain(rhs.data, pt, result_ct)
            return BFVValue(result_ct, rhs.ctx, is_cipher=True)
        except RuntimeError as e:
            # See comment above regarding "transparent ciphertext"
            if "transparent" in str(e):
                # Fallback for zero plaintext
                result_ct = sealapi.Ciphertext()
                rhs.ctx.encryptor.encrypt_zero(result_ct)
                return BFVValue(result_ct, rhs.ctx, is_cipher=True)
            raise e

    # Handle Plaintext * Plaintext (TensorValue * TensorValue)
    if isinstance(lhs, TensorValue) and isinstance(rhs, TensorValue):
        return TensorValue.wrap(lhs.unwrap() * rhs.unwrap())
    raise TypeError(f"Unsupported types for mul: {type(lhs)}, {type(rhs)}")


@bfv.relinearize_p.def_impl
def relinearize_impl(
    interpreter: Interpreter,
    op: Operation,
    ciphertext: BFVValue,
    rk: BFVPublicContextValue,
) -> BFVValue:
    # rk is BFVPublicContextValue (same as ciphertext.ctx)

    # Optimization: Handle transparent ciphertext (zero)
    if ciphertext.data.is_transparent():
        return ciphertext

    # Check if relinearization is needed (size > 2)
    if ciphertext.data.size() > 2:
        new_ct = sealapi.Ciphertext()
        ciphertext.ctx.evaluator.relinearize(ciphertext.data, rk.relin_keys, new_ct)
        return BFVValue(new_ct, ciphertext.ctx, is_cipher=True)

    return ciphertext


@bfv.rotate_p.def_impl
def rotate_impl(
    interpreter: Interpreter,
    op: Operation,
    ciphertext: BFVValue,
    gk: BFVPublicContextValue,
) -> BFVValue:
    """Implement rotation using low-level SEAL API directly."""
    steps = op.attrs.get("steps", 0)
    if steps == 0:
        return ciphertext

    # Optimization: Handle transparent ciphertext (zero)
    if ciphertext.data.is_transparent():
        return ciphertext

    # ciphertext is BFVValue
    # gk is BFVPublicContextValue

    new_ct = sealapi.Ciphertext()
    ciphertext.ctx.evaluator.rotate_rows(ciphertext.data, steps, gk.galois_keys, new_ct)
    return BFVValue(new_ct, ciphertext.ctx, is_cipher=True)


@bfv.rotate_columns_p.def_impl
def rotate_columns_impl(
    interpreter: Interpreter,
    op: Operation,
    ciphertext: BFVValue,
    gk: BFVPublicContextValue,
) -> BFVValue:
    """Swap the two rows in SIMD batching (row 0 <-> row 1)."""
    new_ct = sealapi.Ciphertext()
    ciphertext.ctx.evaluator.rotate_columns(ciphertext.data, gk.galois_keys, new_ct)
    return BFVValue(new_ct, ciphertext.ctx, is_cipher=True)
