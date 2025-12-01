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

from dataclasses import dataclass
from typing import Any

import numpy as np
import tenseal as ts
import tenseal.sealapi as sealapi

from mplang.v2.dialects import bfv
from mplang.v2.edsl.graph import Operation
from mplang.v2.edsl.interpreter import Interpreter


class BFVPublicContext:
    """Wraps TenSEAL context and exposes low-level SEAL objects (Public only)."""

    def __init__(self, ts_ctx: ts.Context):
        self.ts_ctx = ts_ctx

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


class BFVSecretContext(BFVPublicContext):
    """Wraps TenSEAL context and exposes low-level SEAL objects (including Secret)."""

    def __init__(self, ts_ctx: ts.Context):
        if not ts_ctx.has_secret_key():
            raise ValueError("Context does not have a secret key")

        super().__init__(ts_ctx)

        self.secret_key = self.ts_ctx.secret_key().data
        self.decryptor = sealapi.Decryptor(self.cpp_ctx, self.secret_key)

    def make_public(self) -> BFVPublicContext:
        """Create a public-only version of this context."""
        # Serialize without secret key
        serialized = self.ts_ctx.serialize(save_secret_key=False)
        # Deserialize to create a new context
        new_ts_ctx = ts.context_from(serialized)
        return BFVPublicContext(new_ts_ctx)


@dataclass
class BFVValue:
    """Runtime value holding a SEAL Ciphertext or Plaintext."""

    data: sealapi.Ciphertext | sealapi.Plaintext
    ctx: BFVPublicContext
    is_cipher: bool = True


# =============================================================================
# Keygen Cache (Optimization: avoid regenerating keys for same parameters)
# =============================================================================
_KEYGEN_CACHE: dict[tuple[int, int], tuple[BFVPublicContext, BFVSecretContext]] = {}


def clear_keygen_cache() -> None:
    """Clear the keygen cache."""
    _KEYGEN_CACHE.clear()


@bfv.keygen_p.def_impl
def keygen_impl(
    interpreter: Interpreter, op: Operation, *args: Any
) -> tuple[BFVPublicContext, BFVSecretContext]:
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

    full_context = BFVSecretContext(ts_ctx)
    public_context = full_context.make_public()

    # Cache the result
    result = (public_context, full_context)
    _KEYGEN_CACHE[cache_key] = result

    # Return (PK, SK)
    return result


@bfv.make_relin_keys_p.def_impl
def make_relin_keys_impl(
    interpreter: Interpreter, op: Operation, sk: BFVSecretContext
) -> BFVSecretContext:
    return sk


@bfv.make_galois_keys_p.def_impl
def make_galois_keys_impl(
    interpreter: Interpreter, op: Operation, sk: BFVSecretContext
) -> BFVSecretContext:
    return sk


@bfv.create_encoder_p.def_impl
def create_encoder_impl(interpreter: Interpreter, op: Operation) -> dict[str, Any]:
    return {"poly_modulus_degree": op.attrs.get("poly_modulus_degree", 4096)}


@bfv.encode_p.def_impl
def encode_impl(
    interpreter: Interpreter, op: Operation, data: Any, encoder: dict[str, Any]
) -> np.ndarray:
    # Return raw data as "Logical Plaintext"
    return np.array(data)


@bfv.encrypt_p.def_impl
def encrypt_impl(
    interpreter: Interpreter, op: Operation, plaintext: np.ndarray, pk: BFVPublicContext
) -> BFVValue:
    # plaintext is numpy array (from encode_impl)
    # pk is BFVPublicContext

    # 1. Create Plaintext
    pt = sealapi.Plaintext()

    # 2. Encode
    # We need to handle types. Assuming int64 vector.
    vec = [int(x) for x in plaintext]
    pk.batch_encoder.encode(vec, pt)

    # 3. Encrypt
    ct = sealapi.Ciphertext()
    pk.encryptor.encrypt(pt, ct)

    return BFVValue(ct, pk, is_cipher=True)


@bfv.decrypt_p.def_impl
def decrypt_impl(
    interpreter: Interpreter, op: Operation, ciphertext: BFVValue, sk: BFVSecretContext
) -> BFVValue:
    # ciphertext is BFVValue
    # sk is BFVSecretContext

    pt = sealapi.Plaintext()
    sk.decryptor.decrypt(ciphertext.data, pt)

    return BFVValue(pt, sk, is_cipher=False)


@bfv.decode_p.def_impl
def decode_impl(
    interpreter: Interpreter, op: Operation, plaintext: BFVValue, encoder: Any
) -> np.ndarray:
    # plaintext is BFVValue(Plaintext)
    # encoder is dummy config

    vec = plaintext.ctx.batch_encoder.decode_int64(plaintext.data)
    return np.array(vec)


def _ensure_plaintext(ctx: BFVPublicContext, data: Any) -> sealapi.Plaintext:
    """Convert data to sealapi.Plaintext using the given context."""
    if isinstance(data, BFVValue):
        if data.is_cipher:
            raise TypeError("Expected Plaintext, got Ciphertext")
        return data.data

    # Assume data is raw values (list/numpy)
    pt = sealapi.Plaintext()
    # Handle numpy types
    if hasattr(data, "tolist"):
        data = data.tolist()
    # Ensure int
    vec = [int(x) for x in data]
    ctx.batch_encoder.encode(vec, pt)
    return pt


@bfv.add_p.def_impl
def add_impl(
    interpreter: Interpreter, op: Operation, lhs: Any, rhs: Any
) -> BFVValue | np.ndarray:
    # Case 1: Both are BFVValues
    if isinstance(lhs, BFVValue) and isinstance(rhs, BFVValue):
        result_ct = sealapi.Ciphertext()

        if lhs.is_cipher and rhs.is_cipher:
            lhs.ctx.evaluator.add(lhs.data, rhs.data, result_ct)
            return BFVValue(result_ct, lhs.ctx, is_cipher=True)
        elif lhs.is_cipher and not rhs.is_cipher:
            lhs.ctx.evaluator.add_plain(lhs.data, rhs.data, result_ct)
            return BFVValue(result_ct, lhs.ctx, is_cipher=True)
        elif not lhs.is_cipher and rhs.is_cipher:
            rhs.ctx.evaluator.add_plain(rhs.data, lhs.data, result_ct)
            return BFVValue(result_ct, rhs.ctx, is_cipher=True)
        else:
            raise NotImplementedError(
                "BFV Plaintext + Plaintext addition not implemented yet"
            )

    # Case 2: One is BFVValue (Ciphertext), other is Raw
    if isinstance(lhs, BFVValue) and lhs.is_cipher:
        pt = _ensure_plaintext(lhs.ctx, rhs)
        result_ct = sealapi.Ciphertext()
        lhs.ctx.evaluator.add_plain(lhs.data, pt, result_ct)
        return BFVValue(result_ct, lhs.ctx, is_cipher=True)

    if isinstance(rhs, BFVValue) and rhs.is_cipher:
        pt = _ensure_plaintext(rhs.ctx, lhs)
        result_ct = sealapi.Ciphertext()
        rhs.ctx.evaluator.add_plain(rhs.data, pt, result_ct)
        return BFVValue(result_ct, rhs.ctx, is_cipher=True)

    # Handle Plaintext + Plaintext (numpy + numpy)
    return lhs + rhs  # type: ignore


@bfv.sub_p.def_impl
def sub_impl(
    interpreter: Interpreter, op: Operation, lhs: Any, rhs: Any
) -> BFVValue | np.ndarray:
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

    # Handle Plaintext + Plaintext (numpy + numpy)
    return lhs - rhs  # type: ignore


@bfv.mul_p.def_impl
def mul_impl(
    interpreter: Interpreter, op: Operation, lhs: Any, rhs: Any
) -> BFVValue | np.ndarray:
    # Case 1: Both are BFVValues
    if isinstance(lhs, BFVValue) and isinstance(rhs, BFVValue):
        result_ct = sealapi.Ciphertext()

        if lhs.is_cipher and rhs.is_cipher:
            lhs.ctx.evaluator.multiply(lhs.data, rhs.data, result_ct)
            return BFVValue(result_ct, lhs.ctx, is_cipher=True)
        elif lhs.is_cipher and not rhs.is_cipher:
            lhs.ctx.evaluator.multiply_plain(lhs.data, rhs.data, result_ct)
            return BFVValue(result_ct, lhs.ctx, is_cipher=True)
        elif not lhs.is_cipher and rhs.is_cipher:
            rhs.ctx.evaluator.multiply_plain(rhs.data, lhs.data, result_ct)
            return BFVValue(result_ct, rhs.ctx, is_cipher=True)
        else:
            raise NotImplementedError(
                "BFV Plaintext * Plaintext multiplication not implemented yet"
            )

    # Case 2: One is BFVValue (Ciphertext), other is Raw
    if isinstance(lhs, BFVValue) and lhs.is_cipher:
        # Check for zero plaintext to avoid "transparent ciphertext" error
        if isinstance(rhs, (int, float)) and rhs == 0:
            result_ct = sealapi.Ciphertext()
            lhs.ctx.encryptor.encrypt_zero(result_ct)
            return BFVValue(result_ct, lhs.ctx, is_cipher=True)
        if isinstance(rhs, np.ndarray) and np.all(rhs == 0):
            result_ct = sealapi.Ciphertext()
            lhs.ctx.encryptor.encrypt_zero(result_ct)
            return BFVValue(result_ct, lhs.ctx, is_cipher=True)

        pt = _ensure_plaintext(lhs.ctx, rhs)
        result_ct = sealapi.Ciphertext()
        lhs.ctx.evaluator.multiply_plain(lhs.data, pt, result_ct)
        return BFVValue(result_ct, lhs.ctx, is_cipher=True)

    if isinstance(rhs, BFVValue) and rhs.is_cipher:
        # Check for zero plaintext to avoid "transparent ciphertext" error
        if isinstance(lhs, (int, float)) and lhs == 0:
            result_ct = sealapi.Ciphertext()
            rhs.ctx.encryptor.encrypt_zero(result_ct)
            return BFVValue(result_ct, rhs.ctx, is_cipher=True)
        if isinstance(lhs, np.ndarray) and np.all(lhs == 0):
            result_ct = sealapi.Ciphertext()
            rhs.ctx.encryptor.encrypt_zero(result_ct)
            return BFVValue(result_ct, rhs.ctx, is_cipher=True)

        pt = _ensure_plaintext(rhs.ctx, lhs)
        result_ct = sealapi.Ciphertext()
        rhs.ctx.evaluator.multiply_plain(rhs.data, pt, result_ct)
        return BFVValue(result_ct, rhs.ctx, is_cipher=True)

    return lhs * rhs  # type: ignore


@bfv.relinearize_p.def_impl
def relinearize_impl(
    interpreter: Interpreter, op: Operation, ciphertext: BFVValue, rk: BFVPublicContext
) -> BFVValue:
    # rk is BFVPublicContext (same as ciphertext.ctx)

    # Check if relinearization is needed (size > 2)
    if ciphertext.data.size() > 2:
        new_ct = sealapi.Ciphertext()
        ciphertext.ctx.evaluator.relinearize(
            ciphertext.data, ciphertext.ctx.relin_keys, new_ct
        )
        return BFVValue(new_ct, ciphertext.ctx, is_cipher=True)

    return ciphertext


@bfv.rotate_p.def_impl
def rotate_impl(
    interpreter: Interpreter, op: Operation, ciphertext: BFVValue, gk: BFVPublicContext
) -> BFVValue:
    """Implement rotation using low-level SEAL API directly."""
    steps = op.attrs.get("steps", 0)
    if steps == 0:
        return ciphertext

    # ciphertext is BFVValue
    # gk is BFVPublicContext

    new_ct = sealapi.Ciphertext()
    ciphertext.ctx.evaluator.rotate_rows(
        ciphertext.data, steps, ciphertext.ctx.galois_keys, new_ct
    )
    return BFVValue(new_ct, ciphertext.ctx, is_cipher=True)


@bfv.rotate_columns_p.def_impl
def rotate_columns_impl(
    interpreter: Interpreter, op: Operation, ciphertext: BFVValue, gk: BFVPublicContext
) -> BFVValue:
    """Swap the two rows in SIMD batching (row 0 <-> row 1)."""
    new_ct = sealapi.Ciphertext()
    ciphertext.ctx.evaluator.rotate_columns(
        ciphertext.data, ciphertext.ctx.galois_keys, new_ct
    )
    return BFVValue(new_ct, ciphertext.ctx, is_cipher=True)
