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

"""
Serde registration for backend runtime objects.

This module registers backend implementation objects like BFVContext,
which wrap C++ libraries and need special serialization handling.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from mplang.v2.edsl import serde

if TYPE_CHECKING:
    from mplang.v2.backends.bfv_impl import BFVPublicContext, BFVSecretContext, BFVValue


# =============================================================================
# BFV Runtime Objects
# =============================================================================


def _register_bfv_runtime() -> None:
    """Register BFV runtime objects for serialization."""
    # Lazy import to avoid circular imports and missing dependencies
    try:
        import tenseal as ts

        from mplang.v2.backends.bfv_impl import (
            BFVPublicContext,
            BFVSecretContext,
            BFVValue,
        )
    except ImportError:
        # TenSEAL not installed, skip registration
        return

    # BFVPublicContext
    BFVPublicContext._serde_kind = "bfv_impl.BFVPublicContext"  # type: ignore[attr-defined]

    def pub_ctx_to_json(self: BFVPublicContext) -> dict:
        # Serialize TenSEAL context (without secret key)
        serialized = self.ts_ctx.serialize(save_secret_key=False)
        return {"ctx_bytes": base64.b64encode(serialized).decode("ascii")}

    @classmethod  # type: ignore[misc]
    def pub_ctx_from_json(cls: type[BFVPublicContext], data: dict) -> BFVPublicContext:
        ctx_bytes = base64.b64decode(data["ctx_bytes"])
        ts_ctx = ts.context_from(ctx_bytes)
        return cls(ts_ctx)

    BFVPublicContext.to_json = pub_ctx_to_json  # type: ignore[attr-defined]
    BFVPublicContext.from_json = pub_ctx_from_json  # type: ignore[attr-defined]
    serde.register_class(BFVPublicContext)

    # BFVSecretContext
    BFVSecretContext._serde_kind = "bfv_impl.BFVSecretContext"  # type: ignore[attr-defined]

    def sec_ctx_to_json(self: BFVSecretContext) -> dict:
        # Serialize TenSEAL context (with secret key)
        serialized = self.ts_ctx.serialize(save_secret_key=True)
        return {"ctx_bytes": base64.b64encode(serialized).decode("ascii")}

    @classmethod  # type: ignore[misc]
    def sec_ctx_from_json(cls: type[BFVSecretContext], data: dict) -> BFVSecretContext:
        ctx_bytes = base64.b64decode(data["ctx_bytes"])
        ts_ctx = ts.context_from(ctx_bytes)
        return cls(ts_ctx)

    BFVSecretContext.to_json = sec_ctx_to_json  # type: ignore[attr-defined]
    BFVSecretContext.from_json = sec_ctx_from_json  # type: ignore[attr-defined]
    serde.register_class(BFVSecretContext)

    # BFVValue (holds Ciphertext or Plaintext)
    BFVValue._serde_kind = "bfv_impl.BFVValue"  # type: ignore[attr-defined]

    def _get_seal_temp_path() -> str:
        """Get a temp file path for SEAL serialization.

        Uses /dev/shm on Linux for better performance (RAM-based tmpfs),
        falls back to regular tempfile on other platforms.
        """
        import os
        import uuid

        # Try /dev/shm first (Linux RAM-based tmpfs, ~30% faster)
        shm_dir = "/dev/shm"
        if os.path.isdir(shm_dir) and os.access(shm_dir, os.W_OK):
            return os.path.join(shm_dir, f"seal_{uuid.uuid4().hex}.bin")

        # Fallback to regular temp directory
        import tempfile

        return os.path.join(tempfile.gettempdir(), f"seal_{uuid.uuid4().hex}.bin")

    def bfv_value_to_json(self: BFVValue) -> dict:
        import os

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

        # Also need to serialize the context reference
        ctx_json = serde.to_json(self.ctx)
        return {
            "data_bytes": base64.b64encode(data_bytes).decode("ascii"),
            "is_cipher": self.is_cipher,
            "ctx": ctx_json,
        }

    @classmethod  # type: ignore[misc]
    def bfv_value_from_json(cls: type[BFVValue], data: dict) -> BFVValue:
        import os

        import tenseal.sealapi as sealapi

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

    BFVValue.to_json = bfv_value_to_json  # type: ignore[attr-defined]
    BFVValue.from_json = bfv_value_from_json  # type: ignore[attr-defined]
    serde.register_class(BFVValue)


# =============================================================================
# Crypto Runtime Objects (coincurve + RuntimeKey dataclasses)
# =============================================================================


def _register_crypto_runtime() -> None:
    """Register crypto runtime objects (coincurve keys, RuntimeKey dataclasses)."""
    # Register RuntimeKey dataclasses from crypto_impl
    try:
        from mplang.v2.backends.crypto_impl import (
            RuntimePrivateKey,
            RuntimePublicKey,
            RuntimeSymmetricKey,
        )

        # RuntimePublicKey
        RuntimePublicKey._serde_kind = "crypto_impl.RuntimePublicKey"  # type: ignore[attr-defined]

        def rpk_to_json(self: RuntimePublicKey) -> dict:
            return {
                "suite": self.suite,
                "key_bytes": base64.b64encode(self.key_bytes).decode("ascii"),
            }

        @classmethod  # type: ignore[misc]
        def rpk_from_json(cls: type, data: dict) -> RuntimePublicKey:
            return cls(
                suite=data["suite"],
                key_bytes=base64.b64decode(data["key_bytes"]),
            )

        RuntimePublicKey.to_json = rpk_to_json  # type: ignore[attr-defined]
        RuntimePublicKey.from_json = rpk_from_json  # type: ignore[attr-defined]
        serde.register_class(RuntimePublicKey)

        # RuntimePrivateKey
        RuntimePrivateKey._serde_kind = "crypto_impl.RuntimePrivateKey"  # type: ignore[attr-defined]

        def rsk_to_json(self: RuntimePrivateKey) -> dict:
            return {
                "suite": self.suite,
                "key_bytes": base64.b64encode(self.key_bytes).decode("ascii"),
            }

        @classmethod  # type: ignore[misc]
        def rsk_from_json(cls: type, data: dict) -> RuntimePrivateKey:
            return cls(
                suite=data["suite"],
                key_bytes=base64.b64decode(data["key_bytes"]),
            )

        RuntimePrivateKey.to_json = rsk_to_json  # type: ignore[attr-defined]
        RuntimePrivateKey.from_json = rsk_from_json  # type: ignore[attr-defined]
        serde.register_class(RuntimePrivateKey)

        # RuntimeSymmetricKey
        RuntimeSymmetricKey._serde_kind = "crypto_impl.RuntimeSymmetricKey"  # type: ignore[attr-defined]

        def rsym_to_json(self: RuntimeSymmetricKey) -> dict:
            return {
                "suite": self.suite,
                "key_bytes": base64.b64encode(self.key_bytes).decode("ascii"),
            }

        @classmethod  # type: ignore[misc]
        def rsym_from_json(cls: type, data: dict) -> RuntimeSymmetricKey:
            return cls(
                suite=data["suite"],
                key_bytes=base64.b64decode(data["key_bytes"]),
            )

        RuntimeSymmetricKey.to_json = rsym_to_json  # type: ignore[attr-defined]
        RuntimeSymmetricKey.from_json = rsym_from_json  # type: ignore[attr-defined]
        serde.register_class(RuntimeSymmetricKey)

    except ImportError:
        pass

    # Register coincurve keys
    try:
        import coincurve
    except ImportError:
        return

    # coincurve.PublicKey
    coincurve.PublicKey._serde_kind = "crypto.coincurve.PublicKey"  # type: ignore[attr-defined]

    def pk_to_json(self: coincurve.PublicKey) -> dict:
        return {"data": base64.b64encode(self.format()).decode("ascii")}

    @classmethod  # type: ignore[misc]
    def pk_from_json(cls: type, data: dict) -> coincurve.PublicKey:
        return coincurve.PublicKey(base64.b64decode(data["data"]))

    coincurve.PublicKey.to_json = pk_to_json  # type: ignore[attr-defined]
    coincurve.PublicKey.from_json = pk_from_json  # type: ignore[attr-defined]
    serde.register_class(coincurve.PublicKey)

    # coincurve.PrivateKey
    coincurve.PrivateKey._serde_kind = "crypto.coincurve.PrivateKey"  # type: ignore[attr-defined]

    def sk_to_json(self: coincurve.PrivateKey) -> dict:
        return {"data": base64.b64encode(self.secret).decode("ascii")}

    @classmethod  # type: ignore[misc]
    def sk_from_json(cls: type, data: dict) -> coincurve.PrivateKey:
        return coincurve.PrivateKey(base64.b64decode(data["data"]))

    coincurve.PrivateKey.to_json = sk_to_json  # type: ignore[attr-defined]
    coincurve.PrivateKey.from_json = sk_from_json  # type: ignore[attr-defined]
    serde.register_class(coincurve.PrivateKey)


# =============================================================================
# SPU Runtime Objects
# =============================================================================


def _register_spu_runtime() -> None:
    """Register SPU runtime objects (Share) for serialization."""
    try:
        from spu import libspu
    except ImportError:
        # SPU not installed, skip registration
        return

    # libspu.Share - SPU secret share
    libspu.Share._serde_kind = "libspu.Share"  # type: ignore[attr-defined]

    def share_to_json(self: libspu.Share) -> dict:
        # Share has meta (bytes) and share_chunks (list of bytes)
        return {
            "meta": base64.b64encode(self.meta).decode("ascii"),
            "share_chunks": [
                base64.b64encode(chunk).decode("ascii") for chunk in self.share_chunks
            ],
        }

    def share_from_json(cls: type, data: dict) -> libspu.Share:
        share = cls()
        share.meta = base64.b64decode(data["meta"])
        # Assign share_chunks directly (pybind11 list)
        share.share_chunks = [
            base64.b64decode(chunk_b64) for chunk_b64 in data["share_chunks"]
        ]
        return share

    libspu.Share.to_json = share_to_json  # type: ignore[attr-defined]
    libspu.Share.from_json = classmethod(share_from_json)  # type: ignore[attr-defined]
    serde.register_class(libspu.Share)


# =============================================================================
# TEE Runtime Objects
# =============================================================================


def _register_tee_runtime() -> None:
    """Register TEE runtime objects (MockQuote) for serialization."""
    try:
        from mplang.v2.backends.tee_impl import MockQuote
    except ImportError:
        return

    # MockQuote
    MockQuote._serde_kind = "tee_impl.MockQuote"  # type: ignore[attr-defined]

    def quote_to_json(self: MockQuote) -> dict:
        return {
            "platform": self.platform,
            "bound_pk": base64.b64encode(self.bound_pk).decode("ascii"),
            "suite": self.suite,
        }

    @classmethod  # type: ignore[misc]
    def quote_from_json(cls: type, data: dict) -> MockQuote:
        return cls(
            platform=data["platform"],
            bound_pk=base64.b64decode(data["bound_pk"]),
            suite=data["suite"],
        )

    MockQuote.to_json = quote_to_json  # type: ignore[attr-defined]
    MockQuote.from_json = quote_from_json  # type: ignore[attr-defined]
    serde.register_class(MockQuote)


# =============================================================================
# Register all runtime objects
# =============================================================================


def _register_all() -> None:
    """Register all backend runtime objects with serde."""
    _register_bfv_runtime()
    _register_crypto_runtime()
    _register_spu_runtime()
    _register_tee_runtime()


# Auto-register on import
_register_all()
