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

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

try:
    import trustflow.attestation.verification as verification
    from trustflow.attestation.common import (
        AttestationAttribute,
        AttestationGenerationParams,
        AttestationPolicy,
        AttestationReport,
        AttestationReportParams,
    )

    HAS_TRUSTFLOW = True
except ImportError:
    HAS_TRUSTFLOW = False

    # Define dummy classes for when trustflow is not available
    class AttestationReport:
        def to_json(self):
            return ""

        @classmethod
        def from_json(cls, json_str):
            return cls()

    class AttestationGenerationParams:
        pass

    class AttestationReportParams:
        pass

    class AttestationAttribute:
        pass

    class AttestationPolicy:
        def __init__(self, **kwargs):
            pass


from mplang.core.pfunc import PFunction, TensorHandler
from mplang.core.tensor import TensorLike
from mplang.utils.crypto import blake2b


@dataclass
class Quote:
    """Simple quote structure for the mock TEE backend (no payload)."""

    report_data: bytes  # e.g., H(program_hash||nonce||H(epk)) in real impl

    def to_array(self) -> np.ndarray:
        data = self.report_data
        return np.frombuffer(data if data else b"\x00", dtype=np.uint8)


class MockTeeHandler(TensorHandler):
    """TEE Handler with a mock implementation that binds provided pk.

    WARNING: This is a mock implementation for demos/tests. It does NOT perform
    real verification of vendor quotes, measurements, or program hashes, and it
    embeds payload bytes into the quote for easy extraction. Do not use in
    production. The production design uses TEE ephemeral key binding and KEM.

    PFunctions:
    - tee.quote(pk): returns quote binding the provided public key
    - tee.attest(quote): verifies and returns a gating byte

    This mock does not perform real attestation. It emulates the flow so the
    IR/plumbing/API work end-to-end. Quotes and payloads are byte arrays.
    """

    QUOTE_GEN = "tee.quote"
    QUOTE_VERIFY_AND_EXTRACT = "tee.attest"

    def setup(self, rank: int) -> None:  # override
        self._rank = rank
        # Derive a deterministic per-rank seed for testing stability
        seed = int(os.environ.get("MPLANG_TEE_SEED", "0")) + rank * 10007
        self._rng = np.random.default_rng(seed)

        logging.info(f"Using MockTeeHandler (not secure) on rank {rank}")

    def teardown(self) -> None:  # override
        ...

    def list_fn_names(self) -> list[str]:  # override
        return [self.QUOTE_GEN, self.QUOTE_VERIFY_AND_EXTRACT]

    def _quote_from_pk(self, pk: np.ndarray) -> np.ndarray:
        # Mock quote structure: 1-byte header + 32-byte pk
        header = np.array([1], dtype=np.uint8)
        pk32 = np.asarray(pk, dtype=np.uint8).reshape(32)
        ret: np.ndarray = np.concatenate([header, pk32]).astype(np.uint8)
        return ret

    def _execute_quote_gen(
        self, args: list[TensorLike], pfunc: PFunction
    ) -> list[TensorLike]:
        # Expect one arg (pk: u8[32]); return single quote tensor
        if len(args) != 1:
            raise ValueError("tee.quote expects exactly one argument (pk)")
        pk = np.asarray(args[0], dtype=np.uint8)
        q = self._quote_from_pk(pk)
        return [q]

    def _execute_quote_verify_and_extract(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:
        # Mock attest: parse and return pk from quote (no real verification)
        if len(args) != 1:
            raise ValueError("tee.attest expects exactly one argument (quote)")
        quote = np.asarray(args[0], dtype=np.uint8)
        if quote.size != 33:
            raise ValueError("mock quote must be 33 bytes (1 header + 32 pk)")
        pk = quote[1:33].astype(np.uint8)
        return [pk]

    def execute(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:  # override
        if pfunc.fn_type == self.QUOTE_GEN:
            return self._execute_quote_gen(args, pfunc)
        elif pfunc.fn_type == self.QUOTE_VERIFY_AND_EXTRACT:
            return self._execute_quote_verify_and_extract(pfunc, args)
        else:
            raise ValueError(f"Unsupported function type: {pfunc.fn_type}")


class TeeHandler(TensorHandler):
    """TEE Handler with a real TEE implementation using TrustFlow Attestation Library.

    PFunctions:
    - tee.quote(pk): returns quote binding the provided public key
    - tee.attest(quote): verifies and returns a gating byte

    This implementation base on Intel TDX platform and requires TrustFlow Attestation Library.
    """

    QUOTE_GEN = "tee.quote"
    QUOTE_VERIFY_AND_EXTRACT = "tee.attest"

    def __init__(self):
        if not HAS_TRUSTFLOW:
            raise ImportError(
                "TeeHandler requires trustflow dependencies. Please install trustflow packages."
            )

    def setup(self, rank: int) -> None:  # override
        self._rank = rank

        logging.info(f"Using TeeHandler on rank {rank}")

    def teardown(self) -> None:  # override
        ...

    def list_fn_names(self) -> list[str]:  # override
        return [self.QUOTE_GEN, self.QUOTE_VERIFY_AND_EXTRACT]

    def _build_quote(self, pk: np.ndarray, report_json: str) -> np.ndarray:
        # quote structure: 1-byte header + 32-byte pk + report json bytes
        header = np.array([2], dtype=np.uint8)
        pk32 = np.asarray(pk, dtype=np.uint8).reshape(32)
        ret: np.ndarray = np.concatenate([
            header,
            pk32,
            report_json.encode("utf-8"),
        ]).astype(np.uint8)
        return ret

    def _execute_quote_gen(
        self, args: list[TensorLike], pfunc: PFunction
    ) -> list[TensorLike]:
        if not HAS_TRUSTFLOW:
            raise ImportError("tee.quote requires trustflow dependencies.")

        import trustflow.attestation.generation as tdx_generation

        # Expect one arg (pk: u8[32]);
        if len(args) != 1:
            raise ValueError("tee.quote expects exactly one argument (pk)")
        pk = np.asarray(args[0], dtype=np.uint8)
        if pk.size != 32:
            raise ValueError("pk must be 32 bytes")

        # Generate TDX attestation report binding the provided pk
        params = AttestationGenerationParams()
        params.tee_identity = "tdx_instance"
        params.report_type = "Passport"
        params.report_params = AttestationReportParams()
        params.report_params.hex_user_data = blake2b(pk.tobytes()).hex()
        report: AttestationReport = tdx_generation.generate_report(params)
        report_json = report.to_json()

        return [self._build_quote(pk, report_json)]

    def _execute_quote_verify_and_extract(
        self, args: list[TensorLike], pfunc: PFunction
    ) -> list[TensorLike]:
        if not HAS_TRUSTFLOW:
            raise ImportError("tee.attest requires trustflow dependencies.")

        # Verify and extract pk from quote
        if len(args) != 1:
            raise ValueError("tee.attest expects exactly one argument (quote)")
        quote = np.asarray(args[0], dtype=np.uint8)
        if quote.size < 33:
            raise ValueError(
                "quote must be at least 33 bytes (1 header + 32 pk + report)"
            )
        if quote[0] != 2:
            raise ValueError("invalid quote header")
        pk = quote[1:33].astype(np.uint8)
        report_json = quote[33:].tobytes().decode("utf-8")

        logging.info(f"Verifying quote, tdx report json: {report_json}")

        report = AttestationReport.from_json(report_json)

        # Verify the attestation report
        attrs = AttestationAttribute()
        attrs.str_tee_platform = "TDX"
        attrs.hex_user_data = blake2b(pk.tobytes()).hex()
        attrs.bool_debug_disabled = "true"  # require non-debug for real use
        status = verification.report_verify(
            report,
            AttestationPolicy(main_attributes=[attrs]),
        )
        if status.code != 0:
            raise ValueError(
                f"Attestation verification failed: {status.message}, detail: {status.detail}"
            )

        return [pk]

    def execute(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:  # override
        if pfunc.fn_type == self.QUOTE_GEN:
            return self._execute_quote_gen(args, pfunc)
        elif pfunc.fn_type == self.QUOTE_VERIFY_AND_EXTRACT:
            return self._execute_quote_verify_and_extract(args, pfunc)
        else:
            raise ValueError(f"Unsupported function type: {pfunc.fn_type}")
