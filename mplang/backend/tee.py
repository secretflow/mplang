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

import os
from dataclasses import dataclass
from typing import Any  # noqa: F401

import numpy as np

from mplang.core.pfunc import PFunction, TensorHandler
from mplang.core.tensor import TensorLike
from mplang.utils.crypto import blake2b  # noqa: F401


@dataclass
class Quote:
    """Simple quote structure for the mock TEE backend (no payload)."""

    report_data: bytes  # e.g., H(program_hash||nonce||H(epk)) in real impl

    def to_array(self) -> np.ndarray:
        data = self.report_data
        return np.frombuffer(data if data else b"\x00", dtype=np.uint8)


class TeeHandler(TensorHandler):
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

    def teardown(self) -> None:  # override
        ...

    def list_fn_names(self) -> list[str]:  # override
        return [self.QUOTE_GEN, self.QUOTE_VERIFY_AND_EXTRACT]

    def _quote_from_pk(self, pk: np.ndarray) -> np.ndarray:
        # Bind the provided pk (mock: only first byte) into report_data
        if pk.size == 0:
            report = b"REPORTDATA:\x00"
        else:
            report = b"REPORTDATA:" + bytes([int(pk.flatten()[0])])
        q = Quote(report_data=report).to_array()
        return q.astype(np.uint8)

    def _execute_quote_gen(
        self, args: list[TensorLike], pfunc: PFunction
    ) -> list[TensorLike]:
        # Expect one arg: pk[u8[32]]; return single quote tensor
        if len(args) != 1:
            raise ValueError("tee.quote expects exactly one argument (pk)")
        pk = np.asarray(args[0], dtype=np.uint8)
        q = self._quote_from_pk(pk)
        return [q]

    def _execute_quote_verify_and_extract(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:
        # Mock attest: return a single-byte 1 to indicate verification passed
        if len(args) != 1:
            raise ValueError("tee.attest expects exactly one argument (quote)")
        return [np.array([1], dtype=np.uint8)]

    def execute(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:  # override
        if pfunc.fn_type == self.QUOTE_GEN:
            return self._execute_quote_gen(args, pfunc)
        elif pfunc.fn_type == self.QUOTE_VERIFY_AND_EXTRACT:
            return self._execute_quote_verify_and_extract(pfunc, args)
        else:
            raise ValueError(f"Unsupported function type: {pfunc.fn_type}")
