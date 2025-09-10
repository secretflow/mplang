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


@dataclass
class Quote:
    """Simple quote structure for the mock TEE backend."""

    report_data: bytes
    payload: bytes  # could carry key commitment or encrypted key

    def to_array(self) -> np.ndarray:
        data = self.report_data + b"|" + self.payload
        return np.frombuffer(data, dtype=np.uint8)


class TeeHandler(TensorHandler):
    """TEE Handler with a mock implementation (payload-based).

    PFunctions:
    - tee.quote_gen(payloads...): returns list of quotes (one per payload)
    - tee.quote_verify_and_extract(quote): verifies and extracts payload

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

    def _quote_bytes(self, payload: bytes) -> np.ndarray:
        # Mock report_data binds to a fake session tuple; here we just hash-ish
        # Keep it simple for now: prefix + key (to allow extraction)
        prefix = b"REPORTDATA:"
        q = Quote(report_data=prefix, payload=payload)
        arr = q.to_array()
        if arr.size == 0:
            # Guarantee at least 1 byte to satisfy placeholder TensorType
            return np.array([0], dtype=np.uint8)
        return arr

    def _execute_quote_gen(
        self, args: list[TensorLike], pfunc: PFunction
    ) -> list[TensorLike]:
        # For each payload, build a quote embedding it.
        quotes: list[TensorLike] = []
        for payload in args:
            pb = np.asarray(payload, dtype=np.uint8).tobytes()
            quote_arr = self._quote_bytes(pb)
            quotes.append(quote_arr)
        return quotes

    def _execute_quote_verify_and_extract(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:
        if len(args) != 1:
            raise ValueError(
                "tee.quote_verify_and_extract expects exactly one argument (quote)"
            )
        quote_arr = args[0]
        if not isinstance(quote_arr, np.ndarray):
            quote_arr = np.asarray(quote_arr)

        # Split on the first '|' (124) byte to recover payload
        sep_idx = (
            int(np.where(quote_arr == 124)[0][0]) if np.any(quote_arr == 124) else -1
        )
        if sep_idx <= 0:
            # invalid mock quote
            raise RuntimeError("Invalid quote format: separator not found")

        payload = quote_arr[sep_idx + 1 :].astype(np.uint8)
        return [payload]

    def execute(
        self, pfunc: PFunction, args: list[TensorLike]
    ) -> list[TensorLike]:  # override
        if pfunc.fn_type == self.QUOTE_GEN:
            return self._execute_quote_gen(args, pfunc)
        elif pfunc.fn_type == self.QUOTE_VERIFY_AND_EXTRACT:
            return self._execute_quote_verify_and_extract(pfunc, args)
        else:
            raise ValueError(f"Unsupported function type: {pfunc.fn_type}")
