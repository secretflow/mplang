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
import warnings

import numpy as np
from numpy.typing import NDArray

from mplang.core.pfunc import PFunction
from mplang.kernels.base import cur_kctx, kernel_def

__all__: list[str] = []


def _rng() -> np.random.Generator:
    kctx = cur_kctx()
    pocket = kctx.state.setdefault("tee", {})
    r = pocket.get("rng")
    if r is None:
        seed = int(os.environ.get("MPLANG_TEE_SEED", "0")) + kctx.rank * 10007
        r = np.random.default_rng(seed)
        pocket["rng"] = r
    return r


def _quote_from_pk(pk: np.ndarray) -> NDArray[np.uint8]:
    header = np.array([1], dtype=np.uint8)
    pk32 = np.asarray(pk, dtype=np.uint8).reshape(32)
    out: NDArray[np.uint8] = np.concatenate([header, pk32]).astype(np.uint8)  # type: ignore[assignment]
    return out


@kernel_def("mock_tee.quote")
def _tee_quote(pfunc: PFunction, pk: object) -> NDArray[np.uint8]:
    warnings.warn(
        "Insecure mock TEE kernel 'mock_tee.quote' in use. NOT secure; for local testing only.",
        stacklevel=3,
    )
    pk = np.asarray(pk, dtype=np.uint8)
    # rng access ensures deterministic seeding per rank even if unused now
    _rng()
    return _quote_from_pk(pk)


@kernel_def("mock_tee.attest")
def _tee_attest(pfunc: PFunction, quote: object) -> NDArray[np.uint8]:
    warnings.warn(
        "Insecure mock TEE kernel 'mock_tee.attest' in use. NOT secure; for local testing only.",
        stacklevel=3,
    )
    quote = np.asarray(quote, dtype=np.uint8)
    if quote.size != 33:
        raise ValueError("mock quote must be 33 bytes (1 header + 32 pk)")
    return quote[1:33].astype(np.uint8)
