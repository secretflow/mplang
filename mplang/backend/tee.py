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

import numpy as np

from mplang.backend.base import backend_kernel, cur_kctx
from mplang.core.pfunc import PFunction

__all__ = []


def _rng():
    kctx = cur_kctx()
    pocket = kctx.kernel_state.setdefault("tee", {})
    r = pocket.get("rng")
    if r is None:
        seed = int(os.environ.get("MPLANG_TEE_SEED", "0")) + kctx.rank * 10007
        r = np.random.default_rng(seed)
        pocket["rng"] = r
    return r


def _quote_from_pk(pk: np.ndarray) -> np.ndarray:
    header = np.array([1], dtype=np.uint8)
    pk32 = np.asarray(pk, dtype=np.uint8).reshape(32)
    return np.concatenate([header, pk32]).astype(np.uint8)


@backend_kernel("tee.quote")
def _tee_quote(pfunc: PFunction, args: tuple) -> tuple:
    if len(args) != 1:
        raise ValueError("tee.quote expects exactly one argument (pk)")
    pk = np.asarray(args[0], dtype=np.uint8)
    # rng access ensures deterministic seeding per rank even if unused now
    _rng()
    q = _quote_from_pk(pk)
    return (q,)


@backend_kernel("tee.attest")
def _tee_attest(pfunc: PFunction, args: tuple) -> tuple:
    if len(args) != 1:
        raise ValueError("tee.attest expects exactly one argument (quote)")
    quote = np.asarray(args[0], dtype=np.uint8)
    if quote.size != 33:
        raise ValueError("mock quote must be 33 bytes (1 header + 32 pk)")
    pk = quote[1:33].astype(np.uint8)
    return (pk,)


class MockTeeHandler:  # pragma: no cover - deprecated shim
    def __init__(self, *_, **__):
        raise RuntimeError(
            "MockTeeHandler removed; use flat tee.* kernels via @backend_kernel"
        )
