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

from mplang.v1.core import PFunction
from mplang.v1.kernels.base import cur_kctx, kernel_def
from mplang.v1.kernels.value import TensorValue

__all__: list[str] = []


def _rng() -> np.random.Generator:
    kctx = cur_kctx()
    rt = kctx.runtime
    r = rt.get_state("tee.rng")
    if r is None:
        seed = int(os.environ.get("MPLANG_TEE_SEED", "0")) + kctx.rank * 10007
        r = np.random.default_rng(seed)
        rt.set_state("tee.rng", r)
    assert isinstance(r, np.random.Generator)  # type narrowing for mypy
    return r


def _quote_from_pk(pk: np.ndarray) -> NDArray[np.uint8]:
    header = np.array([1], dtype=np.uint8)
    pk32 = np.asarray(pk, dtype=np.uint8).reshape(32)
    out: NDArray[np.uint8] = np.concatenate([header, pk32]).astype(np.uint8)  # type: ignore[assignment]
    return out


@kernel_def("mock_tee.quote_gen")
def _tee_quote_gen(pfunc: PFunction, pk: TensorValue) -> TensorValue:
    warnings.warn(
        "Insecure mock TEE kernel 'mock_tee.quote_gen' in use. NOT secure; for local testing only.",
        stacklevel=3,
    )
    pk_arr = pk.to_numpy().astype(np.uint8, copy=False)
    # rng access ensures deterministic seeding per rank even if unused now
    _rng()
    quote = _quote_from_pk(pk_arr)
    return TensorValue(np.array(quote, copy=True))


@kernel_def("mock_tee.attest")
def _tee_attest(pfunc: PFunction, quote: TensorValue) -> TensorValue:
    warnings.warn(
        "Insecure mock TEE kernel 'mock_tee.attest' in use. NOT secure; for local testing only.",
        stacklevel=3,
    )
    quote_arr = quote.to_numpy().astype(np.uint8, copy=False)
    if quote_arr.size != 33:
        raise ValueError("mock quote must be 33 bytes (1 header + 32 pk)")
    attest = quote_arr[1:33].astype(np.uint8, copy=True)
    return TensorValue(attest)
