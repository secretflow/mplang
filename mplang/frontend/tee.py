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

from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core.dtype import UINT8
from mplang.core.mpobject import MPObject
from mplang.core.pfunc import PFunction
from mplang.core.tensor import TensorType
from mplang.frontend.base import FEOp


class QuoteGen(FEOp):
    """TEE quote generation FEOp (payload-based).

    Takes one key or a list of keys as payloads and returns a list of quotes.
    """

    def __call__(
        self, payloads: list[MPObject]
    ) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        if not payloads:
            raise ValueError("payloads list must not be empty")
        in_tys = tuple(TensorType.from_obj(p) for p in payloads)
        outs_info = tuple(TensorType(UINT8, (1,)) for _ in payloads)
        pfunc = PFunction(
            fn_type="tee.quote",
            ins_info=in_tys,
            outs_info=outs_info,
            num_quotes=len(payloads),
        )
        _, treedef = tree_flatten([TensorType(UINT8, (1,)) for _ in payloads])
        return pfunc, payloads, treedef


class QuoteVerifyAndExtract(FEOp):
    """TEE quote verification FEOp that returns the embedded payload (key)."""

    def __call__(self, quote: MPObject) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        quote_ty = TensorType.from_obj(quote)
        key_ty = TensorType(UINT8, (1,))
        pfunc = PFunction(
            fn_type="tee.attest",
            ins_info=(quote_ty,),
            outs_info=(key_ty,),
        )
        _, treedef = tree_flatten(key_ty)
        return pfunc, [quote], treedef


# Public instances (similar to frontend.builtin pattern)
quote = QuoteGen()
attest = QuoteVerifyAndExtract()
