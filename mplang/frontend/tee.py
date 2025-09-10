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
    """TEE quote generation FEOp binding the provided ephemeral public key.

    API (mock): quote(pk[u8[32]]) -> quote[u8[1]]
    """

    def __call__(self, pk: MPObject) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        pk_ty = TensorType.from_obj(pk)
        quote_ty = TensorType(UINT8, (1,))
        pfunc = PFunction(
            fn_type="tee.quote",
            ins_info=(pk_ty,),
            outs_info=(quote_ty,),
        )
        _, treedef = tree_flatten(quote_ty)
        return pfunc, [pk], treedef


class QuoteVerifyAndExtract(FEOp):
    """TEE quote verification FEOp returning a gating byte (1 for success).

    Mock behavior: returns u8[1] = 1 on success.
    """

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
