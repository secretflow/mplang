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

    API (mock): quote(pk[u8[32]]) -> quote[u8[33]]
    The mock encodes a 1-byte header + 32-byte pk.
    """

    def __call__(self, pk: MPObject) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        pk_ty = TensorType.from_obj(pk)
        quote_ty = TensorType(UINT8, (33,))
        pfunc = PFunction(
            fn_type="tee.quote",
            ins_info=(pk_ty,),
            outs_info=(quote_ty,),
        )
        _, treedef = tree_flatten(quote_ty)
        return pfunc, [pk], treedef


class QuoteVerifyAndExtract(FEOp):
    """TEE quote verification FEOp returning the attested TEE public key.

    API (mock): attest(quote[u8[33]]) -> tee_pk[u8[32]]
    """

    def __call__(self, quote: MPObject) -> tuple[PFunction, list[MPObject], PyTreeDef]:
        quote_ty = TensorType.from_obj(quote)
        pk_ty = TensorType(UINT8, (32,))
        pfunc = PFunction(
            fn_type="tee.attest",
            ins_info=(quote_ty,),
            outs_info=(pk_ty,),
        )
        _, treedef = tree_flatten(pk_ty)
        return pfunc, [quote], treedef


# Public instances (similar to frontend.builtin pattern)
quote = QuoteGen()
attest = QuoteVerifyAndExtract()
