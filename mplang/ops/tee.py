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
from mplang.ops.base import stateless_mod

_TEE_MOD = stateless_mod("tee")


@_TEE_MOD.simple_op()
def quote_gen(pk: TensorType) -> TensorType:
    """TEE quote generation binding the provided ephemeral public key."""
    _ = pk  # Mark as used for the decorator
    return TensorType(UINT8, (-1,))


@_TEE_MOD.op_def()
def attest(
    quote: MPObject, platform: str
) -> tuple[PFunction, list[MPObject], PyTreeDef]:
    """TEE quote verification returning the attested TEE public key."""

    ins_info = [TensorType.from_obj(quote)]
    outs_info = [TensorType(UINT8, (32,))]  # pk is always 32 bytes for x25519
    pfunc = PFunction(
        fn_type="tee.attest",
        ins_info=ins_info,
        outs_info=outs_info,
        platform=platform,
    )
    _, treedef = tree_flatten(outs_info[0])

    return pfunc, [quote], treedef
