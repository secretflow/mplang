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

from mplang.core import primitive as prim
from mplang.core.mask import Mask
from mplang.core.mpobject import MPObject
from mplang.core.pfunc import PFunction
from mplang.core.table import TableLike, TableType
from mplang.core.tensor import TensorLike, TensorType


def encrypt_data(
    plaintext: MPObject,
    frm_rank: int,
    to_rank: int,
    tee_rank: int,
) -> PFunction:
    # Create input info for the data
    in_type: TableType | TensorType
    if isinstance(plaintext, TableLike):
        in_type = TableType.from_tablelike(plaintext)
    elif isinstance(plaintext, TensorLike):
        in_type = TensorType.from_obj(plaintext)

    out_type = in_type

    # Create the PFunction
    pfunc = PFunction(
        fn_type="tee.encrypt",
        ins_info=[in_type],
        outs_info=[out_type],
        fn_name="encrypt",
        to_rank=to_rank,
        frm_rank=frm_rank,
        tee_rank=tee_rank,
    )

    return pfunc


def decrypt_data(
    ciphertext: MPObject,
    frm_rank: int,
) -> PFunction:
    # Create input info for the data
    in_type: TableType | TensorType
    if isinstance(ciphertext, TableLike):
        in_type = TableType.from_tablelike(ciphertext)
    elif isinstance(ciphertext, TensorLike):
        in_type = TensorType.from_obj(ciphertext)

    out_type = in_type

    # Create the PFunction
    pfunc = PFunction(
        fn_type="tee.decrypt",
        ins_info=[in_type],
        outs_info=[out_type],
        fn_name="decrypt",
        frm_rank=frm_rank,
    )

    return pfunc


def sealTo(
    plaintext: MPObject,
    frm_rank: int,
    to_rank: int,
    tee_rank: int,
) -> MPObject:
    if plaintext.pmask is None:
        raise ValueError("SealTo does not support dynamic masks.")

    if plaintext.pmask != Mask.from_ranks(frm_rank):
        raise ValueError(
            f"Cannot seal from {Mask.from_ranks(frm_rank)} to {plaintext.pmask}, "
        )

    # TODO: sealTo only allowed between non-tee party and tee party

    # TODO: what if frm_rank == to_rank, no need to encrypt

    pfunc = encrypt_data(plaintext, frm_rank, to_rank, tee_rank)
    encrypted = prim.peval(pfunc, [plaintext], Mask.from_ranks(frm_rank))
    return encrypted[0]


def reveal(ciphertext: MPObject, frm_rank: int) -> MPObject:
    if ciphertext.pmask is None:
        raise ValueError("Reveal does not support dynamic masks.")

    if ciphertext.pmask != Mask.from_ranks(frm_rank):
        raise ValueError(
            f"Cannot reveal from {Mask.from_ranks(frm_rank)} to {ciphertext.pmask}, "
        )

    pfunc = decrypt_data(ciphertext, frm_rank)
    plaintext = prim.peval(pfunc, [ciphertext], Mask.from_ranks(frm_rank))
    return plaintext[0]
