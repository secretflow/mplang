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

import logging

import mplang.core.primitive as prim
import mplang.utils.mask_utils as mask_utils
from mplang.core.base import Mask, MPObject, Rank


# scatter :: [m a] -> m Rank -> m a
@prim.primitive
def scatter_m(to_mask: Mask, root: Rank, args: list[MPObject]) -> MPObject:
    """Scatter the object from root to the parties in pmask.

    Args:
        to_mask: The mask of the parties that will receive the object.
        root: The rank of the root party.
        args: The objects to be scattered, which must hold by root and length of pmask'ed parties.
    """
    # sanity check, ensure all args are in the to_mask.
    for arg in args:
        if arg.pmask is None:
            logging.warning(f"Scattering dynamic {arg} from static root {root}")
        else:
            if not mask_utils.is_subset(1 << root, arg.pmask):
                raise ValueError(f"Expect root {root} in {arg.pmask}, got {arg}.")

    to_ranks = list(mask_utils.enum_mask(to_mask))
    if len(args) != len(to_ranks):
        raise ValueError(f"Expect {len(to_ranks)} args, got {len(args)}. ")

    scattered = [
        prim.pshfl_s(arg, 1 << to_rank, [root]) for to_rank, arg in zip(to_ranks, args)
    ]

    result = prim.pconv(scattered)
    assert result.pmask == to_mask, (result.pmask, to_mask)
    return result


# gather :: m a -> m Rank -> [m a]
@prim.primitive
def gather_m(src_mask: Mask, root: Rank, arg: MPObject) -> list[MPObject]:
    """Gather the object from pmask'ed parties to the root party.

    Args:
        src_pmask: The mask of the parties that will gather the object.
        root: The rank of the root party.
        arg: The object to be gathered, which must be the subset of pmask.

    Returns:
        A list of objects, with length equal to the number of parties in pmask.
    """
    # static pmask check.
    if arg.pmask is None:
        logging.warning(f"Gathering {arg} from {src_mask}, may raise RuntimeError.")
    else:
        if not mask_utils.is_subset(src_mask, arg.pmask):
            raise ValueError(f"Expect {src_mask} in {arg.pmask}, got {arg}.")

    result = []
    root_mask = Mask(1 << root)
    for src_rank in mask_utils.enum_mask(src_mask):
        # Shuffle data from src_rank to root
        gathered_data = prim.pshfl_s(arg, root_mask, [src_rank])
        result.append(gathered_data)

    assert len(result) == mask_utils.bit_count(src_mask), (result, src_mask)
    return result


# bcast :: m a -> m Rank -> m a
@prim.function
def bcast_m(pmask: Mask, root: Rank, obj: MPObject) -> MPObject:
    """Broadcast the object from the root party to the parties in pmask."""
    if obj.pmask is None:
        logging.warning(f"Broadcasting {obj} from {root}, may raise RuntimeError.")
    else:
        if not mask_utils.is_subset(1 << root, obj.pmask):
            raise ValueError(f"Expect root {root} in obj mask {obj.pmask}.")

    result = prim.pshfl_s(obj, pmask, [root] * mask_utils.bit_count(pmask))

    assert result.pmask == pmask, (result.pmask, pmask)
    return result


# p2p :: m Rank -> m Rank -> m a -> m a
@prim.function
def p2p(frm: Rank, to: Rank, obj: MPObject) -> MPObject:
    """Point-to-point communication from frm to to."""

    # sanity check, ensure the object is in the frm mask.
    if obj.pmask is None:
        logging.warning(f"P2P {obj} from {frm} to {to}, may raise RuntimeError.")
    else:
        if not mask_utils.is_subset(1 << frm, obj.pmask):
            raise ValueError(f"Expect {frm} in {obj.pmask}, got {obj}.")

    if frm == to:
        return obj

    return prim.pshfl_s(obj, Mask(1 << to), [frm])


# allgather :: m a -> [m a]
@prim.function
def allgather_m(pmask: Mask, arg: MPObject) -> list[MPObject]:
    """Gather the object from all parties in pmask and return a list of objects."""

    if arg.pmask is None:
        logging.warning(f"Allgathering {arg} from {pmask}, may raise RuntimeError.")

    if not mask_utils.is_subset(pmask, arg.pmask):
        raise ValueError(f"Expect {pmask} in {arg.pmask}, got {arg}.")

    # TODO(jint): implement me.
    raise NotImplementedError("Allgather not implemented")


@prim.function
def pshfl(val: MPObject, index: MPObject) -> MPObject:
    """Shuffle the value to the index party from the source party."""
    return prim.pshfl(val, index)
