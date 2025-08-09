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


from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from functools import wraps

from jax.tree_util import tree_unflatten

import mplang.mpi as mpi
import mplang.utils.mask_utils as mask_utils
from mplang.core import primitive as prim
from mplang.core.base import Mask, MPObject, Rank
from mplang.core.context_mgr import cur_ctx
from mplang.plib.spu_fe import SpuFE, Visibility


class SecureAPI(ABC):
    """Base class for secure APIs."""

    @abstractmethod
    def seal(self, obj: MPObject, frm_mask: Mask | None) -> list[MPObject]: ...

    @abstractmethod
    def sealFrom(self, obj: MPObject, root: Rank) -> MPObject: ...

    @abstractmethod
    def seval(self, fe_type: str, pyfn: Callable, *args, **kwargs):
        """Run a function in the secure environment."""

    @abstractmethod
    def reveal(self, obj: MPObject, to_mask: Mask) -> MPObject: ...

    @abstractmethod
    def revealTo(self, obj: MPObject, to_rank: Rank) -> MPObject: ...


class Delegation(SecureAPI):
    """Delegate to a trusted third-party to perform secure operations."""

    def seal(self, obj: MPObject, frm_mask: Mask | None = None) -> list[MPObject]:
        raise NotImplementedError("TODO")

    def sealFrom(self, obj: MPObject, root: Rank) -> MPObject:
        raise NotImplementedError("TODO")

    def seval(self, fe_type: str, pyfn: Callable, *args, **kwargs):
        raise NotImplementedError("TODO")

    def reveal(self, obj: MPObject, to_mask: Mask) -> MPObject:
        raise NotImplementedError("TODO")

    def revealTo(self, obj: MPObject, to_rank: Rank) -> MPObject:
        raise NotImplementedError("TODO")


class SPU(SecureAPI):
    """Use SPU to  perform secure operations."""

    def seal(self, obj: MPObject, frm_mask: Mask | None = None) -> list[MPObject]:
        spu_mask: Mask = cur_ctx().attr("spu_mask")
        if obj.pmask is None:
            if frm_mask is None:
                # NOTE: The length of the return list is statically determined by obj_mask,
                # so only static masks are supported here.
                raise ValueError("Seal does not support dynamic masks.")
            else:
                # Force seal from the given mask, the runtime will raise error if the mask
                # does not match obj.pmask.
                # TODO(jint): add set_pmask primitive.
                pass
        else:
            if frm_mask is None:
                frm_mask = obj.pmask
            else:
                if not mask_utils.is_subset(frm_mask, obj.pmask):
                    raise ValueError(f"Cannot seal from {frm_mask} to {obj.pmask}, ")

        # Get the world_size from spu_mask (number of parties in SPU computation)
        spu = SpuFE(world_size=mask_utils.bit_count(spu_mask))

        # make shares on each party.
        pfunc = spu.makeshares(obj, visibility=Visibility.SECRET)
        shares = prim.peval(pfunc, [obj], frm_mask)

        # scatter the shares to each party.
        return [
            mpi.scatter_m(spu_mask, rank, shares)
            for rank in mask_utils.enum_mask(frm_mask)
        ]

    def sealFrom(self, obj: MPObject, root: Rank) -> MPObject:
        results = seal(obj, frm_mask=1 << root)
        assert len(results) == 1, f"Expected one result, got {len(results)}"
        return results[0]

    def seval(self, fe_type: str, pyfn: Callable, *args, **kwargs):
        if fe_type != "jax":
            raise ValueError(f"Unsupported fe_type: {fe_type}")

        spu_mask = cur_ctx().attr("spu_mask")

        spu = SpuFE(world_size=mask_utils.bit_count(spu_mask))
        is_mpobject = lambda x: isinstance(x, MPObject)
        pfunc, in_vars, out_tree = spu.compile_jax(is_mpobject, pyfn, *args, **kwargs)
        assert all(var.pmask == spu_mask for var in in_vars), in_vars

        out_flat = prim.peval(pfunc, in_vars, spu_mask)

        return tree_unflatten(out_tree, out_flat)

    def reveal(self, obj: MPObject, to_mask: Mask) -> MPObject:
        spu_mask = cur_ctx().attr("spu_mask")

        assert obj.pmask == spu_mask, (obj.pmask, spu_mask)

        # (n_parties, n_shares)
        shares = [
            mpi.bcast_m(to_mask, rank, obj) for rank in mask_utils.enum_mask(spu_mask)
        ]
        assert len(shares) == mask_utils.bit_count(spu_mask), (shares, spu_mask)
        assert all(share.pmask == to_mask for share in shares)

        # Reconstruct the original object from shares
        spu = SpuFE(world_size=mask_utils.bit_count(spu_mask))
        pfunc = spu.reconstruct(shares)
        return prim.peval(pfunc, shares, to_mask)[0]

    def revealTo(self, obj: MPObject, to_rank: Rank) -> MPObject:
        return self.reveal(obj, to_mask=1 << to_rank)


class SEE(Enum):
    """Secure Execution Environment."""

    MOCK = 0
    SPU = 1
    TEE = 2


# TODO(jint): move me to options.py
mode: SEE = SEE.SPU


def _get_sapi() -> SecureAPI:
    """Get the current secure API based on the mode."""
    if mode == SEE.MOCK:
        return Delegation()
    elif mode == SEE.SPU:
        return SPU()
    elif mode == SEE.TEE:
        raise NotImplementedError("TEE is not implemented yet")
    else:
        raise ValueError(f"Unknown mode: {mode}")


# seal :: m a -> [s a]
def seal(obj: MPObject, frm_mask: Mask | None = None) -> list[MPObject]:
    """Seal an simp object, result a list of sealed objects, with
    the i'th element as the secret from the i'th party.
    """
    return _get_sapi().seal(obj, frm_mask=frm_mask)


# sealFrom :: m a -> m Rank -> s a
def sealFrom(obj: MPObject, root: Rank) -> MPObject:
    """Seal an simp object from a specific root party."""
    return _get_sapi().sealFrom(obj, root)


# reveal :: s a -> m a
def reveal(obj: MPObject, to_mask: Mask | None = None) -> MPObject:
    """Reveal a sealed object to pmask'ed parties."""
    to_mask = to_mask or ((1 << prim.psize()) - 1)
    return _get_sapi().reveal(obj, to_mask)


# revealTo :: s a -> m Rank -> m a
def revealTo(obj: MPObject, to_rank: Rank) -> MPObject:
    return _get_sapi().revealTo(obj, to_rank)


# srun :: (a -> a) -> s a -> s a
def srun(pyfn: Callable, *, fe_type: str = "jax"):
    @wraps(pyfn)
    def wrapped(*args, **kwargs):
        return _get_sapi().seval(fe_type, pyfn, *args, **kwargs)
        # if fe_type == "jax":
        #     is_mpobject = lambda x: isinstance(x, MPObject)
        #     # suppose the arguments is already sealed, so it's a mpobject.
        #     flat_fn, in_vars = PyPFunction.from_jax_function(
        #         is_mpobject, pyfn, *args, **kwargs
        #     )
        #     out_flat = _get_sapi().seval(flat_fn, in_vars, {})
        #     return tree_unflatten(flat_fn.out_tree, out_flat)
        # else:
        #     raise ValueError("SPU only support JAX for now")

    return wrapped
