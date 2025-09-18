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
from typing import Any

from jax.tree_util import tree_unflatten

from mplang.core import Mask, MPObject, Rank, peval, psize
from mplang.core.context_mgr import cur_ctx
from mplang.frontend import spu
from mplang.simp import mpi


class SecureAPI(ABC):
    """Base class for secure APIs."""

    @abstractmethod
    def seal(self, obj: MPObject, frm_mask: Mask | None) -> list[MPObject]: ...

    @abstractmethod
    def sealFrom(self, obj: MPObject, root: Rank) -> MPObject: ...

    @abstractmethod
    def seval(self, fe_type: str, pyfn: Callable, *args: Any, **kwargs: Any) -> Any:
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

    def seval(self, fe_type: str, pyfn: Callable, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("TODO")

    def reveal(self, obj: MPObject, to_mask: Mask) -> MPObject:
        raise NotImplementedError("TODO")

    def revealTo(self, obj: MPObject, to_rank: Rank) -> MPObject:
        raise NotImplementedError("TODO")


class SPU(SecureAPI):
    """Use SPU to  perform secure operations."""

    def get_spu_mask(self) -> Mask:
        spu_devices = cur_ctx().cluster_spec.get_devices_by_kind("SPU")
        if not spu_devices:
            raise ValueError("No SPU device found in the cluster specification")
        if len(spu_devices) > 1:
            raise ValueError("Multiple SPU devices found in the cluster specification")
        spu_device = spu_devices[0]
        spu_mask = Mask.from_ranks([member.rank for member in spu_device.members])
        return spu_mask

    def seal(self, obj: MPObject, frm_mask: Mask | None = None) -> list[MPObject]:
        spu_mask: Mask = self.get_spu_mask()
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
                if not Mask(frm_mask).is_subset(obj.pmask):
                    raise ValueError(f"Cannot seal from {frm_mask} to {obj.pmask}, ")

        # Get the world_size from spu_mask (number of parties in SPU computation)
        world_size = Mask(spu_mask).num_parties()
        pfunc, ins, _ = spu.makeshares(
            obj, world_size=world_size, visibility=spu.Visibility.SECRET
        )
        assert len(ins) == 1
        shares = peval(pfunc, ins, frm_mask)

        # scatter the shares to each party.
        return [mpi.scatter_m(spu_mask, rank, shares) for rank in Mask(frm_mask)]

    def sealFrom(self, obj: MPObject, root: Rank) -> MPObject:
        results = seal(obj, frm_mask=Mask.from_ranks(root))
        assert len(results) == 1, f"Expected one result, got {len(results)}"
        return results[0]

    def seval(self, fe_type: str, pyfn: Callable, *args: Any, **kwargs: Any) -> Any:
        if fe_type != "jax":
            raise ValueError(f"Unsupported fe_type: {fe_type}")

        spu_mask = self.get_spu_mask()
        pfunc, in_vars, out_tree = spu.jax_compile(pyfn, *args, **kwargs)
        assert all(var.pmask == spu_mask for var in in_vars), in_vars
        out_flat = peval(pfunc, in_vars, spu_mask)
        return tree_unflatten(out_tree, out_flat)

    def reveal(self, obj: MPObject, to_mask: Mask) -> MPObject:
        spu_mask = self.get_spu_mask()

        assert obj.pmask == spu_mask, (obj.pmask, spu_mask)

        # (n_parties, n_shares)
        shares = [mpi.bcast_m(to_mask, rank, obj) for rank in Mask(spu_mask)]
        assert len(shares) == Mask(spu_mask).num_parties(), (shares, spu_mask)
        assert all(share.pmask == to_mask for share in shares)

        # Reconstruct the original object from shares
        pfunc, ins, _ = spu.reconstruct(*shares)
        return peval(pfunc, ins, to_mask)[0]  # type: ignore[no-any-return]

    def revealTo(self, obj: MPObject, to_rank: Rank) -> MPObject:
        return self.reveal(obj, to_mask=Mask.from_ranks(to_rank))


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
    to_mask = to_mask or Mask.all(psize())
    return _get_sapi().reveal(obj, to_mask)


# revealTo :: s a -> m Rank -> m a
def revealTo(obj: MPObject, to_rank: Rank) -> MPObject:
    return _get_sapi().revealTo(obj, to_rank)


# srun :: (a -> a) -> s a -> s a
def srun(pyfn: Callable, *, fe_type: str = "jax") -> Callable:
    @wraps(pyfn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return _get_sapi().seval(fe_type, pyfn, *args, **kwargs)

    return wrapped
