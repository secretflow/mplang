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

from collections.abc import Callable
from typing import Any

from jax.tree_util import tree_unflatten

from mplang.core import Mask, MPObject, Rank, peval, psize
from mplang.core.context_mgr import cur_ctx
from mplang.ops import spu
from mplang.simp import mpi


def _get_spu_mask() -> Mask:
    spu_devices = cur_ctx().cluster_spec.get_devices_by_kind("SPU")
    if not spu_devices:
        raise ValueError("No SPU device found in the cluster specification")
    if len(spu_devices) > 1:
        raise ValueError("Multiple SPU devices found in the cluster specification")
    spu_device = spu_devices[0]
    spu_mask = Mask.from_ranks([member.rank for member in spu_device.members])
    return spu_mask


def spu_seal(obj: MPObject, frm_mask: Mask | None = None) -> list[MPObject]:
    spu_mask: Mask = _get_spu_mask()
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


def spu_seal_at(root: Rank, obj: MPObject) -> MPObject:
    results = spu_seal(obj, frm_mask=Mask.from_ranks(root))
    assert len(results) == 1, f"Expected one result, got {len(results)}"
    return results[0]


def spu_run(fe_type: str, pyfn: Callable, *args: Any, **kwargs: Any) -> Any:
    if fe_type != "jax":
        raise ValueError(f"Unsupported fe_type: {fe_type}")

    spu_mask = _get_spu_mask()
    pfunc, in_vars, out_tree = spu.jax_compile(pyfn, *args, **kwargs)
    assert all(var.pmask == spu_mask for var in in_vars), in_vars
    out_flat = peval(pfunc, in_vars, spu_mask)
    return tree_unflatten(out_tree, out_flat)


def spu_reveal(obj: MPObject, to_mask: Mask) -> MPObject:
    spu_mask = _get_spu_mask()

    assert obj.pmask == spu_mask, (obj.pmask, spu_mask)

    # (n_parties, n_shares)
    shares = [mpi.bcast_m(to_mask, rank, obj) for rank in Mask(spu_mask)]
    assert len(shares) == Mask(spu_mask).num_parties(), (shares, spu_mask)
    assert all(share.pmask == to_mask for share in shares)

    # Reconstruct the original object from shares
    pfunc, ins, _ = spu.reconstruct(*shares)
    return peval(pfunc, ins, to_mask)[0]  # type: ignore[no-any-return]


def spu_reveal_to(to_rank: Rank, obj: MPObject) -> MPObject:
    return spu_reveal(obj, to_mask=Mask.from_ranks(to_rank))


# -----------------------------------------------------------------------


# seal :: m a -> [s a]
def seal(obj: MPObject, frm_mask: Mask | None = None) -> list[MPObject]:
    """Seal an simp object, result a list of sealed objects, with
    the i'th element as the secret from the i'th party.
    """
    return spu_seal(obj, frm_mask=frm_mask)


# seal_at :: m Rank -> m a -> s a
def seal_at(root: Rank, obj: MPObject) -> MPObject:
    """Seal a simp object from a specific root party."""
    return spu_seal_at(root, obj)


# reveal :: s a -> m a
def reveal(obj: MPObject, to_mask: Mask | None = None) -> MPObject:
    """Reveal a sealed object to pmask'ed parties."""
    to_mask = to_mask or Mask.all(psize())
    return spu_reveal(obj, to_mask)


# reveal_to :: m Rank -> s a -> m a
def reveal_to(to_rank: Rank, obj: MPObject) -> MPObject:
    return spu_reveal_to(to_rank, obj)


def srun_jax(jax_fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run a jax function on sealed values securely.

    This function executes a JAX computation on sealed (secret-shared) values
    using secure multi-party computation (MPC).

    Args:
        jax_fn: A JAX function to run on sealed values
        *args: Positional arguments (sealed values)
        **kwargs: Keyword arguments (sealed values)

    Returns:
        The result of the computation, still in sealed form
    """
    return spu_run("jax", jax_fn, *args, **kwargs)
