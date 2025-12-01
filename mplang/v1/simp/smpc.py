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

"""
SMPC on simp: conventions and object semantics

Overview
- simp is party-centric. Objects produced purely by simp code carry only an execution
    mask ("pmask") and have no security device semantics by default.
- Secure semantics (secret sharing, protected execution, declassification) are introduced
    only when using the device API or the helpers in this module: "seal", "srun", "reveal".

Definitions
- "__device__" attribute is attached by the device API to indicate the concrete device
    an object is bound to (e.g., an SPU/TEE/PPU name). See mplang.device.DEVICE_ATTR_NAME.
- pmask describes which parties currently hold/execute the value under the simp model.

Conventions
1) If an object has NO "__device__" attribute (i.e., it has not gone through the device API):
     - It is a simp object, privately owned on the parties indicated by its pmask.
     - When sealed via "seal(obj)", we infer target PPU device(s) from pmask:
         • one-hot pmask {pi} → route to PPU(pi).
         • multi-party pmask → fan out per party and seal independently to each party's PPU.
     - Such objects CANNOT be passed to "srun"/"reveal" directly; seal first.

2) If an object HAS a "__device__" attribute:
     - Its behavior follows the bound device (e.g., SPU/TEE/PPU) and its membership.
     - "srun" executes on that device; "reveal" declassifies from that device to the requested parties.
     - pmask must be consistent with the device membership during transitions; inconsistencies raise errors.

Notes
- "seal"/"seal_from" construct secret shares on the chosen secure device and attach the
    "__device__" attribute to outputs. "srun"/"reveal" assume inputs are already sealed
    (device-bound) and validate pmask ↔ device-membership consistency.
- These rules align with "design/simp_vs_device.md" and keep routing unambiguous.

Examples (obj state → interpretation/behavior)
- {pmask={A}, dev_attr=None}: simp plaintext on party A. "seal" routes to PPU(A);
    must "seal" before "srun"/"reveal".
- {pmask={A,B}, dev_attr=None}: simp plaintext held by A and B. "seal" produces two
    per-party sealed objects via PPU(A) and PPU(B), respectively.
- {pmask={A,B}, dev_attr="spu:spu0"}: device object on SPU(spu0) whose members are {A,B};
    "srun" runs on spu0; "reveal(to={A})" reveals result to party A.
- {pmask={A}, dev_attr="ppu:A"}: device object on PPU(A); "reveal(to={A})" returns A's plaintext.
- {pmask=None, dev_attr=None}: dynamic pmask; "seal" is unsupported and will error.
- {pmask={A}, dev_attr="spu:spu0"} where A ∉ members(spu0): inconsistent; operations will error.
"""

from collections.abc import Callable
from typing import Any

from mplang.v1 import _device
from mplang.v1.core import Mask, MPObject, Rank, psize
from mplang.v1.core.cluster import Device
from mplang.v1.core.context_mgr import cur_ctx
from mplang.v1.core.primitive import pconv
from mplang.v1.simp.api import set_mask
from mplang.v1.utils.func_utils import normalize_fn


def _determine_secure_device(*args: MPObject) -> Device:
    """Determine secure device from args, or find any available if no args."""
    if not args:
        # Find an available secure device (fallback when no args provided).
        devices = cur_ctx().cluster_spec.get_devices_by_kind("SPU")
        if devices:
            return devices[0]

        devices = cur_ctx().cluster_spec.get_devices_by_kind("TEE")
        if devices:
            return devices[0]

        raise ValueError(
            "No secure device (SPU or TEE) found in the cluster specification"
        )

    dev_names: list[str] = []
    for arg in args:
        if not _device.is_device_obj(arg):
            raise ValueError(
                "srun/reveal expect sealed inputs with a device attribute; "
                f"got an unsealed object: {arg}. Please call seal()/seal_from() first."
            )
        dev_names.append(_device.get_dev_attr(arg))

    if len(set(dev_names)) != 1:
        raise ValueError(f"Ambiguous secure devices among arguments: {dev_names}")

    dev_name = dev_names[0]

    cluster_spec = cur_ctx().cluster_spec
    assert dev_name in cluster_spec.devices
    return cluster_spec.devices[dev_name]


def _get_ppu_from_rank(rank: Rank) -> Device:
    """Get the PPU device for a specific rank."""
    for dev in cur_ctx().cluster_spec.get_devices_by_kind("PPU"):
        assert len(dev.members) == 1, "Expected single member PPU devices."
        if dev.members[0].rank == rank:
            return dev
    raise ValueError(f"No PPU device found for rank {rank}.")


def seal(obj: MPObject) -> list[MPObject] | MPObject:
    """Seal a simp object to a secure device.

    Args:
        obj: The simp object to seal.

    Returns:
        The sealed object(s). If the input is a plaintext simp object with a multi-party
        mask, a list of sealed objects (one per party) is returned. Otherwise, a
        single sealed object is returned.
    """

    if obj.pmask is None:
        raise ValueError("Seal does not support dynamic masks.")

    if _device.is_device_obj(obj):
        sdev = _determine_secure_device()
        return _device._d2d(sdev.name, obj)
    else:
        # it's a normal plaintext simp object, treat as a list of PPU objects
        rets: list[MPObject] = []
        for rank in obj.pmask:
            ppu_obj = set_mask(obj, Mask.from_ranks([rank]))
            _device.set_dev_attr(ppu_obj, _get_ppu_from_rank(rank).name)
            sealed = seal(ppu_obj)
            assert isinstance(sealed, MPObject), (
                "Expected single sealed object per rank"
            )
            rets.append(sealed)
        return rets


def seal_from(from_rank: Rank, obj: MPObject) -> MPObject:
    """Seal a simp object from a specific party to its PPU.

    Args:
        from_rank: The party rank from which to seal the object.
        obj: The simp object to seal.

    Returns:
        The sealed object.
    """
    obj = set_mask(obj, Mask.from_ranks([from_rank]))
    out = seal(obj)
    assert isinstance(out, list), "seal_from should return a list of sealed objects."
    assert len(out) == 1, "seal_from should return a single sealed object."
    return out[0]


# reveal :: s a -> m a
def reveal(obj: MPObject, to_mask: Mask | None = None) -> MPObject:
    """Reveal a sealed object to pmask'ed parties."""
    assert isinstance(obj, MPObject), "reveal expects an MPObject."

    if not _device.is_device_obj(obj):
        raise ValueError(f"reveal does not support non-device object={obj}.")

    if to_mask is None:
        ranks = []
        for rank in range(psize()):
            try:
                _get_ppu_from_rank(rank)
            except ValueError:
                continue
            ranks.append(rank)
        to_mask = Mask.from_ranks(ranks)
    rets = [reveal_to(rank, obj) for rank in to_mask]
    return pconv(rets)


def reveal_to(to_rank: Rank, obj: MPObject) -> MPObject:
    """Reveal a sealed object to a specific party."""
    if not _device.is_device_obj(obj):
        raise ValueError("reveal_to expects a device object (sealed value).")

    to_dev = _get_ppu_from_rank(to_rank)
    return _device._d2d(to_dev.name, obj)


def srun(fe_type: str, pyfn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run a function on sealed values securely.

    This function executes a computation on sealed (secret-shared) values
    using secure multi-party computation (MPC).

    Args:
        fe_type: The front-end type, e.g., "jax"
        pyfn: A function to run on sealed values
        *args: Positional arguments (sealed values)
        **kwargs: Keyword arguments (sealed values)

    Returns:
        The result of the computation, still in sealed form
    """

    fn_flat, args_flat = normalize_fn(
        pyfn, args, kwargs, lambda x: isinstance(x, MPObject)
    )

    dev_info = _determine_secure_device(*args_flat)

    dev_kind = dev_info.kind.upper()
    if dev_kind in {"SPU", "TEE"}:
        return _device.device(dev_info.name, fe_type=fe_type)(fn_flat)(args_flat)
    else:
        raise ValueError(f"Unsupported secure device kind: {dev_kind}")


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
    return srun("jax", jax_fn, *args, **kwargs)
