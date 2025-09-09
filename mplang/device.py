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
This module provides the device oriented programming interface for MPC.

The device oriented programming interface is designed to provide a high-level
abstraction for the MPC programming. It allows the user to write the program
in a device-oriented manner, and the runtime will take care of the data
transformation between devices.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial, wraps
from typing import Any

from jax.tree_util import tree_map

import mplang.api as mapi
from mplang import simp
from mplang.core import InterpContext, MPObject, primitive
from mplang.core.cluster import ClusterSpec
from mplang.simp import mpi, smpc
from mplang.utils.func_utils import normalize_fn

# Automatic transfer between devices when parameter is not on the target device.
g_auto_trans: bool = True


# `function` decorator could also compile device-level apis.
function = primitive.function


class Utils:
    DEV_ID = "dev_id"

    @classmethod
    def is_device_obj(cls, obj: MPObject) -> bool:
        if not isinstance(obj, MPObject):
            return False
        return cls.DEV_ID in obj.attrs

    @classmethod
    def set_devid(cls, obj: MPObject, dev_id: str) -> MPObject:
        if not isinstance(obj, MPObject):
            raise TypeError(f"Input must be an instance of Object, {obj}")
        obj.attrs[cls.DEV_ID] = dev_id
        return obj

    @classmethod
    def get_devid(cls, obj: MPObject) -> str:
        if not isinstance(obj, MPObject):
            raise TypeError("Input must be an instance of Object")

        return obj.attrs[cls.DEV_ID]  # type: ignore[no-any-return]


def device(dev_id: str, *, fe_type: str = "jax") -> Callable:
    """Decorator to mark a function to be executed on a specific device.

    Args:
        dev_id: The device id.
        fe_type: The frontend type of the device, currently only "jax" is supported.

    Example:
        >>> @device("P0")
        ... def foo(x, y):
        ...     return x + y
    """

    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            nfn, args_flat = normalize_fn(
                fn, args, kwargs, lambda x: isinstance(x, MPObject)
            )
            if not all(Utils.is_device_obj(arg) for arg in args_flat):
                raise ValueError(f"All arguments must be device objects. {args_flat}")

            cluster_spec = mapi.cur_ctx().cluster_spec
            if g_auto_trans:
                args_flat = [_d2d(dev_id, arg) for arg in args_flat]

            assert all(Utils.get_devid(arg) == dev_id for arg in args_flat)
            dev_info = cluster_spec.devices[dev_id]
            if dev_info.kind.upper() == "SPU":
                var = smpc.srun(nfn, fe_type=fe_type)(args_flat)
                return tree_map(partial(Utils.set_devid, dev_id=dev_id), var)
            elif dev_info.kind.upper() == "PPU":
                assert len(dev_info.members) == 1
                rank = dev_info.members[0].rank
                var = simp.runAt(rank, nfn)(args_flat)
                return tree_map(partial(Utils.set_devid, dev_id=dev_id), var)
            else:
                raise ValueError(f"Unknown device type: {dev_info.type}")

        return wrapped

    return deco


def _d2d(to_dev_id: str, obj: MPObject) -> MPObject:
    assert isinstance(obj, MPObject)
    frm_dev_id = Utils.get_devid(obj)

    if frm_dev_id == to_dev_id:
        return obj

    cluster_spec: ClusterSpec = mapi.cur_ctx().cluster_spec
    frm_dev = cluster_spec.devices[frm_dev_id]
    to_dev = cluster_spec.devices[to_dev_id]
    frm_to_pair = (frm_dev.kind.upper(), to_dev.kind.upper())

    if frm_to_pair == ("SPU", "SPU"):
        raise NotImplementedError("Only one SPU is supported for now.")
    elif frm_to_pair == ("SPU", "PPU"):
        assert len(to_dev.members) == 1
        to_rank = to_dev.members[0].rank
        var = smpc.revealTo(obj, to_rank)
        return tree_map(partial(Utils.set_devid, dev_id=to_dev_id), var)  # type: ignore[no-any-return]
    elif frm_to_pair == ("PPU", "SPU"):
        assert len(frm_dev.members) == 1
        frm_rank = frm_dev.members[0].rank
        var = smpc.sealFrom(obj, frm_rank)
        return tree_map(partial(Utils.set_devid, dev_id=to_dev_id), var)  # type: ignore[no-any-return]
    elif frm_to_pair == ("PPU", "PPU"):
        assert len(frm_dev.members) == 1 and len(to_dev.members) == 1
        frm_rank = frm_dev.members[0].rank
        to_rank = to_dev.members[0].rank
        var = mpi.p2p(frm_rank, to_rank, obj)
        return tree_map(partial(Utils.set_devid, dev_id=to_dev_id), var)  # type: ignore[no-any-return]
    else:
        raise ValueError(f"Unsupported device transfer: {frm_to_pair}")


def put(to_dev_id: str, obj: Any) -> MPObject:
    if not isinstance(obj, MPObject):
        return device(to_dev_id)(lambda x: x)(obj)  # type: ignore[no-any-return]
    assert isinstance(obj, MPObject)
    return _d2d(to_dev_id, obj)


def _fetch(interp: InterpContext, obj: MPObject) -> Any:
    dev_id = Utils.get_devid(obj)
    cluster_spec = interp.cluster_spec
    dev_kind = cluster_spec.devices[dev_id].kind.upper()

    if dev_kind == "SPU":
        revealed = smpc.reveal(obj)
        result = mapi.fetch(interp, revealed)
        return result[0]
    elif dev_kind == "PPU":
        dev_info = cluster_spec.devices[dev_id]
        assert len(dev_info.members) == 1
        rank = dev_info.members[0].rank
        result = mapi.fetch(interp, obj)
        return result[rank]
    else:
        raise ValueError(f"Unknown device id: {dev_id}")


def fetch(interp: InterpContext, objs: Any) -> list[Any]:
    ctx = interp or mapi.cur_ctx()
    assert isinstance(ctx, InterpContext), f"Expect InterpContext, got {ctx}"
    return list(tree_map(partial(_fetch, ctx), objs))
