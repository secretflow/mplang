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
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import Any

from jax.tree_util import tree_map

import mplang.api as mapi
from mplang import mpi, simp, smpc
from mplang.core.context_mgr import set_ctx
from mplang.core.interp import InterpContext
from mplang.core.mask import Mask
from mplang.core.mpobject import MPObject
from mplang.core.primitive import primitive
from mplang.frontend import teeu
from mplang.runtime.driver import ExecutorDriver
from mplang.runtime.simulation import Simulator
from mplang.utils.func_utils import normalize_fn


@dataclass
class DeviceInfo:
    """A device description."""

    type: str
    """The backend type information."""

    node_ids: list[str]
    """The nodes that runs together to construct the device.
    For device that requires multiple nodes to run together, we assume that
    they run in a SPMD manner.
    """

    configs: dict = field(default_factory=dict)
    """The device runtime configurations"""


SAMPLE_DEVICE_CONF = {
    "SP0": {
        "type": "SPU",
        "node_ids": ["node:0", "node:1", "node:2"],
        "configs": {
            "protocol": "SEMI2K",
            "field": "FM128",
            "enable_pphlo_profile": True,
        },
    },
    "TEE0": {
        "type": "TEEU",
        "node_ids": ["node:0", "node:1", "node:3"],
        "configs": {"tee_mode": "sim", "tee_node": "node:3"},
    },
    "P0": {"type": "PPU", "node_ids": ["node:0"]},
    "P1": {"type": "PPU", "node_ids": ["node:1"]},
}


def parse_device_conf(conf: dict) -> dict[str, DeviceInfo]:
    return {dev_id: DeviceInfo(**desc) for dev_id, desc in conf.items()}


class DeviceContext:
    def __init__(self, dev_infos: dict[str, DeviceInfo], auto_trans: bool = True):
        super().__init__()
        self.dev_infos = dev_infos
        self.auto_trans = auto_trans
        # Note: ensure node_ids maintain a consistent order for deterministic behavior
        node_ids = [nid for info in dev_infos.values() for nid in info.node_ids]
        node_ids = sorted(set(node_ids))
        assert len(node_ids) == len(set(node_ids)), (
            f"node_ids must be unique. {node_ids}"
        )
        self.node_ids = node_ids

    def id2rank(self, dev_id: str) -> int:
        dev_info = self.dev_infos[dev_id]
        if dev_info.type == "SPU":
            raise ValueError("SPU does not have a rank")
        if dev_info.type == "TEEU":
            # return tee node rank
            tee_id = dev_info.configs.get("tee_node")
            if tee_id is None:
                raise ValueError("miss field `tee_node`")
            return self.node_ids.index(tee_id)
        assert len(dev_info.node_ids) == 1
        return self.node_ids.index(dev_info.node_ids[0])


def init(device_def: dict, nodes_def: dict | None = None) -> None:
    # no 'real' party defined, we are in simulation mode
    if nodes_def is None:
        nodes_def = {}
    device_conf = parse_device_conf(device_def)
    device_ctx = DeviceContext(device_conf)

    spu_conf = [dev for dev in device_conf.values() if dev.type == "SPU"]
    assert len(spu_conf) <= 1, "Only one SPU is supported for now."

    tee_conf = [dev for dev in device_conf.values() if dev.type == "TEEU"]
    assert len(spu_conf) <= 1, "Only one TEEU is supported for now."

    # unique node_ids across all devices
    node_ids = [nid for dev_info in device_conf.values() for nid in dev_info.node_ids]
    node_ids = sorted(set(node_ids))
    world_size = len(node_ids)
    spu_mask = Mask.none()
    tee_mask = None
    tee_rank = None
    tee_mode = "sim"
    if len(spu_conf) > 0:
        for nid in spu_conf[0].node_ids:
            spu_mask |= Mask.from_ranks(node_ids.index(nid))
    if len(tee_conf) > 0:
        tee_mode = tee_conf[0].configs["tee_mode"]
        tee_node_id = tee_conf[0].configs["tee_node"]
        tee_mask = Mask.none()
        for nid in tee_conf[0].node_ids:
            tee_mask |= Mask.from_ranks(node_ids.index(nid))
            if nid == tee_node_id:
                tee_rank = node_ids.index(nid)

    driver: InterpContext
    if not nodes_def:
        driver = Simulator(
            world_size,
            spu_mask=spu_mask,
            tee_mask=tee_mask,
            tee_rank=tee_rank,
            device_ctx=device_ctx,
        )
    else:
        driver = ExecutorDriver(
            nodes_def,
            spu_mask=spu_mask,
            tee_mask=tee_mask,
            tee_rank=tee_rank,
            tee_mode=tee_mode,
            device_ctx=device_ctx,
        )

    set_ctx(driver)


function = primitive


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

            device_ctx = mapi.cur_ctx().attr("device_ctx")
            if device_ctx.auto_trans:
                args_flat = [_d2d(dev_id, arg) for arg in args_flat]

            assert all(Utils.get_devid(arg) == dev_id for arg in args_flat)
            dev_info = device_ctx.dev_infos[dev_id]
            if dev_info.type == "SPU":
                var = smpc.srun(nfn, fe_type=fe_type)(args_flat)
                return tree_map(partial(Utils.set_devid, dev_id=dev_id), var)
            elif dev_info.type == "PPU":
                rank = device_ctx.id2rank(dev_id)
                var = simp.runAt(rank, nfn)(args_flat)
                return tree_map(partial(Utils.set_devid, dev_id=dev_id), var)
            elif dev_info.type == "TEEU":
                rank = device_ctx.id2rank(dev_id)
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

    device_ctx: DeviceContext = mapi.cur_ctx().attr("device_ctx")
    frm_to_pair = (
        device_ctx.dev_infos[frm_dev_id].type,
        device_ctx.dev_infos[to_dev_id].type,
    )

    if frm_to_pair == ("SPU", "SPU"):
        raise NotImplementedError("Only one SPU is supported for now.")
    elif frm_to_pair == ("SPU", "PPU"):
        var = smpc.revealTo(obj, device_ctx.id2rank(to_dev_id))
        return tree_map(partial(Utils.set_devid, dev_id=to_dev_id), var)  # type: ignore[no-any-return]
    elif frm_to_pair == ("PPU", "SPU"):
        var = smpc.sealFrom(obj, device_ctx.id2rank(frm_dev_id))
        return tree_map(partial(Utils.set_devid, dev_id=to_dev_id), var)  # type: ignore[no-any-return]
    elif frm_to_pair == ("PPU", "PPU"):
        frm_rank = device_ctx.id2rank(frm_dev_id)
        to_rank = device_ctx.id2rank(to_dev_id)
        var = mpi.p2p(frm_rank, to_rank, obj)
        return tree_map(partial(Utils.set_devid, dev_id=to_dev_id), var)  # type: ignore[no-any-return]
    elif frm_to_pair == ("TEEU", "PPU"):
        frm_rank = device_ctx.id2rank(frm_dev_id)
        to_rank = device_ctx.id2rank(to_dev_id)
        encrypted_var = teeu.sealTo(obj, frm_rank, to_rank, frm_rank)
        var = teeu.reveal(encrypted_var, to_rank)
        return tree_map(partial(Utils.set_devid, dev_id=to_dev_id), var)  # type: ignore[no-any-return]
    elif frm_to_pair == ("PPU", "TEEU"):
        frm_rank = device_ctx.id2rank(frm_dev_id)
        to_rank = device_ctx.id2rank(to_dev_id)
        encrypted_var = teeu.sealTo(obj, frm_rank, to_rank, to_rank)
        var = teeu.reveal(encrypted_var, to_rank)
        return tree_map(partial(Utils.set_devid, dev_id=to_dev_id), var)  # type: ignore[no-any-return]
    elif frm_to_pair == ("TEEU", "SPU"):
        raise NotImplementedError("TEEU to SPU transfer not yet implemented.")
    elif frm_to_pair == ("SPU", "TEEU"):
        raise NotImplementedError("SPU to TEEU transfer not yet implemented.")
    elif frm_to_pair == ("TEEU", "TEEU"):
        raise NotImplementedError("Only one TEEU is supported for now.")
    else:
        raise ValueError(f"Unsupported device transfer: {frm_to_pair}")


def put(to_dev_id: str, obj: Any) -> MPObject:
    if not isinstance(obj, MPObject):
        return device(to_dev_id)(lambda x: x)(obj)  # type: ignore[no-any-return]
    assert isinstance(obj, MPObject)
    return _d2d(to_dev_id, obj)


def _fetch(interp: InterpContext, obj: MPObject) -> Any:
    dev_id = Utils.get_devid(obj)
    device_ctx = interp.attr("device_ctx")
    dev_type = device_ctx.dev_infos[dev_id].type

    if dev_type == "SPU":
        revealed = smpc.reveal(obj)
        result = mapi.fetch(interp, revealed)
        return result[0]
    elif dev_type == "PPU":
        rank = device_ctx.id2rank(dev_id)
        result = mapi.fetch(interp, obj)
        return result[rank]
    elif dev_type == "TEEU":
        # TODO: need encrypt and decypt
        revealed = smpc.reveal(obj)
        result = mapi.fetch(interp, revealed)
        return result[0]
    else:
        raise ValueError(f"Unknown device id: {dev_id}")


def fetch(interp: InterpContext, objs: Any) -> list[Any]:
    ctx = interp or mapi.cur_ctx()
    assert isinstance(ctx, InterpContext), f"Expect InterpContext, got {ctx}"
    return list(tree_map(partial(_fetch, ctx), objs))
