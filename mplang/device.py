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
from mplang.core.cluster import ClusterSpec, Device
from mplang.core.context_mgr import cur_ctx
from mplang.core.tensor import TensorType
from mplang.frontend import builtin, crypto, ibis_cc, jax_cc, tee
from mplang.frontend.base import FeOperation
from mplang.frontend.ibis_cc import IbisCompiler
from mplang.frontend.jax_cc import JaxCompiler
from mplang.simp import mpi, smpc

# Automatic transfer between devices when parameter is not on the target device.
g_auto_trans: bool = True

_HKDF_INFO_LITERAL: str = "mplang/device/tee/v1"
# Default KEM suite for TEE session establishment; make configurable via ClusterSpec in future.
_TEE_KEM_SUITE: str = "x25519"


# `function` decorator could also compile device-level apis.
function = primitive.function

# magic attribute name to mark a MPObject as a device object
DEVICE_ATTR_NAME = "_devid_"


def _is_device_obj(obj: Any) -> bool:
    if not isinstance(obj, MPObject):
        return False
    return DEVICE_ATTR_NAME in obj.attrs


def _set_devid(obj: MPObject, dev_id: str) -> MPObject:
    if not isinstance(obj, MPObject):
        raise TypeError(f"Input must be an instance of Object, {obj}")
    obj.attrs[DEVICE_ATTR_NAME] = dev_id
    return obj


def _get_devid(obj: MPObject) -> str:
    if not isinstance(obj, MPObject):
        raise TypeError("Input must be an instance of Object")

    return obj.attrs[DEVICE_ATTR_NAME]  # type: ignore[no-any-return]


_is_mpobj = lambda x: isinstance(x, MPObject)


def _device_run_spu(
    dev_info: Device, op: FeOperation, *args: Any, **kwargs: Any
) -> Any:
    if not isinstance(op, JaxCompiler):
        raise ValueError("SPU device only supports JAX frontend.")
    fn, *aargs = args
    var = smpc.srun(fn)(*aargs, **kwargs)
    return tree_map(partial(_set_devid, dev_id=dev_info.name), var)


def _device_run_tee(
    dev_info: Device, op: FeOperation, *args: Any, **kwargs: Any
) -> Any:
    if not isinstance(op, JaxCompiler) and not isinstance(op, IbisCompiler):
        raise ValueError("TEE device only supports JAX and Ibis frontend.")
    assert len(dev_info.members) == 1
    rank = dev_info.members[0].rank
    var = simp.runAt(rank, op)(*args, **kwargs)
    return tree_map(partial(_set_devid, dev_id=dev_info.name), var)


def _device_run_ppu(
    dev_info: Device, op: FeOperation, *args: Any, **kwargs: Any
) -> Any:
    assert len(dev_info.members) == 1
    rank = dev_info.members[0].rank
    var = simp.runAt(rank, op)(*args, **kwargs)
    return tree_map(partial(_set_devid, dev_id=dev_info.name), var)


def _device_run(dev_id: str, op: FeOperation, *args: Any, **kwargs: Any) -> Any:
    assert isinstance(op, FeOperation)
    cluster_spec = mapi.cur_ctx().cluster_spec
    if dev_id not in cluster_spec.devices:
        raise ValueError(f"Device {dev_id} not found in cluster spec.")

    if g_auto_trans:

        def trans(obj: Any) -> Any:
            if _is_mpobj(obj):
                assert _is_device_obj(obj)
                return _d2d(dev_id, obj)
            else:
                return obj

        args, kwargs = tree_map(trans, (args, kwargs))

    dev_info = cluster_spec.devices[dev_id]
    if dev_info.kind.upper() == "SPU":
        return _device_run_spu(dev_info, op, *args, **kwargs)
    elif dev_info.kind.upper() == "TEE":
        return _device_run_tee(dev_info, op, *args, **kwargs)
    elif dev_info.kind.upper() == "PPU":
        return _device_run_ppu(dev_info, op, *args, **kwargs)
    else:
        raise ValueError(f"Unknown device type: {dev_info.kind}")


def device(dev_id: str, *, fe_type: str = "jax") -> Callable:
    """Decorator to mark a function to be executed on a specific device.

    Args:
        dev_id: The device id.
        fe_type: The frontend type of the device, could be "jax" or "ibis".

    Note: 'fe_type' is not needed if the decorated function is already a FeOperation.

    Example:
        >>> @device("P0")
        ... def foo(x, y):
        ...     return x + y
    """

    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if isinstance(fn, FeOperation):
                return _device_run(dev_id, fn, *args, **kwargs)
            else:
                if fe_type == "jax":
                    return _device_run(dev_id, jax_cc.jax_compile, fn, *args, **kwargs)
                elif fe_type == "ibis":
                    return _device_run(
                        dev_id, ibis_cc.ibis_compile, fn, *args, **kwargs
                    )
                else:
                    raise ValueError(f"Unsupported frontend type: {fe_type}")

        return wrapped

    return deco


def _d2d(to_dev_id: str, obj: MPObject) -> MPObject:
    assert isinstance(obj, MPObject)
    frm_dev_id = _get_devid(obj)

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
        return tree_map(partial(_set_devid, dev_id=to_dev_id), var)  # type: ignore[no-any-return]
    elif frm_to_pair == ("PPU", "SPU"):
        assert len(frm_dev.members) == 1
        frm_rank = frm_dev.members[0].rank
        var = smpc.sealFrom(obj, frm_rank)
        return tree_map(partial(_set_devid, dev_id=to_dev_id), var)  # type: ignore[no-any-return]
    elif frm_to_pair == ("PPU", "PPU"):
        assert len(frm_dev.members) == 1 and len(to_dev.members) == 1
        frm_rank = frm_dev.members[0].rank
        to_rank = to_dev.members[0].rank
        var = mpi.p2p(frm_rank, to_rank, obj)
        return tree_map(partial(_set_devid, dev_id=to_dev_id), var)  # type: ignore[no-any-return]
    elif frm_to_pair == ("PPU", "TEE"):
        # Transparent handshake + encryption for the first transfer; reuse thereafter
        assert len(frm_dev.members) == 1 and len(to_dev.members) == 1
        frm_rank = frm_dev.members[0].rank
        tee_rank = to_dev.members[0].rank
        # Ensure sessions (both directions) exist for this PPU<->TEE pair
        sess_p, sess_t = _ensure_tee_session(frm_dev_id, to_dev_id, frm_rank, tee_rank)
        # Bytes-only path: pack -> enc -> p2p -> dec -> unpack (with static out type)
        obj_ty = TensorType.from_obj(obj)
        b = simp.runAt(frm_rank, builtin.pack)(obj)
        ct = simp.runAt(frm_rank, crypto.enc)(b, sess_p)
        ct_at_tee = mpi.p2p(frm_rank, tee_rank, ct)
        b_at_tee = simp.runAt(tee_rank, crypto.dec)(ct_at_tee, sess_t)
        pt_at_tee = simp.runAt(tee_rank, builtin.unpack)(b_at_tee, out_ty=obj_ty)
        return tree_map(partial(_set_devid, dev_id=to_dev_id), pt_at_tee)  # type: ignore[no-any-return]
    elif frm_to_pair == ("TEE", "PPU"):
        # Transparent encryption from TEE to a specific PPU using the reverse-direction session key
        assert len(frm_dev.members) == 1 and len(to_dev.members) == 1
        tee_rank = frm_dev.members[0].rank
        ppu_rank = to_dev.members[0].rank
        # Ensure bidirectional session established for this pair
        sess_p, sess_t = _ensure_tee_session(to_dev_id, frm_dev_id, ppu_rank, tee_rank)
        obj_ty = TensorType.from_obj(obj)
        b = simp.runAt(tee_rank, builtin.pack)(obj)
        ct = simp.runAt(tee_rank, crypto.enc)(b, sess_t)
        ct_at_ppu = mpi.p2p(tee_rank, ppu_rank, ct)
        b_at_ppu = simp.runAt(ppu_rank, crypto.dec)(ct_at_ppu, sess_p)
        pt_at_ppu = simp.runAt(ppu_rank, builtin.unpack)(b_at_ppu, out_ty=obj_ty)
        return tree_map(partial(_set_devid, dev_id=to_dev_id), pt_at_ppu)  # type: ignore[no-any-return]
    else:
        supported = [
            ("SPU", "PPU"),
            ("PPU", "SPU"),
            ("PPU", "PPU"),
            ("PPU", "TEE"),
            ("TEE", "PPU"),
        ]
        raise ValueError(
            f"Unsupported device transfer: {frm_to_pair}. Supported pairs: {supported}."
        )


def _ensure_tee_session(
    frm_dev_id: str, to_dev_id: str, frm_rank: int, tee_rank: int
) -> tuple[MPObject, MPObject]:
    """Ensure a TEE session (sess_p at sender, sess_t at TEE) exists.

    Returns (sess_p, sess_t).
    """
    ctx = cur_ctx().root()
    if not hasattr(ctx, "_tee_sessions"):
        ctx._tee_sessions = {}  # type: ignore[attr-defined]
    cache: dict[tuple[str, str], tuple[MPObject, MPObject]] = ctx._tee_sessions  # type: ignore

    key = (frm_dev_id, to_dev_id)
    if key in cache:
        return cache[key]

    # 1) TEE generates (sk, pk) and quote(pk)
    # KEM suite currently constant; future: read from tee device config (e.g. cluster_spec.devices[dev_id].config)
    tee_sk, tee_pk = simp.runAt(tee_rank, crypto.kem_keygen)(_TEE_KEM_SUITE)
    quote = simp.runAt(tee_rank, tee.quote)(tee_pk)

    # 2) Send quote to sender and attest to obtain TEE pk
    quote_at_sender = mpi.p2p(tee_rank, frm_rank, quote)
    tee_pk_at_sender = simp.runAt(frm_rank, tee.attest)(quote_at_sender)

    # 3) Sender generates its ephemeral keypair and sends its pk to TEE
    v_sk, v_pk = simp.runAt(frm_rank, crypto.kem_keygen)(_TEE_KEM_SUITE)
    v_pk_at_tee = mpi.p2p(frm_rank, tee_rank, v_pk)

    # 4) Both sides derive the shared secret and session key
    shared_p = simp.runAt(frm_rank, crypto.kem_derive)(
        v_sk, tee_pk_at_sender, _TEE_KEM_SUITE
    )
    shared_t = simp.runAt(tee_rank, crypto.kem_derive)(
        tee_sk, v_pk_at_tee, _TEE_KEM_SUITE
    )
    # Use a fixed ASCII string literal for HKDF info on both sides
    sess_p = simp.runAt(frm_rank, crypto.hkdf)(shared_p, _HKDF_INFO_LITERAL)
    sess_t = simp.runAt(tee_rank, crypto.hkdf)(shared_t, _HKDF_INFO_LITERAL)

    cache[key] = (sess_p, sess_t)
    return sess_p, sess_t


def put(to_dev_id: str, obj: Any) -> MPObject:
    if not isinstance(obj, MPObject):
        return device(to_dev_id)(lambda x: x)(obj)  # type: ignore[no-any-return]
    assert isinstance(obj, MPObject)
    return _d2d(to_dev_id, obj)


def _fetch(interp: InterpContext, obj: MPObject) -> Any:
    dev_id = _get_devid(obj)
    cluster_spec = interp.cluster_spec
    dev_kind = cluster_spec.devices[dev_id].kind.upper()

    dev_info = cluster_spec.devices[dev_id]
    if dev_kind == "SPU":
        revealed = mapi.evaluate(interp, smpc.reveal, obj)
        result = mapi.fetch(interp, revealed)
        # now all members have the same value, return the one at rank 0
        return result[dev_info.members[0].rank]
    elif dev_kind == "PPU":
        assert len(dev_info.members) == 1
        rank = dev_info.members[0].rank
        result = mapi.fetch(interp, obj)
        return result[rank]
    elif dev_kind == "TEE":
        assert len(dev_info.members) == 1
        rank = dev_info.members[0].rank
        result = mapi.fetch(interp, obj)
        return result[rank]
    else:
        raise ValueError(f"Unknown device id: {dev_id}")


def fetch(interp: InterpContext, objs: Any) -> Any:
    ctx = interp or mapi.cur_ctx()
    assert isinstance(ctx, InterpContext), f"Expect InterpContext, got {ctx}"
    return tree_map(partial(_fetch, ctx), objs)
