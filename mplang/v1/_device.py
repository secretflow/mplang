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
from typing import Any, cast

from jax.tree_util import tree_map, tree_unflatten

from mplang.v1.core import (
    ClusterSpec,
    Device,
    Mask,
    MPContext,
    MPObject,
    TableLike,
    TensorLike,
    cur_ctx,
    peval,
)
from mplang.v1.ops import basic, crypto, jax_cc, nnx_cc, spu, tee
from mplang.v1.ops.base import FeOperation
from mplang.v1.ops.jax_cc import JaxRunner
from mplang.v1.simp import mpi
from mplang.v1.simp.api import run_at

# Automatic transfer between devices when parameter is not on the target device.
g_auto_trans: bool = True

_HKDF_INFO_LITERAL: str = "mplang/device/tee/v1"
# Default KEM suite for TEE session establishment; make configurable via ClusterSpec in future.
_TEE_KEM_SUITE: str = "x25519"


# Context-aware session management
def _get_context_id(ctx: MPContext) -> int:
    """
    Get unique identifier for a context.

    Args:
        ctx: The context object (TraceContext or InterpContext)

    Returns:
        Unique integer ID for this context instance
    """
    return id(ctx)


# magic attribute name to mark a MPObject as a device object
DEVICE_ATTR_NAME = "__device__"


def is_device_obj(obj: Any) -> bool:
    if not isinstance(obj, MPObject):
        return False
    return DEVICE_ATTR_NAME in obj.attrs


def set_dev_attr(obj: MPObject, dev_id: str) -> MPObject:
    if not isinstance(obj, MPObject):
        raise TypeError(f"Input must be an instance of MPObject, {obj}")
    obj.attrs[DEVICE_ATTR_NAME] = dev_id
    return obj


def get_dev_attr(obj: MPObject) -> str:
    if not isinstance(obj, MPObject):
        raise TypeError("Input must be an instance of MPObject")

    return str(obj.attrs[DEVICE_ATTR_NAME])


def _infer_device_from_args(*args: Any, **kwargs: Any) -> str:
    """Infer target device from function arguments.

    Inference strategy:
    1. Collect all MPObject arguments and check device attributes
       - If MPObject has no device attr -> error (user must set_devid)
       - If no MPObject arguments -> error (explicit device required)

    2. Analyze device distribution
       2.1 All objects on same device -> use that device
       2.2 Multiple devices with g_auto_trans enabled:
           - Single SPU (+ PPUs) -> use SPU (auto-transfer from PPUs)
           - Single TEE (+ PPUs) -> use TEE (auto-transfer from PPUs)
           - Multiple PPUs only -> error (ambiguous, need explicit device)
           - Multiple SPUs -> error (ambiguous)
           - Multiple TEEs -> error (ambiguous)
           - SPU + TEE -> error (conflicting secure devices)
       2.3 Multiple devices with g_auto_trans disabled -> error

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Device id string

    Raises:
        ValueError: When inference fails or is ambiguous
    """
    from jax.tree_util import tree_flatten

    # Step 1: Collect all MPObject arguments and validate device attributes
    all_args = tree_flatten((args, kwargs))[0]
    device_objs = []

    for obj in all_args:
        if isinstance(obj, MPObject):
            if not is_device_obj(obj):
                raise ValueError(
                    "MPObject is missing device attribute. "
                    "If you're mixing device-level and simp-level code, "
                    "use set_dev_attr(obj, 'device_id') to mark the device explicitly."
                )
            device_objs.append(obj)

    if not device_objs:
        raise ValueError(
            "Cannot infer device: no MPObject arguments found. "
            "Please specify device explicitly using device('device_id')(fn)."
        )

    # Step 2: Extract all unique devices
    devices = {get_dev_attr(obj) for obj in device_objs}

    if len(devices) == 1:
        return devices.pop()  # All arguments on same device

    # Step 3: Multiple devices - check if auto-transfer is enabled
    if not g_auto_trans:
        raise ValueError(
            f"Cannot infer device: arguments from multiple devices {devices} "
            f"but auto-transfer is disabled (g_auto_trans=False). "
            f"Please enable auto-transfer or put all data on same device first."
        )

    # Step 4: Analyze device kinds for auto-transfer scenario
    cluster_spec = cur_ctx().cluster_spec
    device_kinds = {
        dev_id: cluster_spec.devices[dev_id].kind.upper() for dev_id in devices
    }

    # Count devices by type
    spu_devs = [d for d, k in device_kinds.items() if k == "SPU"]
    tee_devs = [d for d, k in device_kinds.items() if k == "TEE"]
    ppu_devs = [d for d, k in device_kinds.items() if k == "PPU"]

    # Decision logic
    # Case 1: Only PPUs -> ambiguous
    if not spu_devs and not tee_devs:
        raise ValueError(
            f"Cannot infer device: arguments from multiple PPU devices {ppu_devs}. "
            f"Please specify device explicitly or use put() to consolidate data."
        )

    # Case 2: Single SPU (possibly with PPUs) -> use SPU
    if len(spu_devs) == 1 and len(tee_devs) == 0:
        return spu_devs[0]

    # Case 3: Single TEE (possibly with PPUs) -> use TEE
    if len(tee_devs) == 1 and len(spu_devs) == 0:
        return tee_devs[0]

    # Case 4: Multiple SPUs -> ambiguous
    if len(spu_devs) > 1:
        raise ValueError(
            f"Ambiguous device inference: arguments from multiple SPU devices {spu_devs}. "
            f"Please specify which SPU to use explicitly."
        )

    # Case 5: Multiple TEEs -> ambiguous
    if len(tee_devs) > 1:
        raise ValueError(
            f"Ambiguous device inference: arguments from multiple TEE devices {tee_devs}. "
            f"Please specify which TEE to use explicitly."
        )

    # Case 6: Both SPU and TEE -> conflicting
    if spu_devs and tee_devs:
        raise ValueError(
            f"Ambiguous device inference: arguments from both SPU {spu_devs} and TEE {tee_devs}. "
            f"Please specify which secure device to use explicitly."
        )

    # Should never reach here
    raise ValueError(f"Unexpected device configuration: {devices}")


def _device_run_spu(
    dev_info: Device, op: FeOperation, fn: Callable, *args: Any, **kwargs: Any
) -> Any:
    if not isinstance(op, JaxRunner):
        raise ValueError("SPU device only supports JAX frontend.")
    spu_mask = Mask.from_ranks([member.rank for member in dev_info.members])
    pfunc, in_vars, out_tree = spu.jax_compile(fn, *args, **kwargs)
    assert all(var.pmask == spu_mask for var in in_vars), in_vars
    out_flat = peval(pfunc, in_vars, spu_mask)
    result = tree_unflatten(out_tree, out_flat)
    return tree_map(partial(set_dev_attr, dev_id=dev_info.name), result)


def _device_run_tee(
    dev_info: Device, op: FeOperation, *args: Any, **kwargs: Any
) -> Any:
    # TODO(jint): should we filter out all IO operations?
    assert len(dev_info.members) == 1
    rank = dev_info.members[0].rank
    var = run_at(rank, op, *args, **kwargs)
    return tree_map(partial(set_dev_attr, dev_id=dev_info.name), var)


def _device_run_ppu(
    dev_info: Device, op: FeOperation, *args: Any, **kwargs: Any
) -> Any:
    assert len(dev_info.members) == 1
    rank = dev_info.members[0].rank
    var = run_at(rank, op, *args, **kwargs)
    return tree_map(partial(set_dev_attr, dev_id=dev_info.name), var)


def _device_run(dev_id: str, op: FeOperation, *args: Any, **kwargs: Any) -> Any:
    assert isinstance(op, FeOperation)
    cluster_spec = cur_ctx().cluster_spec
    if dev_id not in cluster_spec.devices:
        raise ValueError(f"Device {dev_id} not found in cluster spec.")
    dev_info = cluster_spec.devices[dev_id]

    if g_auto_trans:

        def trans(obj: Any) -> Any:
            if isinstance(obj, MPObject):
                assert is_device_obj(obj)
                return _d2d(dev_id, obj)
            else:
                return obj

        args, kwargs = tree_map(trans, (args, kwargs))

    if dev_info.kind.upper() == "SPU":
        return _device_run_spu(dev_info, op, *args, **kwargs)
    elif dev_info.kind.upper() == "TEE":
        return _device_run_tee(dev_info, op, *args, **kwargs)
    elif dev_info.kind.upper() == "PPU":
        return _device_run_ppu(dev_info, op, *args, **kwargs)
    else:
        raise ValueError(f"Unknown device type: {dev_info.kind}")


def device(
    dev_or_fn: str | Callable | None = None, *, fe_type: str = "jax"
) -> Callable:
    """Decorator to mark a function to be executed on a specific device.

    Supports both explicit device specification and automatic device inference:

    1. Explicit device placement:
       @device("P0")
       def foo(x, y): return x + y

    2. Auto device inference:
       @device
       def foo(x, y): return x + y
       # Device is inferred from x, y at runtime

    3. Inline usage:
       result = device(lambda x, y: x + y)(x_on_p0, y_on_p0)
       # Automatically infers device from arguments

    Args:
        dev_or_fn: Either a device id string ("P0", "SPU", etc.) for explicit placement,
                   a callable function for auto inference, or None (same as not providing arg).
        fe_type: The frontend type of the device, could be "jax" or "nnx".
                 Not needed if the decorated function is already a FeOperation.

    Returns:
        A decorator (when dev_or_fn is a string or None) or decorated function (when callable).

    Raises:
        TypeError: When dev_or_fn is not a string, callable, or None.
        ValueError: When device cannot be inferred or inference is ambiguous.

    Device Inference Strategy:
        - Same device: All arguments on device D -> execute on D
        - PPU + SPU: Arguments from PPU and SPU -> execute on SPU (secure compute)
        - PPU + TEE: Arguments from PPU and TEE -> execute on TEE (trusted execution)
        - Multiple PPUs: Ambiguous -> error (explicit device required)
        - No device objects: Cannot infer -> error (explicit device required)

    Example:
        >>> # Explicit device
        >>> @device("P0")
        ... def add_explicit(x, y):
        ...     return x + y
        >>>
        >>> # Auto inference
        >>> @device
        ... def add_auto(x, y):
        ...     return x + y
        >>>
        >>> x_on_p0 = ...  # data on P0
        >>> y_on_p0 = ...  # data on P0
        >>> result = add_auto(x_on_p0, y_on_p0)  # Inferred to P0
        >>>
        >>> x_on_spu = ...  # data on SPU
        >>> y_on_p1 = ...  # data on P1
        >>> result = add_auto(x_on_spu, y_on_p1)  # Inferred to SPU
    """

    def _execute_on_device(dev_id: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Helper to execute function on specified device with appropriate frontend."""
        if isinstance(fn, FeOperation):
            return _device_run(dev_id, fn, *args, **kwargs)
        else:
            if fe_type == "jax":
                return _device_run(dev_id, jax_cc.run_jax, fn, *args, **kwargs)
            elif fe_type == "nnx":
                return _device_run(dev_id, nnx_cc.run_nnx, fn, *args, **kwargs)
            else:
                raise ValueError(f"Unsupported frontend type: {fe_type}")

    # Case 1: device("P0") - Explicit device specification
    if isinstance(dev_or_fn, str):
        dev_id = dev_or_fn

        def deco(fn: Callable) -> Callable:
            @wraps(fn)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                return _execute_on_device(dev_id, fn, *args, **kwargs)

            return wrapped

        return deco

    # Case 2: device(fn) or @device - Auto device inference
    elif callable(dev_or_fn):
        fn = dev_or_fn

        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                dev_id = _infer_device_from_args(*args, **kwargs)
            except ValueError as e:
                # Enhance error message with function context
                raise ValueError(
                    f"Cannot infer device for function '{fn.__name__}': {e!s}"
                ) from e

            return _execute_on_device(dev_id, fn, *args, **kwargs)

        return wrapped

    # Case 3: device() or @device() - Return auto-inference decorator
    elif dev_or_fn is None:

        def deco(fn: Callable) -> Callable:
            return device(fn, fe_type=fe_type)

        return deco

    else:
        # More helpful error message for common mistakes
        raise TypeError(
            f"device() expects a device id (string), a function (callable), or nothing. "
            f"Got: {type(dev_or_fn).__name__}.\n"
            f"Usage:\n"
            f"  - Explicit device: @device('P0') or device('P0')(fn)\n"
            f"  - Auto inference: @device or device(fn)"
        )


def _spu_reveal(spu_dev: Device, obj: MPObject, to_mask: Mask) -> MPObject:
    spu_mask = Mask.from_ranks([m.rank for m in spu_dev.members])
    assert obj.pmask == spu_mask, (obj.pmask, spu_mask)

    # (n_parties, n_shares)
    shares = [mpi.bcast_m(to_mask, rank, obj) for rank in Mask(spu_mask)]
    assert len(shares) == Mask(spu_mask).num_parties(), (shares, spu_mask)
    assert all(share.pmask == to_mask for share in shares)

    # Reconstruct the original object from shares
    pfunc, ins, _ = spu.reconstruct(*shares)
    return peval(pfunc, ins, to_mask)[0]  # type: ignore[no-any-return]


def _spu_seal(spu_dev: Device, obj: MPObject) -> list[MPObject]:
    """Seal plaintext into SPU shares on a specific SPU device.

    Low-level API: device id is mandatory to avoid ambiguity.
    """
    if obj.pmask is None:
        raise ValueError("Seal can not apply to dynamic mask objects.")

    spu_mask = Mask.from_ranks([member.rank for member in spu_dev.members])
    spu_wsize = Mask(spu_mask).num_parties()
    pfunc, ins, _ = spu.makeshares(
        obj, world_size=spu_wsize, visibility=spu.Visibility.SECRET
    )
    assert len(ins) == 1
    shares = peval(pfunc, ins)

    # scatter the shares to each party.
    outs = [mpi.scatter_m(spu_mask, rank, shares) for rank in obj.pmask]
    return outs


def _d2d(to_dev_id: str, obj: MPObject) -> MPObject:
    assert isinstance(obj, MPObject)
    frm_dev_id = get_dev_attr(obj)

    if frm_dev_id == to_dev_id:
        return obj

    cluster_spec: ClusterSpec = cur_ctx().cluster_spec
    frm_dev = cluster_spec.devices[frm_dev_id]
    to_dev = cluster_spec.devices[to_dev_id]
    frm_to_pair = (frm_dev.kind.upper(), to_dev.kind.upper())

    if frm_to_pair == ("SPU", "SPU"):
        raise NotImplementedError("Only one SPU is supported for now.")
    elif frm_to_pair == ("SPU", "PPU"):
        assert len(to_dev.members) == 1
        to_rank = to_dev.members[0].rank
        var = _spu_reveal(frm_dev, obj, Mask.from_ranks([to_rank]))
        return tree_map(partial(set_dev_attr, dev_id=to_dev_id), var)  # type: ignore[no-any-return]
    elif frm_to_pair == ("PPU", "SPU"):
        assert len(frm_dev.members) == 1
        frm_rank = frm_dev.members[0].rank
        vars = _spu_seal(to_dev, obj)
        assert len(vars) == 1, "Expected single share from PPU to SPU seal."
        return tree_map(partial(set_dev_attr, dev_id=to_dev_id), vars[0])  # type: ignore[no-any-return]
    elif frm_to_pair == ("PPU", "PPU"):
        assert len(frm_dev.members) == 1 and len(to_dev.members) == 1
        frm_rank = frm_dev.members[0].rank
        to_rank = to_dev.members[0].rank
        var = mpi.p2p(frm_rank, to_rank, obj)
        return tree_map(partial(set_dev_attr, dev_id=to_dev_id), var)  # type: ignore[no-any-return]
    elif frm_to_pair == ("PPU", "TEE"):
        # Transparent handshake + encryption for the first transfer; reuse thereafter
        assert len(frm_dev.members) == 1 and len(to_dev.members) == 1
        frm_rank = frm_dev.members[0].rank
        tee_rank = to_dev.members[0].rank
        # Ensure sessions (both directions) exist for this PPU<->TEE pair
        sess_p, sess_t = _ensure_tee_session(frm_dev_id, to_dev_id, frm_rank, tee_rank)
        # Bytes-only path: pack -> enc -> p2p -> dec -> unpack (with static out type)
        obj_ty = obj.mptype.raw_type()
        b = run_at(frm_rank, basic.pack, obj)
        ct = run_at(frm_rank, crypto.enc, b, sess_p)
        ct_at_tee = mpi.p2p(frm_rank, tee_rank, ct)
        b_at_tee = run_at(tee_rank, crypto.dec, ct_at_tee, sess_t)
        pt_at_tee = run_at(tee_rank, basic.unpack, b_at_tee, out_ty=obj_ty)
        return tree_map(partial(set_dev_attr, dev_id=to_dev_id), pt_at_tee)  # type: ignore[no-any-return]
    elif frm_to_pair == ("TEE", "PPU"):
        # Transparent encryption from TEE to a specific PPU using the reverse-direction session key
        assert len(frm_dev.members) == 1 and len(to_dev.members) == 1
        tee_rank = frm_dev.members[0].rank
        ppu_rank = to_dev.members[0].rank
        # Ensure bidirectional session established for this pair
        sess_p, sess_t = _ensure_tee_session(to_dev_id, frm_dev_id, ppu_rank, tee_rank)
        obj_ty = obj.mptype.raw_type()
        b = run_at(tee_rank, basic.pack, obj)
        ct = run_at(tee_rank, crypto.enc, b, sess_t)
        ct_at_ppu = mpi.p2p(tee_rank, ppu_rank, ct)
        b_at_ppu = run_at(ppu_rank, crypto.dec, ct_at_ppu, sess_p)
        pt_at_ppu = run_at(ppu_rank, basic.unpack, b_at_ppu, out_ty=obj_ty)
        return tree_map(partial(set_dev_attr, dev_id=to_dev_id), pt_at_ppu)  # type: ignore[no-any-return]
    elif frm_to_pair == ("TEE", "SPU"):
        assert len(frm_dev.members) == 1
        frm_rank = frm_dev.members[0].rank
        vars = _spu_seal(to_dev, obj)
        assert len(vars) == 1, "Expected single share from TEE to SPU seal."
        return tree_map(partial(set_dev_attr, dev_id=to_dev_id), vars[0])  # type: ignore[no-any-return]
    elif frm_to_pair == ("SPU", "TEE"):
        assert len(to_dev.members) == 1
        to_rank = to_dev.members[0].rank
        var = _spu_reveal(frm_dev, obj, Mask.from_ranks([to_rank]))
        return tree_map(partial(set_dev_attr, dev_id=to_dev_id), var)  # type: ignore[no-any-return]
    else:
        supported = [
            ("SPU", "PPU"),
            ("PPU", "SPU"),
            ("PPU", "PPU"),
            ("PPU", "TEE"),
            ("TEE", "PPU"),
            ("TEE", "SPU"),
            ("SPU", "TEE"),
        ]
        raise ValueError(
            f"Unsupported device transfer: {frm_to_pair}. Supported pairs: {supported}."
        )


def _ensure_tee_session(
    frm_dev_id: str, to_dev_id: str, frm_rank: int, tee_rank: int
) -> tuple[MPObject, MPObject]:
    """Ensure a TEE session (sess_p at sender, sess_t at TEE) exists.

    Context-aware version: caches include context ID to ensure isolation
    between different TraceContext instances, preventing TraceVar pollution.

    Returns (sess_p, sess_t).
    """
    # Get current context and its unique ID
    current_ctx = cur_ctx()
    current_context_id = _get_context_id(current_ctx)

    # Get root context for cache storage
    root_ctx = current_ctx.root()
    if not hasattr(root_ctx, "_tee_sessions"):
        root_ctx._tee_sessions = {}  # type: ignore[attr-defined]
    cache: dict[tuple[str, str], tuple[int, MPObject, MPObject]] = (
        root_ctx._tee_sessions  # type: ignore[attr-defined]
    )

    key = (frm_dev_id, to_dev_id)

    # Check cache with context awareness
    if key in cache:
        cached_context_id, sess_p, sess_t = cache[key]

        # Only reuse cache from the same context
        if cached_context_id == current_context_id:
            return sess_p, sess_t
        else:
            # Different context, cannot reuse cache, clean up old entry
            del cache[key]

    # 1) TEE generates (sk, pk) and quote(pk)
    # KEM suite currently constant; future: read from tee device config (e.g. cluster_spec.devices[dev_id].config)
    tee_sk, tee_pk = run_at(tee_rank, crypto.kem_keygen, _TEE_KEM_SUITE)
    quote = run_at(tee_rank, tee.quote_gen, tee_pk)

    # 2) Send quote to sender and attest to obtain TEE pk
    quote_at_sender = mpi.p2p(tee_rank, frm_rank, quote)
    tee_pk_at_sender = run_at(frm_rank, tee.attest, quote_at_sender)

    # 3) Sender generates its ephemeral keypair and sends its pk to TEE
    v_sk, v_pk = run_at(frm_rank, crypto.kem_keygen, _TEE_KEM_SUITE)
    v_pk_at_tee = mpi.p2p(frm_rank, tee_rank, v_pk)

    # 4) Both sides derive the shared secret and session key
    shared_p = run_at(
        frm_rank, crypto.kem_derive, v_sk, tee_pk_at_sender, _TEE_KEM_SUITE
    )
    shared_t = run_at(tee_rank, crypto.kem_derive, tee_sk, v_pk_at_tee, _TEE_KEM_SUITE)
    # Use a fixed ASCII string literal for HKDF info on both sides
    sess_p = run_at(frm_rank, crypto.hkdf, shared_p, _HKDF_INFO_LITERAL)
    sess_t = run_at(tee_rank, crypto.hkdf, shared_t, _HKDF_INFO_LITERAL)

    # Cache with context ID for isolation
    cache[key] = (current_context_id, sess_p, sess_t)
    return sess_p, sess_t


def _host_to_device(to_dev_id: str, obj: Any) -> MPObject:
    if isinstance(obj, TensorLike):
        # run jax identity on the target device to put the tensor there
        return device(to_dev_id)(lambda x: x)(obj)  # type: ignore[no-any-return]
    elif isinstance(obj, TableLike):
        dev_info = cur_ctx().cluster_spec.devices[to_dev_id]
        if dev_info.kind.upper() not in ["PPU", "TEE"]:
            raise ValueError(
                f"TableLike put() only supports PPU or TEE devices, got {dev_info.kind}"
            )
        assert len(dev_info.members) == 1
        rank = dev_info.members[0].rank
        obj_mp = cast(MPObject, run_at(rank, basic.constant, obj))
        set_dev_attr(obj_mp, to_dev_id)
        return obj_mp
    else:
        raise TypeError(
            f"put() only supports TensorLike or TableLike objects, got {type(obj)}"
        )


def put(to_dev_id: str, obj: Any) -> MPObject:
    if not isinstance(obj, MPObject):
        return _host_to_device(to_dev_id, obj)
    assert isinstance(obj, MPObject)
    return _d2d(to_dev_id, obj)
