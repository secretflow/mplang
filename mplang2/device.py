"""
Device-oriented programming interface for MPLang2.

This module provides high-level abstractions for device placement and data movement.
It allows users to write programs in a device-centric way, handling data transfers
and execution dispatch automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial, wraps
from typing import Any

from jax.tree_util import tree_flatten, tree_map

from mplang2.backends import load_builtins
from mplang2.cluster import Device, get_global_cluster
from mplang2.dialects import simp, spu
from mplang2.edsl.object import Object

# Load built-in backends (SPU, Tensor, etc.)
load_builtins()

# Magic attribute name to mark an Object as a device object
DEVICE_ATTR_NAME = "__device__"

# Automatic transfer between devices when parameter is not on the target device.
g_auto_trans: bool = True


def is_device_obj(obj: Any) -> bool:
    """Check if an object is a device object (has device attribute)."""
    if not isinstance(obj, Object):
        return False
    return hasattr(obj, DEVICE_ATTR_NAME)


def set_dev_attr(obj: Object, dev_id: str) -> Object:
    """Mark an object as residing on a specific device."""
    if not isinstance(obj, Object):
        raise TypeError(f"Input must be an instance of Object, got {type(obj)}")
    setattr(obj, DEVICE_ATTR_NAME, dev_id)
    return obj


def get_dev_attr(obj: Object) -> str:
    """Get the device ID of an object."""
    if not isinstance(obj, Object):
        raise TypeError("Input must be an instance of Object")
    if not hasattr(obj, DEVICE_ATTR_NAME):
        raise ValueError("Object does not have a device attribute")
    return str(getattr(obj, DEVICE_ATTR_NAME))


def _infer_device_from_args(*args: Any, **kwargs: Any) -> str:
    """Infer target device from function arguments."""
    all_args = tree_flatten((args, kwargs))[0]
    device_objs = []

    for obj in all_args:
        if isinstance(obj, Object):
            if not is_device_obj(obj):
                # Skip non-device objects (they might be purely local/host values)
                continue
            device_objs.append(obj)

    if not device_objs:
        raise ValueError(
            "Cannot infer device: no device-bound Object arguments found. "
            "Please specify device explicitly using @device('device_id')."
        )

    devices = {get_dev_attr(obj) for obj in device_objs}

    if len(devices) == 1:
        return devices.pop()  # All arguments on same device

    if not g_auto_trans:
        raise ValueError(
            f"Cannot infer device: arguments from multiple devices {devices} "
            f"but auto-transfer is disabled (g_auto_trans=False). "
            f"Please enable auto-transfer or put all data on same device first."
        )

    cluster = get_global_cluster()
    device_kinds = {dev_id: cluster.devices[dev_id].kind.upper() for dev_id in devices}

    # Count devices by type
    spu_devs = [d for d, k in device_kinds.items() if k == "SPU"]
    tee_devs = [d for d, k in device_kinds.items() if k == "TEE"]
    ppu_devs = [d for d, k in device_kinds.items() if k == "PPU"]

    # Decision logic
    # Case 1: Only PPUs -> ambiguous (unless we want to pick one arbitrarily, but safer to error)
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


def _device_run_spu(dev_info: Device, fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run function on SPU device."""
    spu_parties = tuple(m.rank for m in dev_info.members)

    # SPU execution uses spu.run_jax to compile and execute the function on the SPU.
    # Inputs are expected to be already on the SPU (handled by _d2d).
    # We wrap spu.run_jax in simp.pcall_static to execute it on all SPU parties.
    result = simp.pcall_static(
        spu_parties, lambda *a, **k: spu.run_jax(fn, *a, **k), *args, **kwargs
    )

    return tree_map(partial(set_dev_attr, dev_id=dev_info.name), result)


def _device_run_ppu(dev_info: Device, fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run function on PPU device."""
    assert len(dev_info.members) == 1
    rank = dev_info.members[0].rank

    # Note on Single Controller (God View) Programming Model:
    # MPLang2 adopts a strict separation between Control Plane (Host) and Data Plane (Device).
    # - @device functions represent pure computation tasks running on a specific device.
    # - Data movement (put, transfer) and orchestration happen exclusively on the Host.
    #
    # Therefore, we do NOT need to wrap 'fn' to restore device attributes (metadata)
    # because 'fn' should not contain any control logic (like nested @device calls or put())
    # that relies on this metadata. 'fn' sees only pure data (TraceObjects).

    result = simp.pcall_static((rank,), fn, *args, **kwargs)
    return tree_map(partial(set_dev_attr, dev_id=dev_info.name), result)


def _device_run(dev_id: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Execute function on the specified device."""
    cluster = get_global_cluster()
    if dev_id not in cluster.devices:
        raise ValueError(f"Device {dev_id} not found in cluster spec.")
    dev_info = cluster.devices[dev_id]

    if g_auto_trans:

        def trans(obj: Any) -> Any:
            if isinstance(obj, Object) and is_device_obj(obj):
                return _d2d(dev_id, obj)
            else:
                return obj

        args, kwargs = tree_map(trans, (args, kwargs))

    if dev_info.kind.upper() == "SPU":
        return _device_run_spu(dev_info, fn, *args, **kwargs)
    elif dev_info.kind.upper() == "TEE":
        raise NotImplementedError("TEE device support not yet implemented.")
    elif dev_info.kind.upper() == "PPU":
        return _device_run_ppu(dev_info, fn, *args, **kwargs)
    else:
        raise ValueError(f"Unknown device type: {dev_info.kind}")


def device(
    dev_or_fn: str | Callable | None = None,
) -> Callable:
    """Decorator to mark a function to be executed on a specific device."""

    def _execute_on_device(dev_id: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        return _device_run(dev_id, fn, *args, **kwargs)

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
                raise ValueError(
                    f"Cannot infer device for function '{fn.__name__}': {e!s}"
                ) from e
            return _execute_on_device(dev_id, fn, *args, **kwargs)

        return wrapped

    # Case 3: device() or @device() - Return auto-inference decorator
    elif dev_or_fn is None:

        def deco(fn: Callable) -> Callable:
            return device(fn)

        return deco

    else:
        raise TypeError(
            f"device() expects a device id (string), a function (callable), or nothing. "
            f"Got: {type(dev_or_fn).__name__}."
        )


def _d2d(to_dev_id: str, obj: Object) -> Object:
    """Transfer object to target device."""
    if not isinstance(obj, Object):
        raise TypeError(f"Expected Object, got {type(obj)}")

    frm_dev_id = get_dev_attr(obj)
    if frm_dev_id == to_dev_id:
        return obj

    cluster = get_global_cluster()
    frm_dev = cluster.devices[frm_dev_id]
    to_dev = cluster.devices[to_dev_id]
    frm_to_pair = (frm_dev.kind.upper(), to_dev.kind.upper())

    # PPU -> PPU
    if frm_to_pair == ("PPU", "PPU"):
        assert len(frm_dev.members) == 1 and len(to_dev.members) == 1
        to_rank = to_dev.members[0].rank
        frm_rank = frm_dev.members[0].rank

        var = simp.shuffle_static(obj, {to_rank: frm_rank})
        return set_dev_attr(var, to_dev_id)

    # PPU -> SPU (Seal)
    elif frm_to_pair == ("PPU", "SPU"):
        assert len(frm_dev.members) == 1
        frm_rank = frm_dev.members[0].rank
        spu_parties = tuple(m.rank for m in to_dev.members)

        # 1. Generate shares on source
        # We call spu.make_shares inside pcall on the source party
        shares_on_source = simp.pcall_static(
            (frm_rank,), lambda x: spu.make_shares(x, count=len(spu_parties)), obj
        )

        # 2. Distribute shares
        distributed_shares = []
        for i, target_rank in enumerate(spu_parties):
            # Extract i-th share (still on source)
            # shares_on_source is MP[tuple[SS, ...], (frm_rank)]
            # We need to extract the i-th element.
            # Since pcall returns MPType, we can't index it directly if it's a tuple of shares.
            # Wait, pcall returns a PyTree of MPObjects if the function returns a PyTree.
            # So shares_on_source IS a tuple of MPObjects.
            share_i = shares_on_source[i]

            share_at_target = simp.shuffle_static(share_i, {target_rank: frm_rank})
            distributed_shares.append(share_at_target)

        # 3. Converge
        var = simp.converge(*distributed_shares)
        return set_dev_attr(var, to_dev_id)

    # SPU -> PPU (Reveal)
    elif frm_to_pair == ("SPU", "PPU"):
        assert len(to_dev.members) == 1
        to_rank = to_dev.members[0].rank
        spu_parties = tuple(m.rank for m in frm_dev.members)

        # 1. Gather shares to target
        gathered_shares = []
        for source_rank in spu_parties:
            # Extract share from logical variable
            share_on_source = simp.pcall_static((source_rank,), lambda x: x, obj)

            # Move to target
            share_at_target = simp.shuffle_static(
                share_on_source, {to_rank: source_rank}
            )
            gathered_shares.append(share_at_target)

        # 2. Reconstruct on target
        # We call spu.reconstruct inside pcall on the target party
        var = simp.pcall_static(
            (to_rank,), lambda *s: spu.reconstruct(s), *gathered_shares
        )
        return set_dev_attr(var, to_dev_id)

    # SPU -> SPU
    elif frm_to_pair == ("SPU", "SPU"):
        raise NotImplementedError("SPU to SPU transfer not implemented yet.")

    else:
        raise ValueError(f"Unsupported device transfer: {frm_to_pair}")


def put(obj: Any, to_dev_id: str) -> Object:
    """Put data onto a device.

    If obj is already a device object, it moves it to the target device.
    If obj is a host object (e.g. numpy array), it uploads it to the target device.
    """
    cluster = get_global_cluster()
    if to_dev_id not in cluster.devices:
        raise ValueError(f"Device {to_dev_id} not found in cluster spec.")

    if isinstance(obj, Object) and is_device_obj(obj):
        return _d2d(to_dev_id, obj)

    # Host -> Device
    dev_info = cluster.devices[to_dev_id]

    if dev_info.kind.upper() == "PPU":
        assert len(dev_info.members) == 1
        rank = dev_info.members[0].rank

        var = simp.constant((rank,), obj)
        return set_dev_attr(var, to_dev_id)

    elif dev_info.kind.upper() == "SPU":
        # Host -> SPU: Run identity function on SPU.
        # Note: This results in a Public (replicated) value on the SPU.
        # SPU operations will automatically promote it to Secret if needed.
        return device(to_dev_id)(lambda x: x)(obj)

    else:
        raise ValueError(f"Cannot put to device kind {dev_info.kind}")
