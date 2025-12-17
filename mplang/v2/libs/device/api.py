# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Device-oriented programming interface for MPLang2.

This module provides high-level abstractions for device placement and data movement.
It allows users to write programs in a device-centric way, handling data transfers
and execution dispatch automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial, wraps
from typing import Any, cast

from jax.tree_util import tree_flatten, tree_map

from mplang.v2.backends import load_builtins
from mplang.v2.dialects import crypto, simp, spu, tee
from mplang.v2.edsl.object import Object
from mplang.v2.libs.device.cluster import Device

load_builtins()


def _resolve_cluster() -> Any:
    """Resolve the active ClusterSpec by traversing the context stack.

    Traverses from the top of the stack (most recent) to find the nearest
    Interpreter with a _cluster_spec attribute. This allows nested contexts
    to override the cluster if needed.
    """
    from mplang.v2.edsl.context import find_context

    ctx = find_context(lambda c: getattr(c, "_cluster_spec", None) is not None)
    if ctx is not None:
        return ctx._cluster_spec  # type: ignore[attr-defined]

    raise RuntimeError(
        "No active device context found. Please use 'with simulator:' "
        "or 'push_context(sim)' to set the execution environment."
    )


# Magic attribute name to mark an Object as a device object
DEVICE_ATTR_NAME = "__device__"

# Default KEM suite for TEE session establishment
_TEE_KEM_SUITE: str = "x25519"

# Global cache for TEE sessions (keyed by (frm_dev_id, to_dev_id))
# Each entry is (context_id, sess_frm, sess_tee) where context_id ensures
# sessions are not reused across different trace/interp contexts.
_tee_session_cache: dict[tuple[str, str], tuple[int, Object, Object]] = {}

# Automatic transfer between devices when parameter is not on the target device.
g_auto_trans: bool = True


class DeviceError(Exception):
    """Base exception for device-related errors."""


class DeviceNotFoundError(DeviceError):
    """Raised when a device ID is not found in the cluster."""


class DeviceInferenceError(DeviceError):
    """Raised when device cannot be inferred from arguments."""


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


def _maybe_set_dev_attr(dev_id: str, obj: Any) -> Any:
    """Set device attribute if obj is an Object, otherwise return as-is."""
    if isinstance(obj, Object):
        return set_dev_attr(obj, dev_id)
    return obj


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
        raise DeviceInferenceError(
            "Cannot infer device: no device-bound Object arguments found. "
            "Please specify device explicitly using device('device_id')."
        )

    devices = {get_dev_attr(obj) for obj in device_objs}

    if len(devices) == 1:
        return devices.pop()  # All arguments on same device

    if not g_auto_trans:
        raise DeviceInferenceError(
            f"Cannot infer device: arguments from multiple devices {devices} "
            f"but auto-transfer is disabled (g_auto_trans=False). "
            f"Please enable auto-transfer or put all data on same device first."
        )

    cluster = _resolve_cluster()
    device_kinds = {dev_id: cluster.devices[dev_id].kind.upper() for dev_id in devices}

    # Count devices by type
    spu_devs = [d for d, k in device_kinds.items() if k == "SPU"]
    tee_devs = [d for d, k in device_kinds.items() if k == "TEE"]
    ppu_devs = [d for d, k in device_kinds.items() if k == "PPU"]

    # Decision logic
    # Case 1: Only PPUs -> ambiguous (unless we want to pick one arbitrarily, but safer to error)
    if not spu_devs and not tee_devs:
        raise DeviceInferenceError(
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
        raise DeviceInferenceError(
            f"Ambiguous device inference: arguments from multiple SPU devices {spu_devs}. "
            f"Please specify which SPU to use explicitly."
        )

    # Case 5: Multiple TEEs -> ambiguous
    if len(tee_devs) > 1:
        raise DeviceInferenceError(
            f"Ambiguous device inference: arguments from multiple TEE devices {tee_devs}. "
            f"Please specify which TEE to use explicitly."
        )

    # Case 6: Both SPU and TEE -> conflicting
    if spu_devs and tee_devs:
        raise DeviceInferenceError(
            f"Ambiguous device inference: arguments from both SPU {spu_devs} and TEE {tee_devs}. "
            f"Please specify which secure device to use explicitly."
        )

    # Should never reach here
    raise DeviceInferenceError(f"Unexpected device configuration: {devices}")


def _device_run_spu(dev_info: Device, fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run function on SPU device."""
    spu_parties = tuple(m.rank for m in dev_info.members)

    # SPU execution uses spu.run_jax to compile and execute the function on the SPU.
    # Inputs are expected to be already on the SPU (handled by _d2d).
    # We wrap spu.run_jax in simp.pcall_static to execute it on all SPU parties.
    spu_config = spu.SPUConfig.from_dict(dev_info.config)
    result = simp.pcall_static(
        spu_parties,
        spu.run_jax,
        spu_config,
        fn,
        *args,
        **kwargs,
    )

    return tree_map(partial(set_dev_attr, dev_id=dev_info.name), result)


def _device_run_ppu(
    dev_info: Device,
    fn: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run function on PPU device."""
    assert len(dev_info.members) == 1
    rank = dev_info.members[0].rank

    result = simp.pcall_static((rank,), fn, *args, **kwargs)
    return tree_map(partial(_maybe_set_dev_attr, dev_info.name), result)


def _device_run_tee(
    dev_info: Device,
    fn: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run function on TEE device.

    TEE devices execute functions in a trusted execution environment.
    The execution is similar to PPU but runs in an isolated enclave.
    """
    assert len(dev_info.members) == 1
    rank = dev_info.members[0].rank

    result = simp.pcall_static((rank,), fn, *args, **kwargs)
    return tree_map(partial(_maybe_set_dev_attr, dev_info.name), result)


def _device_run(
    dev_id: str,
    fn: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute function on the specified device."""
    cluster = _resolve_cluster()
    if dev_id not in cluster.devices:
        available = list(cluster.devices.keys())
        raise DeviceNotFoundError(
            f"Device '{dev_id}' not found in cluster. Available devices: {available}"
        )
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
        return _device_run_tee(dev_info, fn, *args, **kwargs)
    elif dev_info.kind.upper() == "PPU":
        return _device_run_ppu(dev_info, fn, *args, **kwargs)
    else:
        raise DeviceError(f"Unknown device type: {dev_info.kind}")


class DeviceContext:
    """Context for device-specific operations.

    Supports explicit device specification or auto-inference from arguments.

    Examples:
        # Explicit device
        @device("P0")
        def add(a, b): ...

        # Auto-infer device from arguments
        @device()
        def add(a, b): ...

        # JAX frontend via .jax property (recommended for PPU)
        @device("P0").jax
        def add(a, b): return a + b

        # Or use separate decorators (equivalent)
        @device("P0")
        @jax_fn
        def add(a, b): return a + b

        # Inline call style
        result = device("P0").jax(fn)(x, y)
    """

    def __init__(self, dev_id: str | None = None):
        """Create a DeviceContext.

        Args:
            dev_id: Device ID (e.g., "P0", "SP0") or None for auto-inference.
        """
        self.dev_id = dev_id

    def _resolve_device(self, *args: Any, **kwargs: Any) -> str:
        """Resolve device ID, inferring from args if needed."""
        if self.dev_id is not None:
            return self.dev_id
        return _infer_device_from_args(*args, **kwargs)

    def _is_spu_device(self) -> bool:
        """Check if this device context targets an SPU device."""
        if self.dev_id is None:
            return False
        cluster = _resolve_cluster()
        if self.dev_id not in cluster.devices:
            return False
        return bool(cluster.devices[self.dev_id].kind.upper() == "SPU")

    @property
    def jax(self) -> Callable[[Callable], Callable]:
        """Return a decorator that wraps JAX functions for this device.

        For PPU/TEE: applies tensor.jax_fn to compile JAX code via StableHLO.
        For SPU: no-op wrapper, as SPU natively uses JAX via spu.run_jax.

        This is syntax sugar for using jax_fn adaptor:
            device("P0").jax(fn)  ==  device("P0")(jax_fn(fn))

        Examples:
            # As decorator
            @device("P0").jax
            def add(a, b): return a + b

            # As inline call
            result = device("P0").jax(fn)(x, y)
        """

        def wrapper(fn: Callable) -> Callable:
            # SPU natively uses JAX via spu.run_jax, no extra wrapping needed
            if self._is_spu_device():
                return self(fn)
            # PPU/TEE need tensor.jax_fn to compile JAX code
            from mplang.v2.dialects.tensor import jax_fn

            return self(jax_fn(fn))

        return wrapper

    def __call__(self, fn: Callable) -> Callable:
        """Wrap function for execution on this device."""

        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            dev_id = self._resolve_device(*args, **kwargs)
            return _device_run(dev_id, fn, *args, **kwargs)

        return wrapped


def device(dev_id: str | None = None) -> DeviceContext:
    """Create a device context for device-specific execution.

    Args:
        dev_id: Device ID (e.g., "P0", "SP0") or None for auto-inference.

    Returns:
        DeviceContext that wraps functions for device execution.

    Usage patterns:
        # Explicit device + generic function
        @device("P0")
        def fn(a, b): ...

        # Auto-infer device from arguments
        @device()
        def fn(a, b): ...

        # JAX frontend via .jax property (recommended for PPU)
        @device("P0").jax
        def add(a, b): return a + b

        # Inline call
        result = device("P0").jax(fn)(x, y)

        # Separate decorators (equivalent to above)
        @device("P0")
        @jax_fn
        def add(a, b): return a + b
    """
    return DeviceContext(dev_id)


def _ensure_tee_session(
    frm_dev_id: str,
    to_dev_id: str,
    frm_rank: int,
    tee_rank: int,
) -> tuple[Object, Object]:
    """Ensure a TEE session (sess_frm at sender, sess_tee at TEE) exists.

    Performs remote attestation and establishes an encrypted channel between
    a PPU and a TEE device. The session keys are cached to avoid repeated
    handshakes within the same execution context.

    The protocol:
    1. TEE generates keypair (sk, pk) and creates attestation quote binding pk
    2. Quote is sent to the sender (PPU) for verification
    3. Sender verifies quote and extracts TEE's attested public key
    4. Sender generates its own ephemeral keypair and sends pk to TEE
    5. Both sides derive shared secret using ECDH (X25519)
    6. The shared secret is used directly as the session key for AES-GCM

    Args:
        frm_dev_id: Source device ID (PPU)
        to_dev_id: Target device ID (TEE)
        frm_rank: Rank of the source party
        tee_rank: Rank of the TEE party

    Returns:
        Tuple of (sess_frm, sess_tee) where each is a symmetric key Object
    """
    import mplang.v2.edsl as el

    # Get current context ID for cache isolation
    current_ctx = el.get_current_context()
    current_context_id = id(current_ctx)

    # Check cache
    key = (frm_dev_id, to_dev_id)
    if key in _tee_session_cache:
        cached_context_id, sess_frm, sess_tee = _tee_session_cache[key]
        if cached_context_id == current_context_id:
            return sess_frm, sess_tee
        else:
            # Different context, cannot reuse
            del _tee_session_cache[key]

    # 1. TEE generates keypair and attestation quote
    tee_sk, tee_pk = simp.pcall_static((tee_rank,), crypto.kem_keygen, _TEE_KEM_SUITE)
    quote = simp.pcall_static((tee_rank,), tee.quote_gen, tee_pk)

    # 2. Send quote to sender for attestation verification
    quote_at_sender = simp.shuffle_static(quote, {frm_rank: tee_rank})

    # 3. Sender verifies quote and extracts TEE's public key
    tee_pk_at_sender = simp.pcall_static((frm_rank,), tee.attest, quote_at_sender)

    # 4. Sender generates ephemeral keypair and sends pk to TEE
    v_sk, v_pk = simp.pcall_static((frm_rank,), crypto.kem_keygen, _TEE_KEM_SUITE)
    v_pk_at_tee = simp.shuffle_static(v_pk, {tee_rank: frm_rank})

    # 5. Both sides derive shared secret (symmetric key) via ECDH
    # The shared secret from X25519 ECDH is suitable for direct use as AES key
    sess_frm = simp.pcall_static((frm_rank,), crypto.kem_derive, v_sk, tee_pk_at_sender)
    sess_tee = simp.pcall_static((tee_rank,), crypto.kem_derive, tee_sk, v_pk_at_tee)

    # Cache the session
    _tee_session_cache[key] = (current_context_id, sess_frm, sess_tee)

    return sess_frm, sess_tee


def _d2d(to_dev_id: str, obj: Object) -> Object:
    """Transfer object to target device."""
    if not isinstance(obj, Object):
        raise TypeError(f"Expected Object, got {type(obj)}")

    frm_dev_id = get_dev_attr(obj)
    if frm_dev_id == to_dev_id:
        return obj

    cluster = _resolve_cluster()
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
        spu_config = spu.SPUConfig.from_dict(to_dev.config)

        # 1. Generate shares on source
        # We call spu.make_shares inside pcall on the source party
        shares_on_source = simp.pcall_static(
            (frm_rank,),
            spu.make_shares,
            spu_config,
            obj,
            count=len(spu_parties),
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
        spu_config = spu.SPUConfig.from_dict(frm_dev.config)

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
            (to_rank,), lambda *s: spu.reconstruct(spu_config, s), *gathered_shares
        )
        return set_dev_attr(var, to_dev_id)

    # SPU -> SPU
    elif frm_to_pair == ("SPU", "SPU"):
        raise NotImplementedError("SPU to SPU transfer not implemented yet.")

    # PPU -> TEE (Encrypted transfer)
    elif frm_to_pair == ("PPU", "TEE"):
        assert len(frm_dev.members) == 1 and len(to_dev.members) == 1
        frm_rank = frm_dev.members[0].rank
        tee_rank = to_dev.members[0].rank

        # Establish encrypted session (includes remote attestation)
        sess_frm, sess_tee = _ensure_tee_session(
            frm_dev_id, to_dev_id, frm_rank, tee_rank
        )

        # Encrypt on sender and send to TEE
        ct = simp.pcall_static((frm_rank,), crypto.sym_encrypt, sess_frm, obj)
        ct_at_tee = simp.shuffle_static(ct, {tee_rank: frm_rank})

        # Decrypt on TEE
        var = simp.pcall_static(
            (tee_rank,),
            crypto.sym_decrypt,
            sess_tee,
            ct_at_tee,
            obj.type.value_type if hasattr(obj.type, "value_type") else obj.type,
        )
        return set_dev_attr(var, to_dev_id)

    # TEE -> PPU (Encrypted transfer)
    elif frm_to_pair == ("TEE", "PPU"):
        assert len(frm_dev.members) == 1 and len(to_dev.members) == 1
        tee_rank = frm_dev.members[0].rank
        ppu_rank = to_dev.members[0].rank

        # Establish encrypted session (reuse existing or create new)
        # Note: We pass (ppu, tee) order to match the session key derivation
        sess_ppu, sess_tee = _ensure_tee_session(
            to_dev_id, frm_dev_id, ppu_rank, tee_rank
        )

        # Encrypt on TEE and send to PPU
        ct = simp.pcall_static((tee_rank,), crypto.sym_encrypt, sess_tee, obj)
        ct_at_ppu = simp.shuffle_static(ct, {ppu_rank: tee_rank})

        # Decrypt on PPU
        var = simp.pcall_static(
            (ppu_rank,),
            crypto.sym_decrypt,
            sess_ppu,
            ct_at_ppu,
            obj.type.value_type if hasattr(obj.type, "value_type") else obj.type,
        )
        return set_dev_attr(var, to_dev_id)

    # TEE -> SPU (TEE acts like a PPU for SPU sealing)
    elif frm_to_pair == ("TEE", "SPU"):
        assert len(frm_dev.members) == 1
        frm_rank = frm_dev.members[0].rank
        spu_parties = tuple(m.rank for m in to_dev.members)
        spu_config = spu.SPUConfig.from_dict(to_dev.config)

        # Generate shares on TEE (same logic as PPU -> SPU)
        shares_on_source = simp.pcall_static(
            (frm_rank,),
            spu.make_shares,
            spu_config,
            obj,
            count=len(spu_parties),
        )

        # Distribute shares to SPU parties
        distributed_shares = []
        for i, target_rank in enumerate(spu_parties):
            share_i = shares_on_source[i]
            share_at_target = simp.shuffle_static(share_i, {target_rank: frm_rank})
            distributed_shares.append(share_at_target)

        # Converge shares
        var = simp.converge(*distributed_shares)
        return set_dev_attr(var, to_dev_id)

    # SPU -> TEE (Reveal to TEE)
    elif frm_to_pair == ("SPU", "TEE"):
        assert len(to_dev.members) == 1
        to_rank = to_dev.members[0].rank
        spu_parties = tuple(m.rank for m in frm_dev.members)
        spu_config = spu.SPUConfig.from_dict(frm_dev.config)

        # Gather shares to TEE (same logic as SPU -> PPU)
        gathered_shares = []
        for source_rank in spu_parties:
            share_on_source = simp.pcall_static((source_rank,), lambda x: x, obj)
            share_at_target = simp.shuffle_static(
                share_on_source, {to_rank: source_rank}
            )
            gathered_shares.append(share_at_target)

        # Reconstruct on TEE
        var = simp.pcall_static(
            (to_rank,), lambda *s: spu.reconstruct(spu_config, s), *gathered_shares
        )
        return set_dev_attr(var, to_dev_id)

    # TEE -> TEE
    elif frm_to_pair == ("TEE", "TEE"):
        raise NotImplementedError("TEE to TEE transfer not implemented yet.")

    else:
        raise DeviceError(f"Unsupported device transfer: {frm_to_pair}")


def put(to_dev_id: str, obj: Any) -> Object:
    """Put data onto a device.

    Args:
        to_dev_id: Target device ID (e.g., "P0", "SP0").
        obj: The object to put onto the device.

    If obj is already a device object, it moves it to the target device.
    If obj is a host object (e.g. numpy array), it uploads it to the target device.
    """
    cluster = _resolve_cluster()
    if to_dev_id not in cluster.devices:
        available = list(cluster.devices.keys())
        raise DeviceNotFoundError(
            f"Device '{to_dev_id}' not found in cluster. Available devices: {available}"
        )

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
        return cast(Object, device(to_dev_id)(lambda x: x)(obj))

    elif dev_info.kind.upper() == "TEE":
        # Host -> TEE: Similar to PPU, create constant on TEE device
        assert len(dev_info.members) == 1
        rank = dev_info.members[0].rank

        var = simp.constant((rank,), obj)
        return set_dev_attr(var, to_dev_id)

    else:
        raise DeviceError(f"Cannot put to device kind '{dev_info.kind}'")


def fetch(obj: Object) -> Any:
    """Fetch data from device to host based on device attribute.

    This function fetches data from the device the object resides on.
    For PPU/TEE: fetches from the single member rank.
    For SPU: fetches from all parties (returns reconstructed value).

    Args:
        obj: Object with device attribute to fetch.

    Returns:
        Python value (numpy array, scalar, etc.)
    """
    from mplang.v2.backends.simp_driver.state import SimpDriver
    from mplang.v2.backends.simp_driver.values import DriverVar
    from mplang.v2.edsl.context import get_current_context
    from mplang.v2.runtime.interpreter import InterpObject, Interpreter
    from mplang.v2.runtime.value import WrapValue

    def _unwrap_value(val: Any) -> Any:
        """Unwrap WrapValue to get the underlying data."""
        if isinstance(val, WrapValue):
            return val.data
        return val

    # 1. Ensure is object and is device obj
    if not is_device_obj(obj):
        raise DeviceError(
            "Object does not have device attribute. Use mp.fetch() directly."
        )

    # 2. Get device information according to device id
    dev_id = get_dev_attr(obj)
    cluster = _resolve_cluster()
    dev_info = cluster.devices[dev_id]

    # Get interpreter context
    ctx = get_current_context()
    if not isinstance(ctx, Interpreter):
        raise RuntimeError("No interpreter context available for fetch")

    simp_state = ctx.get_dialect_state("simp")
    assert isinstance(simp_state, SimpDriver), "DriverVar requires simp state"

    # Unwrap InterpObject to get runtime value
    assert isinstance(obj, InterpObject), f"Expected InterpObject, got {type(obj)}"
    runtime_obj = obj.runtime_obj

    def _fetch_from_rank(rank: int) -> Any:
        """Fetch value from a rank (DriverVar values are always URIs)."""
        uri = runtime_obj.values[rank]
        assert isinstance(uri, str) and "://" in uri, f"Expected URI, got {uri}"
        return simp_state.fetch(rank, uri).result()

    # 3. Match device type and do corresponding fetch action
    if isinstance(runtime_obj, DriverVar):
        # 3.1 PPU/TEE: single member, fetch directly
        if dev_info.kind.upper() in ("PPU", "TEE"):
            assert len(dev_info.members) == 1
            result = _fetch_from_rank(dev_info.members[0].rank)
            # 4. Unwrap if WrapValue
            return _unwrap_value(result)

        # 3.2 SPU: fetch from all ranks and reconstruct
        elif dev_info.kind.upper() == "SPU":
            # Fetch shares from all SPU members
            shares = [_fetch_from_rank(m.rank) for m in dev_info.members]
            # For now, just return the first share (TODO: implement spu.reconstruct)
            # In practice, SPU values should be revealed to a PPU first
            result = shares[0] if shares else None
            # 4. Unwrap if WrapValue
            return _unwrap_value(result)

    # Direct value (not DriverVar)
    return _unwrap_value(runtime_obj)
