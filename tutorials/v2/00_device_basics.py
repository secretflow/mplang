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

"""Device Basics: Placement, Masks, and Auto Device (MPLang2 version)

Learning objectives:
1. Understand device types: PPU (public), SPU (MPC), TEE (trusted)
2. Use @mp.device for explicit and automatic device placement
3. Use @mp.device("P0").jax for JAX function execution on PPU
4. Understand pmask (compile-time) and rmask (runtime) concepts
5. Master auto device inference rules

Key concepts:
- PPU: plaintext computation, data visible to that party
- SPU: secure multi-party computation, data secret-shared
- TEE: trusted execution environment, isolated computation
- Auto device: infers from argument placement when device name omitted

API patterns:
- mp.put("P0", value): place data on a specific device
- mp.device("P0")(fn)(*args): generic tracing execution on device
- mp.device("P0").jax(fn)(*args): JAX execution (for numpy-like operators on PPU)
- mp.device()(fn): auto-infer device from arguments

Migration notes (mplang -> mplang2):
- import mplang.v2 as mp (instead of mplang)
- For JAX functions on PPU, use .jax property:
    mp.device("P0").jax(lambda a, b: a + b)(x, y)
- For SPU, no special frontend needed (JAX is native)
- For constants, use mp.put("P0", value)
"""

import random

import jax.numpy as jnp

import mplang.v2 as mp

# Define a simple 3-party cluster with SPU, 2 PPUs, and TEE
cluster_spec = mp.ClusterSpec.from_dict({
    "nodes": [
        {"name": "node_0", "endpoint": "127.0.0.1:61920"},
        {"name": "node_1", "endpoint": "127.0.0.1:61921"},
        {"name": "node_2", "endpoint": "127.0.0.1:61922"},
    ],
    "devices": {
        "SP0": {
            "kind": "SPU",
            "members": ["node_0", "node_1", "node_2"],
            "config": {"protocol": "SEMI2K", "field": "FM128"},
        },
        "P0": {"kind": "PPU", "members": ["node_0"], "config": {}},
        "P1": {"kind": "PPU", "members": ["node_1"], "config": {}},
        "TEE0": {"kind": "TEE", "members": ["node_2"], "config": {}},
    },
})


# ============================================================================
# Pattern 1: Explicit Device Placement
# ============================================================================


def millionaire():
    """Explicitly specify device for each operation."""
    # Generate data on specific parties using mp.put
    x = mp.put("P0", jnp.array(random.randint(0, 10)))
    y = mp.put("P1", jnp.array(random.randint(0, 10)))

    # Secure comparison on SPU - SPU uses JAX natively, no frontend needed
    result = mp.device("SP0")(lambda a, b: a < b)(x, y)

    return x, y, result


# ============================================================================
# Pattern 2: Auto Device Inference with JAX
# ============================================================================


def add_jax(a, b):
    # Use jax_fn for JAX operators
    return a + b


def auto_device():
    """Demonstrate auto device inference with explicit device calls."""
    # Use mp.put to place constants on devices
    x = mp.put("P0", jnp.array(1))
    y = mp.put("P0", jnp.array(2))
    z = mp.put("P1", jnp.array(3))

    # Same PPU (P0): use device("P0").jax for JAX operators
    sum_x_y = mp.device("P0").jax(add_jax)(x, y)
    assert mp.get_dev_attr(sum_x_y) == "P0"

    sealed_sum = mp.put("SP0", sum_x_y)

    # Cross devices: SPU uses JAX natively
    product = mp.device("SP0")(add_jax)(sealed_sum, z)
    assert mp.get_dev_attr(product) == "SP0"

    return product


# ============================================================================
# Pattern 3: Device Attributes and mp.put
# ============================================================================


def device_movement():
    """Move data between devices using mp.put."""
    # Data starts on P0
    x = mp.put("P0", jnp.array(42))

    # Move to SPU for secure computation
    x_on_spu = mp.put("SP0", x)

    # Compute on SPU - SPU uses JAX natively
    result_spu = mp.device("SP0")(lambda v: v * 2)(x_on_spu)

    # Move result back to P0
    result_p0 = mp.put("P0", result_spu)

    return result_p0


# ============================================================================
# Main: Demonstrate All Patterns
# ============================================================================


def main():
    print("=" * 70)
    print("Device Basics: Placement, Masks, and Auto Device (MPLang2)")
    print("=" * 70)

    sim = mp.make_simulator(3, cluster_spec=cluster_spec)
    mp.set_root_context(sim)  # Set global context (JAX-like pattern)

    # Pattern 1: Explicit placement
    print("\n--- Pattern 1: Explicit Device Placement (Millionaire) ---")
    x, y, result = mp.evaluate(millionaire)
    # fetch uses device attribute to get value from correct rank
    print(f"P0 value: {mp.fetch(x)}")
    print(f"P1 value: {mp.fetch(y)}")
    # SPU result is secret-shared; to reveal, move it to a PPU first
    result_revealed = mp.evaluate(lambda: mp.put("P0", result))
    print(f"x < y (revealed to P0): {mp.fetch(result_revealed)}")

    # Pattern 2: Auto device inference
    print("\n--- Pattern 2: Auto Device Inference ---")
    product = mp.evaluate(auto_device)
    # SPU result needs to be revealed
    product_revealed = mp.evaluate(lambda: mp.put("P0", product))
    print(f"Result (revealed to P0): {mp.fetch(product_revealed)}")

    # Pattern 3: Device movement
    print("\n--- Pattern 3: Device Movement with mp.put ---")
    moved = mp.evaluate(device_movement)
    print(f"Final result on P0: {mp.fetch(moved)}")

    # Show device attributes
    print("\n--- Device Attributes ---")
    x_ref = mp.evaluate(lambda: mp.put("P0", jnp.array(10)))
    print(f"Device of x_ref: {mp.get_dev_attr(x_ref)}")

    print("\n" + "=" * 70)
    print("Key takeaways:")
    print("1. mp.put('DevName', value): place data on a specific device")
    print("2. mp.device('Dev').jax(fn): execute JAX function on PPU")
    print("3. mp.device('Dev')(fn): generic traced execution (SPU uses JAX natively)")
    print("4. SPU results are secret-shared; use mp.put to reveal to a PPU")
    print("5. mp.get_dev_attr: inspect device of a variable")
    print("=" * 70)


if __name__ == "__main__":
    main()
