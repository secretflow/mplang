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

"""Device Basics: Placement, Masks, and Auto Device

Learning objectives:
1. Understand device types: PPU (public), SPU (MPC), TEE (trusted)
2. Use @mp.device for explicit and automatic device placement
3. Understand pmask (compile-time) and rmask (runtime) concepts
4. Master auto device inference rules

Key concepts:
- PPU: plaintext computation, data visible to that party
- SPU: secure multi-party computation, data secret-shared
- TEE: trusted execution environment, isolated computation
- Auto device: infers from argument placement when device name omitted
"""

import random

import mplang.v1 as mp

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
    # Generate data on specific parties
    x = mp.device("P0")(lambda: random.randint(0, 10))()
    y = mp.device("P1")(lambda: random.randint(0, 10))()

    # Secure comparison on SPU
    result = mp.device("SP0")(lambda a, b: a < b)(x, y)

    return x, y, result


# ============================================================================
# Pattern 2: Auto Device Inference
# ============================================================================


@mp.device
def add(a, b):
    # Auto infers device from arguments
    return a + b


def auto_device():
    """Use @mp.device decorator for reusable device-inferred functions."""
    x = mp.device("P0")(lambda: 1)()
    y = mp.device("P0")(lambda: 2)()
    z = mp.device("P1")(lambda: 3)()

    # Same PPU (P0): auto-infers P0
    sum_x_y = add(x, y)
    assert mp.get_dev_attr(sum_x_y) == "P0"

    sealed_sum = mp.put("SP0", sum_x_y)

    # Cross P0 and SP0: auto-infers to SPU
    product = add(sealed_sum, z)
    assert mp.get_dev_attr(product) == "SP0"

    return product


# ============================================================================
# Pattern 3: Device Attributes and mp.put
# ============================================================================


def device_movement():
    """Move data between devices using mp.put."""
    # Data starts on P0
    x = mp.device("P0")(lambda: 42)()

    # Move to SPU for secure computation
    x_on_spu = mp.put("SP0", x)

    # Compute on SPU
    result_spu = mp.device("SP0")(lambda v: v * 2)(x_on_spu)

    # Move result back to P0
    result_p0 = mp.put("P0", result_spu)

    return result_p0


# ============================================================================
# Main: Demonstrate All Patterns
# ============================================================================


def main():
    print("=" * 70)
    print("Device Basics: Placement, Masks, and Auto Device")
    print("=" * 70)

    sim = mp.Simulator(cluster_spec)

    # Pattern 1: Explicit placement
    print("\n--- Pattern 1: Explicit Device Placement (Millionaire) ---")
    x, y, result = mp.evaluate(sim, millionaire)
    print(f"P0 value: {mp.fetch(sim, x)}")
    print(f"P1 value: {mp.fetch(sim, y)}")
    print(f"x < y (computed on SPU): {mp.fetch(sim, result)}")

    # Pattern 2: Auto device inference
    print("\n--- Pattern 2: Auto Device Inference ---")
    product = mp.evaluate(sim, auto_device)
    print(f"Result (cross-PPU auto-inferred to SPU): {mp.fetch(sim, product)}")

    # Pattern 3: Device movement
    print("\n--- Pattern 3: Device Movement with mp.put ---")
    moved = mp.evaluate(sim, device_movement)
    print(f"Final result on P0: {mp.fetch(sim, moved)}")

    # Show device attributes
    print("\n--- Device Attributes ---")
    x_ref = mp.evaluate(sim, lambda: mp.device("P0")(lambda: 10)())
    print(f"Device of x_ref: {mp.get_dev_attr(x_ref)}")

    print("\n" + "=" * 70)
    print("Key takeaways:")
    print("1. Explicit: mp.device('DevName')(fn) for specific placement")
    print("2. Auto: @mp.device decorator infers device from arguments")
    print("3. Cross-PPU: auto-infers to SPU when args from different PPUs")
    print("4. mp.put: move data between devices explicitly")
    print("5. mp.get_dev_attr: inspect device of a variable")
    print("=" * 70)


if __name__ == "__main__":
    main()
