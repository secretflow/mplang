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

"""Device: Running JAX Functions Across Devices

Learning objectives:
1. Run JAX functions on PPU (plaintext)
2. Run JAX functions on SPU (secure MPC)
3. Understand device-specific performance characteristics

Key concepts:
- PPU: fast, no privacy protection
- SPU: slower, privacy-preserving

Note: TEE support not yet available in mplang2.

Migrated from mplang v1 to mplang2.
"""

import jax.numpy as jnp

import mplang.v2 as mp

cluster_spec = mp.ClusterSpec.from_dict({
    "nodes": [
        {"name": "node_0", "endpoint": "127.0.0.1:61920"},
        {"name": "node_1", "endpoint": "127.0.0.1:61921"},
    ],
    "devices": {
        "SP0": {
            "kind": "SPU",
            "members": ["node_0", "node_1"],
            "config": {"protocol": "SEMI2K", "field": "FM128"},
        },
        "P0": {"kind": "PPU", "members": ["node_0"]},
        "P1": {"kind": "PPU", "members": ["node_1"]},
    },
})


@mp.function
def jax_on_ppu():
    """Pattern 1: Run JAX on PPU (plaintext computation)."""
    # Generate data on P0 using mp.put for constants
    x = mp.put("P0", jnp.array([1.0, 2.0, 3.0]))

    # Run JAX computation on P0
    # Use .jax for functions that use JAX operators
    result = mp.device("P0").jax(lambda v: jnp.sum(v**2))(x)

    return result


@mp.function
def jax_on_spu():
    """Pattern 2: Run JAX on SPU (secure multi-party computation)."""
    # Generate private inputs using mp.put
    x = mp.put("P0", jnp.array([1.0, 2.0]))
    y = mp.put("P1", jnp.array([3.0, 4.0]))

    # Secure computation on SPU - SPU always uses JAX semantics natively
    # No .jax_fn needed for SPU!
    dot_product = mp.device("SP0")(lambda a, b: jnp.dot(a, b))(x, y)

    # Reveal result to P0
    result = mp.put("P0", dot_product)

    return result


@mp.function
def cross_device_pipeline():
    """Pattern 3: Multi-stage pipeline across devices (PPU -> SPU -> PPU)."""
    # Stage 1: Generate on P0
    x = mp.put("P0", jnp.array([10, 20, 30]))

    # Stage 2: Process on SPU - SPU always uses JAX semantics natively
    processed = mp.device("SP0")(lambda v: v * 2)(x)

    # Stage 3: Bring back to P0 and compute sum
    # PPU needs .jax for JAX operators
    tmp_p0 = mp.put("P0", processed)
    final = mp.device("P0").jax(lambda v: jnp.sum(v))(tmp_p0)

    return final


def main():
    print("=" * 70)
    print("Device: Running JAX Functions Across Devices")
    print("=" * 70)

    sim = mp.make_simulator(2, cluster_spec=cluster_spec)
    mp.set_root_context(sim)

    # Pattern 1: PPU
    print("\n--- Pattern 1: JAX on PPU ---")
    r1 = mp.evaluate(jax_on_ppu)
    print(f"Sum of squares [1,2,3]: {mp.fetch(r1)}")

    # Pattern 2: SPU
    print("\n--- Pattern 2: JAX on SPU ---")
    r2 = mp.evaluate(jax_on_spu)
    print(f"Secure dot product [1,2]·[3,4] = {mp.fetch(r2)}")

    # Pattern 3: Cross-device pipeline
    print("\n--- Pattern 3: Cross-Device Pipeline ---")
    r3 = mp.evaluate(cross_device_pipeline)
    print(f"Pipeline result (sum of [10,20,30]*2): {mp.fetch(r3)}")

    print("\n" + "=" * 70)
    print("Key takeaways:")
    print("1. PPU: fast plaintext JAX, no privacy")
    print("2. SPU: secure JAX via MPC, slower but privacy-preserving")
    print("3. mp.put: move results between devices as needed")
    print("=" * 70)


if __name__ == "__main__":
    main()
