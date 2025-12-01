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
3. Run JAX functions on TEE (trusted execution)
4. Understand device-specific performance characteristics

Key concepts:
- PPU: fast, no privacy protection
- SPU: slower, privacy-preserving
- TEE: moderate speed, requires trust in hardware
"""

import random

import jax.numpy as jnp

import mplang.v1 as mp

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
        "P0": {"kind": "PPU", "members": ["node_0"]},
        "P1": {"kind": "PPU", "members": ["node_1"]},
        "TEE0": {"kind": "TEE", "members": ["node_2"]},
    },
})


@mp.function
def jax_on_ppu():
    """Pattern 1: Run JAX on PPU (plaintext computation)."""
    # Generate data on P0
    x = mp.device("P0")(lambda: jnp.array([1.0, 2.0, 3.0]))()

    # Run JAX computation on P0
    result = mp.device("P0")(lambda v: jnp.sum(v**2))(x)

    return result


@mp.function
def jax_on_spu():
    """Pattern 2: Run JAX on SPU (secure multi-party computation)."""
    # Generate private inputs
    x = mp.device("P0")(lambda: jnp.array([1.0, 2.0]))()
    y = mp.device("P1")(lambda: jnp.array([3.0, 4.0]))()

    # Secure computation on SPU
    dot_product = mp.device("SP0")(lambda a, b: jnp.dot(a, b))(x, y)

    # Reveal result
    result = mp.put("P0", dot_product)

    return result


@mp.function
def jax_on_tee():
    """Pattern 3: Run JAX on TEE (trusted execution environment)."""
    x = mp.device("P0")(lambda: random.randint(0, 100))()
    y = mp.device("P1")(lambda: random.randint(0, 100))()

    # Run comparison in TEE
    comparison = mp.device("TEE0")(lambda a, b: a < b)(x, y)

    # Bring result back
    result = mp.put("P0", comparison)

    return x, y, result


@mp.function
def cross_device_pipeline():
    """Pattern 4: Multi-stage pipeline across devices (PPU -> SPU -> PPU)."""
    # Stage 1: Generate on P0
    x = mp.device("P0")(lambda: jnp.array([10, 20, 30]))()

    # Stage 2: Process on SPU
    processed = mp.device("SP0")(lambda v: v * 2)(x)

    # Stage 3: Bring back to P0 and compute sum
    # Note: SPU->TEE transfer not yet supported, so we use SPU->PPU
    tmp_p0 = mp.put("P0", processed)

    final = mp.device("TEE0")(lambda v: jnp.sum(v))(tmp_p0)

    return final


def main():
    print("=" * 70)
    print("Device: Running JAX Functions Across Devices")
    print("=" * 70)

    sim = mp.Simulator(cluster_spec)

    # Pattern 1: PPU
    print("\n--- Pattern 1: JAX on PPU ---")
    r1 = mp.evaluate(sim, jax_on_ppu)
    print(f"Result: {mp.fetch(sim, r1)}")

    # Pattern 2: SPU
    print("\n--- Pattern 2: JAX on SPU ---")
    r2 = mp.evaluate(sim, jax_on_spu)
    print(f"Secure dot product: {mp.fetch(sim, r2)}")

    # Pattern 3: TEE
    print("\n--- Pattern 3: JAX on TEE ---")
    # Setup TEE bindings for mock
    tee_bindings = {
        "tee.quote_gen": "mock_tee.quote_gen",
        "tee.attest": "mock_tee.attest",
    }
    for n in cluster_spec.nodes.values():
        n.runtime_info.op_bindings.update(tee_bindings)
    sim_tee = mp.Simulator(cluster_spec)

    x, y, r3 = mp.evaluate(sim_tee, jax_on_tee)
    print(
        f"TEE comparison: {mp.fetch(sim_tee, x)} < {mp.fetch(sim_tee, y)} = {mp.fetch(sim_tee, r3)}"
    )

    # Pattern 4: Cross-device pipeline
    print("\n--- Pattern 4: Cross-Device Pipeline ---")
    # TODO(jint): fixme, hang - cross-device pipeline involving SPU->TEE transfer is not yet supported and causes deadlock. See issue tracker for details.
    # r4 = mp.evaluate(sim, cross_device_pipeline)
    # print(f"Pipeline result: {mp.fetch(sim, r4)}")

    print("\n" + "=" * 70)
    print("Key takeaways:")
    print("1. PPU: fast plaintext JAX, no privacy")
    print("2. SPU: secure JAX via MPC, slower but privacy-preserving")
    print("3. TEE: trusted JAX in isolated environment")
    print("4. mp.put: move results between devices as needed")
    print("=" * 70)


if __name__ == "__main__":
    main()
