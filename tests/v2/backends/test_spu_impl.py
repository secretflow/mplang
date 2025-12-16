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

"""Tests for SPU backend implementation."""

import numpy as np

import mplang.v2 as mp
import mplang.v2.backends.tensor_impl  # noqa: F401
import mplang.v2.edsl as el
from mplang.v2.dialects import simp, spu


def test_spu_e2e_simulation():
    """Test SPU end-to-end flow using SimpSimulator."""
    # 1. Setup
    world_size = 3
    sim = simp.make_simulator(world_size=world_size)
    mp.set_root_context(sim)
    spu_parties = (0, 1, 2)
    spu_config = spu.SPUConfig()

    # 2. Define computation
    def secure_add(x, y):
        return x + y

    # 3. Define workflow graph
    def workflow():
        # Create data on party 0
        # We use pcall to create MP-typed data on party 0
        x_mp = simp.constant((0,), np.array([1.0, 2.0], dtype=np.float32))
        y_mp = simp.constant((0,), np.array([3.0, 4.0], dtype=np.float32))

        # Encrypt (Public -> SPU)
        # Manual encrypt for x
        # 1. Make shares (Local on source)
        x_shares = simp.pcall_static((0,), spu.make_shares, spu_config, x_mp, count=3)
        # 2. Distribute
        x_dist = []
        for i, target in enumerate(spu_parties):
            x_dist.append(simp.shuffle_static(x_shares[i], {target: 0}))
        # 3. Converge
        x_enc = simp.converge(*x_dist)

        # Manual encrypt for y
        y_shares = simp.pcall_static((0,), spu.make_shares, spu_config, y_mp, count=3)
        y_dist = []
        for i, target in enumerate(spu_parties):
            y_dist.append(simp.shuffle_static(y_shares[i], {target: 0}))
        y_enc = simp.converge(*y_dist)

        # Execute (SPU -> SPU)
        z_enc = simp.pcall_static(
            spu_parties,
            spu.run_jax,
            spu_config,
            secure_add,
            x_enc,
            y_enc,
        )

        # Decrypt (SPU -> Public)
        # Manual decrypt
        z_shares = []
        for source in spu_parties:
            share = simp.pcall_static((source,), lambda x: x, z_enc)
            z_shares.append(simp.shuffle_static(share, {0: source}))

        z_mp = simp.pcall_static(
            (0,), lambda *s: spu.reconstruct(spu_config, s), *z_shares
        )
        return z_mp

    # Trace to get graph
    traced = el.trace(workflow)
    graph = traced.graph

    try:
        # 4. Execute on all parties
        # sim is now an Interpreter directly (from simp.make_simulator)
        results_list = sim.evaluate_graph(graph, [])
        # evaluate_graph returns list; extract single output
        results_var = results_list[0]

        # Fetch results
        values = mp.fetch(results_var)

        # 5. Verify
        # Result from party 0 should be the tensor (wrapped in TensorValue)
        res_p0 = values[0]

        assert res_p0 is not None
        # Unwrap TensorValue if needed
        if hasattr(res_p0, "unwrap"):
            res_p0 = res_p0.unwrap()
        np.testing.assert_allclose(res_p0, [4.0, 6.0])

        assert values[1] is None
        assert values[2] is None

    finally:
        # Shutdown the cluster via the interpreter's reference
        if hasattr(sim, "_simp_cluster"):
            sim._simp_cluster.shutdown()
