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
import pytest

import mplang as mp
import mplang.backends.tensor_impl
import mplang.edsl as el
from mplang.dialects import simp, spu
from mplang.edsl import typing as elt


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


def test_spu_channels_mode_simulation():
    """Test SPU using Channels mode (no BRPC endpoints, reuse simp communicator)."""
    # 1. Setup
    world_size = 3
    sim = simp.make_simulator(world_size=world_size)
    mp.set_root_context(sim)
    spu_parties = (0, 1, 2)
    spu_config = spu.SPUConfig()

    # 2. Define computation
    def secure_mul(x, y):
        return x * y

    # 3. Define workflow graph (no spu_endpoints -> Channels mode)
    def workflow():
        # Create data
        x_mp = simp.constant((0,), np.array([2.0, 3.0], dtype=np.float32))
        y_mp = simp.constant((0,), np.array([4.0, 5.0], dtype=np.float32))

        # Encrypt
        x_shares = simp.pcall_static((0,), spu.make_shares, spu_config, x_mp, count=3)
        x_dist = [
            simp.shuffle_static(x_shares[i], {target: 0})
            for i, target in enumerate(spu_parties)
        ]
        x_enc = simp.converge(*x_dist)

        y_shares = simp.pcall_static((0,), spu.make_shares, spu_config, y_mp, count=3)
        y_dist = [
            simp.shuffle_static(y_shares[i], {target: 0})
            for i, target in enumerate(spu_parties)
        ]
        y_enc = simp.converge(*y_dist)

        # Execute (Channels mode - no spu_endpoints)
        z_enc = simp.pcall_static(
            spu_parties,
            spu.run_jax,
            spu_config,
            secure_mul,
            x_enc,
            y_enc,
        )

        # Decrypt
        z_shares = [
            simp.shuffle_static(
                simp.pcall_static((source,), lambda x: x, z_enc), {0: source}
            )
            for source in spu_parties
        ]
        z_mp = simp.pcall_static(
            (0,), lambda *s: spu.reconstruct(spu_config, s), *z_shares
        )
        return z_mp

    # Trace and execute
    traced = el.trace(workflow)
    graph = traced.graph

    try:
        results_list = sim.evaluate_graph(graph, [])
        results_var = results_list[0]
        values = mp.fetch(results_var)

        # Verify
        res_p0 = values[0]
        if hasattr(res_p0, "unwrap"):
            res_p0 = res_p0.unwrap()
        np.testing.assert_allclose(res_p0, [8.0, 15.0])

        assert values[1] is None
        assert values[2] is None

    finally:
        if hasattr(sim, "_simp_cluster"):
            sim._simp_cluster.shutdown()


def test_spu_run_dynamic_shape():
    """Test SPU backend with dynamic shapes (using -1 for unknown dimensions).

    This test follows the exact pattern from test_tensor_run_dynamic_shape:
    1. Compile with dynamic shape (-1) - shape unknown at compile time
    2. Execute multiple times with different actual data to demonstrate dynamic behavior
    """
    import shutil

    # Check if stablehlo-opt is available
    if not shutil.which("stablehlo-opt"):
        pytest.skip("stablehlo-opt not available, skipping dynamic shape test")

    # Phase 1: Setup and compilation

    # Define computation
    def jax_add(x, y):
        return x + y

    # Define the SPU execution function
    def exec_spu(x, y):
        # Create SPU config and parties
        spu_config = spu.SPUConfig()
        spu_parties = (0, 1, 2)

        # Manual encryption using make_shares
        x_shares = simp.pcall_static((0,), spu.make_shares, spu_config, x, count=3)
        x_enc = simp.converge(*[
            simp.shuffle_static(x_shares[i], {spu_parties[i]: 0}) for i in range(3)
        ])

        y_shares = simp.pcall_static((0,), spu.make_shares, spu_config, y, count=3)
        y_enc = simp.converge(*[
            simp.shuffle_static(y_shares[i], {spu_parties[i]: 0}) for i in range(3)
        ])

        # Execute on SPU
        z_enc = simp.pcall_static(
            spu_parties, spu.run_jax, spu_config, jax_add, x_enc, y_enc
        )

        # Decrypt (SPU -> Public)
        z_shares = [
            simp.shuffle_static(
                simp.pcall_static((source,), lambda x: x, z_enc), {0: source}
            )
            for source in spu_parties
        ]
        z_mp = simp.pcall_static(
            (0,), lambda *s: spu.reconstruct(spu_config, s), *z_shares
        )

        return z_mp

    # Compile with dynamic shape tensor
    tracer = el.Tracer()
    tensor_type = elt.TensorType(elt.f32, (-1,))
    x_obj = tracer._new_arg(tensor_type)
    y_obj = tracer._new_arg(tensor_type)
    traced_fn = mp.compile(exec_spu, x_obj, y_obj)
    out_type = traced_fn.graph.outputs[0].type
    assert isinstance(out_type, elt.MPType)
    out_type = out_type.value_type

    # print(f"Graph structure:\n{traced_fn.graph.to_string(verbose=True)}")
    assert isinstance(out_type, elt.TensorType) and out_type.shape == (-1,)
    # Verify the graph contains dynamic shape notation
    assert "Tensor[f32, (-1)" in traced_fn.graph.to_string(verbose=True)

    # Phase 2: Execution with various test inputs
    test_cases = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),  # Size 3
        np.array([0.5, -1.5, 2.0, 3.5], dtype=np.float32),  # Size 4
        np.arange(5, dtype=np.float32),  # Size 5
    ]

    world_size = 3
    sim = simp.make_simulator(world_size=world_size)
    mp.set_root_context(sim)

    try:
        for input_data in test_cases:
            # Need to run within simp context since this uses simp operations
            with sim:
                # Create MP objects from the input data
                x_mp = simp.constant((0,), input_data)
                y_mp = simp.constant((0,), input_data)

                # Execute the compiled function with MP objects
                result = mp.evaluate(traced_fn, x_mp, y_mp)

            # Verify the result - need to fetch from DriverVar
            expected = input_data + input_data
            result_values = mp.fetch(result)
            np.testing.assert_allclose(result_values[0], expected)
    finally:
        # Shutdown the cluster via the interpreter's reference
        if hasattr(sim, "_simp_cluster"):
            sim._simp_cluster.shutdown()
