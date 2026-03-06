# Copyright 2026 Ant Group Co., Ltd.
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

"""Tests for make_compile_context (worker-free compilation)."""

import pytest

import mplang as mp


class TestMakeCompileContext:
    """Test mp.make_compile_context for worker-free compilation."""

    def test_basic_compile_with_world_size(self):
        """Compile a simple function using only world_size (no cluster_spec)."""
        ctx = mp.make_compile_context(2)

        def job():
            x = mp.put("P0", 100)
            y = mp.put("P1", 200)
            return x, y

        traced = mp.compile(job, context=ctx)
        assert traced is not None
        assert len(traced.graph.outputs) == 2

    def test_basic_compile_with_cluster_spec(self):
        """Compile using an explicit ClusterSpec."""
        cluster_spec = mp.ClusterSpec.from_dict({
            "nodes": [
                {"name": "node_0", "endpoint": "127.0.0.1:61930"},
                {"name": "node_1", "endpoint": "127.0.0.1:61931"},
            ],
            "devices": {
                "SP0": {
                    "kind": "SPU",
                    "members": ["node_0", "node_1"],
                    "config": {"protocol": "SEMI2K", "field": "FM128"},
                },
                "P0": {"kind": "PPU", "members": ["node_0"], "config": {}},
                "P1": {"kind": "PPU", "members": ["node_1"], "config": {}},
            },
        })

        ctx = mp.make_compile_context(cluster_spec=cluster_spec)

        def millionaire():
            x = mp.put("P0", 100)
            y = mp.put("P1", 200)
            z = mp.device("SP0")(lambda a, b: a < b)(x, y)
            r = mp.put("P0", z)
            return x, y, z, r

        traced = mp.compile(millionaire, context=ctx)
        assert traced is not None
        assert len(traced.graph.outputs) == 4
        ir_text = traced.compiler_ir()
        assert "put" in ir_text or "constant" in ir_text

    def test_compile_with_mp_function(self):
        """Compile a @mp.function wrapped function."""
        ctx = mp.make_compile_context(2)

        @mp.function
        def my_job():
            x = mp.put("P0", 42)
            return x

        traced = mp.compile(my_job, context=ctx)
        assert traced is not None
        assert len(traced.graph.operations) > 0

    def test_compile_context_as_context_manager(self):
        """Use compile context via 'with' statement."""
        ctx = mp.make_compile_context(2)

        def job():
            x = mp.put("P0", 100)
            return x

        with ctx:
            traced = mp.compile(job)

        assert traced is not None

    def test_requires_world_size_or_cluster_spec(self):
        """Error when neither world_size nor cluster_spec is provided."""
        with pytest.raises(ValueError, match="requires at least one"):
            mp.make_compile_context()

    def test_world_size_inferred_from_cluster_spec(self):
        """world_size is inferred from cluster_spec when not explicit."""
        cluster_spec = mp.ClusterSpec.simple(3)
        ctx = mp.make_compile_context(cluster_spec=cluster_spec)

        simp_state = ctx.get_dialect_state("simp")
        assert simp_state.world_size == 3

    def test_ir_matches_simulator(self):
        """IR from compile context matches IR from full simulator."""
        cluster_spec = mp.ClusterSpec.simple(2)

        def job():
            x = mp.put("P0", 100)
            y = mp.put("P1", 200)
            return x, y

        # Compile with lightweight context
        ctx_light = mp.make_compile_context(cluster_spec=cluster_spec)
        traced_light = mp.compile(job, context=ctx_light)

        # Compile with full simulator
        sim = mp.make_simulator(2, cluster_spec=cluster_spec)
        traced_sim = mp.compile(job, context=sim)

        # IR should be structurally identical
        assert len(traced_light.graph.operations) == len(traced_sim.graph.operations)
        assert len(traced_light.graph.outputs) == len(traced_sim.graph.outputs)
        assert traced_light.compiler_ir() == traced_sim.compiler_ir()
