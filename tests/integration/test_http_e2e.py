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

"""End-to-end functional tests for HttpDriver evaluate and fetch operations.

This version uses the shared `http_servers` fixture for process management.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import mplang
import mplang.simp as simp
from mplang.core.cluster import ClusterSpec, Device, Node, RuntimeInfo
from tests.utils.server_fixtures import http_servers  # noqa: F401


def create_e2e_cluster_spec(
    node_addrs: dict[str, str], spu_nodes: list[str]
) -> ClusterSpec:
    """Create a ClusterSpec for e2e testing with specific SPU nodes."""
    nodes = {}

    # Create nodes for all parties
    for node_id, addr in node_addrs.items():
        # Extract rank from node_id (e.g., "P0" -> 0)
        rank = int(node_id[1:])
        nodes[f"node{rank}"] = Node(
            name=f"node{rank}",
            rank=rank,
            endpoint=addr,
            runtime_info=RuntimeInfo(
                version="test",
                platform="test",
                backends=["__all__"],
            ),
        )

    # Create local devices for each node
    local_devices = {}
    for _node_name, node in nodes.items():
        local_devices[f"local_{node.rank}"] = Device(
            name=f"local_{node.rank}",
            kind="local",
            members=[node],
        )

    # Create SPU device with specified nodes
    spu_node_ranks = [int(node_id[1:]) for node_id in spu_nodes]
    spu_members = [nodes[f"node{rank}"] for rank in spu_node_ranks]

    spu_device = Device(
        name="SPU_0",
        kind="SPU",
        members=spu_members,
        config={
            "protocol": "SEMI2K",
            "field": "FM128",
        },
    )

    devices = {**local_devices, "SPU_0": spu_device}

    return ClusterSpec(nodes=nodes, devices=devices)


@pytest.fixture
def http_driver(http_servers):  # type: ignore  # noqa: F811
    node_ids = ["P0", "P1", "P2", "P3", "P4"]
    node_addrs = dict(zip(node_ids, http_servers.addresses, strict=True))
    cluster_spec = create_e2e_cluster_spec(node_addrs, ["P1", "P2", "P3"])
    return mplang.Driver(cluster_spec)


@pytest.mark.parametrize("http_servers", [5], indirect=True)
def test_simple_addition_e2e(http_driver):
    """Test simple addition computation using HttpDriver with 5 parties."""
    # Create test data
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    def constant_add_fn(x, y):
        x_const = simp.constant(x)
        y_const = simp.constant(y)
        # Use JAX function for addition
        return simp.run(jnp.add)(x_const, y_const)

    # Evaluate the computation
    result = mplang.evaluate(http_driver, constant_add_fn, x, y)

    # Fetch the result
    fetched = mplang.fetch(http_driver, result)

    # Verify result - all 5 parties should get the same result
    expected = x + y
    for i, actual in enumerate(fetched):
        assert np.allclose(actual, expected), (
            f"Mismatch at rank {i}: {actual} vs {expected}"
        )


@pytest.mark.parametrize("http_servers", [5], indirect=True)
def test_secure_comparison_e2e(http_driver):
    """Test secure comparison computation using HttpDriver with 5 parties."""
    # Create test data - data comes from different parties
    x = np.array([5.0])  # From P0
    y = np.array([3.0])  # From P4

    @mplang.function
    def secure_compare():
        # Create constants
        x_const = simp.constant(x)
        y_const = simp.constant(y)

        # Seal them for secure computation - data from P0 and P4
        x_sealed = simp.sealFrom(x_const, 0)  # P0 provides data
        y_sealed = simp.sealFrom(y_const, 4)  # P4 provides data

        # Perform secure comparison
        import jax.numpy as jnp

        result = simp.srun(lambda a, b: jnp.where(a > b, True, False))(
            x_sealed, y_sealed
        )

        # Reveal the result
        revealed = simp.reveal(result)
        return revealed

    # Evaluate the computation
    result = mplang.evaluate(http_driver, secure_compare)

    # Fetch the result
    fetched = mplang.fetch(http_driver, result)

    # Verify result - all 5 parties should get the same result
    expected = x > y  # Should be True since 5.0 > 3.0
    for i in range(5):
        assert fetched[i] == expected, (
            f"Mismatch at rank {i}: {fetched[i]} vs {expected}"
        )


@pytest.mark.parametrize("http_servers", [5], indirect=True)
def test_three_way_comparison_e2e(http_driver):
    """Test multi-party comparison (millionaire problem) using HttpDriver with 5 parties."""
    # Create test data - wealth from different parties
    wealth_a = np.array([50.0])  # P0's wealth
    wealth_b = np.array([30.0])  # P2's wealth
    wealth_c = np.array([70.0])  # P4's wealth

    @mplang.function
    def millionaire_problem():
        # Create constants for each party's wealth
        wealth_a_const = simp.constant(wealth_a)
        wealth_b_const = simp.constant(wealth_b)
        wealth_c_const = simp.constant(wealth_c)

        # Seal the wealth values for secure computation
        wealth_a_sealed = simp.sealFrom(wealth_a_const, 0)  # P0 provides wealth
        wealth_b_sealed = simp.sealFrom(wealth_b_const, 2)  # P2 provides wealth
        wealth_c_sealed = simp.sealFrom(wealth_c_const, 4)  # P4 provides wealth

        # Perform secure comparison to find the richest
        import jax.numpy as jnp

        # Find who has the maximum wealth
        max_ab = simp.srun(jnp.maximum)(wealth_a_sealed, wealth_b_sealed)
        max_wealth = simp.srun(jnp.maximum)(max_ab, wealth_c_sealed)

        # Check if each party is the richest
        a_is_richest = simp.srun(lambda a, max_w: a >= max_w)(
            wealth_a_sealed, max_wealth
        )
        b_is_richest = simp.srun(lambda b, max_w: b >= max_w)(
            wealth_b_sealed, max_wealth
        )
        c_is_richest = simp.srun(lambda c, max_w: c >= max_w)(
            wealth_c_sealed, max_wealth
        )

        # Reveal the results
        a_result = simp.reveal(a_is_richest)
        b_result = simp.reveal(b_is_richest)
        c_result = simp.reveal(c_is_richest)

        return a_result, b_result, c_result

    # Evaluate the computation
    a_result, b_result, c_result = mplang.evaluate(http_driver, millionaire_problem)

    # Fetch the results
    fetched_a = mplang.fetch(http_driver, a_result)
    fetched_b = mplang.fetch(http_driver, b_result)
    fetched_c = mplang.fetch(http_driver, c_result)

    # Verify results - P4 (wealth_c = 70.0) should be the richest
    for i in range(5):
        assert not fetched_a[i], f"P0 should not be richest at rank {i}: {fetched_a[i]}"
        assert not fetched_b[i], f"P2 should not be richest at rank {i}: {fetched_b[i]}"
        assert fetched_c[i], f"P4 should be richest at rank {i}: {fetched_c[i]}"


@pytest.mark.parametrize("http_servers", [5], indirect=True)
def test_multiple_operations_e2e(http_driver):
    """Test multiple operations in sequence using HttpDriver with 5 parties."""
    # Test data - from different parties
    a = np.array([10.0, 20.0])  # From P0
    b = np.array([5.0, 15.0])  # From P3

    @mplang.function
    def multi_operations():
        # Create constants
        a_const = simp.constant(a)
        b_const = simp.constant(b)

        # Seal for secure computation - data from P0 and P3
        a_sealed = simp.sealFrom(a_const, 0)  # P0 provides data
        b_sealed = simp.sealFrom(b_const, 3)  # P3 provides data

        # Multiple operations
        import jax.numpy as jnp

        # Addition
        sum_result = simp.srun(jnp.add)(a_sealed, b_sealed)

        # Multiplication
        mul_result = simp.srun(jnp.multiply)(a_sealed, b_sealed)

        # Comparison
        cmp_result = simp.srun(lambda x, y: x > y)(a_sealed, b_sealed)

        # Reveal all results
        sum_revealed = simp.reveal(sum_result)
        mul_revealed = simp.reveal(mul_result)
        cmp_revealed = simp.reveal(cmp_result)

        return sum_revealed, mul_revealed, cmp_revealed

    # Evaluate the computation
    sum_result, mul_result, cmp_result = mplang.evaluate(http_driver, multi_operations)

    # Fetch the results
    fetched_sum = mplang.fetch(http_driver, sum_result)
    fetched_mul = mplang.fetch(http_driver, mul_result)
    fetched_cmp = mplang.fetch(http_driver, cmp_result)

    # Verify results
    expected_sum = a + b
    expected_mul = a * b
    expected_cmp = a > b

    # All 5 parties should get the same results
    for i in range(5):
        assert np.allclose(fetched_sum[i], expected_sum), f"Sum mismatch at rank {i}"
        assert np.allclose(fetched_mul[i], expected_mul), f"Mul mismatch at rank {i}"
        assert np.array_equal(fetched_cmp[i], expected_cmp), f"Cmp mismatch at rank {i}"
