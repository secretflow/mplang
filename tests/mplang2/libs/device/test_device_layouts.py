"""Tests for different device layouts in mplang2.

This module tests various cluster configurations including:
- 3-party computation with SPU (ABY3 protocol)
- Mixed PPU/SPU/TEE configurations
- Different SPU protocols (SEMI2K, ABY3, CHEETAH)
- Edge cases like single-party SPU, many-party scenarios
"""

import jax.numpy as jnp
import pytest

from mplang2.backends.simp_simulator import SimpSimulator
from mplang2.dialects import spu
from mplang2.edsl.context import pop_context, push_context
from mplang2.libs.device import (
    ClusterSpec,
    device,
    get_dev_attr,
    put,
    set_global_cluster,
)

# =============================================================================
# Fixtures for different cluster configurations
# =============================================================================


@pytest.fixture
def cluster_3pc_aby3():
    """3-party computation cluster with ABY3 protocol (like 3pc.yaml)."""
    config = {
        "nodes": [
            {"name": "node_0", "endpoint": "http://127.0.0.1:61920"},
            {"name": "node_1", "endpoint": "http://127.0.0.1:61921"},
            {"name": "node_2", "endpoint": "http://127.0.0.1:61922"},
            {"name": "node_3", "endpoint": "http://127.0.0.1:61923"},
        ],
        "devices": {
            "SP0": {
                "kind": "SPU",
                "members": ["node_0", "node_1", "node_2"],
                "config": {
                    "protocol": "ABY3",
                    "field": "FM64",
                },
            },
            "P0": {"kind": "PPU", "members": ["node_0"], "config": {}},
            "P1": {"kind": "PPU", "members": ["node_3"], "config": {}},
            "TEE0": {"kind": "TEE", "members": ["node_1"], "config": {}},
        },
    }
    return ClusterSpec.from_dict(config)


@pytest.fixture
def cluster_2pc_semi2k():
    """2-party computation cluster with SEMI2K protocol."""
    config = {
        "nodes": [
            {"name": "alice", "endpoint": "http://127.0.0.1:9000"},
            {"name": "bob", "endpoint": "http://127.0.0.1:9001"},
        ],
        "devices": {
            "SP0": {
                "kind": "SPU",
                "members": ["alice", "bob"],
                "config": {
                    "protocol": "SEMI2K",
                    "field": "FM128",
                },
            },
            "P_alice": {"kind": "PPU", "members": ["alice"], "config": {}},
            "P_bob": {"kind": "PPU", "members": ["bob"], "config": {}},
        },
    }
    return ClusterSpec.from_dict(config)


@pytest.fixture
def cluster_4pc_cheetah():
    """4-party computation cluster with CHEETAH protocol."""
    config = {
        "nodes": [
            {"name": "p0", "endpoint": "http://127.0.0.1:8000"},
            {"name": "p1", "endpoint": "http://127.0.0.1:8001"},
            {"name": "p2", "endpoint": "http://127.0.0.1:8002"},
            {"name": "p3", "endpoint": "http://127.0.0.1:8003"},
        ],
        "devices": {
            "SP0": {
                "kind": "SPU",
                "members": ["p0", "p1"],
                "config": {
                    "protocol": "CHEETAH",
                    "field": "FM64",
                },
            },
            "SP1": {
                "kind": "SPU",
                "members": ["p2", "p3"],
                "config": {
                    "protocol": "SEMI2K",
                    "field": "FM128",
                },
            },
            "P0": {"kind": "PPU", "members": ["p0"], "config": {}},
            "P1": {"kind": "PPU", "members": ["p1"], "config": {}},
            "P2": {"kind": "PPU", "members": ["p2"], "config": {}},
            "P3": {"kind": "PPU", "members": ["p3"], "config": {}},
        },
    }
    return ClusterSpec.from_dict(config)


@pytest.fixture
def cluster_ppu_only():
    """PPU-only cluster (no secure devices)."""
    config = {
        "nodes": [
            {"name": "node_0", "endpoint": "http://127.0.0.1:5000"},
            {"name": "node_1", "endpoint": "http://127.0.0.1:5001"},
        ],
        "devices": {
            "P0": {"kind": "PPU", "members": ["node_0"], "config": {}},
            "P1": {"kind": "PPU", "members": ["node_1"], "config": {}},
        },
    }
    return ClusterSpec.from_dict(config)


# =============================================================================
# Test ClusterSpec parsing and validation
# =============================================================================


class TestClusterSpecParsing:
    """Tests for ClusterSpec parsing from dict/yaml format."""

    def test_parse_3pc_aby3(self, cluster_3pc_aby3):
        """Test parsing 3-party ABY3 cluster config."""
        cluster = cluster_3pc_aby3

        # Check nodes
        assert len(cluster.nodes) == 4
        assert "node_0" in cluster.nodes
        assert cluster.nodes["node_0"].rank == 0
        assert cluster.nodes["node_3"].rank == 3

        # Check devices
        assert len(cluster.devices) == 4
        assert "SP0" in cluster.devices
        assert "P0" in cluster.devices
        assert "P1" in cluster.devices
        assert "TEE0" in cluster.devices

        # Check SPU config
        sp0 = cluster.devices["SP0"]
        assert sp0.kind == "SPU"
        assert len(sp0.members) == 3
        assert sp0.config["protocol"] == "ABY3"
        assert sp0.config["field"] == "FM64"

    def test_parse_2pc_semi2k(self, cluster_2pc_semi2k):
        """Test parsing 2-party SEMI2K cluster config."""
        cluster = cluster_2pc_semi2k

        assert len(cluster.nodes) == 2
        assert len(cluster.devices) == 3

        sp0 = cluster.devices["SP0"]
        assert len(sp0.members) == 2
        assert sp0.config["protocol"] == "SEMI2K"

    def test_parse_multi_spu(self, cluster_4pc_cheetah):
        """Test parsing cluster with multiple SPU devices."""
        cluster = cluster_4pc_cheetah

        assert len(cluster.devices) == 6  # 2 SPU + 4 PPU

        sp0 = cluster.devices["SP0"]
        sp1 = cluster.devices["SP1"]

        assert sp0.config["protocol"] == "CHEETAH"
        assert sp1.config["protocol"] == "SEMI2K"

    def test_get_devices_by_kind(self, cluster_3pc_aby3):
        """Test filtering devices by kind."""
        cluster = cluster_3pc_aby3

        spu_devs = cluster.get_devices_by_kind("SPU")
        ppu_devs = cluster.get_devices_by_kind("PPU")
        tee_devs = cluster.get_devices_by_kind("TEE")

        assert len(spu_devs) == 1
        assert len(ppu_devs) == 2
        assert len(tee_devs) == 1


# =============================================================================
# Test SPUConfig
# =============================================================================


class TestSPUConfig:
    """Tests for SPUConfig dataclass."""

    def test_default_config(self):
        """Test default SPUConfig values."""
        config = spu.SPUConfig()
        assert config.protocol == "SEMI2K"
        assert config.field == "FM128"
        assert config.fxp_fraction_bits == 18

    def test_from_dict(self):
        """Test SPUConfig.from_dict."""
        config = spu.SPUConfig.from_dict({
            "protocol": "ABY3",
            "field": "FM64",
            "fxp_fraction_bits": 20,
        })
        assert config.protocol == "ABY3"
        assert config.field == "FM64"
        assert config.fxp_fraction_bits == 20

    def test_from_dict_partial(self):
        """Test SPUConfig.from_dict with partial config."""
        config = spu.SPUConfig.from_dict({"protocol": "CHEETAH"})
        assert config.protocol == "CHEETAH"
        assert config.field == "FM128"  # default
        assert config.fxp_fraction_bits == 18  # default

    def test_frozen(self):
        """Test that SPUConfig is immutable."""
        config = spu.SPUConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.protocol = "ABY3"


# =============================================================================
# Test device operations with different layouts
# =============================================================================


class TestDeviceOperations3PC:
    """Test device operations with 3-party ABY3 cluster."""

    @pytest.fixture(autouse=True)
    def setup(self, cluster_3pc_aby3):
        """Set up cluster and contexts."""
        set_global_cluster(cluster_3pc_aby3)

        # SimpSimulator is itself an Interpreter subclass (via SimpHost)
        # It handles both tracing and execution
        sim = SimpSimulator(world_size=4)
        push_context(sim)

        yield

        pop_context()

    def test_put_to_ppu(self):
        """Test putting data to PPU devices."""
        x = jnp.array([1, 2, 3])

        x_p0 = put("P0", x)
        assert get_dev_attr(x_p0) == "P0"

        x_p1 = put("P1", x)
        assert get_dev_attr(x_p1) == "P1"

    def test_put_to_spu(self):
        """Test putting data to SPU device."""
        x = jnp.array([1.0, 2.0, 3.0])
        x_sp0 = put("SP0", x)
        assert get_dev_attr(x_sp0) == "SP0"

    def test_transfer_ppu_to_ppu(self):
        """Test data transfer between PPU devices."""
        x = put("P0", jnp.array([1, 2, 3]))
        x_p1 = put("P1", x)
        assert get_dev_attr(x_p1) == "P1"

    def test_transfer_ppu_to_spu(self):
        """Test data transfer from PPU to SPU (seal)."""
        x = put("P0", jnp.array([1.0, 2.0, 3.0]))
        x_sp0 = put("SP0", x)
        assert get_dev_attr(x_sp0) == "SP0"

    def test_transfer_spu_to_ppu(self):
        """Test data transfer from SPU to PPU (reveal)."""
        x = put("SP0", jnp.array([1.0, 2.0, 3.0]))
        x_p0 = put("P0", x)
        assert get_dev_attr(x_p0) == "P0"

    def test_device_decorator_explicit(self):
        """Test @device decorator with explicit device."""

        @device("SP0")
        def secure_add(a, b):
            return a + b

        x = put("SP0", jnp.array([1.0, 2.0]))
        y = put("SP0", jnp.array([3.0, 4.0]))

        z = secure_add(x, y)
        assert get_dev_attr(z) == "SP0"

    def test_device_decorator_auto_infer(self):
        """Test @device decorator with auto device inference."""

        @device
        def add(a, b):
            return a + b

        # Both on P0 -> execute on P0
        x = put("P0", jnp.array([1, 2]))
        y = put("P0", jnp.array([3, 4]))
        z = add(x, y)
        assert get_dev_attr(z) == "P0"

        # Both on SP0 -> execute on SP0
        x2 = put("SP0", jnp.array([1.0, 2.0]))
        y2 = put("SP0", jnp.array([3.0, 4.0]))
        z2 = add(x2, y2)
        assert get_dev_attr(z2) == "SP0"

    def test_auto_transfer_ppu_to_spu(self):
        """Test automatic data transfer when args are on different devices."""

        @device
        def add(a, b):
            return a + b

        x = put("P0", jnp.array([1.0, 2.0]))
        y = put("SP0", jnp.array([3.0, 4.0]))

        # Should auto-transfer x to SP0 and execute there
        z = add(x, y)
        assert get_dev_attr(z) == "SP0"


class TestDeviceOperations2PC:
    """Test device operations with 2-party SEMI2K cluster."""

    @pytest.fixture(autouse=True)
    def setup(self, cluster_2pc_semi2k):
        """Set up cluster and contexts."""
        set_global_cluster(cluster_2pc_semi2k)

        sim = SimpSimulator(world_size=2)
        push_context(sim)

        yield

        pop_context()

    def test_2pc_basic_operations(self):
        """Test basic operations in 2-party setting."""
        x = put("P_alice", jnp.array([1.0, 2.0]))
        y = put("P_bob", jnp.array([3.0, 4.0]))

        # Transfer to SPU
        x_sp = put("SP0", x)
        y_sp = put("SP0", y)

        assert get_dev_attr(x_sp) == "SP0"
        assert get_dev_attr(y_sp) == "SP0"

    def test_2pc_secure_computation(self):
        """Test secure computation in 2-party setting."""

        @device("SP0")
        def secure_multiply(a, b):
            return a * b

        x = put("SP0", jnp.array([2.0, 3.0]))
        y = put("SP0", jnp.array([4.0, 5.0]))

        z = secure_multiply(x, y)
        assert get_dev_attr(z) == "SP0"


class TestDeviceOperationsPPUOnly:
    """Test device operations with PPU-only cluster."""

    @pytest.fixture(autouse=True)
    def setup(self, cluster_ppu_only):
        """Set up cluster and contexts."""
        set_global_cluster(cluster_ppu_only)

        sim = SimpSimulator(world_size=2)
        push_context(sim)

        yield

        pop_context()

    def test_ppu_only_transfer(self):
        """Test data transfer in PPU-only cluster."""
        x = put("P0", jnp.array([1, 2, 3]))
        x_p1 = put("P1", x)

        assert get_dev_attr(x_p1) == "P1"

    def test_ppu_only_computation(self):
        """Test computation on PPU devices."""

        @device("P0")
        def double(x):
            return x * 2

        x = put("P0", jnp.array([1, 2, 3]))
        y = double(x)

        assert get_dev_attr(y) == "P0"

    def test_ppu_only_ambiguous_device(self):
        """Test error when device inference is ambiguous."""

        @device
        def add(a, b):
            return a + b

        x = put("P0", jnp.array([1]))
        y = put("P1", jnp.array([2]))

        # Should raise because both are PPU and ambiguous
        with pytest.raises(ValueError, match="multiple PPU devices"):
            add(x, y)


# =============================================================================
# Test error handling
# =============================================================================


class TestDeviceErrors:
    """Test error handling in device operations."""

    @pytest.fixture(autouse=True)
    def setup(self, cluster_3pc_aby3):
        """Set up cluster and contexts."""
        set_global_cluster(cluster_3pc_aby3)

        sim = SimpSimulator(world_size=4)
        push_context(sim)

        yield

        pop_context()

    def test_invalid_device_id(self):
        """Test error when putting to invalid device."""
        x = jnp.array([1, 2, 3])

        with pytest.raises(ValueError, match="not found"):
            put("INVALID_DEVICE", x)

    def test_tee_not_implemented(self):
        """Test that TEE operations are not yet implemented."""
        x = put("P0", jnp.array([1, 2, 3]))

        # TEE0 exists but transfer is not implemented
        with pytest.raises(NotImplementedError):
            put("TEE0", x)


class TestMultipleSPUs:
    """Test cluster with multiple SPU devices."""

    @pytest.fixture(autouse=True)
    def setup(self, cluster_4pc_cheetah):
        """Set up cluster and contexts."""
        set_global_cluster(cluster_4pc_cheetah)

        sim = SimpSimulator(world_size=4)
        push_context(sim)

        yield

        pop_context()

    def test_multiple_spu_explicit(self):
        """Test explicit device specification with multiple SPUs."""

        @device("SP0")
        def add_sp0(a, b):
            return a + b

        @device("SP1")
        def add_sp1(a, b):
            return a + b

        x0 = put("SP0", jnp.array([1.0, 2.0]))
        y0 = put("SP0", jnp.array([3.0, 4.0]))
        z0 = add_sp0(x0, y0)
        assert get_dev_attr(z0) == "SP0"

        x1 = put("SP1", jnp.array([1.0, 2.0]))
        y1 = put("SP1", jnp.array([3.0, 4.0]))
        z1 = add_sp1(x1, y1)
        assert get_dev_attr(z1) == "SP1"

    def test_multiple_spu_ambiguous(self):
        """Test error when device inference is ambiguous with multiple SPUs."""

        @device
        def add(a, b):
            return a + b

        x = put("SP0", jnp.array([1.0, 2.0]))
        y = put("SP1", jnp.array([3.0, 4.0]))

        # Should raise because both are different SPUs
        with pytest.raises(ValueError, match="multiple SPU devices"):
            add(x, y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
