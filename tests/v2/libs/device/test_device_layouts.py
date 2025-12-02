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

"""Tests for device operations across different cluster layouts.

This module tests:
- ClusterSpec parsing and validation
- SPUConfig dataclass behavior
- Device operations (put, transfer, computation) across various layouts
- Multi-SPU configurations
"""

import jax.numpy as jnp
import pytest

from mplang.v2.dialects import spu
from mplang.v2.libs.device import (
    DeviceInferenceError,
    device,
    get_dev_attr,
    put,
)

# =============================================================================
# ClusterSpec Parsing
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
        assert len(cluster.devices) == 5  # SP0 + P0, P1, P2 + TEE0
        assert "SP0" in cluster.devices
        assert "P0" in cluster.devices
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
        assert len(cluster.devices) == 3  # SP0 + P_alice + P_bob

        sp0 = cluster.devices["SP0"]
        assert len(sp0.members) == 2
        assert sp0.config["protocol"] == "SEMI2K"

    def test_parse_multi_spu(self, cluster_4pc_multi_spu):
        """Test parsing cluster with multiple SPU devices."""
        cluster = cluster_4pc_multi_spu

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
        assert len(ppu_devs) == 3  # P0, P1, P2
        assert len(tee_devs) == 1


# =============================================================================
# SPUConfig
# =============================================================================


class TestSPUConfig:
    """Tests for SPUConfig dataclass."""

    def test_default_config(self):
        """Test default SPUConfig values."""
        config = spu.SPUConfig()
        assert config.protocol == "SEMI2K"
        assert config.field == "FM128"
        assert config.fxp_fraction_bits == 18

    def test_from_dict_full(self):
        """Test SPUConfig.from_dict with full config."""
        config = spu.SPUConfig.from_dict({
            "protocol": "ABY3",
            "field": "FM64",
            "fxp_fraction_bits": 20,
        })
        assert config.protocol == "ABY3"
        assert config.field == "FM64"
        assert config.fxp_fraction_bits == 20

    def test_from_dict_partial(self):
        """Test SPUConfig.from_dict with partial config uses defaults."""
        config = spu.SPUConfig.from_dict({"protocol": "CHEETAH"})
        assert config.protocol == "CHEETAH"
        assert config.field == "FM128"  # default
        assert config.fxp_fraction_bits == 18  # default

    def test_frozen_immutable(self):
        """Test that SPUConfig is immutable (frozen dataclass)."""
        config = spu.SPUConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.protocol = "ABY3"


# =============================================================================
# Device Operations - 3PC Layout
# =============================================================================


class TestDeviceOperations3PC:
    """Test device operations with 3-party ABY3 cluster."""

    def test_put_to_ppu(self, ctx_3pc):
        """Test putting data to PPU devices."""
        x = jnp.array([1, 2, 3])

        x_p0 = put("P0", x)
        assert get_dev_attr(x_p0) == "P0"

        x_p1 = put("P1", x)
        assert get_dev_attr(x_p1) == "P1"

    def test_put_to_spu(self, ctx_3pc):
        """Test putting data to SPU device."""
        x = jnp.array([1.0, 2.0, 3.0])
        x_sp0 = put("SP0", x)
        assert get_dev_attr(x_sp0) == "SP0"

    def test_transfer_ppu_to_ppu(self, ctx_3pc):
        """Test data transfer between PPU devices."""
        x = put("P0", jnp.array([1, 2, 3]))
        x_p1 = put("P1", x)
        assert get_dev_attr(x_p1) == "P1"

    def test_transfer_ppu_to_spu(self, ctx_3pc):
        """Test data transfer from PPU to SPU (seal)."""
        x = put("P0", jnp.array([1.0, 2.0, 3.0]))
        x_sp0 = put("SP0", x)
        assert get_dev_attr(x_sp0) == "SP0"

    def test_transfer_spu_to_ppu(self, ctx_3pc):
        """Test data transfer from SPU to PPU (reveal)."""
        x = put("SP0", jnp.array([1.0, 2.0, 3.0]))
        x_p0 = put("P0", x)
        assert get_dev_attr(x_p0) == "P0"

    def test_device_decorator_explicit(self, ctx_3pc):
        """Test @device decorator with explicit device."""

        @device("SP0")
        def secure_add(a, b):
            return a + b

        x = put("SP0", jnp.array([1.0, 2.0]))
        y = put("SP0", jnp.array([3.0, 4.0]))

        z = secure_add(x, y)
        assert get_dev_attr(z) == "SP0"

    def test_device_decorator_auto_infer_ppu(self, ctx_3pc):
        """Test @device() auto-inference for PPU arguments."""

        @device()
        def add(a, b):
            return a + b

        x = put("P0", jnp.array([1, 2]))
        y = put("P0", jnp.array([3, 4]))
        z = add(x, y)
        assert get_dev_attr(z) == "P0"

    def test_device_decorator_auto_infer_spu(self, ctx_3pc):
        """Test @device() auto-inference for SPU arguments."""

        @device()
        def add(a, b):
            return a + b

        x = put("SP0", jnp.array([1.0, 2.0]))
        y = put("SP0", jnp.array([3.0, 4.0]))
        z = add(x, y)
        assert get_dev_attr(z) == "SP0"

    def test_auto_transfer_ppu_to_spu(self, ctx_3pc):
        """Test automatic transfer when args are on different devices."""

        @device()
        def add(a, b):
            return a + b

        x = put("P0", jnp.array([1.0, 2.0]))
        y = put("SP0", jnp.array([3.0, 4.0]))

        # Should auto-transfer x to SP0 and execute there
        z = add(x, y)
        assert get_dev_attr(z) == "SP0"


# =============================================================================
# Device Operations - 2PC Layout
# =============================================================================


class TestDeviceOperations2PC:
    """Test device operations with 2-party SEMI2K cluster."""

    def test_basic_operations(self, ctx_2pc):
        """Test basic put operations in 2-party setting."""
        x = put("P_alice", jnp.array([1.0, 2.0]))
        y = put("P_bob", jnp.array([3.0, 4.0]))

        # Transfer to SPU
        x_sp = put("SP0", x)
        y_sp = put("SP0", y)

        assert get_dev_attr(x_sp) == "SP0"
        assert get_dev_attr(y_sp) == "SP0"

    def test_secure_computation(self, ctx_2pc):
        """Test secure computation in 2-party setting."""

        @device("SP0")
        def secure_multiply(a, b):
            return a * b

        x = put("SP0", jnp.array([2.0, 3.0]))
        y = put("SP0", jnp.array([4.0, 5.0]))

        z = secure_multiply(x, y)
        assert get_dev_attr(z) == "SP0"


# =============================================================================
# Device Operations - PPU Only Layout
# =============================================================================


class TestDeviceOperationsPPUOnly:
    """Test device operations with PPU-only cluster."""

    def test_ppu_transfer(self, ctx_ppu_only):
        """Test data transfer in PPU-only cluster."""
        x = put("P0", jnp.array([1, 2, 3]))
        x_p1 = put("P1", x)

        assert get_dev_attr(x_p1) == "P1"

    def test_ppu_computation(self, ctx_ppu_only):
        """Test computation on PPU devices."""

        @device("P0")
        def double(x):
            return x * 2

        x = put("P0", jnp.array([1, 2, 3]))
        y = double(x)

        assert get_dev_attr(y) == "P0"

    def test_ambiguous_device_error(self, ctx_ppu_only):
        """Test error when device inference is ambiguous."""

        @device()
        def add(a, b):
            return a + b

        x = put("P0", jnp.array([1]))
        y = put("P1", jnp.array([2]))

        with pytest.raises(DeviceInferenceError, match="multiple PPU devices"):
            add(x, y)


# =============================================================================
# Multi-SPU Operations
# =============================================================================


class TestMultipleSPUs:
    """Test cluster with multiple SPU devices."""

    def test_explicit_spu_selection(self, ctx_4pc):
        """Test explicit device specification with multiple SPUs."""

        @device("SP0")
        def add_sp0(a, b):
            return a + b

        @device("SP1")
        def add_sp1(a, b):
            return a + b

        # Operations on SP0
        x0 = put("SP0", jnp.array([1.0, 2.0]))
        y0 = put("SP0", jnp.array([3.0, 4.0]))
        z0 = add_sp0(x0, y0)
        assert get_dev_attr(z0) == "SP0"

        # Operations on SP1
        x1 = put("SP1", jnp.array([1.0, 2.0]))
        y1 = put("SP1", jnp.array([3.0, 4.0]))
        z1 = add_sp1(x1, y1)
        assert get_dev_attr(z1) == "SP1"

    def test_ambiguous_spu_error(self, ctx_4pc):
        """Test error when device inference is ambiguous with multiple SPUs."""

        @device()
        def add(a, b):
            return a + b

        x = put("SP0", jnp.array([1.0, 2.0]))
        y = put("SP1", jnp.array([3.0, 4.0]))

        with pytest.raises(DeviceInferenceError, match="multiple SPU devices"):
            add(x, y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
