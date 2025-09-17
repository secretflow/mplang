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

"""
Unit tests for the mplang.core.cluster module.
"""

import pytest

from mplang.core.cluster import (
    ClusterSpec,
    Device,
    Node,
    RuntimeInfo,
)


class TestRuntimeInfo:
    """Test cases for RuntimeInfo dataclass."""

    def test_basic_creation(self):
        """Test basic RuntimeInfo creation."""
        runtime = RuntimeInfo(
            version="2.0.0",
            platform="cpu",
            backends=["builtin", "spu"],
        )
        assert runtime.version == "2.0.0"
        assert runtime.platform == "cpu"
        assert runtime.backends == ["builtin", "spu"]
        assert runtime.extra == {}

    def test_creation_with_extra(self):
        """Test RuntimeInfo creation with extra fields."""
        runtime = RuntimeInfo(
            version="2.0.0",
            platform="cpu",
            backends=["builtin"],
            extra={"custom_field": "value", "number": 42},
        )
        assert runtime.extra == {"custom_field": "value", "number": 42}

    def test_to_dict(self):
        """Test RuntimeInfo.to_dict() method."""
        runtime = RuntimeInfo(
            version="2.0.0",
            platform="cpu",
            backends=["builtin", "spu"],
            extra={"custom": "value"},
        )
        expected = {
            "version": "2.0.0",
            "platform": "cpu",
            "backends": ["builtin", "spu"],
            "custom": "value",
        }
        assert runtime.to_dict() == expected

    def test_immutability(self):
        """Test that RuntimeInfo is immutable."""
        runtime = RuntimeInfo(version="2.0.0", platform="cpu", backends=["builtin"])
        with pytest.raises(AttributeError):
            runtime.version = "3.0.0"  # type: ignore


class TestPhysicalNode:
    """Test cases for PhysicalNode dataclass."""

    def test_basic_creation(self):
        """Test basic PhysicalNode creation."""
        runtime = RuntimeInfo(version="2.0.0", platform="cpu", backends=["builtin"])
        node = Node(
            name="alice_node",
            rank=0,
            endpoint="127.0.0.1:9001",
            runtime_info=runtime,
        )
        assert node.name == "alice_node"
        assert node.rank == 0
        assert node.endpoint == "127.0.0.1:9001"
        assert node.runtime_info == runtime

    def test_to_dict(self):
        """Test PhysicalNode.to_dict() method."""
        runtime = RuntimeInfo(version="2.0.0", platform="cpu", backends=["builtin"])
        node = Node(
            name="alice_node",
            rank=0,
            endpoint="127.0.0.1:9001",
            runtime_info=runtime,
        )
        expected = {
            "name": "alice_node",
            "endpoint": "127.0.0.1:9001",
            "runtime_info": {
                "version": "2.0.0",
                "platform": "cpu",
                "backends": ["builtin"],
            },
        }
        assert node.to_dict() == expected

    def test_immutability(self):
        """Test that PhysicalNode is immutable."""
        runtime = RuntimeInfo(version="2.0.0", platform="cpu", backends=["builtin"])
        node = Node(
            name="alice_node",
            rank=0,
            endpoint="127.0.0.1:9001",
            runtime_info=runtime,
        )
        with pytest.raises(AttributeError):
            node.name = "bob_node"  # type: ignore


class TestLogicalDevice:
    """Test cases for LogicalDevice dataclass."""

    def test_basic_creation(self):
        """Test basic LogicalDevice creation."""
        runtime = RuntimeInfo(version="2.0.0", platform="cpu", backends=["builtin"])
        node1 = Node("alice_node", 0, "127.0.0.1:9001", runtime)
        node2 = Node("bob_node", 1, "127.0.0.1:9002", runtime)

        device = Device(
            name="spu_device",
            kind="spu",
            members=[node1, node2],
            config={"protocol": "SEMI2K"},
        )
        assert device.name == "spu_device"
        assert device.kind == "spu"
        assert device.members == [node1, node2]
        assert device.config == {"protocol": "SEMI2K"}

    def test_member_ranks_property(self):
        """Test LogicalDevice.member_ranks property."""
        runtime = RuntimeInfo(version="2.0.0", platform="cpu", backends=["builtin"])
        node1 = Node("alice_node", 0, "127.0.0.1:9001", runtime)
        node2 = Node("bob_node", 1, "127.0.0.1:9002", runtime)
        node3 = Node("carol_node", 2, "127.0.0.1:9003", runtime)

        device = Device(
            name="spu_device",
            kind="spu",
            members=[node3, node1, node2],  # Unordered
        )
        assert device.member_ranks == [0, 1, 2]  # Should be sorted

    def test_to_dict(self):
        """Test LogicalDevice.to_dict() method."""
        runtime = RuntimeInfo(version="2.0.0", platform="cpu", backends=["builtin"])
        node1 = Node("alice_node", 0, "127.0.0.1:9001", runtime)
        node2 = Node("bob_node", 1, "127.0.0.1:9002", runtime)

        device = Device(
            name="spu_device",
            kind="spu",
            members=[node1, node2],
            config={"protocol": "SEMI2K"},
        )
        expected = {
            "kind": "spu",
            "members": ["alice_node", "bob_node"],
            "config": {"protocol": "SEMI2K"},
        }
        assert device.to_dict() == expected

    def test_empty_config_default(self):
        """Test LogicalDevice with default empty config."""
        runtime = RuntimeInfo(version="2.0.0", platform="cpu", backends=["builtin"])
        node = Node("alice_node", 0, "127.0.0.1:9001", runtime)

        device = Device(name="alice", kind="local", members=[node])
        assert device.config == {}


class TestClusterSpec:
    """Test cases for ClusterSpec dataclass."""

    def test_basic_creation(self):
        """Test basic ClusterSpec creation."""
        runtime = RuntimeInfo(version="2.0.0", platform="cpu", backends=["builtin"])
        node1 = Node("alice_node", 0, "127.0.0.1:9001", runtime)
        node2 = Node("bob_node", 1, "127.0.0.1:9002", runtime)

        device = Device("alice", "local", [node1])

        cluster = ClusterSpec(
            nodes={"alice_node": node1, "bob_node": node2},
            devices={"alice": device},
        )
        assert len(cluster.nodes) == 2
        assert len(cluster.devices) == 1

    def test_get_node(self):
        """Test ClusterSpec.get_node() method."""
        runtime = RuntimeInfo(version="2.0.0", platform="cpu", backends=["builtin"])
        node = Node("alice_node", 0, "127.0.0.1:9001", runtime)

        cluster = ClusterSpec(nodes={"alice_node": node}, devices={})
        retrieved_node = cluster.get_node("alice_node")
        assert retrieved_node == node

    def test_get_node_not_found(self):
        """Test ClusterSpec.get_node() with non-existent node."""
        cluster = ClusterSpec(nodes={}, devices={})
        with pytest.raises(KeyError):
            cluster.get_node("nonexistent")

    def test_get_device(self):
        """Test ClusterSpec.get_device() method."""
        runtime = RuntimeInfo(version="2.0.0", platform="cpu", backends=["builtin"])
        node = Node("alice_node", 0, "127.0.0.1:9001", runtime)
        device = Device("alice", "local", [node])

        cluster = ClusterSpec(nodes={"alice_node": node}, devices={"alice": device})
        retrieved_device = cluster.get_device("alice")
        assert retrieved_device == device

    def test_get_device_not_found(self):
        """Test ClusterSpec.get_device() with non-existent device."""
        cluster = ClusterSpec(nodes={}, devices={})
        with pytest.raises(KeyError):
            cluster.get_device("nonexistent")

    def test_get_node_by_rank(self):
        """Test ClusterSpec.get_node_by_rank() method."""
        runtime = RuntimeInfo(version="2.0.0", platform="cpu", backends=["builtin"])
        node1 = Node("alice_node", 0, "127.0.0.1:9001", runtime)
        node2 = Node("bob_node", 1, "127.0.0.1:9002", runtime)

        cluster = ClusterSpec(
            nodes={"alice_node": node1, "bob_node": node2}, devices={}
        )
        assert cluster.get_node_by_rank(0) == node1
        assert cluster.get_node_by_rank(1) == node2

    def test_get_node_by_rank_not_found(self):
        """Test ClusterSpec.get_node_by_rank() with non-existent rank."""
        cluster = ClusterSpec(nodes={}, devices={})
        with pytest.raises(KeyError, match="No Physical Node found with rank 99"):
            cluster.get_node_by_rank(99)

    def test_to_dict(self):
        """Test ClusterSpec.to_dict() method."""
        runtime = RuntimeInfo(version="2.0.0", platform="cpu", backends=["builtin"])
        node1 = Node("alice_node", 0, "127.0.0.1:9001", runtime)
        node2 = Node("bob_node", 1, "127.0.0.1:9002", runtime)
        device = Device("alice", "local", [node1])

        cluster = ClusterSpec(
            nodes={"alice_node": node1, "bob_node": node2},
            devices={"alice": device},
        )

        result = cluster.to_dict()
        assert "nodes" in result
        assert "devices" in result
        assert len(result["nodes"]) == 2
        assert len(result["devices"]) == 1


class TestFromDict:
    """Test cases for the ClusterSpec.from_dict() function."""

    def test_basic_parsing(self):
        """Test basic config parsing."""
        config = {
            "nodes": [
                {
                    "name": "alice_node",
                    "endpoint": "127.0.0.1:9001",
                    "runtime_info": {
                        "version": "2.0.0",
                        "platform": "cpu",
                        "backends": ["builtin", "spu"],
                    },
                }
            ],
            "devices": {
                "alice": {"kind": "local", "members": ["alice_node"], "config": {}}
            },
        }

        cluster = ClusterSpec.from_dict(config)
        assert len(cluster.nodes) == 1
        assert len(cluster.devices) == 1

        node = cluster.get_node("alice_node")
        assert node.name == "alice_node"
        assert node.rank == 0  # Implicit rank assignment
        assert node.endpoint == "127.0.0.1:9001"
        assert node.runtime_info.version == "2.0.0"

        device = cluster.get_device("alice")
        assert device.name == "alice"
        assert device.kind == "local"
        assert len(device.members) == 1

    def test_implicit_rank_assignment(self):
        """Test that ranks are assigned based on list order."""
        config = {
            "nodes": [
                {"name": "alice_node", "endpoint": "127.0.0.1:9001"},
                {"name": "bob_node", "endpoint": "127.0.0.1:9002"},
                {"name": "carol_node", "endpoint": "127.0.0.1:9003"},
            ],
            "devices": {},
        }

        cluster = ClusterSpec.from_dict(config)
        assert cluster.get_node("alice_node").rank == 0
        assert cluster.get_node("bob_node").rank == 1
        assert cluster.get_node("carol_node").rank == 2

    def test_runtime_info_extra_fields(self):
        """Test that extra runtime_info fields are preserved."""
        config = {
            "nodes": [
                {
                    "name": "alice_node",
                    "endpoint": "127.0.0.1:9001",
                    "runtime_info": {
                        "version": "2.0.0",
                        "platform": "cpu",
                        "backends": ["builtin"],
                        "custom_field": "custom_value",
                        "number_field": 42,
                    },
                }
            ],
            "devices": {},
        }

        cluster = ClusterSpec.from_dict(config)
        node = cluster.get_node("alice_node")
        assert node.runtime_info.extra == {
            "custom_field": "custom_value",
            "number_field": 42,
        }

    def test_missing_runtime_info_defaults(self):
        """Test that missing runtime_info fields get default values."""
        config = {
            "nodes": [{"name": "alice_node", "endpoint": "127.0.0.1:9001"}],
            "devices": {},
        }

        cluster = ClusterSpec.from_dict(config)
        node = cluster.get_node("alice_node")
        assert node.runtime_info.version == "N/A"
        assert node.runtime_info.platform == "N/A"
        assert node.runtime_info.backends == []

    def test_device_config_preservation(self):
        """Test that device configs are preserved with correct types."""
        config = {
            "nodes": [{"name": "alice_node", "endpoint": "127.0.0.1:9001"}],
            "devices": {
                "spu": {
                    "kind": "spu",
                    "members": ["alice_node"],
                    "config": {
                        "protocol": "SEMI2K",
                        "field": "FM64",
                        "fxp_fraction_bits": 18,
                        "enable_profile": True,
                    },
                }
            },
        }

        cluster = ClusterSpec.from_dict(config)
        device = cluster.get_device("spu")
        expected_config = {
            "protocol": "SEMI2K",
            "field": "FM64",
            "fxp_fraction_bits": 18,
            "enable_profile": True,
        }
        assert device.config == expected_config

    def test_missing_nodes_section(self):
        """Test error handling for missing 'nodes' section."""
        config = {"devices": {}}
        with pytest.raises(ValueError, match="must contain 'nodes' and 'devices'"):
            ClusterSpec.from_dict(config)

    def test_missing_devices_section(self):
        """Test error handling for missing 'devices' section."""
        config = {"nodes": []}
        with pytest.raises(ValueError, match="must contain 'nodes' and 'devices'"):
            ClusterSpec.from_dict(config)

    def test_duplicate_node_names(self):
        """Test error handling for duplicate node names."""
        config = {
            "nodes": [
                {"name": "alice_node", "endpoint": "127.0.0.1:9001"},
                {"name": "alice_node", "endpoint": "127.0.0.1:9002"},  # Duplicate
            ],
            "devices": {},
        }
        with pytest.raises(ValueError, match="Duplicate node name found: alice_node"):
            ClusterSpec.from_dict(config)

    def test_device_references_nonexistent_node(self):
        """Test error handling when device references non-existent node."""
        config = {
            "nodes": [{"name": "alice_node", "endpoint": "127.0.0.1:9001"}],
            "devices": {
                "spu": {
                    "kind": "spu",
                    "members": ["alice_node", "nonexistent_node"],
                }
            },
        }
        with pytest.raises(
            ValueError, match="Node 'nonexistent_node' in device 'spu' not defined"
        ):
            ClusterSpec.from_dict(config)

    def test_roundtrip_conversion(self):
        """Test that ClusterSpec.from_dict(cluster.to_dict()) produces equivalent results."""
        original_config = {
            "nodes": [
                {
                    "name": "alice_node",
                    "endpoint": "127.0.0.1:9001",
                    "runtime_info": {
                        "version": "2.0.0",
                        "platform": "cpu",
                        "backends": ["builtin", "spu"],
                        "custom": "value",
                    },
                },
                {
                    "name": "bob_node",
                    "endpoint": "127.0.0.1:9002",
                    "runtime_info": {
                        "version": "2.0.0",
                        "platform": "cpu",
                        "backends": ["builtin"],
                    },
                },
            ],
            "devices": {
                "alice": {"kind": "local", "members": ["alice_node"], "config": {}},
                "spu": {
                    "kind": "spu",
                    "members": ["alice_node", "bob_node"],
                    "config": {"protocol": "SEMI2K"},
                },
            },
        }

        # Parse original config
        cluster1 = ClusterSpec.from_dict(original_config)

        # Convert to dict and parse again
        dict_representation = cluster1.to_dict()
        cluster2 = ClusterSpec.from_dict(dict_representation)

        # They should be equivalent
        assert len(cluster1.nodes) == len(cluster2.nodes)
        assert len(cluster1.devices) == len(cluster2.devices)

        for name, node1 in cluster1.nodes.items():
            node2 = cluster2.nodes[name]
            assert node1.name == node2.name
            assert node1.rank == node2.rank
            assert node1.endpoint == node2.endpoint
            assert node1.runtime_info.version == node2.runtime_info.version

    def test_dict_key_name_mismatch_validation(self):
        """Test that ClusterSpec validates dict keys match object names."""
        runtime = RuntimeInfo(version="1.0", platform="cpu", backends=[])

        # Test node name mismatch
        node = Node(
            name="correct_name",
            rank=0,
            endpoint="localhost:5000",
            runtime_info=runtime,
        )

        with pytest.raises(
            ValueError,
            match=r"Node key 'wrong_key' does not match node\.name 'correct_name'",
        ):
            ClusterSpec(
                nodes={"wrong_key": node},
                devices={},
            )

        # Test device name mismatch
        device = Device(
            name="correct_device_name",
            kind="SPU",
            members=[node],
        )

        with pytest.raises(
            ValueError,
            match=r"Device key 'wrong_device_key' does not match device\.name 'correct_device_name'",
        ):
            ClusterSpec(
                nodes={"correct_name": node},
                devices={"wrong_device_key": device},
            )
