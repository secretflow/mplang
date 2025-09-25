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
This module provides the formal data structures and parsing logic for the
MPLang cluster configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RuntimeInfo:
    """
    Structured representation of a Physical Node's runtime capabilities.
    """

    version: str
    platform: str
    backends: list[str]

    # A catch-all for any other custom or future properties
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert RuntimeInfo to a dictionary."""
        result = {
            "version": self.version,
            "platform": self.platform,
            "backends": self.backends,
        }
        result.update(self.extra)
        return result


@dataclass(frozen=True)
class Node:
    """
    Represents a single physical node (PN) in the cluster.
    This is an immutable description of a compute resource.
    """

    name: str
    rank: int
    endpoint: str
    runtime_info: RuntimeInfo

    def to_dict(self) -> dict[str, Any]:
        """Convert PhysicalNode to a dictionary."""
        return {
            "name": self.name,
            "endpoint": self.endpoint,
            "runtime_info": self.runtime_info.to_dict(),
        }


@dataclass(frozen=True)
class Device:
    """
    Represents a logical device (LD), which is a user-facing computational entity.
    It is composed of one or more Physical Nodes.
    """

    name: str
    kind: str
    members: list[Node]
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def member_ranks(self) -> list[int]:
        """Returns the ranks of the member PNs."""
        return sorted([node.rank for node in self.members])

    def to_dict(self) -> dict[str, Any]:
        """Convert LogicalDevice to a dictionary."""
        return {
            "kind": self.kind,
            "members": [node.name for node in self.members],
            "config": self.config,
        }


@dataclass(frozen=True)
class ClusterSpec:
    """
    The formal, validated representation of the entire cluster.
    This object is the "first-class citizen" representing the cluster topology.
    """

    nodes: dict[str, Node]
    devices: dict[str, Device]

    def __post_init__(self) -> None:
        for key, node in self.nodes.items():
            if key != node.name:
                raise ValueError(
                    f"Node key '{key}' does not match node.name '{node.name}'"
                )

        for key, device in self.devices.items():
            if key != device.name:
                raise ValueError(
                    f"Device key '{key}' does not match device.name '{device.name}'"
                )

        # check all device members are valid nodes
        node_names = set(self.nodes.keys())
        for device in self.devices.values():
            for member in device.members:
                if member.name not in node_names:
                    raise ValueError(
                        f"Device '{device.name}' has member '{member.name}' "
                        "which is not defined in nodes"
                    )

        # ensure local devices have exactly one member
        for device in self.devices.values():
            if device.kind.lower() == "local" and len(device.members) != 1:
                raise ValueError(
                    f"Local device '{device.name}' must have exactly one member"
                )

    def get_node(self, name: str) -> Node:
        """Get a Physical Node by its unique name."""
        return self.nodes[name]

    def get_device(self, name: str) -> Device:
        """Get a Logical Device by its unique name."""
        return self.devices[name]

    def get_devices_by_kind(self, kind: str) -> list[Device]:
        """Get all Logical Devices of a specific kind."""
        lowered = kind.lower()
        return [dev for dev in self.devices.values() if dev.kind.lower() == lowered]

    def get_node_by_rank(self, rank: int) -> Node:
        """Get a Physical Node by its unique rank."""
        # This might require an internal mapping for efficiency if called often
        for node in self.nodes.values():
            if node.rank == rank:
                return node
        raise KeyError(f"No Physical Node found with rank {rank}")

    def to_dict(self) -> dict[str, Any]:
        """Convert ClusterSpec to a dictionary."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "devices": {
                name: device.to_dict() for name, device in self.devices.items()
            },
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ClusterSpec:
        """Parses a raw config dictionary and returns a validated ClusterSpec."""
        # 1. Validate top-level keys
        if "nodes" not in config or "devices" not in config:
            raise ValueError(
                "Cluster config must contain 'nodes' and 'devices' sections."
            )

        # 2. Parse Physical Nodes, using the list index as the rank
        nodes_map: dict[str, Node] = {}
        known_runtime_fields = {"version", "platform", "backends"}
        for i, node_cfg in enumerate(config["nodes"]):
            if "rank" in node_cfg:
                # Optionally, we can log a warning that the explicit 'rank' is ignored.
                pass

            runtime_info_cfg = node_cfg.get("runtime_info", {})
            extra_runtime_info = {
                k: v
                for k, v in runtime_info_cfg.items()
                if k not in known_runtime_fields
            }

            runtime_info = RuntimeInfo(
                version=runtime_info_cfg.get("version", "N/A"),
                platform=runtime_info_cfg.get("platform", "N/A"),
                backends=runtime_info_cfg.get("backends", []),
                extra=extra_runtime_info,
            )

            node = Node(
                name=node_cfg["name"],
                rank=i,  # Implicit rank assignment
                endpoint=node_cfg["endpoint"],
                runtime_info=runtime_info,
            )

            if node.name in nodes_map:
                raise ValueError(f"Duplicate node name found: {node.name}")
            nodes_map[node.name] = node

        # 3. Parse Logical Devices
        devices_map: dict[str, Device] = {}
        for dev_name, dev_cfg in config["devices"].items():
            member_nodes = []
            for member_name in dev_cfg["members"]:
                if member_name not in nodes_map:
                    raise ValueError(
                        f"Node '{member_name}' in device '{dev_name}' not defined in 'nodes' section."
                    )
                member_nodes.append(nodes_map[member_name])

            devices_map[dev_name] = Device(
                name=dev_name,
                kind=dev_cfg["kind"],
                members=member_nodes,
                config=dev_cfg.get("config", {}),
            )

        return cls(nodes=nodes_map, devices=devices_map)

    @classmethod
    def simple(cls, world_size: int) -> ClusterSpec:
        """Creates a simple cluster spec for simulation with the given number of parties."""
        nodes = {
            f"node{i}": Node(
                name=f"node{i}",
                rank=i,
                endpoint=f"localhost:{5000 + i}",
                runtime_info=RuntimeInfo(
                    version="simulated",
                    platform="simulated",
                    backends=["__all__"],
                ),
            )
            for i in range(world_size)
        }

        devices = {
            "SP0": Device(
                name="SP0",
                kind="SPU",
                members=list(nodes.values()),
                config={
                    "protocol": "SEMI2K",
                    "field": "FM128",
                },
            )
        }

        return cls(nodes=nodes, devices=devices)
