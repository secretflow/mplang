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
from functools import cached_property
from typing import Any


@dataclass(frozen=True)
class RuntimeInfo:
    """Per-physical-node runtime configuration.

    ``op_bindings`` is a per-node override map (logical_op -> kernel_id) merged
    into that node's ``RuntimeContext``. Unknown future / auxiliary fields are
    preserved in ``extra``.
    """

    version: str
    platform: str
    # Per-node partial override dispatch table (merged over project defaults).
    op_bindings: dict[str, str] = field(default_factory=dict)

    # A catch-all for any other custom or future properties (must not collide
    # with reserved keys: version, platform, op_bindings).
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert RuntimeInfo to a dictionary (stable field names)."""
        result = {
            "version": self.version,
            "platform": self.platform,
            "op_bindings": self.op_bindings,
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
    def member_ranks(self) -> tuple[int, ...]:
        """Returns the ranks of the member PNs."""
        return tuple(sorted([node.rank for node in self.members]))

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

    cluster_id: str
    nodes: dict[str, Node]
    devices: dict[str, Device]

    @property
    def world_size(self) -> int:
        """Total number of physical nodes (parties)."""
        return len(self.nodes)

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

        # ensure ppu devices have exactly one member
        for device in self.devices.values():
            if device.kind.lower() == "ppu" and len(device.members) != 1:
                raise ValueError(
                    f"PPU device '{device.name}' must have exactly one member"
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
            "cluster_id": self.cluster_id,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "devices": {
                name: device.to_dict() for name, device in self.devices.items()
            },
        }

    @cached_property
    def endpoints(self) -> list[str]:
        eps: list[str] = []
        for n in sorted(
            self.nodes.values(),
            key=lambda x: x.rank,  # type: ignore[attr-defined]
        ):
            eps.append(n.endpoint)
        return eps

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
        # Reserved runtime info keys we recognize explicitly.
        known_runtime_fields = {"version", "platform", "op_bindings"}
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
                op_bindings=runtime_info_cfg.get("op_bindings", {}) or {},
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

        # Get cluster_id from config or generate from filename
        cluster_id = config.get("cluster_id", f"cluster_{len(nodes_map)}")
        return cls(cluster_id=cluster_id, nodes=nodes_map, devices=devices_map)

    @classmethod
    def simple(
        cls,
        world_size: int,
        *,
        endpoints: list[str] | None = None,
        spu_world_size: int | None = None,
        spu_protocol: str = "SEMI2K",
        spu_field: str = "FM128",
        runtime_version: str = "simulated",
        runtime_platform: str = "simulated",
        op_bindings: list[dict[str, str]] | None = None,
        enable_ppu_device: bool = True,
        enable_spu_device: bool = True,
    ) -> ClusterSpec:
        """Convenience constructor used heavily in tests.

        Parameters
        ----------
        world_size:
            Number of parties (physical nodes).
        endpoints:
            Optional explicit endpoint list of length ``world_size``. Each element may
            include scheme (``http://``) or not; stored verbatim. If not provided we
            synthesize ``localhost:{5000 + i}`` (5000 is a fixed default; pass explicit
            endpoints for control).
        spu_protocol / spu_field:
            SPU device config values.
        runtime_version / runtime_platform:
            Populated into each node's ``RuntimeInfo``.
        op_bindings:
            Optional list of length ``world_size`` supplying per-node op_bindings
            override dicts (defaults to empty dicts).
        enable_ppu_device:
            If True (default), create one ``P{rank}`` PPU device per node.
        enable_spu_device:
            If True (default) create a shared SPU device named ``SP0``.
        """
        base_port = 5000

        if endpoints is not None and len(endpoints) != world_size:
            raise ValueError(
                "len(endpoints) must equal world_size when provided: "
                f"{len(endpoints)} != {world_size}"
            )

        if op_bindings is not None and len(op_bindings) != world_size:
            raise ValueError(
                "len(op_bindings) must equal world_size when provided: "
                f"{len(op_bindings)} != {world_size}"
            )

        if not enable_ppu_device and not enable_spu_device:
            raise ValueError(
                "At least one of enable_ppu_device or enable_spu_device must be True"
            )

        nodes: dict[str, Node] = {}
        for i in range(world_size):
            ep = endpoints[i] if endpoints is not None else f"localhost:{base_port + i}"
            node_op_bindings = op_bindings[i] if op_bindings is not None else {}
            nodes[f"node{i}"] = Node(
                name=f"node{i}",
                rank=i,
                endpoint=ep,
                runtime_info=RuntimeInfo(
                    version=runtime_version,
                    platform=runtime_platform,
                    op_bindings=node_op_bindings,
                ),
            )

        devices: dict[str, Device] = {}
        # Optional per-node PPU devices
        if enable_ppu_device:
            for i in range(world_size):
                devices[f"P{i}"] = Device(
                    name=f"P{i}",
                    kind="ppu",
                    members=[nodes[f"node{i}"]],
                )

        # Shared SPU device
        if enable_spu_device:
            if spu_world_size is None:
                spu_world_size = world_size
            spu_members = [nodes[f"node{i}"] for i in range(spu_world_size)]

            devices["SP0"] = Device(
                name="SP0",
                kind="SPU",
                members=spu_members,
                config={
                    "protocol": spu_protocol,
                    "field": spu_field,
                },
            )

        return cls(cluster_id=f"__sim_{world_size}", nodes=nodes, devices=devices)
