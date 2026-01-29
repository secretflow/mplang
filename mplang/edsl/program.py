# Copyright 2026 Ant Group Co., Ltd.
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

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, ClassVar

from mplang.edsl import serde
from mplang.edsl.graph import Graph


@dataclass(frozen=True)
class FlatIOSignature:
    """Portable I/O signature for source-free execution.

    Only supports flat positional inputs/outputs corresponding to
    `graph.inputs` / `graph.outputs`.
    """

    kind: ClassVar[str] = "flat_list_v0"
    input_arity: int
    output_arity: int

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "input_arity": self.input_arity,
            "output_arity": self.output_arity,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FlatIOSignature:
        if data.get("kind") != cls.kind:
            raise ValueError(f"Unsupported signature kind: {data.get('kind')}")
        return cls(
            input_arity=int(data["input_arity"]),
            output_arity=int(data["output_arity"]),
        )


@serde.register_class
@dataclass
class CompiledProgram:
    """Executable program decoupled from user source.

    This is a *logical model*; packaging (file/zip/etc.) is handled by tool layer.

    Current constraints:
    - signature is flat positional list I/O.
    - no closure captures.
    - no constant outputs (out_imms) unless future signature captures them.
    """

    _serde_kind: ClassVar[str] = "mplang.CompiledProgram"

    graph: Graph
    signature: FlatIOSignature
    required_opcodes: list[str]
    graph_digest: str
    required_world_size: int | None = None
    created_at: str | None = None
    mplang_version: str | None = None
    schema_version: int = 1
    name: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "graph": serde.to_json(self.graph),
            "signature": self.signature.to_json(),
            "required_opcodes": list(self.required_opcodes),
            "graph_digest": self.graph_digest,
            "required_world_size": self.required_world_size,
            "created_at": self.created_at,
            "mplang_version": self.mplang_version,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CompiledProgram:
        if "schema_version" not in data:
            raise KeyError("Missing required field: schema_version")
        schema_version = int(data["schema_version"])
        if schema_version != 1:
            raise ValueError(
                f"Unsupported CompiledProgram schema_version: {schema_version}"
            )

        graph = serde.from_json(data["graph"])
        if not isinstance(graph, Graph):
            raise TypeError(
                f"Expected graph to deserialize to Graph, got {type(graph).__name__}"
            )

        signature = FlatIOSignature.from_json(data["signature"])

        required_world_size = data.get("required_world_size")
        if required_world_size is not None:
            required_world_size = int(required_world_size)
        return cls(
            graph=graph,
            signature=signature,
            required_opcodes=list(data.get("required_opcodes", [])),
            graph_digest=str(data["graph_digest"]),
            required_world_size=required_world_size,
            created_at=data.get("created_at"),
            mplang_version=data.get("mplang_version"),
            schema_version=schema_version,
            name=data.get("name"),
        )


def compute_graph_digest(graph: Graph) -> str:
    """Compute a deterministic digest for a Graph.

    We intentionally avoid `serde.dumps()` because it doesn't sort keys.
    """

    canonical = json.dumps(serde.to_json(graph), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
