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

"""Core Session model (pure, no global registries).

Contents:
    * SessionState dataclass
    * LinkCommFactory (SPU link reuse cache)
    * Session (topology derivation, runtime init, SPU env seeding, local symbol/computation storage)

Process-wide registries (sessions, global symbols) live in the server layer
(`server.py`) so this module remains portable and easy to unit test.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

import spu.libspu as libspu

from mplang.core.expr.ast import Expr
from mplang.core.expr.evaluator import IEvaluator, create_evaluator
from mplang.core.mask import Mask
from mplang.kernels.context import RuntimeContext
from mplang.kernels.spu import PFunction  # type: ignore
from mplang.runtime.communicator import HttpCommunicator
from mplang.runtime.exceptions import ResourceNotFound
from mplang.runtime.link_comm import LinkCommunicator
from mplang.utils.spu_utils import parse_field, parse_protocol

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from mplang.core.cluster import ClusterSpec, Node, RuntimeInfo


class LinkCommFactory:
    """Factory for creating and caching link communicators."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, tuple[str, ...]], LinkCommunicator] = {}

    def create_link(self, rel_rank: int, addrs: list[str]) -> LinkCommunicator:
        key = (rel_rank, tuple(addrs))
        link = self._cache.get(key)
        if link is not None:
            return link
        logging.info(f"LinkCommunicator created: rel_rank={rel_rank} addrs={addrs}")
        link = LinkCommunicator(rel_rank, addrs)
        self._cache[key] = link
        return link


# Shared link factory (module-local, not global registry of sessions)
g_link_factory = LinkCommFactory()


@dataclass
class Symbol:
    name: str
    mptype: Any
    data: Any


@dataclass
class Computation:
    name: str
    expr: Expr


@dataclass
class SessionState:
    runtime: RuntimeContext | None = None
    computations: dict[str, Computation] = field(default_factory=dict)
    symbols: dict[str, Symbol] = field(default_factory=dict)
    spu_seeded: bool = False
    created_ts: float = field(default_factory=time.time)
    last_access_ts: float = field(default_factory=time.time)


class Session:
    """Represents the per-rank execution context.

    Immutable config: name, rank, cluster_spec.
    Derived: node, runtime_info, endpoints, spu_device, spu_mask, protocol/field, is_spu_party.
    Mutable: state (runtime object, symbols, computations, seeded flag).
    """

    def __init__(self, name: str, rank: int, cluster_spec: ClusterSpec):
        self.name = name
        self.rank = rank
        self.cluster_spec = cluster_spec
        self.state = SessionState()
        self.communicator = HttpCommunicator(
            session_name=name, rank=rank, endpoints=self.endpoints
        )

    # --- Derived topology ---
    @cached_property
    def node(self) -> Node:
        return self.cluster_spec.get_node_by_rank(self.rank)

    @property
    def runtime_info(self) -> RuntimeInfo:
        return self.node.runtime_info

    @cached_property
    def endpoints(self) -> list[str]:
        eps: list[str] = []
        for n in sorted(
            self.cluster_spec.nodes.values(),
            key=lambda x: x.rank,  # type: ignore[attr-defined]
        ):
            ep = n.endpoint
            if not ep.startswith(("http://", "https://")):
                ep = f"http://{ep}"
            eps.append(ep)
        return eps

    @cached_property
    def spu_device(self):  # type: ignore
        devs = self.cluster_spec.get_devices_by_kind("SPU")
        if len(devs) != 1:
            raise RuntimeError(
                f"Expected exactly one SPU device, got {len(devs)} (session={self.name})"
            )
        return devs[0]

    @cached_property
    def spu_mask(self) -> Mask:
        return Mask.from_ranks([m.rank for m in self.spu_device.members])

    @property
    def spu_protocol(self) -> str:
        return cast(str, self.spu_device.config.get("protocol", "SEMI2K"))

    @property
    def spu_field(self) -> str:
        return cast(str, self.spu_device.config.get("field", "FM64"))

    @property
    def is_spu_party(self) -> bool:
        return self.rank in self.spu_mask

    # --- Runtime helpers ---
    def ensure_runtime(self) -> RuntimeContext:
        if self.state.runtime is None:
            self.state.runtime = RuntimeContext(
                rank=self.rank,
                world_size=len(self.cluster_spec.nodes),  # type: ignore[attr-defined]
                initial_bindings=(
                    self.runtime_info.op_bindings if self.runtime_info else {}
                ),
            )
        return self.state.runtime

    def ensure_spu_env(self) -> None:
        """Ensure SPU kernel env (config/world[/link]) registered on this runtime.

        Previous logic only seeded SPU parties; non-participating ranks then raised
        a hard error when the evaluator encountered SPU ops in the global program,
        because the kernel pocket lacked config/world. For now we register the
        config/world on ALL parties (idempotent) and only attach a link context for
        participating SPU ranks. Non-parties will still error later if they try to
        execute a link-dependent SPU kernel (which should be guarded by masks in the
        IR), but they will no longer fail early with a misleading
        "SPU kernel state not initialized" message.
        """
        if self.state.spu_seeded:
            return

        link_ctx = None
        # TODO(jint): reuse same port for mplang and spu.
        SPU_PORT_OFFSET = 100

        if self.is_spu_party:
            # Build SPU address list across all endpoints for ranks in mask
            spu_addrs: list[str] = []
            for r, addr in enumerate(self.communicator.endpoints):
                if r in self.spu_mask:
                    if "//" not in addr:
                        addr = f"//{addr}"
                    parsed = urlparse(addr)
                    assert isinstance(parsed.port, int)
                    new_addr = f"{parsed.hostname}:{parsed.port + SPU_PORT_OFFSET}"
                    spu_addrs.append(new_addr)
            rel_index = sum(1 for r in range(self.rank) if r in self.spu_mask)
            link_ctx = g_link_factory.create_link(rel_index, spu_addrs)

        spu_config = libspu.RuntimeConfig(
            protocol=parse_protocol(self.spu_protocol),
            field=parse_field(self.spu_field),
            fxp_fraction_bits=18,
        )
        seed_pfunc = PFunction(
            fn_type="spu.seed_env",
            ins_info=(),
            outs_info=(),
            config=spu_config,
            world=self.spu_mask.num_parties(),
            link=link_ctx,
        )
        self.ensure_runtime().run_kernel(seed_pfunc, [])
        self.state.spu_seeded = True

    # --- Computations & Symbols (instance-local) ---
    def add_computation(self, computation: Computation) -> None:
        self.state.computations[computation.name] = computation

    def get_computation(self, name: str) -> Computation | None:
        return self.state.computations.get(name)

    def add_symbol(self, symbol: Symbol) -> None:
        self.state.symbols[symbol.name] = symbol

    def get_symbol(self, name: str) -> Symbol | None:
        return self.state.symbols.get(name)

    def list_symbols(self) -> list[str]:  # pragma: no cover - trivial
        return list(self.state.symbols.keys())

    def delete_symbol(self, name: str) -> bool:
        if name in self.state.symbols:
            del self.state.symbols[name]
            return True
        return False

    def list_computations(self) -> list[str]:  # pragma: no cover - trivial
        return list(self.state.computations.keys())

    def delete_computation(self, name: str) -> bool:
        if name in self.state.computations:
            del self.state.computations[name]
            return True
        return False

    # --- Execution ---
    def execute(
        self, computation: Computation, input_names: list[str], output_names: list[str]
    ) -> None:
        env: dict[str, Any] = {}
        for in_name in input_names:
            sym = self.get_symbol(in_name)
            if sym is None:
                raise ResourceNotFound(
                    f"Input symbol '{in_name}' not found in session '{self.name}'"
                )
            env[in_name] = sym.data
        rt = self.ensure_runtime()
        self.ensure_spu_env()
        evaluator: IEvaluator = create_evaluator(
            rank=self.rank, env=env, comm=self.communicator, runtime=rt
        )
        results = evaluator.evaluate(computation.expr)
        if results and len(results) != len(output_names):
            raise RuntimeError(
                f"Expected {len(output_names)} results, got {len(results)}"
            )
        for name, val in zip(output_names, results, strict=True):
            self.add_symbol(Symbol(name=name, mptype={}, data=val))

    # --- Convenience constructor ---
    @classmethod
    def from_cluster_spec_dict(cls, name: str, rank: int, spec_dict: dict) -> Session:
        from mplang.core.cluster import ClusterSpec  # local import to avoid cycles

        spec = ClusterSpec.from_dict(spec_dict)
        if len(spec.get_devices_by_kind("SPU")) == 0:
            raise RuntimeError("No SPU device found in cluster_spec")
        return cls(name=name, rank=rank, cluster_spec=spec)
