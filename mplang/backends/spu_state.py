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

"""SPU Dialect State.

Manages SPU Runtime lifecycle as a dialect state, enabling reuse across
multiple executions while binding to the Interpreter's lifecycle.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import spu.api as spu_api
import spu.libspu as libspu

from mplang.runtime.dialect_state import DialectState

if TYPE_CHECKING:
    from mplang.backends.simp_worker.infra import WorkerInfra
    from mplang.dialects import spu

logger = logging.getLogger(__name__)


def _opt_int_env(name: str) -> int | None:
    """Read an optional int env var; return None when unset/empty."""
    val = os.getenv(name)
    return int(val) if val else None


@dataclass(frozen=True)
class BrpcLinkConfig:
    """Tunables for the brpc-backed SPU link.

    Defaults come from environment variables so deployments can switch
    protocol/timeouts without code changes; explicit constructor args
    still win for tests and programmatic overrides.

    Set ``protocol`` to ``"http"`` (HTTP/1.1) or ``"h2"`` (HTTP/2) when
    traffic must traverse an L7 gateway that does not understand the
    default ``baidu_std`` binary protocol. HTTP/1.1 requires
    ``connection_type="pooled"`` to avoid head-of-line blocking.

    ``http_timeout_ms`` defaults to ``None`` (use brpc's own default) —
    long-running MPC jobs traversing a gateway with idle timeout should
    set this explicitly to cover the longest single RPC.

    Frozen so instances are hashable and safe to embed in cache keys.
    """

    protocol: str = field(
        default_factory=lambda: os.getenv("MPLANG_BRPC_PROTOCOL") or "http"
    )
    connection_type: str = field(
        default_factory=lambda: os.getenv("MPLANG_BRPC_CONN_TYPE") or "pooled"
    )
    recv_timeout_ms: int = field(
        default_factory=lambda: int(os.getenv("MPLANG_BRPC_RECV_TIMEOUT_MS") or 100000)
    )
    http_max_payload_size: int = field(
        default_factory=lambda: int(
            os.getenv("MPLANG_BRPC_HTTP_MAX_PAYLOAD") or 32 * 1024 * 1024
        )
    )
    # Per-RPC HTTP timeout. None = inherit brpc default. Set when going
    # through a gateway with its own idle/request timeout.
    http_timeout_ms: int | None = field(
        default_factory=lambda: _opt_int_env("MPLANG_BRPC_HTTP_TIMEOUT_MS")
    )
    # Initial connect retry — tolerates peers that come up later.
    connect_retry_times: int = field(
        default_factory=lambda: int(os.getenv("MPLANG_BRPC_CONNECT_RETRY_TIMES") or 60)
    )
    connect_retry_interval_ms: int = field(
        default_factory=lambda: int(
            os.getenv("MPLANG_BRPC_CONNECT_RETRY_INTERVAL_MS") or 1000
        )
    )


class SPUState(DialectState):
    """SPU Runtime cache as dialect state.

    Caches SPU Runtime and Io objects per (local_rank, world_size, config, link_mode)
    to enable reuse across multiple SPU kernel executions.

    When created with a ``WorkerInfra`` reference (per-request mode), template
    links are obtained from the shared infra (thread-safe) and then spawned
    via ``link.spawn()`` for per-request isolation.

    This replaces the previous global `_SPU_RUNTIMES` cache with a properly
    lifecycle-managed dialect state.
    """

    dialect_name: str = "spu"

    def __init__(
        self,
        infra: WorkerInfra | None = None,
        brpc_config: BrpcLinkConfig | None = None,
    ) -> None:
        # Optional shared infrastructure (for per-request isolation via link.spawn)
        self._infra = infra
        self._brpc_config = brpc_config or BrpcLinkConfig()
        # Key: (local_rank, world_size, protocol, field, fxp_fraction_bits,
        # link_mode, spu_endpoints, brpc_config). ``brpc_config`` participates
        # only when link_mode == "brpc" (else None) so callers passing custom
        # configs don't silently reuse the first config's link.
        # Value: (Runtime, Io)
        self._runtimes: dict[
            tuple[
                int,
                int,
                str,
                str,
                int,
                str,
                tuple[str, ...] | None,
                BrpcLinkConfig | None,
            ],
            tuple[spu_api.Runtime, spu_api.Io],
        ] = {}
        # Local template link cache (used when no WorkerInfra is provided)
        self._template_links: dict[tuple, libspu.link.Context] = {}

    def _effective_brpc_config(self, config: spu.SPUConfig) -> BrpcLinkConfig:
        """Merge per-SPU link_desc over this state's default brpc config."""
        link_desc = config.link_desc
        if link_desc is None:
            return self._brpc_config

        overrides: dict[str, Any] = {}
        if link_desc.brpc_channel_protocol is not None:
            overrides["protocol"] = link_desc.brpc_channel_protocol
        if link_desc.brpc_channel_connection_type is not None:
            overrides["connection_type"] = link_desc.brpc_channel_connection_type
        if link_desc.recv_timeout_ms is not None:
            overrides["recv_timeout_ms"] = link_desc.recv_timeout_ms
        if link_desc.http_max_payload_size is not None:
            overrides["http_max_payload_size"] = link_desc.http_max_payload_size
        if link_desc.http_timeout_ms is not None:
            overrides["http_timeout_ms"] = link_desc.http_timeout_ms
        if link_desc.connect_retry_times is not None:
            overrides["connect_retry_times"] = link_desc.connect_retry_times
        if link_desc.connect_retry_interval_ms is not None:
            overrides["connect_retry_interval_ms"] = link_desc.connect_retry_interval_ms

        if not overrides:
            return self._brpc_config
        return replace(self._brpc_config, **overrides)

    def _get_template_link(
        self,
        cache_key: tuple,
        local_rank: int,
        spu_world_size: int,
        communicator: object | None,
        parties: list[int] | None,
        spu_endpoints: list[str] | None,
        brpc_config: BrpcLinkConfig | None = None,
    ) -> libspu.link.Context:
        """Get or create a template link for the given configuration.

        With ``WorkerInfra``: uses infra's thread-safe shared cache.
        Without: uses a local (non-thread-safe) cache on this SPUState.
        """

        def _create() -> libspu.link.Context:
            if spu_endpoints:
                cfg = brpc_config or self._brpc_config
                return self._create_brpc_link(local_rank, spu_endpoints, cfg)
            elif communicator is not None:
                if parties is None:
                    raise ValueError("parties required when using communicator")
                return self._create_channels_link(
                    local_rank, spu_world_size, communicator, parties
                )
            else:
                return self._create_mem_link(local_rank, spu_world_size)

        if self._infra is not None:
            return self._infra.get_or_create_spu_link(cache_key, _create)

        if cache_key not in self._template_links:
            self._template_links[cache_key] = _create()
        return self._template_links[cache_key]

    def get_or_create(
        self,
        local_rank: int,
        spu_world_size: int,
        config: spu.SPUConfig,
        spu_endpoints: list[str] | None = None,
        communicator: object | None = None,
        parties: list[int] | None = None,
    ) -> tuple[spu_api.Runtime, spu_api.Io]:
        """Get or create SPU Runtime and Io for the given configuration.

        Link mode priority: spu_endpoints (BRPC) > communicator (Channels) > mem.
        When ``spu_endpoints`` is provided it always takes precedence, even if
        a ``communicator`` is also supplied.

        Args:
            local_rank: The local rank within the SPU device (0-indexed).
            spu_world_size: The number of parties in the SPU device.
            config: SPU configuration including protocol settings.
            spu_endpoints: Optional list of BRPC endpoints. Takes highest
                priority when provided.
            communicator: Optional v2 communicator (ThreadCommunicator/HttpCommunicator).
                Used only when ``spu_endpoints`` is not provided.
            parties: Optional list of global ranks for SPU parties.
                Required when communicator is provided.

        Returns:
            A tuple of (Runtime, Io) for this party.
        """
        from mplang.backends.spu_impl import to_runtime_config

        # Determine link mode
        if spu_endpoints:
            link_mode = "brpc"
        elif communicator is not None:
            link_mode = "channels"
        else:
            link_mode = "mem"

        brpc_config = (
            self._effective_brpc_config(config) if link_mode == "brpc" else None
        )
        cache_key = (
            local_rank,
            spu_world_size,
            config.protocol,
            config.field,
            config.fxp_fraction_bits,
            link_mode,
            tuple(spu_endpoints) if spu_endpoints else None,
            brpc_config,
        )

        if cache_key in self._runtimes:
            return self._runtimes[cache_key]

        # Unified path: get-or-create template link, then spawn for isolation
        template_link = self._get_template_link(
            cache_key,
            local_rank,
            spu_world_size,
            communicator,
            parties,
            spu_endpoints,
            brpc_config,
        )
        link = template_link.spawn()

        # Create Runtime and Io
        runtime_config = to_runtime_config(config)
        runtime = spu_api.Runtime(link, runtime_config)
        io = spu_api.Io(spu_world_size, runtime_config)

        self._runtimes[cache_key] = (runtime, io)
        return runtime, io

    def _create_mem_link(
        self, local_rank: int, spu_world_size: int
    ) -> libspu.link.Context:
        """Create in-memory link for simulation."""
        desc = libspu.link.Desc()  # type: ignore
        desc.recv_timeout_ms = 30 * 1000
        for i in range(spu_world_size):
            desc.add_party(f"P{i}", f"mem:{i}")
        return libspu.link.create_mem(desc, local_rank)

    def _create_channels_link(
        self,
        local_rank: int,
        spu_world_size: int,
        communicator: Any,
        parties: list[int],
    ) -> libspu.link.Context:
        """Create link using custom channels (reuse v2 communicator).

        Args:
            local_rank: SPU local rank (0-indexed, already converted from global)
            spu_world_size: Number of SPU parties
            communicator: v2 communicator (ThreadCommunicator/HttpCommunicator)
            parties: List of global ranks for SPU parties (ordered by local rank)

        Returns:
            libspu link context using BaseChannel adapters
        """
        from mplang.backends.channel import BaseChannel

        # Get this worker's global rank
        global_rank = parties[local_rank]

        # Create channels list (world_size elements, self = None)
        channels = []
        for idx, peer_global_rank in enumerate(parties):
            if idx == local_rank:
                # Self channel must be None
                channel = None
            else:
                # Create channel to peer
                channel = BaseChannel(communicator, global_rank, peer_global_rank)
            channels.append(channel)

        # Create link descriptor
        desc = libspu.link.Desc()  # type: ignore
        desc.recv_timeout_ms = 100 * 1000  # 100 seconds

        # Add party info (required for world_size inference)
        for idx in range(spu_world_size):
            desc.add_party(f"P{idx}", f"dummy_{parties[idx]}")

        return libspu.link.create_with_channels(desc, local_rank, channels)

    def _create_brpc_link(
        self,
        local_rank: int,
        spu_endpoints: list[str],
        brpc_config: BrpcLinkConfig | None = None,
    ) -> libspu.link.Context:
        """Create BRPC link for distributed execution."""
        cfg = brpc_config or self._brpc_config
        desc = libspu.link.Desc()  # type: ignore
        desc.recv_timeout_ms = cfg.recv_timeout_ms
        desc.http_max_payload_size = cfg.http_max_payload_size
        desc.brpc_channel_protocol = cfg.protocol
        desc.brpc_channel_connection_type = cfg.connection_type
        desc.connect_retry_times = cfg.connect_retry_times
        desc.connect_retry_interval_ms = cfg.connect_retry_interval_ms
        if cfg.http_timeout_ms is not None:
            desc.http_timeout_ms = cfg.http_timeout_ms

        for i, endpoint in enumerate(spu_endpoints):
            desc.add_party(f"P{i}", endpoint)

        logger.info(
            "Creating SPU brpc link: rank=%d, endpoints=%s, protocol=%s, "
            "connection_type=%s, recv_timeout_ms=%d, http_timeout_ms=%s, "
            "http_max_payload_size=%d, connect_retry_times=%d, "
            "connect_retry_interval_ms=%d",
            local_rank,
            spu_endpoints,
            cfg.protocol,
            cfg.connection_type,
            cfg.recv_timeout_ms,
            cfg.http_timeout_ms,
            cfg.http_max_payload_size,
            cfg.connect_retry_times,
            cfg.connect_retry_interval_ms,
        )

        return libspu.link.create_brpc(desc, local_rank)

    def shutdown(self) -> None:
        """Clear all cached runtimes and template links."""
        self._runtimes.clear()
        self._template_links.clear()
