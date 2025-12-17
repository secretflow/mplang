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

from typing import TYPE_CHECKING

import spu.api as spu_api
import spu.libspu as libspu

from mplang.v2.runtime.dialect_state import DialectState

if TYPE_CHECKING:
    from mplang.v2.dialects import spu


class SPUState(DialectState):
    """SPU Runtime cache as dialect state.

    Caches SPU Runtime and Io objects per (local_rank, world_size, config, link_mode)
    to enable reuse across multiple SPU kernel executions.

    This replaces the previous global `_SPU_RUNTIMES` cache with a properly
    lifecycle-managed dialect state.
    """

    dialect_name: str = "spu"

    def __init__(self) -> None:
        # Key: (local_rank, world_size, protocol, field, link_mode)
        # Value: (Runtime, Io)
        self._runtimes: dict[
            tuple[int, int, str, str, str], tuple[spu_api.Runtime, spu_api.Io]
        ] = {}

    def get_or_create(
        self,
        local_rank: int,
        spu_world_size: int,
        config: spu.SPUConfig,
        spu_endpoints: list[str] | None = None,
    ) -> tuple[spu_api.Runtime, spu_api.Io]:
        """Get or create SPU Runtime and Io for the given configuration.

        Args:
            local_rank: The local rank within the SPU device (0-indexed).
            spu_world_size: The number of parties in the SPU device.
            config: SPU configuration including protocol settings.
            spu_endpoints: Optional list of BRPC endpoints. If None, use mem link.

        Returns:
            A tuple of (Runtime, Io) for this party.
        """
        from mplang.v2.backends.spu_impl import to_runtime_config

        link_mode = "brpc" if spu_endpoints else "mem"
        cache_key = (
            local_rank,
            spu_world_size,
            config.protocol,
            config.field,
            link_mode,
        )

        if cache_key in self._runtimes:
            return self._runtimes[cache_key]

        # Create Link
        if spu_endpoints:
            link = self._create_brpc_link(local_rank, spu_endpoints)
        else:
            link = self._create_mem_link(local_rank, spu_world_size)

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

    def _create_brpc_link(
        self, local_rank: int, spu_endpoints: list[str]
    ) -> libspu.link.Context:
        """Create BRPC link for distributed execution."""
        desc = libspu.link.Desc()  # type: ignore
        desc.recv_timeout_ms = 100 * 1000  # 100 seconds
        desc.http_max_payload_size = 32 * 1024 * 1024  # 32MB

        for i, endpoint in enumerate(spu_endpoints):
            desc.add_party(f"P{i}", endpoint)

        return libspu.link.create_brpc(desc, local_rank)

    def shutdown(self) -> None:
        """Clear all cached runtimes."""
        self._runtimes.clear()
