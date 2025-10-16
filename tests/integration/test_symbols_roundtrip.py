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
"""Integration: global symbols read -> compute -> write new global symbol.

Flow:
 1. Spawn two HTTP runtimes via the shared fixture infrastructure.
 2. Use the client SDK to upload ``x`` only to party P0's global symbol table.
 3. Evaluate a function that reads ``x`` on P0, adds 1 on P0, sends the result to P1, and writes ``y`` from P1.
 4. Confirm the computation result is now held by P1 only.
 5. Use the client SDK to fetch ``y`` from the P1 runtime and verify ``y == x + 1``.

This tests end-to-end interaction between:
 - Global symbol CRUD endpoints
 - ``basic.read`` path using ``symbols://`` scheme
 - Driver evaluation and fetch
 - HttpExecutorClient helpers for symbol CRUD
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import mplang as mp
from mplang.kernels.value import TensorValue  # Internal type not in __init__
from mplang.ops import basic  # Frontend module not in __init__
from mplang.runtime.client import HttpExecutorClient  # Runtime client not in __init__

pytest_plugins = ("tests.utils.server_fixtures",)


@pytest.mark.parametrize("http_servers", [2], indirect=True)
def test_global_symbol_roundtrip(http_servers):
    node_ids = ["P0", "P1"]
    node_addrs = dict(zip(node_ids, http_servers.addresses, strict=True))

    # Cluster spec
    nodes = {}
    for nid, addr in node_addrs.items():
        r = int(nid[1:])
        nodes[f"node{r}"] = mp.Node(
            name=f"node{r}",
            rank=r,
            endpoint=addr,
            runtime_info=mp.RuntimeInfo(
                version="test", platform="test", op_bindings={}
            ),
        )
    local_devices = {
        f"local_{n.rank}": mp.Device(name=f"local_{n.rank}", kind="ppu", members=[n])
        for n in nodes.values()
    }
    spu_device = mp.Device(
        name="SPU_0",
        kind="SPU",
        members=list(nodes.values()),
        config={"protocol": "SEMI2K", "field": "FM64"},
    )
    spec = mp.ClusterSpec(nodes=nodes, devices={**local_devices, "SPU_0": spu_device})
    driver = mp.Driver(spec)

    arr = np.arange(5, dtype=np.int32)
    tensor_meta = {"tensor": {"dtype": "I32", "shape": list(arr.shape)}}

    async def upload_symbol(addr: str, symbol_name: str, data: np.ndarray) -> None:
        client = HttpExecutorClient(addr)
        try:
            # Wrap numpy array in TensorValue before sending
            await client.create_global_symbol(
                symbol_name, TensorValue(data), tensor_meta
            )
        finally:
            await client.close()

    # Seed only P0 with the initial value.
    asyncio.run(upload_symbol(http_servers.addresses[0], "x", arr))

    @mp.function
    def compute_chain():
        ty = mp.TensorType(mp.DType.from_any(np.int32), (5,))
        # P0 reads x from its local global symbol table.
        x_p0 = mp.run_at(0, basic.read, path="symbols://x", ty=ty)

        # P0 computes x + 1 using rjax for plain Python function.
        x_inc_p0 = mp.run_jax(lambda a: a + 1, x_p0)

        # Transfer the result to P1 and write it out as symbols://y from P1.
        x_p1 = mp.p2p(0, 1, x_inc_p0)
        written = mp.run_at(1, basic.write, x_p1, path="symbols://y")

        return written

    out = mp.evaluate(driver, compute_chain)
    fetched = mp.fetch(driver, out)

    # Only P1 should now hold the materialized value.
    assert fetched[0] is None, "P0 should not retain the result after p2p transfer"
    np.testing.assert_array_equal(fetched[1], arr + 1)

    async def fetch_y_from_p1() -> np.ndarray:
        client = HttpExecutorClient(http_servers.addresses[1])
        try:
            result = await client.get_global_symbol("y")
            return np.asarray(result)
        finally:
            await client.close()

    y_val = asyncio.run(fetch_y_from_p1())
    np.testing.assert_array_equal(y_val, arr + 1)
