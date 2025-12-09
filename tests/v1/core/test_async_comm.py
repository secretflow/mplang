import asyncio

import pytest

from mplang.v1.core.async_comm import AsyncThreadCommunicator


@pytest.mark.asyncio
async def test_async_p2p():
    world_size = 2
    comms = [AsyncThreadCommunicator(i, world_size) for i in range(world_size)]
    for comm in comms:
        comm.set_peers(comms)

    # P0 sends to P1
    async def p0_task():
        await comms[0].p2p(0, 1, "hello")
        return "done"

    async def p1_task():
        data = await comms[1].p2p(0, 1, None)
        return data

    results = await asyncio.gather(p0_task(), p1_task())
    assert results[1] == "hello"


@pytest.mark.asyncio
async def test_async_gather():
    world_size = 3
    comms = [AsyncThreadCommunicator(i, world_size) for i in range(world_size)]
    for comm in comms:
        comm.set_peers(comms)

    async def task(rank):
        data = f"data-{rank}"
        return await comms[rank].gather(0, data)

    results = await asyncio.gather(*[task(i) for i in range(world_size)])

    # Rank 0 should get all data
    assert results[0] == ["data-0", "data-1", "data-2"]
    # Others get None list
    assert results[1] == [None, None, None]
    assert results[2] == [None, None, None]


@pytest.mark.asyncio
async def test_async_scatter():
    world_size = 3
    comms = [AsyncThreadCommunicator(i, world_size) for i in range(world_size)]
    for comm in comms:
        comm.set_peers(comms)

    data_to_scatter = ["d0", "d1", "d2"]

    async def task(rank):
        if rank == 0:
            return await comms[rank].scatter(0, data_to_scatter)
        else:
            return await comms[rank].scatter(0, [None] * 3)  # args ignored for non-root

    results = await asyncio.gather(*[task(i) for i in range(world_size)])

    assert results[0] == "d0"
    assert results[1] == "d1"
    assert results[2] == "d2"


@pytest.mark.asyncio
async def test_async_bcast():
    world_size = 3
    comms = [AsyncThreadCommunicator(i, world_size) for i in range(world_size)]
    for comm in comms:
        comm.set_peers(comms)

    async def task(rank):
        if rank == 0:
            return await comms[rank].bcast(0, "broadcast_data")
        else:
            return await comms[rank].bcast(0, None)

    results = await asyncio.gather(*[task(i) for i in range(world_size)])

    # bcast returns the data for everyone in the mask, including the root
    assert results[0] == "broadcast_data"
    assert results[1] == "broadcast_data"
    assert results[2] == "broadcast_data"
