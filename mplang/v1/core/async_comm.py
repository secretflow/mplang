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

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from mplang.v1.core.comm import ICommunicator
from mplang.v1.core.mask import Mask


class IAsyncCommunicator(ICommunicator):
    """Base class for asynchronous communicators."""

    @abstractmethod
    async def async_send(self, to: int, key: str, data: Any) -> None:
        """Send data to peer with the given key asynchronously"""

    @abstractmethod
    async def async_recv(self, frm: int, key: str) -> Any:
        """Receive data from peer with the given key asynchronously"""


class IAsyncCollective(ABC):
    """Interface for asynchronous collective communication"""

    @abstractmethod
    async def p2p(self, frm: int, to: int, data: Any) -> Any:
        """Perform point-to-point communication"""

    @abstractmethod
    async def gather(self, root: int, data: Any) -> list[Any]:
        """Gather data from all processes to root"""

    @abstractmethod
    async def gather_m(self, pmask: int, root: int, data: Any) -> list[Any]:
        """Gather data from parties in pmask to root"""

    @abstractmethod
    async def scatter(self, root: int, args: list[Any]) -> Any:
        """Scatter data from root to all processes"""

    @abstractmethod
    async def scatter_m(self, pmask: int, root: int, args: list[Any]) -> Any:
        """Scatter data from root to parties in pmask"""

    @abstractmethod
    async def allgather(self, arg: Any) -> list[Any]:
        """Gather data from all processes to all processes"""

    @abstractmethod
    async def allgather_m(self, pmask: int, arg: Any) -> list[Any]:
        """Gather data from parties in pmask to all processes"""

    @abstractmethod
    async def bcast(self, root: int, arg: Any) -> Any:
        """Broadcast data from root to all processes"""

    @abstractmethod
    async def bcast_m(self, pmask: int, root: int, arg: Any) -> Any:
        """Broadcast data from root to parties in pmask"""


class AsyncCollectiveMixin(IAsyncCommunicator, IAsyncCollective):
    """Mixin class providing default implementations of asynchronous collective communication algorithms"""

    # Note: These will be provided by mixing classes as properties
    @property
    def rank(self) -> int:
        raise NotImplementedError

    @property
    def world_size(self) -> int:
        raise NotImplementedError

    def send(self, to: int, key: str, data: Any) -> None:
        raise NotImplementedError

    def recv(self, frm: int, key: str) -> Any:
        raise NotImplementedError

    async def async_send(self, to: int, key: str, data: Any) -> None:
        raise NotImplementedError

    async def async_recv(self, frm: int, key: str) -> Any:
        raise NotImplementedError

    def new_id(self) -> str:
        raise NotImplementedError

    async def p2p(self, frm: int, to: int, data: Any) -> Any:
        assert 0 <= frm < self.world_size
        assert 0 <= to < self.world_size

        cid = self.new_id()

        send_coro = None
        if self.rank == frm:
            send_coro = self.async_send(to, cid, data)

        recv_coro = None
        if self.rank == to:
            recv_coro = self.async_recv(frm, cid)

        if send_coro and recv_coro:
            _, res = await asyncio.gather(send_coro, recv_coro)
            return res
        elif send_coro:
            await send_coro
            return None
        elif recv_coro:
            return await recv_coro
        else:
            return None

    async def gather_m(self, pmask: int, root: int, data: Any) -> list[Any]:
        assert 0 <= root < self.world_size
        cid = self.new_id()
        mask = Mask(pmask)

        # 1. Send if we are in mask
        if self.rank in mask:
            await self.async_send(root, cid, data)

        # 2. Recv if we are root
        if self.rank == root:
            # Create futures for all expected receives
            futures = []
            for idx in mask:
                futures.append(self.async_recv(idx, cid))

            # Wait for all concurrently
            results = await asyncio.gather(*futures)
            return results
        else:
            return [None] * mask.num_parties()

    async def gather(self, root: int, data: Any) -> list[Any]:
        pmask = Mask.all(self.world_size)
        return await self.gather_m(pmask.value, root, data)

    async def scatter_m(self, pmask: int, root: int, args: list[Any]) -> Any:
        logging.debug(
            f"[{self.rank}]: scatter_m: pmask={pmask}, root={root}, args={args}"
        )
        assert 0 <= root < self.world_size
        mask = Mask(pmask)
        assert len(args) == mask.num_parties(), f"{len(args)} != {mask.num_parties()}"

        cid = self.new_id()

        if self.rank == root:
            # Send to all targets concurrently
            send_futures = []
            for idx, arg in zip(mask, args, strict=True):
                send_futures.append(self.async_send(idx, cid, arg))
            await asyncio.gather(*send_futures)

        if self.rank in mask:
            data = await self.async_recv(root, cid)
        else:
            data = None

        return data

    async def scatter(self, root: int, args: list[Any]) -> Any:
        pmask = Mask.all(self.world_size)
        return await self.scatter_m(pmask.value, root, args)

    async def allgather_m(self, pmask: int, arg: Any) -> list[Any]:
        logging.debug(f"allgather_m: pmask={pmask}, arg={arg}")
        cid = self.new_id()
        mask = Mask(pmask)

        # 1. Send to all other parties in mask
        if self.rank in mask:
            send_futures = []
            for idx in mask:
                send_futures.append(self.async_send(idx, cid, arg))
            await asyncio.gather(*send_futures)

            # 2. Recv from all parties in mask
            recv_futures = []
            for idx in mask:
                recv_futures.append(self.async_recv(idx, cid))

            res = await asyncio.gather(*recv_futures)
            return res
        else:
            return [None] * mask.num_parties()

    async def allgather(self, arg: Any) -> list[Any]:
        pmask = Mask.all(self.world_size)
        return await self.allgather_m(pmask.value, arg)

    async def bcast_m(self, pmask: int, root: int, arg: Any) -> Any:
        logging.debug(f"bcast_m: pmask={pmask}, root={root}, arg={arg}")
        assert 0 <= root < self.world_size
        mask = Mask(pmask)
        cid = self.new_id()

        if self.rank == root:
            send_futures = []
            for idx in mask:
                send_futures.append(self.async_send(idx, cid, arg))
            await asyncio.gather(*send_futures)

        if self.rank in mask:
            return await self.async_recv(root, cid)
        else:
            return None

    async def bcast(self, root: int, arg: Any) -> Any:
        pmask = Mask.all(self.world_size)
        return await self.bcast_m(pmask.value, root, arg)


class AsyncCommunicatorBase(IAsyncCommunicator):
    """Base implementation providing message box functionality for local communication using asyncio"""

    def __init__(
        self, rank: int, world_size: int, loop: asyncio.AbstractEventLoop | None = None
    ):
        self._rank = rank
        self._world_size = world_size
        # Map (frm, key) -> Future or Data
        self._msgboxes: dict[tuple[int, str], Any | asyncio.Future] = {}
        self._counter = 0
        self._loop = loop

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError as e:
                raise RuntimeError(
                    "AsyncCommunicatorBase must be used within an asyncio event loop or loop must be provided in init"
                ) from e
        return self._loop

    def new_id(self) -> str:
        # Simple counter, assuming single-threaded access to this method within the loop
        res = self._counter
        self._counter += 1
        return str(res)

    async def async_recv(self, frm: int, key: str) -> Any:
        """Wait until the key is set, returns the value"""
        mkey = (frm, key)

        # Check if data is already there
        if mkey in self._msgboxes:
            val = self._msgboxes[mkey]
            if isinstance(val, asyncio.Future):
                # Already waiting? This shouldn't happen in normal logic unless multiple recvs for same key
                return await val
            else:
                # Data arrived before recv
                del self._msgboxes[mkey]
                return val

        # Not there, create a future
        loop = self._get_loop()
        fut = loop.create_future()
        self._msgboxes[mkey] = fut
        try:
            return await fut
        finally:
            if mkey in self._msgboxes and self._msgboxes[mkey] is fut:
                del self._msgboxes[mkey]

    def onSent(self, frm: int, key: str, data: Any) -> None:
        """Called when a key is sent to self.

        This method must be thread-safe as it might be called from network threads.
        """
        loop = self._get_loop()
        # Use call_soon_threadsafe to handle calls from other threads (e.g. network callbacks)
        # If called from the same loop, it just schedules it for next iteration.
        loop.call_soon_threadsafe(self._on_sent_internal, frm, key, data)

    def _on_sent_internal(self, frm: int, key: str, data: Any) -> None:
        mkey = (frm, key)
        if mkey in self._msgboxes:
            val = self._msgboxes[mkey]
            if isinstance(val, asyncio.Future):
                if not val.done():
                    val.set_result(data)
                # Future is done, we can remove it from msgboxes?
                # No, recv needs to await it. But recv will remove it after await.
                # Wait, if we remove it here, recv might fail if it hasn't awaited yet?
                # Actually, once set_result is called, the future holds the value.
                # We should remove it from _msgboxes so it doesn't grow forever?
                # But recv uses mkey to find the future.
                # So we leave it there. recv will remove it.
            else:
                raise RuntimeError(f"Duplicate message for {mkey}")
        else:
            self._msgboxes[mkey] = data

    async def async_send(self, to: int, key: str, data: Any) -> None:
        # Base implementation for local simulation: directly call peer's onSent
        # In a real distributed setting, this would put data on wire.
        raise NotImplementedError(
            "Must be implemented by subclass or mixin with peer awareness"
        )

    def send(self, to: int, key: str, data: Any) -> None:
        raise NotImplementedError(
            "Synchronous send not supported in AsyncCommunicatorBase"
        )

    def recv(self, frm: int, key: str) -> Any:
        raise NotImplementedError(
            "Synchronous recv not supported in AsyncCommunicatorBase"
        )


class AsyncThreadCommunicator(AsyncCommunicatorBase, AsyncCollectiveMixin):
    """Thread-based async communicator for in-memory communication (simulation)"""

    def __init__(
        self, rank: int, world_size: int, loop: asyncio.AbstractEventLoop | None = None
    ):
        super().__init__(rank, world_size, loop)
        self.peers: list[AsyncThreadCommunicator] = []

    def set_peers(self, peers: list[AsyncThreadCommunicator]) -> None:
        assert self.world_size == len(peers)
        self.peers = peers

    async def async_send(self, to: int, key: str, data: Any) -> None:
        assert 0 <= to < self.world_size
        # In local simulation, we can directly call peer's onSent.
        # Since we are all in the same process (and likely same loop for simulation),
        # we can just call it.
        self.peers[to].onSent(self.rank, key, data)
