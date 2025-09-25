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

import logging
from typing import Any

import cloudpickle as pickle
import spu.libspu as libspu

from mplang.core.comm import ICollective, ICommunicator


class LinkCommunicator(ICommunicator, ICollective):
    """Wraps libspu link communicator for distributed communication"""

    def __init__(self, rank: int, addrs: list[str], *, mem_link: bool = False):
        self._rank = rank
        self._world_size = len(addrs)

        desc = libspu.link.Desc()  # type: ignore
        desc.recv_timeout_ms = 100 * 1000  # 100 seconds
        desc.http_max_payload_size = 32 * 1024 * 1024  # Default set link payload to 32M
        for rank, addr in enumerate(addrs):
            desc.add_party(f"P{rank}", addr)

        if mem_link:
            self.lctx = libspu.link.create_mem(desc, self._rank)
        else:
            self.lctx = libspu.link.create_brpc(desc, self._rank)

        logging.info(
            f"LinkCommunicator initialized with rank={self._rank}, world_size={self._world_size}, addrs={addrs}",
        )

        self._counter = 0

    @property
    def rank(self) -> int:
        return self.lctx.rank  # type: ignore[no-any-return]

    @property
    def world_size(self) -> int:
        return self.lctx.world_size  # type: ignore[no-any-return]

    def get_lctx(self) -> libspu.link.Context:
        """Get the link context"""
        return self.lctx

    # override
    def new_id(self) -> str:
        res = self._counter
        self._counter += 1
        return str(res)

    def wrap(self, obj: Any) -> str:
        data = pickle.dumps(obj)
        return data.hex()  # type: ignore[no-any-return]

    def unwrap(self, obj: str) -> Any:
        data = bytes.fromhex(obj)
        return pickle.loads(data)  # type: ignore[no-any-return]

    def send(self, to: int, key: str, data: Any) -> None:
        serialized = pickle.dumps((key, data))
        self.lctx.send(to, serialized.hex())

    def recv(self, frm: int, key: str) -> Any:
        serialized = self.lctx.recv(frm)
        rkey, data = pickle.loads(bytes.fromhex(serialized.decode()))
        assert key == rkey, f"recv key {key} != {rkey}"
        return data  # type: ignore[no-any-return]

    def p2p(self, frm: int, to: int, data: Any) -> Any:
        assert 0 <= frm < self.world_size
        assert 0 <= to < self.world_size

        # TODO: link handles cid internally?
        cid = self.new_id()

        if self.rank == frm:
            self.send(to, cid, data)
            return None
        elif self.rank == to:
            return self.recv(frm, cid)
        else:
            return None

    def gather(self, root: int, data: Any) -> list[Any]:
        assert 0 <= root < self.world_size
        rets = self.lctx.gather(self.wrap(data), root)
        return [self.unwrap(ret) for ret in rets]

    def scatter(self, root: int, args: list[Any]) -> Any:
        assert 0 <= root < self.world_size
        assert len(args) == self.world_size, f"{len(args)} != {self.world_size}"
        ret = self.lctx.scatter([self.wrap(arg) for arg in args], root)
        return self.unwrap(ret)

    def allgather(self, arg: Any) -> list[Any]:
        rets = self.lctx.all_gather(self.wrap(arg))
        return [self.unwrap(ret) for ret in rets]

    def bcast(self, root: int, arg: Any) -> Any:
        assert 0 <= root < self.world_size
        ret = self.lctx.broadcast(self.wrap(arg), root)
        return self.unwrap(ret)

    def gather_m(self, pmask: int, root: int, data: Any) -> list[Any]:
        raise ValueError("Not supported by LinkCommunicator")

    def scatter_m(self, pmask: int, root: int, args: list[Any]) -> Any:
        raise ValueError("Not supported by LinkCommunicator")

    def allgather_m(self, pmask: int, arg: Any) -> list[Any]:
        raise ValueError("Not supported by LinkCommunicator")

    def bcast_m(self, pmask: int, root: int, arg: Any) -> Any:
        raise ValueError("Not supported by LinkCommunicator")
