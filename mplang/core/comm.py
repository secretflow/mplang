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
from abc import ABC, abstractmethod
from typing import Any

from mplang.utils.mask import enum_mask


class ICommunicator(ABC):
    """Base class for communicators."""

    @property
    @abstractmethod
    def rank(self) -> int:
        """Get the rank of this process"""

    @property
    @abstractmethod
    def world_size(self) -> int:
        """Get the world size of this process"""

    @abstractmethod
    def new_id(self) -> str:
        """Must be implemented by mixing class"""
        raise NotImplementedError

    @abstractmethod
    def send(self, to: int, key: str, data: Any):
        """Send data to peer with the given key"""

    @abstractmethod
    def recv(self, frm: int, key: str) -> Any:
        """Receive data from peer with the given key"""


class ICollective(ABC):
    """Interface for collective communication"""

    @abstractmethod
    def p2p(self, frm: int, to: int, data: Any) -> Any:
        """Perform point-to-point communication"""

    @abstractmethod
    def gather(self, root: int, data: Any) -> list[Any]:
        """Gather data from all processes to root"""

    @abstractmethod
    def gather_m(self, pmask: int, root: int, data: Any) -> list[Any]:
        """Gather data from parties in pmask to root"""

    @abstractmethod
    def scatter(self, root: int, args: list[Any]) -> Any:
        """Scatter data from root to all processes"""

    @abstractmethod
    def scatter_m(self, pmask: int, root: int, args: list[Any]) -> Any:
        """Scatter data from root to parties in pmask"""

    @abstractmethod
    def allgather(self, arg: Any) -> list[Any]:
        """Gather data from all processes to all processes"""

    @abstractmethod
    def allgather_m(self, pmask: int, arg: Any) -> list[Any]:
        """Gather data from parties in pmask to all processes"""

    @abstractmethod
    def bcast(self, root: int, arg: Any) -> Any:
        """Broadcast data from root to all processes"""

    @abstractmethod
    def bcast_m(self, pmask: int, root: int, arg: Any) -> Any:
        """Broadcast data from root to parties in pmask"""


def is_rank_in(rank: int, mask: int) -> bool:
    """Check if the given rank is in the mask"""
    return (1 << rank) & mask != 0


class CollectiveMixin(ICommunicator, ICollective):
    """Mixin class providing default implementations of collective communication algorithms

    This mixin provides implementations based on send/recv primitives.
    Classes using this mixin must implement the ICommunicator interface methods.
    """

    # Note: These will be provided by mixing classes as properties
    @property
    def rank(self) -> int:
        """Must be implemented by mixing class"""
        raise NotImplementedError

    @property
    def world_size(self) -> int:
        """Must be implemented by mixing class"""
        raise NotImplementedError

    def send(self, to: int, key: str, data: Any) -> None:
        """Must be implemented by mixing class"""
        raise NotImplementedError

    def recv(self, frm: int, key: str) -> Any:
        """Must be implemented by mixing class"""
        raise NotImplementedError

    def new_id(self) -> str:
        """Must be implemented by mixing class"""
        raise NotImplementedError

    def p2p(self, frm: int, to: int, data: Any) -> Any:
        """Perform point-to-point communication"""
        # p2p is a special collective operation, with non-sender and non-receiver nodes get None
        assert 0 <= frm < self.world_size
        assert 0 <= to < self.world_size

        cid = self.new_id()

        if self.rank == frm:
            self.send(to, cid, data)

        if self.rank == to:
            return self.recv(frm, cid)
        else:
            return None

    def gather_m(self, pmask: int, root: int, data: Any) -> list[Any]:
        """Gather data from parties in pmask to root"""
        assert 0 <= root < self.world_size
        # wmask = (1 << self.world_size) - 1
        # assert mpt.is_subset(pmask, wmask)

        cid = self.new_id()

        if is_rank_in(self.rank, pmask):
            self.send(root, cid, data)

        if self.rank == root:
            res = [self.recv(idx, cid) for idx in enum_mask(pmask)]
        else:
            res = [None] * pmask.bit_count()

        return res

    def gather(self, root: int, data: Any) -> list[Any]:
        """Gather data from all processes to root"""
        pmask = (1 << self.world_size) - 1
        return self.gather_m(pmask, root, data)

    def scatter_m(self, pmask: int, root: int, args: list[Any]) -> Any:
        """Scatter data from root to parties in pmask"""
        logging.debug(
            f"[{self.rank}]: scatter_m: pmask={pmask}, root={root}, args={args}"
        )
        assert 0 <= root < self.world_size
        assert len(args) == pmask.bit_count(), f"{len(args)} != {pmask.bit_count()}"

        cid = self.new_id()

        if self.rank == root:
            for idx, arg in zip(enum_mask(pmask), args, strict=True):
                self.send(idx, cid, arg)

        if is_rank_in(self.rank, pmask):
            data = self.recv(root, cid)
        else:
            data = None

        return data

    def scatter(self, root: int, args: list[Any]) -> Any:
        """Scatter data from root to all processes"""
        pmask = (1 << self.world_size) - 1
        return self.scatter_m(pmask, root, args)

    def allgather_m(self, pmask: int, arg: Any) -> list[Any]:
        """Gather data from parties in pmask to all parties"""
        logging.debug(f"allgather_m: pmask={pmask}, arg={arg}")
        cid = self.new_id()

        if is_rank_in(self.rank, pmask):
            for idx in enum_mask(pmask):
                self.send(idx, cid, arg)

            res = [self.recv(idx, cid) for idx in enum_mask(pmask)]
        else:
            res = [None] * pmask.bit_count()

        return res

    def allgather(self, arg: Any) -> list[Any]:
        """Gather data from all processes to all processes"""
        pmask = (1 << self.world_size) - 1
        return self.allgather_m(pmask, arg)

    def bcast_m(self, pmask: int, root: int, arg: Any) -> Any:
        """Broadcast data from root to parties in pmask"""
        logging.debug(f"bcast_m: pmask={pmask}, root={root}, arg={arg}")
        assert 0 <= root < self.world_size

        cid = self.new_id()

        if self.rank == root:
            for idx in enum_mask(pmask):
                self.send(idx, cid, arg)

        if is_rank_in(self.rank, pmask):
            return self.recv(root, cid)
        else:
            return None

    def bcast(self, root: int, arg: Any) -> Any:
        """Broadcast data from root to all processes"""
        pmask = (1 << self.world_size) - 1
        return self.bcast_m(pmask, root, arg)
