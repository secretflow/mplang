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

"""Abstract Base Class for OKVS (Oblivious Key-Value Store)."""

from abc import ABC, abstractmethod

import mplang.v2.edsl as el


class OKVS(ABC):
    """Abstract interface for Oblivious Key-Value Store."""

    @abstractmethod
    def encode(self, keys: el.Object, values: el.Object, seed: el.Object) -> el.Object:
        """Encode items into OKVS storage.

        Args:
            keys: (N,) uint64 tensor of keys
            values: (N, D) uint64 tensor of values
            seed: (2,) uint64 tensor seed

        Returns:
            (M, D) uint64 tensor OKVS storage
        """

    @abstractmethod
    def decode(self, keys: el.Object, storage: el.Object, seed: el.Object) -> el.Object:
        """Decode items from OKVS storage.

        Args:
            keys: (N,) uint64 tensor of keys to query
            storage: (M, D) uint64 tensor OKVS storage
            seed: (2,) uint64 tensor seed

        Returns:
            (N, D) uint64 tensor of recovered values
        """
