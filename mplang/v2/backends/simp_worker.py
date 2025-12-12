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

"""SIMP worker module.

Provides Context and Store for executing Graph IR on a single party.
This module only defines the context/store - operation implementations
are registered in simp_impl.py.
"""

from __future__ import annotations

from typing import Any

import mplang.v2.backends.field_impl  # noqa: F401
import mplang.v2.backends.tensor_impl  # noqa: F401
from mplang.v2.runtime.object_store import ObjectStore


class SimpWorkerContext:
    """Worker Context for SIMP execution.

    This context provides capabilities (Store, Communicator) to the Interpreter.
    It does NOT inherit from Interpreter or wrap logic.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        communicator: Any,
        store: ObjectStore,
        spu_endpoints: dict[int, str] | None = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.communicator = communicator
        self.store = store
        self.spu_endpoints = spu_endpoints
        self.current_parties: tuple[int, ...] | None = None
