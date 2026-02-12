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

"""Simp Worker package.

Provides Worker-side state and ops for the simp dialect.
"""

# Import base components first
from mplang.backends.simp_worker.base import (
    CommunicatorProtocol,
    RecvRequest,
    Request,
    RequestStatus,
    SendRequest,
    testall,
    testany,
    wait_all,
    wait_any,
)

# Import http module components after state to avoid circular import
# (http.py imports SimpWorker from state.py directly)
from mplang.backends.simp_worker.http import (
    CommConfig,
    CommStats,
    HttpCommunicator,
    RecvTimeoutError,
    SendTimeoutError,
)
from mplang.backends.simp_worker.mem import LocalMesh, ThreadCommunicator
from mplang.backends.simp_worker.ops import WORKER_HANDLERS
from mplang.backends.simp_worker.state import SimpWorker

__all__ = [
    "WORKER_HANDLERS",
    "CommConfig",
    "CommStats",
    "CommunicatorProtocol",
    "HttpCommunicator",
    "LocalMesh",
    "RecvRequest",
    "RecvTimeoutError",
    "Request",
    "RequestStatus",
    "SendRequest",
    "SendTimeoutError",
    "SimpWorker",
    "ThreadCommunicator",
    "testall",
    "testany",
    "wait_all",
    "wait_any",
]
