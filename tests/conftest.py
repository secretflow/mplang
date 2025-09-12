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

import socket

import pytest


def get_free_port() -> int:
    """Return an ephemeral free TCP port bound on localhost.

    Each call binds to port 0 then closes immediately; a tiny race is still
    possible if something else grabs it before use, but acceptable for tests.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture
def free_port():
    return get_free_port()


def get_free_ports(n: int) -> list[int]:
    """Return a list of n free ephemeral ports.

    Convenience wrapper used by tests needing multiple deterministic ports without
    extra synchronization logic between processes.
    """
    return [get_free_port() for _ in range(n)]
