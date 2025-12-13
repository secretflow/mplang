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

"""Tests for simp_worker/mem.py (ThreadCommunicator, DriverVar)."""

import threading
import time

import pytest

from mplang.v2.backends.simp_driver import DriverVar
from mplang.v2.backends.simp_worker.mem import ThreadCommunicator


def test_thread_communicator_send_recv():
    """Test basic send and receive between two communicators."""
    world_size = 2
    comm0 = ThreadCommunicator(0, world_size)
    comm1 = ThreadCommunicator(1, world_size)
    peers = [comm0, comm1]
    comm0.set_peers(peers)
    comm1.set_peers(peers)

    # Send from 0 to 1
    key = "test_key"
    data = "hello"
    comm0.send(1, key, data)

    # Recv at 1 from 0
    received = comm1.recv(0, key)
    assert received == data


def test_thread_communicator_blocking_recv():
    """Test that recv blocks until data is available."""
    world_size = 2
    comm0 = ThreadCommunicator(0, world_size)
    comm1 = ThreadCommunicator(1, world_size)
    peers = [comm0, comm1]
    comm0.set_peers(peers)
    comm1.set_peers(peers)

    key = "blocking_key"
    data = "delayed_data"

    def delayed_send():
        time.sleep(0.1)
        comm0.send(1, key, data)

    t = threading.Thread(target=delayed_send)
    t.start()

    start_time = time.time()
    received = comm1.recv(0, key)
    end_time = time.time()

    t.join()

    assert received == data
    # Should have waited at least a bit
    assert end_time - start_time >= 0.05


def test_thread_communicator_mailbox_overflow():
    """Test that sending to an occupied mailbox slot raises RuntimeError."""
    world_size = 2
    comm0 = ThreadCommunicator(0, world_size)
    comm1 = ThreadCommunicator(1, world_size)
    peers = [comm0, comm1]
    comm0.set_peers(peers)
    comm1.set_peers(peers)

    key = "overflow_key"
    data1 = "data1"
    data2 = "data2"

    comm0.send(1, key, data1)

    with pytest.raises(RuntimeError, match="Mailbox overflow"):
        comm0.send(1, key, data2)


def test_host_var():
    """Test DriverVar behavior."""
    values = [10, 20, 30]
    hv = DriverVar(values)

    assert hv[0] == 10
    assert hv[1] == 20
    assert hv[2] == 30
    assert repr(hv) == f"DriverVar({values})"
