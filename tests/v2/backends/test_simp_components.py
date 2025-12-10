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

import threading
import time
from unittest.mock import MagicMock

import pytest

from mplang.v2.backends.simp_host import HostVar, SimpHost
from mplang.v2.backends.simp_simulator import ThreadCommunicator
from mplang.v2.edsl.graph import Operation


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
    """Test HostVar behavior."""
    values = [10, 20, 30]
    hv = HostVar(values)

    assert hv[0] == 10
    assert hv[1] == 20
    assert hv[2] == 30
    assert repr(hv) == f"HostVar({values})"


class MockSimpHost(SimpHost):
    """Mock implementation of SimpHost for testing evaluate_graph."""

    def __init__(self, world_size):
        super().__init__(world_size)
        self.submit_calls = []
        self.collect_return = []

    def _submit(self, rank, graph, inputs, job_id=None):
        self.submit_calls.append((rank, graph, inputs))
        return f"future_{rank}"

    def _collect(self, futures):
        return self.collect_return


def test_simp_host_evaluate_graph():
    """Test SimpHost.evaluate_graph logic."""
    world_size = 3
    host = MockSimpHost(world_size)

    # Create a dummy graph with inputs
    graph = Operation(opcode="test_op", inputs=[], outputs=[], attrs={}, regions=[])
    # Mock inputs and outputs for the graph
    in1_val = MagicMock(name="in1")
    in2_val = MagicMock(name="in2")
    graph.inputs = [in1_val, in2_val]
    graph.outputs = [MagicMock(), MagicMock()]  # 2 outputs

    # Inputs as list (matching graph.inputs order)
    # Use Mock objects to represent remote references, not raw values
    # This better reflects that HostVar holds references in a distributed setting
    ref0 = MagicMock(name="ref_party0")
    ref1 = MagicMock(name="ref_party1")
    ref2 = MagicMock(name="ref_party2")
    hv_input = HostVar([ref0, ref1, ref2])

    const_input = 100
    inputs = [hv_input, const_input]

    # Mock collect return: list of (out1, out2) for each party
    # These represent the results fetched from workers
    # Party 0: (11, 101)
    # Party 1: (12, 102)
    # Party 2: (13, 103)
    host.collect_return = [(11, 101), (12, 102), (13, 103)]

    results = host.evaluate_graph(graph, inputs)

    # Check submit calls
    assert len(host.submit_calls) == 3
    for rank in range(3):
        r, g, i = host.submit_calls[rank]
        assert r == rank
        assert g == graph
        # Verify that the correct reference was dispatched to the correct worker
        # i is now a list [hv_input[rank], const_input]
        assert i[0] == hv_input[rank]
        assert i[1] == const_input

    # Check results
    # Should be [HostVar([11, 12, 13]), HostVar([101, 102, 103])]
    assert isinstance(results, list)
    assert len(results) == 2
    assert isinstance(results[0], HostVar)
    assert results[0].values == [11, 12, 13]
    assert isinstance(results[1], HostVar)
    assert results[1].values == [101, 102, 103]


def test_simp_host_evaluate_graph_single_output():
    """Test SimpHost.evaluate_graph with single output."""
    world_size = 2
    host = MockSimpHost(world_size)

    graph = Operation(opcode="test_op", inputs=[], outputs=[], attrs={}, regions=[])
    graph.inputs = []  # no inputs
    graph.outputs = [MagicMock()]  # 1 output

    inputs = []  # empty list

    # Mock collect return: list of single values
    host.collect_return = [42, 43]

    result = host.evaluate_graph(graph, inputs)

    assert isinstance(result, HostVar)
    assert result.values == [42, 43]


def test_simp_host_evaluate_graph_no_output():
    """Test SimpHost.evaluate_graph with no output."""
    world_size = 2
    host = MockSimpHost(world_size)

    graph = Operation(opcode="test_op", inputs=[], outputs=[], attrs={}, regions=[])
    graph.inputs = []  # no inputs
    graph.outputs = []  # 0 outputs

    inputs = []  # empty list
    host.collect_return = [None, None]

    result = host.evaluate_graph(graph, inputs)

    assert result is None
