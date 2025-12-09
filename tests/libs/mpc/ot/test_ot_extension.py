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

"""OT Extension Tests using clean simp.constant API."""

import numpy as np

from mplang.v2.backends.simp_simulator import SimpSimulator
from mplang.v2.dialects import simp
from mplang.v2.edsl import trace
from mplang.v2.libs.mpc.ot import extension as ot_extension


def run_protocol(sim: SimpSimulator, protocol_fn):
    """Helper to trace and run a protocol."""
    traced = trace(protocol_fn)
    return sim.evaluate_graph(traced.graph, [])


def test_transfer_extension():
    """Test IKNP OT extension correctness."""
    sim = SimpSimulator(world_size=2)

    np.random.seed(42)
    num_ots = 128

    # Messages: random bytes (N, 16)
    np_m0 = np.random.randint(0, 255, size=(num_ots, 16), dtype=np.uint8)
    np_m1 = np.random.randint(0, 255, size=(num_ots, 16), dtype=np.uint8)
    np_choices = np.random.randint(0, 2, size=(num_ots,), dtype=np.uint8)

    SENDER = 0
    RECEIVER = 1

    def protocol():
        # Sender holds messages, Receiver holds choices
        m0 = simp.constant((SENDER,), np_m0)
        m1 = simp.constant((SENDER,), np_m1)
        choices = simp.constant((RECEIVER,), np_choices)

        return ot_extension.transfer_extension(
            m0, m1, choices, SENDER, RECEIVER, num_ots
        )

    result = run_protocol(sim, protocol)

    # Result is on receiver
    res = result[RECEIVER].unwrap()

    # Verify: if choice=0, res=m0. if choice=1, res=m1.
    expected = np.where(np_choices[:, None] == 0, np_m0, np_m1)
    np.testing.assert_array_equal(res, expected)

    sim.shutdown()
