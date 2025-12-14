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

"""OT Extension Tests using correct mp API."""

import numpy as np

import mplang.v2 as mp
from mplang.v2.dialects import simp
from mplang.v2.libs.mpc.ot import extension as ot_extension


def test_transfer_extension():
    """Test IKNP OT extension correctness."""
    sim = simp.make_simulator(2)
    mp.set_root_context(sim)

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

    traced = mp.compile(protocol)
    result = mp.evaluate(traced)

    # Fetch result - it's on receiver
    res = mp.fetch(result)[RECEIVER]

    # Verify: if choice=0, res=m0. if choice=1, res=m1.
    expected = np.where(np_choices[:, None] == 0, np_m0, np_m1)
    np.testing.assert_array_equal(res, expected)
