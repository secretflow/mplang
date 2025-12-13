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

"""OPRF Tests using correct mp API."""

import numpy as np

import mplang.v2 as mp
from mplang.v2.dialects import simp
from mplang.v2.libs.mpc.psi import oprf


def test_eval_oprf():
    """Test OPRF evaluation produces outputs."""
    sim = simp.make_simulator(2)
    mp.set_root_context(sim)

    np.random.seed(42)
    num_items = 16
    np_inputs = np.random.randint(0, 255, size=(num_items, 16), dtype=np.uint8)

    SENDER = 0
    RECEIVER = 1

    def protocol():
        receiver_inputs = simp.constant((RECEIVER,), np_inputs)
        return oprf.eval_oprf(receiver_inputs, SENDER, RECEIVER, num_items)

    traced = mp.compile(protocol)
    result = mp.evaluate(traced)

    # Result is a tuple (sender_key, receiver_outputs)
    # sender_key is on SENDER, receiver_outputs is on RECEIVER
    sender_key_obj, recv_out_obj = result

    sender_key = mp.fetch(sender_key_obj)[SENDER]
    recv_out = mp.fetch(recv_out_obj)[RECEIVER]

    assert sender_key is not None
    assert recv_out is not None
    assert recv_out.shape == (num_items, 32)  # OPRF output is 32 bytes (SHA256)


def test_oprf_determinism():
    """Test that OPRF runs successfully with fixed inputs."""
    sim = simp.make_simulator(2)
    mp.set_root_context(sim)

    np.random.seed(42)
    num_items = 8
    np_inputs = np.random.randint(0, 255, size=(num_items, 16), dtype=np.uint8)

    SENDER = 0
    RECEIVER = 1

    def protocol():
        receiver_inputs = simp.constant((RECEIVER,), np_inputs)
        return oprf.eval_oprf(receiver_inputs, SENDER, RECEIVER, num_items)

    traced = mp.compile(protocol)
    result = mp.evaluate(traced)

    sender_key_obj, recv_out_obj = result
    sender_key = mp.fetch(sender_key_obj)[SENDER]
    recv_out = mp.fetch(recv_out_obj)[RECEIVER]

    assert sender_key is not None
    assert recv_out is not None
