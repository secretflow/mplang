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

"""Tests for OT library."""

import numpy as np
import pytest

import mplang.v2 as mp
import mplang.v2.backends.tensor_impl  # noqa: F401 (registers tensor primitives)
from mplang.v2.dialects import simp
from mplang.v2.libs.mpc.ot import base as ot


class TestOTScalar:
    """Test suite for scalar 1-out-of-2 Oblivious Transfer."""

    def setup_method(self):
        """Initialize simulator for each test."""
        self.interp = simp.make_simulator(world_size=2)

    @pytest.mark.parametrize(
        "m0_val, m1_val, choice_val, expected",
        [
            # Basic cases
            (10, 20, 0, 10),
            (10, 20, 1, 20),
            # Zero values
            (0, 100, 0, 0),
            (0, 100, 1, 100),
            # Negative values
            (-42, -100, 0, -42),
            (-42, -100, 1, -100),
            # Mixed signs
            (-42, 42, 0, -42),
            (-42, 42, 1, 42),
            # Large values
            (2**30, 2**31 - 1, 1, 2**31 - 1),
            # Max/Min int64
            (2**63 - 1, 2**63 - 2, 0, 2**63 - 1),
            (-(2**63), -(2**63) + 1, 0, -(2**63)),
            # Equal messages
            (42, 42, 0, 42),
            (42, 42, 1, 42),
            # Powers of two
            (2**10, 2**20, 1, 2**20),
            # Binary values
            (1, 0, 0, 1),
            (1, 0, 1, 0),
        ],
    )
    def test_scalar_ot(self, m0_val, m1_val, choice_val, expected):
        """Test scalar OT with various inputs."""
        with self.interp:
            m0 = simp.constant((0,), m0_val)
            m1 = simp.constant((0,), m1_val)
            choice = simp.constant((1,), choice_val)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

            values = mp.fetch(res)
            result = values[1]
            assert result == expected


class TestOTVector:
    """Test suite for vectorized OT operations."""

    def setup_method(self):
        """Initialize simulator for each test."""
        self.interp = simp.make_simulator(world_size=2)

    def test_vector_all_zeros(self):
        """Test vectorized OT with all choices=0."""
        N = 10
        m0_data = np.arange(N, dtype=np.int64)
        m1_data = np.arange(N, N * 2, dtype=np.int64)
        choices = np.zeros(N, dtype=np.int64)

        with self.interp:
            m0 = simp.constant((0,), m0_data)
            m1 = simp.constant((0,), m1_data)
            choice = simp.constant((1,), choices)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

            values = mp.fetch(res)
            np.testing.assert_array_equal(values[1], m0_data)

    def test_vector_all_ones(self):
        """Test vectorized OT with all choices=1."""
        N = 10
        m0_data = np.arange(N, dtype=np.int64)
        m1_data = np.arange(N, N * 2, dtype=np.int64)
        choices = np.ones(N, dtype=np.int64)

        with self.interp:
            m0 = simp.constant((0,), m0_data)
            m1 = simp.constant((0,), m1_data)
            choice = simp.constant((1,), choices)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

            values = mp.fetch(res)
            np.testing.assert_array_equal(values[1], m1_data)

    def test_vector_mixed_choices(self):
        """Test vectorized OT with mixed choices (0 and 1)."""
        N = 10
        m0_data = np.full(N, 10, dtype=np.int64)
        m1_data = np.full(N, 20, dtype=np.int64)
        # Alternating 0 and 1: [0, 1, 0, 1, ...]
        choices = np.array([i % 2 for i in range(N)], dtype=np.int64)
        expected = np.where(choices == 0, m0_data, m1_data)

        with self.interp:
            m0 = simp.constant((0,), m0_data)
            m1 = simp.constant((0,), m1_data)
            choice = simp.constant((1,), choices)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

            values = mp.fetch(res)
            np.testing.assert_array_equal(values[1], expected)


class TestIKNPCore:
    """Test suite for IKNP OT extension core."""

    def setup_method(self):
        """Initialize simulator for each test."""
        self.sim = simp.make_simulator(2)

    def test_iknp_seeds_relationship(self):
        """Verify IKNP output: Q[i] = T[i] when d[i]=0, Q[i] = T[i]^S when d[i]=1."""
        import jax.numpy as jnp

        import mplang.v2 as mp
        from mplang.v2.dialects import crypto, tensor
        from mplang.v2.libs.mpc.ot import extension as ot

        K = 128
        sender = 0
        receiver = 1

        def job():
            # Receiver creates delta bits
            def _recv_prep():
                d_bytes = crypto.random_bytes(16)
                delta = tensor.run_jax(lambda b: b.view(jnp.uint64).reshape(2), d_bytes)

                def _unpack(d):
                    return jnp.unpackbits(d.view(jnp.uint8), bitorder="little")

                bits_u8 = tensor.run_jax(_unpack, delta)
                bits_reshaped = tensor.reshape(bits_u8, (128, 1))
                return delta, bits_reshaped

            delta_and_bits = simp.pcall_static((receiver,), _recv_prep)
            delta_bits = simp.pcall_static((receiver,), lambda x: x[1], delta_and_bits)

            # Run IKNP
            t_matrix, q_matrix, s_choices = ot.iknp_core(
                delta_bits, sender, receiver, K
            )
            return t_matrix, q_matrix, s_choices, delta_bits

        with self.sim:
            traced = mp.compile(job)
            t_obj, q_obj, s_obj, d_obj = mp.evaluate(traced)

            t_val = mp.fetch(t_obj)[sender]  # (128, 128)
            q_val = mp.fetch(q_obj)[receiver]  # (128, 128)
            s_val = mp.fetch(s_obj)[sender]  # (128,)
            d_val = mp.fetch(d_obj)[receiver]  # (128, 1)

        # Verify: Q[i] = T[i] ^ (d[i] * S)
        d_flat = d_val.reshape(-1)
        s_broad = np.tile(s_val, (128, 1))
        d_mask = d_flat.reshape(-1, 1)
        expected_q = t_val ^ (s_broad * d_mask)

        np.testing.assert_array_equal(
            q_val, expected_q, err_msg="IKNP seeds relationship broken!"
        )
