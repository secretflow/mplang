"""Tests for OT library."""

import mplang2.backends.crypto_impl
import mplang2.backends.tensor_impl  # noqa: F401 (registers tensor primitives)
from mplang2.backends.simp_simulator import SimpSimulator
from mplang2.dialects import simp
from mplang2.libs import ot


class TestOTTransfer:
    """Test suite for 1-out-of-2 Oblivious Transfer."""

    def setup_method(self):
        """Initialize simulator for each test."""
        self.interp = SimpSimulator(world_size=2)

    def test_choice_1_gets_m1(self):
        """Test that choice=1 correctly selects m1."""
        with self.interp:
            m0 = simp.constant((0,), 10)
            m1 = simp.constant((0,), 20)
            choice = simp.constant((1,), 1)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

        assert res.runtime_obj.values[1] == 20

    def test_choice_0_gets_m0(self):
        """Test that choice=0 correctly selects m0."""
        with self.interp:
            m0 = simp.constant((0,), 10)
            m1 = simp.constant((0,), 20)
            choice = simp.constant((1,), 0)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

        assert res.runtime_obj.values[1] == 10

    def test_negative_values(self):
        """Test OT with negative integers."""
        with self.interp:
            m0 = simp.constant((0,), -42)
            m1 = simp.constant((0,), -100)
            choice = simp.constant((1,), 1)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

        assert res.runtime_obj.values[1] == -100

    def test_zero_values(self):
        """Test OT with zero values."""
        with self.interp:
            m0 = simp.constant((0,), 0)
            m1 = simp.constant((0,), 100)
            choice = simp.constant((1,), 0)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

        assert res.runtime_obj.values[1] == 0

    def test_large_values(self):
        """Test OT with large integer values."""
        with self.interp:
            m0 = simp.constant((0,), 2**30)
            m1 = simp.constant((0,), 2**31 - 1)
            choice = simp.constant((1,), 1)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

        assert res.runtime_obj.values[1] == 2**31 - 1

    def test_equal_messages(self):
        """Test OT when both messages are equal."""
        with self.interp:
            m0 = simp.constant((0,), 42)
            m1 = simp.constant((0,), 42)
            choice = simp.constant((1,), 0)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

        assert res.runtime_obj.values[1] == 42

        with self.interp:
            m0 = simp.constant((0,), 42)
            m1 = simp.constant((0,), 42)
            choice = simp.constant((1,), 1)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

        assert res.runtime_obj.values[1] == 42


class TestOTTransferEdgeCases:
    """Test suite for edge cases and different value types."""

    def setup_method(self):
        """Initialize simulator for each test."""
        self.interp = SimpSimulator(world_size=2)

    def test_both_messages_negative(self):
        """Test OT when both messages are negative."""
        with self.interp:
            m0 = simp.constant((0,), -50)
            m1 = simp.constant((0,), -100)
            choice = simp.constant((1,), 0)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

        assert res.runtime_obj.values[1] == -50

    def test_mixed_sign_messages(self):
        """Test OT with one positive and one negative message."""
        with self.interp:
            m0 = simp.constant((0,), -42)
            m1 = simp.constant((0,), 42)
            choice = simp.constant((1,), 1)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

        assert res.runtime_obj.values[1] == 42

    def test_max_int64(self):
        """Test OT with maximum int64 values."""
        max_val = 2**63 - 1
        with self.interp:
            m0 = simp.constant((0,), max_val)
            m1 = simp.constant((0,), max_val - 1)
            choice = simp.constant((1,), 0)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

        assert res.runtime_obj.values[1] == max_val

    def test_min_int64(self):
        """Test OT with minimum int64 values."""
        min_val = -(2**63)
        with self.interp:
            m0 = simp.constant((0,), min_val)
            m1 = simp.constant((0,), min_val + 1)
            choice = simp.constant((1,), 1)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

        assert res.runtime_obj.values[1] == min_val + 1

    def test_powers_of_two(self):
        """Test OT with powers of two."""
        with self.interp:
            m0 = simp.constant((0,), 2**10)
            m1 = simp.constant((0,), 2**20)
            choice = simp.constant((1,), 1)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

        assert res.runtime_obj.values[1] == 2**20

    def test_one_and_zero(self):
        """Test OT with simple 0 and 1 values."""
        with self.interp:
            m0 = simp.constant((0,), 1)
            m1 = simp.constant((0,), 0)
            choice = simp.constant((1,), 0)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

        assert res.runtime_obj.values[1] == 1

        with self.interp:
            m0 = simp.constant((0,), 1)
            m1 = simp.constant((0,), 0)
            choice = simp.constant((1,), 1)

            res = ot.transfer(m0, m1, choice, sender=0, receiver=1)

        assert res.runtime_obj.values[1] == 0


# NOTE: Vectorized OT tests are commented out because the current
# crypto backend implementation does not support vectorized operations.
# The point_to_bytes operation expects a single Point object, not an array.
#
# @pytest.mark.skip(reason="Vectorized OT not yet supported by crypto backend")
# class TestOTTransferVectorized:
#     """Test suite for vectorized OT operations (future work)."""
#
#     def setup_method(self):
#         """Initialize simulator for each test."""
#         self.interp = SimpSimulator(world_size=2)
#
#     def test_vector_choice_0(self):
#         """Test vectorized OT with all choices=0."""
#         N = 10
#         m0_data = np.arange(N, dtype=np.int64)
#         m1_data = np.arange(N, N * 2, dtype=np.int64)
#         choices = np.zeros(N, dtype=np.int64)
#
#         with self.interp:
#             m0 = simp.pcall_static((0,), lambda: tensor.constant(m0_data))
#             m1 = simp.pcall_static((0,), lambda: tensor.constant(m1_data))
#             choice = simp.pcall_static((1,), lambda: tensor.constant(choices))
#
#             res = ot.transfer(m0, m1, choice, sender=0, receiver=1)
#
#         np.testing.assert_array_equal(res.runtime_obj.values[1], m0_data)
