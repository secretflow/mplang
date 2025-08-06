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

import jax.numpy as jnp
import pytest

import mplang
import mplang.mpi as mpi
import mplang.random as mpr
import mplang.simp as simp
import mplang.smpc as smpc


class TestTutorialBasicExamples:
    """Test examples from 0_basic.py tutorial"""

    def test_tutorial_millionaire_basic(self):
        """Test the basic millionaire example from tutorial 0_basic.py"""
        import random
        from functools import partial

        sim2 = mplang.Simulator(2)

        @mplang.function
        def millionaire():
            range_rand = partial(random.randint, 0, 10)

            # P0 make a random number
            x = simp.runAt(0, range_rand)()
            # P1 make a random number too
            y = simp.runAt(1, range_rand)()

            # both of them seal it
            _x = smpc.sealFrom(x, 0)
            _y = smpc.sealFrom(y, 1)

            # compare it securely
            _z = smpc.srun(lambda x, y: x < y)(_x, _y)

            # reveal it to all
            z = smpc.reveal(_z)

            return x, y, z

        x, y, z = mplang.eval(sim2, millionaire)
        x_val, y_val, z_val = mplang.fetch(sim2, (x, y, z))

        # Verify the comparison result
        expected = x_val[0] < y_val[1]  # Party 0's value < Party 1's value
        assert all(result == expected for result in z_val)


class TestTutorialConditionalExamples:
    """Test examples from 1_condition.py tutorial"""

    def test_negate_if_local_cond(self):
        """Test local conditional execution from tutorial"""
        sim3 = mplang.Simulator(3)

        @mplang.function
        def negate_if_local_cond():
            # Parties random a number privately
            x = mpr.prandint(0, 20)

            # Check if local var is less than a given number
            p = simp.run(lambda x: x <= 10)(x)

            # Compute positive and negative of the value
            pos = simp.run(lambda x: x)(x)
            neg = simp.run(lambda x: -x)(x)

            # Each party evaluates only one code path according to p's value
            z = simp.run(jnp.where)(p, pos, neg)

            return x, z

        x, z = mplang.eval(sim3, negate_if_local_cond)
        x_vals, z_vals = mplang.fetch(sim3, (x, z))

        # Verify conditional logic for each party
        for i, (x_val, z_val) in enumerate(zip(x_vals, z_vals)):
            if x_val <= 10:
                assert z_val == x_val  # positive
            else:
                assert z_val == -x_val  # negative

    def test_negate_if_shared_cond(self):
        """Test shared conditional execution from tutorial"""
        sim3 = mplang.Simulator(3)

        @mplang.function
        def negate_if_shared_cond():
            # Seal all parties private
            x = mpr.prandint(0, 10)

            # seal all parties private variable
            _xs = smpc.seal(x)

            # Sum it privately, and compare to 15
            _pred = smpc.srun(lambda xs: jnp.sum(jnp.stack(xs), axis=0) < 15)(_xs)
            # Reveal the comparison result
            pred = smpc.reveal(_pred)

            # if the sum is greater than 15, return it, else return negate of it
            pos = simp.run(lambda x: x)(x)
            neg = simp.run(lambda x: -x)(x)

            z = simp.run(jnp.where)(pred, pos, neg)
            return x, z

        x, z = mplang.eval(sim3, negate_if_shared_cond)
        x_vals, z_vals = mplang.fetch(sim3, (x, z))

        # Verify shared conditional logic
        sum_x = sum(x_vals)
        if sum_x < 15:
            # All parties should have positive values
            for i, (x_val, z_val) in enumerate(zip(x_vals, z_vals)):
                assert z_val == x_val
        else:
            # All parties should have negative values
            for i, (x_val, z_val) in enumerate(zip(x_vals, z_vals)):
                assert z_val == -x_val

    def test_party_branch_on_cond(self):
        """Test party-specific branching from tutorial"""
        sim3 = mplang.Simulator(3)

        @mplang.function
        def party_branch_on_cond():
            x = simp.constant(5)
            y = simp.constant(10)
            pred = simp.run(lambda rank: rank < 2)(simp.prank())
            z = simp.cond(
                pred,
                simp.run(jnp.add),  # Party 0 and 1 will run this branch
                simp.run(jnp.subtract),  # Party 2 will run this branch
                x,
                y,
            )
            return x, z

        x, z = mplang.eval(sim3, party_branch_on_cond)
        x_vals, z_vals = mplang.fetch(sim3, (x, z))

        # Verify party-specific branching
        assert z_vals[0] == 15  # Party 0: 5 + 10
        assert z_vals[1] == 15  # Party 1: 5 + 10
        assert z_vals[2] == -5  # Party 2: 5 - 10


class TestTutorialWhileLoopExamples:
    """Test examples from 2_whileloop.py tutorial"""

    def test_while_party_local(self):
        """Test party-local while loop from tutorial"""
        sim2 = mplang.Simulator(2)

        @mplang.function
        def while_party_local():
            # Parties random a number privately
            x = mpr.prandint(0, 10)

            def cond(x: simp.MPObject):
                return simp.run(lambda x: x < 15)(x)

            def body(x: simp.MPObject):
                return simp.run(lambda x: x + 1)(x)

            # if < 15, each party increases itself
            r = simp.while_loop(cond, body, x)

            return x, r

        x, r = mplang.eval(sim2, while_party_local)
        x_vals, r_vals = mplang.fetch(sim2, (x, r))

        # Each party should reach at least 15
        for r_val in r_vals:
            assert r_val >= 15

    def test_while_sum_greater(self):
        """Test while loop with shared condition from tutorial"""
        sim2 = mplang.Simulator(2)

        @mplang.function
        def while_sum_greater():
            # Parties random a number privately
            x = mpr.prandint(0, 10)

            def cond(x: simp.MPObject):
                # Seal all parties private
                _xs = smpc.seal(x)
                # Sum them and reveal it
                _pred = smpc.srun(lambda i: sum(i) < 15)(_xs)
                return smpc.reveal(_pred)

            def body(x: simp.MPObject):
                return simp.run(lambda x: x + 1)(x)

            # if sum < 15, each party increases itself
            r = simp.while_loop(cond, body, x)

            return x, r

        x, r = mplang.eval(sim2, while_sum_greater)
        x_vals, r_vals = mplang.fetch(sim2, (x, r))

        # Sum should be at least 15
        sum_r = sum(r_vals)
        assert sum_r >= 15

    def test_while_until_ascending(self):
        """Test while loop until ascending order from tutorial (simplified)"""
        sim3 = mplang.Simulator(3)

        @mplang.function
        def while_simple():
            x = mpr.prandint(0, 3)  # Use smaller range for predictable results

            def cond(x: simp.MPObject):
                # Simple condition - stop when all values are >= 2
                return simp.run(lambda x: x < 2)(x)

            def body(x: simp.MPObject):
                return simp.run(lambda x: x + 1)(x)

            z = simp.while_loop(cond, body, x)
            return z

        z = mplang.eval(sim3, while_simple)
        z_vals = mplang.fetch(sim3, z)

        # Each party should have value >= 2
        for val in z_vals:
            assert val >= 2


class TestTutorialAdvancedExamples:
    """Test examples from 6_advanced.py tutorial"""

    def test_pass_and_capture(self):
        """Test parameter passing and variable capture from tutorial"""
        sim3 = mplang.Simulator(3)

        # Make two variables on the simulator
        x = mplang.eval(sim3, simp.prank)
        y = mplang.eval(sim3, lambda: mpr.prandint(0, 100))

        def pass_and_capture(x):
            # pass x as a parameter, and capture y from the outer scope
            return simp.run(jnp.multiply)(x, y)

        z = mplang.eval(sim3, pass_and_capture, x)
        z_vals = mplang.fetch(sim3, z)

        x_vals = mplang.fetch(sim3, x)
        y_vals = mplang.fetch(sim3, y)

        # Verify multiplication
        for i, (x_val, y_val, z_val) in enumerate(zip(x_vals, y_vals, z_vals)):
            assert z_val == x_val * y_val

    def test_jitted_function(self):
        """Test JIT compilation from tutorial"""
        sim3 = mplang.Simulator(3)

        x = mplang.eval(sim3, simp.prank)
        y = mplang.eval(sim3, lambda: mpr.prandint(0, 100))

        def pass_and_capture(x):
            return simp.run(jnp.multiply)(x, y)

        # JIT it
        jitted = mplang.function(pass_and_capture)

        z1 = mplang.eval(sim3, jitted, x)
        z1_vals = mplang.fetch(sim3, z1)

        x_vals = mplang.fetch(sim3, x)
        y_vals = mplang.fetch(sim3, y)

        # Verify jitted function works the same
        for i, (x_val, y_val, z1_val) in enumerate(zip(x_vals, y_vals, z1_vals)):
            assert z1_val == x_val * y_val


class TestTutorialSimulationExamples:
    """Test examples from 4_simulation.py tutorial"""

    def test_simulation_millionaire(self):
        """Test millionaire problem with simulation from tutorial"""
        sim2 = mplang.Simulator(2)

        @mplang.function
        def millionaire():
            # Use prandint instead of random.randint to avoid state capture
            x = mpr.prandint(0, 10)
            y = mpr.prandint(0, 10)

            # both of them seal it
            _x = smpc.sealFrom(x, 0)
            _y = smpc.sealFrom(y, 1)

            # compare it securely
            _z = smpc.srun(lambda x, y: x < y)(_x, _y)

            # reveal it to all
            z = smpc.reveal(_z)

            return x, y, z

        x, y, z = mplang.eval(sim2, millionaire)
        x_vals, y_vals, z_vals = mplang.fetch(sim2, (x, y, z))

        # Verify the comparison result
        expected = x_vals[0] < y_vals[1]  # Party 0's value < Party 1's value
        assert all(result == expected for result in z_vals)


class TestTutorialErrorHandling:
    """Test error handling and edge cases from tutorials"""

    def test_invalid_party_mask(self):
        """Test behavior with invalid party operations"""
        sim2 = mplang.Simulator(2)

        @mplang.function
        def test_runAt_invalid_party():
            # runAt with non-existent party should raise error
            return simp.runAt(5, lambda: 42)()

        # Should raise ValueError when trying to use non-existent party
        with pytest.raises(
            ValueError, match="Specified rmask 32 is not a subset of deduced pmask 3"
        ):
            mplang.eval(sim2, test_runAt_invalid_party)

    def test_mismatched_seal_operations(self):
        """Test error handling for mismatched seal operations"""
        sim2 = mplang.Simulator(2)

        @mplang.function
        def mismatched_seal():
            x = mpr.prandint(0, 10)
            # Try to seal from non-existent party
            return smpc.sealFrom(x, 5)

        with pytest.raises(Exception):  # Should raise some kind of error
            mplang.eval(sim2, mismatched_seal)

    def test_empty_while_loop(self):
        """Test while loop that doesn't execute"""
        sim2 = mplang.Simulator(2)

        @mplang.function
        def empty_while():
            x = simp.constant(10)

            def cond(x):
                return simp.run(lambda x: x < 5)(x)  # Never true

            def body(x):
                return simp.run(lambda x: x + 1)(x)

            return simp.while_loop(cond, body, x)

        result = mplang.eval(sim2, empty_while)
        result_vals = mplang.fetch(sim2, result)

        # Should return original value since loop never executes
        assert all(val == 10 for val in result_vals)


class TestTutorialDataStructures:
    """Test complex data structures from tutorials"""

    def test_list_operations(self):
        """Test simple gather operations"""
        sim2 = mplang.Simulator(2)

        @mplang.function
        def simple_gather():
            # All parties have a value
            val = simp.prank()  # Each party has its rank

            # Gather from all parties to party 0
            gathered = mpi.gather_m((1 << 2) - 1, 0, val)

            return gathered

        result = mplang.eval(sim2, simple_gather)
        result_vals = mplang.fetch(sim2, result)

        # Check that we got values from both parties
        assert len(result_vals) == 2  # Two parties
        # Just verify we got some result structure
        assert result_vals[0] is not None or result_vals[1] is not None

    def test_nested_data_structures(self):
        """Test simple data structures"""
        sim2 = mplang.Simulator(2)

        @mplang.function
        def simple_data():
            x = simp.constant(5)
            y = simp.constant(10)

            return x, y

        result = mplang.eval(sim2, simple_data)
        x_vals, y_vals = mplang.fetch(sim2, result)

        # Verify values are correct
        assert all(val == 5 for val in x_vals)
        assert all(val == 10 for val in y_vals)
