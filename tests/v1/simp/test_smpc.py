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
import numpy as np
import pytest

import mplang
import mplang.v1.core.primitive as prim
import mplang.v1.simp.random as mpr
import mplang.v1.simp.smpc as smpc
from mplang.v1.simp.api import constant, prank, run_jax


class TestSmpcBasics:
    """Test basic SMPC operations"""

    def test_seal_and_reveal(self):
        """Test basic seal and reveal operations"""
        num_parties = 2
        sim = mplang.Simulator.simple(num_parties)

        @mplang.function
        def test_seal_reveal():
            # Each party has some data
            data = prank()
            # Seal the data - returns a list of sealed values
            sealed_list = smpc.seal(data)
            # Use srun_jax to perform computation on sealed values - pass list directly
            sum_result = smpc.srun_jax(lambda xs: xs[0] + xs[1], sealed_list)
            # Reveal the result
            revealed = smpc.reveal(sum_result)
            return data, revealed

        # printer = Printer(indent_size=2)
        # copts = mplang.CompileOptions(world_size=num_parties)
        # compiled = mplang.compile(copts, test_seal_reveal)
        # print("Compiled function IR:")
        # print(printer.print_expr(compiled.make_expr()))

        data, revealed = mplang.evaluate(sim, test_seal_reveal)
        data_vals, revealed_vals = mplang.fetch(sim, (data, revealed))

        # The revealed sum should equal the sum of original data
        expected_sum = sum(data_vals)
        assert all(r == expected_sum for r in revealed_vals)

    def test_srun_basic(self):
        """Test basic secure computation with srun_jax"""
        num_parties = 2
        sim = mplang.Simulator.simple(num_parties)

        @mplang.function
        def test_srun():
            # Each party has some data
            data = mpr.prandint(0, 10)
            # Seal the data - returns a list
            sealed_list = smpc.seal(data)
            # Compute sum securely - pass list directly to lambda
            sum_result = smpc.srun_jax(lambda xs: xs[0] + xs[1], sealed_list)
            # Reveal the result
            revealed_sum = smpc.reveal(sum_result)
            return data, revealed_sum

        data, sum_result = mplang.evaluate(sim, test_srun)
        data_vals, sum_vals = mplang.fetch(sim, (data, sum_result))

        # Verify the sum is correct
        expected_sum = sum(data_vals)
        assert all(s == expected_sum for s in sum_vals)


class TestSMPCComplexScenarios:
    """Test more complex SMPC scenarios"""

    def test_millionaire_problem(self):
        """Test the classic millionaire problem"""
        sim2 = mplang.Simulator.simple(2)

        @mplang.function
        def millionaire():
            # Each party generates a random wealth value
            wealth = mpr.prandint(0, 1000)

            # Seal both parties' values
            sealed_wealth = smpc.seal(wealth)

            # Secure comparison: is party 0's wealth < party 1's wealth?
            comparison = smpc.srun_jax(lambda w0, w1: w0 < w1, *sealed_wealth)

            # Reveal the result
            result = smpc.reveal(comparison)

            return wealth, result

        wealth, result = mplang.evaluate(sim2, millionaire)
        wealth_vals, result_vals = mplang.fetch(sim2, (wealth, result))

        # Verify the comparison result
        expected = wealth_vals[0] < wealth_vals[1]
        assert all(r == expected for r in result_vals)

    def test_federated_average(self):
        """Test federated averaging scenario"""
        sim3 = mplang.Simulator.simple(3)

        @mplang.function
        def federated_avg():
            # Each party has local data
            local_data = mpr.prandint(1, 100, shape=(5,))  # 5-dimensional vector

            # Seal all local data
            sealed_data = smpc.seal(local_data)

            # Compute secure average using all sealed data - pass list to lambda
            avg_result = smpc.srun_jax(
                lambda data_list: jnp.mean(jnp.stack(data_list), axis=0),
                sealed_data,
            )

            # Reveal the average
            revealed_avg = smpc.reveal(avg_result)

            return local_data, revealed_avg

        local_data, avg_result = mplang.evaluate(sim3, federated_avg)
        data_vals, avg_vals = mplang.fetch(sim3, (local_data, avg_result))

        # Verify the average is correct (with lower precision due to floating point)
        expected_avg = np.mean(data_vals, axis=0)
        np.testing.assert_array_almost_equal(avg_vals[0], expected_avg, decimal=3)

    def test_conditional_secure_computation(self):
        """Test conditional execution with secure computation"""
        sim2 = mplang.Simulator.simple(2)

        @mplang.function
        def conditional_secure():
            # Each party has a value
            x = mpr.prandint(0, 20)

            # Seal values for secure computation
            sealed_x = smpc.seal(x)

            # Secure condition: sum > threshold - pass list to lambda
            threshold = 15
            sum_result = smpc.srun_jax(lambda xs: xs[0] + xs[1] > threshold, sealed_x)
            condition = smpc.reveal(sum_result)

            # Conditional execution based on secure condition
            def then_branch(x):
                return run_jax(lambda x: x * 2, x)

            def else_branch(x):
                return run_jax(lambda x: x + 10, x)

            result = prim.uniform_cond(condition, then_branch, else_branch, x)

            return x, condition, result

        x, condition, result = mplang.evaluate(sim2, conditional_secure)
        x_vals, cond_vals, result_vals = mplang.fetch(sim2, (x, condition, result))

        # Verify conditional logic
        sum_x = sum(x_vals)
        expected_condition = sum_x > 15
        assert all(c == expected_condition for c in cond_vals)

        # Verify result computation
        for _i, (x_val, result_val) in enumerate(
            zip(x_vals, result_vals, strict=False)
        ):
            if expected_condition:
                assert result_val == x_val * 2
            else:
                assert result_val == x_val + 10

    def test_iterative_secure_computation(self):
        """Test iterative secure computation with while loop"""
        sim2 = mplang.Simulator.simple(2)

        @mplang.function
        def iterative_secure():
            # Start with small values
            x = constant(1)

            def cond(x):
                # Seal and check if sum < 10 - pass list to lambda
                sealed = smpc.seal(x)
                sum_result = smpc.srun_jax(lambda xs: xs[0] + xs[1] < 10, sealed)
                return smpc.reveal(sum_result)

            def body(x):
                return run_jax(lambda x: x + 1, x)

            result = prim.while_loop(cond, body, x)
            return result

        result = mplang.evaluate(sim2, iterative_secure)
        result_vals = mplang.fetch(sim2, result)

        # Should stop when sum reaches 10 (each party has 5)
        assert sum(result_vals) >= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
