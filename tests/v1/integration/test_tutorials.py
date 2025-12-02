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

# Integration tests derived from tutorial scripts.
# These use multi-party Simulator and cross-module features, so marked as integration.

from __future__ import annotations

import jax.numpy as jnp
import pytest

import mplang.v1 as mp

pytestmark = pytest.mark.integration


class TestTutorialBasicExamples:
    def test_tutorial_millionaire_basic(self):
        import random
        from functools import partial

        sim2 = mp.Simulator.simple(2)

        @mp.function
        def millionaire():
            range_rand = partial(random.randint, 0, 10)
            x = mp.run_jax_at(0, range_rand)
            y = mp.run_jax_at(1, range_rand)
            x_ = mp.seal_from(0, x)
            y_ = mp.seal_from(1, y)
            z_ = mp.srun_jax(lambda x, y: x < y, x_, y_)
            r = mp.reveal(z_)
            return x, y, r

        x, y, z = mp.evaluate(sim2, millionaire)
        x_val, y_val, z_val = mp.fetch(sim2, (x, y, z))
        expected = x_val[0] < y_val[1]
        assert all(result == expected for result in z_val)


class TestTutorialConditionalExamples:
    def test_negate_if_local_cond(self):
        sim3 = mp.Simulator.simple(3)

        @mp.function
        def negate_if_local_cond():
            x = mp.prandint(0, 20)
            p = mp.run_jax(lambda x: x <= 10, x)
            pos = mp.run_jax(lambda x: x, x)
            neg = mp.run_jax(lambda x: -x, x)
            z = mp.run_jax(jnp.where, p, pos, neg)
            return x, z

        x, z = mp.evaluate(sim3, negate_if_local_cond)
        x_vals, z_vals = mp.fetch(sim3, (x, z))
        for _i, (x_val, z_val) in enumerate(zip(x_vals, z_vals, strict=False)):
            if x_val <= 10:
                assert z_val == x_val
            else:
                assert z_val == -x_val

    def test_negate_if_shared_cond(self):
        sim3 = mp.Simulator.simple(3)

        @mp.function
        def negate_if_shared_cond():
            x = mp.prandint(0, 10)
            xs_ = mp.seal(x)
            pred_ = mp.srun_jax(lambda xs: jnp.sum(jnp.stack(xs), axis=0) < 15, xs_)
            pred = mp.reveal(pred_)
            pos = mp.run_jax(lambda x: x, x)
            neg = mp.run_jax(lambda x: -x, x)
            z = mp.run_jax(jnp.where, pred, pos, neg)
            return x, z

        x, z = mp.evaluate(sim3, negate_if_shared_cond)
        x_vals, z_vals = mp.fetch(sim3, (x, z))
        sum_x = sum(x_vals)
        if sum_x < 15:
            for _i, (x_val, z_val) in enumerate(zip(x_vals, z_vals, strict=False)):
                assert z_val == x_val
        else:
            for _i, (x_val, z_val) in enumerate(zip(x_vals, z_vals, strict=False)):
                assert z_val == -x_val

    def test_party_branch_on_cond(self):
        sim3 = mp.Simulator.simple(3)

        @mp.function
        def party_branch_on_cond():
            x = mp.constant(5)
            y = mp.constant(10)
            pred = mp.run_jax(lambda rank: rank < 2, mp.prank())
            z = mp.uniform_cond(
                pred,
                lambda a, b: mp.run_jax(jnp.add, a, b),
                lambda a, b: mp.run_jax(jnp.subtract, a, b),
                x,
                y,
            )
            return x, z

        x, z = mp.evaluate(sim3, party_branch_on_cond)
        _x_vals, z_vals = mp.fetch(sim3, (x, z))
        assert z_vals[0] == 15
        assert z_vals[1] == 15
        assert z_vals[2] == -5


class TestTutorialWhileLoopExamples:
    def test_while_party_local(self):
        sim2 = mp.Simulator.simple(2)

        @mp.function
        def while_party_local():
            x = mp.prandint(0, 10)

            def cond(x: mp.MPObject):
                return mp.run_jax(lambda x: x < 15, x)

            def body(x: mp.MPObject):
                return mp.run_jax(lambda x: x + 1, x)

            r = mp.while_loop(cond, body, x)
            return x, r

        x, r = mp.evaluate(sim2, while_party_local)
        _x_vals, r_vals = mp.fetch(sim2, (x, r))
        for r_val in r_vals:
            assert r_val >= 15

    def test_while_sum_greater(self):
        sim2 = mp.Simulator.simple(2)

        @mp.function
        def while_sum_greater():
            x = mp.prandint(0, 10)

            def cond(x: mp.MPObject):
                xs_ = mp.seal(x)
                pred_ = mp.srun_jax(lambda i: sum(i) < 15, xs_)
                return mp.reveal(pred_)

            def body(x: mp.MPObject):
                return mp.run_jax(lambda x: x + 1, x)

            r = mp.while_loop(cond, body, x)
            return x, r

        x, r = mp.evaluate(sim2, while_sum_greater)
        _x_vals, r_vals = mp.fetch(sim2, (x, r))
        sum_r = sum(r_vals)
        assert sum_r >= 15

    def test_while_until_ascending(self):
        sim3 = mp.Simulator.simple(3)

        @mp.function
        def while_simple():
            x = mp.prandint(0, 3)

            def cond(x: mp.MPObject):
                return mp.run_jax(lambda x: x < 2, x)

            def body(x: mp.MPObject):
                return mp.run_jax(lambda x: x + 1, x)

            z = mp.while_loop(cond, body, x)
            return z

        z = mp.evaluate(sim3, while_simple)
        z_vals = mp.fetch(sim3, z)
        for val in z_vals:
            assert val >= 2


class TestTutorialAdvancedExamples:
    def test_pass_and_capture(self):
        sim3 = mp.Simulator.simple(3)
        x = mp.evaluate(sim3, mp.prank)
        y = mp.evaluate(sim3, lambda: mp.prandint(0, 100))

        def pass_and_capture(x):
            return mp.run_jax(jnp.multiply, x, y)

        z = mp.evaluate(sim3, pass_and_capture, x)
        z_vals = mp.fetch(sim3, z)
        x_vals = mp.fetch(sim3, x)
        y_vals = mp.fetch(sim3, y)
        for _i, (x_val, y_val, z_val) in enumerate(
            zip(x_vals, y_vals, z_vals, strict=False)
        ):
            assert z_val == x_val * y_val

    def test_jitted_function(self):
        sim3 = mp.Simulator.simple(3)
        x = mp.evaluate(sim3, mp.prank)
        y = mp.evaluate(sim3, lambda: mp.prandint(0, 100))

        def pass_and_capture(x):
            return mp.run_jax(jnp.multiply, x, y)

        jitted = mp.function(pass_and_capture)
        z1 = mp.evaluate(sim3, jitted, x)
        z1_vals = mp.fetch(sim3, z1)
        x_vals = mp.fetch(sim3, x)
        y_vals = mp.fetch(sim3, y)
        for _i, (x_val, y_val, z1_val) in enumerate(
            zip(x_vals, y_vals, z1_vals, strict=False)
        ):
            assert z1_val == x_val * y_val


class TestTutorialSimulationExamples:
    def test_simulation_millionaire(self):
        sim2 = mp.Simulator.simple(2)

        @mp.function
        def millionaire():
            x = mp.prandint(0, 10)
            y = mp.prandint(0, 10)
            x_ = mp.seal_from(0, x)
            y_ = mp.seal_from(1, y)
            z_ = mp.srun_jax(lambda x, y: x < y, x_, y_)
            z = mp.reveal(z_)
            return x, y, z

        x, y, z = mp.evaluate(sim2, millionaire)
        x_vals, y_vals, z_vals = mp.fetch(sim2, (x, y, z))
        expected = x_vals[0] < y_vals[1]
        assert all(result == expected for result in z_vals)


class TestTutorialErrorHandling:
    def test_invalid_party_mask(self):
        sim2 = mp.Simulator.simple(2)

        @mp.function
        def test_runAt_invalid_party():
            return mp.run_jax_at(5, lambda: 42)

        with pytest.raises(
            ValueError, match="Specified rmask 32 is not a subset of deduced pmask 3"
        ):
            mp.evaluate(sim2, test_runAt_invalid_party)

    def test_mismatched_seal_operations(self):
        sim2 = mp.Simulator.simple(2)

        @mp.function
        def mismatched_seal():
            x = mp.prandint(0, 10)
            return mp.seal_from(5, x)

        with pytest.raises(Exception):
            mp.evaluate(sim2, mismatched_seal)

    def test_empty_while_loop(self):
        sim2 = mp.Simulator.simple(2)

        @mp.function
        def empty_while():
            x = mp.constant(10)

            def cond(x):
                return mp.run_jax(lambda x: x < 5, x)

            def body(x):
                return mp.run_jax(lambda x: x + 1, x)

            return mp.while_loop(cond, body, x)

        result = mp.evaluate(sim2, empty_while)
        result_vals = mp.fetch(sim2, result)
        assert all(val == 10 for val in result_vals)


class TestTutorialDataStructures:
    def test_list_operations(self):
        sim2 = mp.Simulator.simple(2)

        @mp.function
        def simple_gather():
            val = mp.prank()
            # Build a source mask covering both parties explicitly for type clarity
            src_mask = mp.Mask.from_ranks([0, 1])
            gathered = mp.gather_m(src_mask, 0, val)
            return gathered

        result = mp.evaluate(sim2, simple_gather)
        result_vals = mp.fetch(sim2, result)
        assert len(result_vals) == 2
        assert result_vals[0] is not None or result_vals[1] is not None

    def test_nested_data_structures(self):
        sim2 = mp.Simulator.simple(2)

        @mp.function
        def simple_data():
            x = mp.constant(5)
            y = mp.constant(10)
            return x, y

        result = mp.evaluate(sim2, simple_data)
        x_vals, y_vals = mp.fetch(sim2, result)
        assert all(val == 5 for val in x_vals)
        assert all(val == 10 for val in y_vals)
