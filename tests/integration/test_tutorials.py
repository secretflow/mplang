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

import mplang
import mplang.simp as simp
from mplang.core import Mask

pytestmark = pytest.mark.integration


class TestTutorialBasicExamples:
    def test_tutorial_millionaire_basic(self):
        import random
        from functools import partial

        sim2 = mplang.Simulator.simple(2)

        @mplang.function
        def millionaire():
            range_rand = partial(random.randint, 0, 10)
            x = simp.runAt(0, range_rand)()
            y = simp.runAt(1, range_rand)()
            x_ = simp.sealFrom(x, 0)
            y_ = simp.sealFrom(y, 1)
            z_ = simp.srun(lambda x, y: x < y)(x_, y_)
            z = simp.reveal(z_)
            return x, y, z

        x, y, z = mplang.evaluate(sim2, millionaire)
        x_val, y_val, z_val = mplang.fetch(sim2, (x, y, z))
        expected = x_val[0] < y_val[1]
        assert all(result == expected for result in z_val)


class TestTutorialConditionalExamples:
    def test_negate_if_local_cond(self):
        sim3 = mplang.Simulator.simple(3)

        @mplang.function
        def negate_if_local_cond():
            x = simp.prandint(0, 20)
            p = simp.run(lambda x: x <= 10)(x)
            pos = simp.run(lambda x: x)(x)
            neg = simp.run(lambda x: -x)(x)
            z = simp.run(jnp.where)(p, pos, neg)
            return x, z

        x, z = mplang.evaluate(sim3, negate_if_local_cond)
        x_vals, z_vals = mplang.fetch(sim3, (x, z))
        for _i, (x_val, z_val) in enumerate(zip(x_vals, z_vals, strict=False)):
            if x_val <= 10:
                assert z_val == x_val
            else:
                assert z_val == -x_val

    def test_negate_if_shared_cond(self):
        sim3 = mplang.Simulator.simple(3)

        @mplang.function
        def negate_if_shared_cond():
            x = simp.prandint(0, 10)
            xs_ = simp.seal(x)
            pred_ = simp.srun(lambda xs: jnp.sum(jnp.stack(xs), axis=0) < 15)(xs_)
            pred = simp.reveal(pred_)
            pos = simp.run(lambda x: x)(x)
            neg = simp.run(lambda x: -x)(x)
            z = simp.run(jnp.where)(pred, pos, neg)
            return x, z

        x, z = mplang.evaluate(sim3, negate_if_shared_cond)
        x_vals, z_vals = mplang.fetch(sim3, (x, z))
        sum_x = sum(x_vals)
        if sum_x < 15:
            for _i, (x_val, z_val) in enumerate(zip(x_vals, z_vals, strict=False)):
                assert z_val == x_val
        else:
            for _i, (x_val, z_val) in enumerate(zip(x_vals, z_vals, strict=False)):
                assert z_val == -x_val

    def test_party_branch_on_cond(self):
        sim3 = mplang.Simulator.simple(3)

        @mplang.function
        def party_branch_on_cond():
            x = simp.constant(5)
            y = simp.constant(10)
            pred = simp.run(lambda rank: rank < 2)(simp.prank())
            z = simp.uniform_cond(
                pred,
                simp.run(jnp.add),
                simp.run(jnp.subtract),
                x,
                y,
            )
            return x, z

        x, z = mplang.evaluate(sim3, party_branch_on_cond)
        _x_vals, z_vals = mplang.fetch(sim3, (x, z))
        assert z_vals[0] == 15
        assert z_vals[1] == 15
        assert z_vals[2] == -5


class TestTutorialWhileLoopExamples:
    def test_while_party_local(self):
        sim2 = mplang.Simulator.simple(2)

        @mplang.function
        def while_party_local():
            x = simp.prandint(0, 10)

            def cond(x: simp.MPObject):
                return simp.run(lambda x: x < 15)(x)

            def body(x: simp.MPObject):
                return simp.run(lambda x: x + 1)(x)

            r = simp.while_loop(cond, body, x)
            return x, r

        x, r = mplang.evaluate(sim2, while_party_local)
        _x_vals, r_vals = mplang.fetch(sim2, (x, r))
        for r_val in r_vals:
            assert r_val >= 15

    def test_while_sum_greater(self):
        sim2 = mplang.Simulator.simple(2)

        @mplang.function
        def while_sum_greater():
            x = simp.prandint(0, 10)

            def cond(x: simp.MPObject):
                xs_ = simp.seal(x)
                pred_ = simp.srun(lambda i: sum(i) < 15)(xs_)
                return simp.reveal(pred_)

            def body(x: simp.MPObject):
                return simp.run(lambda x: x + 1)(x)

            r = simp.while_loop(cond, body, x)
            return x, r

        x, r = mplang.evaluate(sim2, while_sum_greater)
        _x_vals, r_vals = mplang.fetch(sim2, (x, r))
        sum_r = sum(r_vals)
        assert sum_r >= 15

    def test_while_until_ascending(self):
        sim3 = mplang.Simulator.simple(3)

        @mplang.function
        def while_simple():
            x = simp.prandint(0, 3)

            def cond(x: simp.MPObject):
                return simp.run(lambda x: x < 2)(x)

            def body(x: simp.MPObject):
                return simp.run(lambda x: x + 1)(x)

            z = simp.while_loop(cond, body, x)
            return z

        z = mplang.evaluate(sim3, while_simple)
        z_vals = mplang.fetch(sim3, z)
        for val in z_vals:
            assert val >= 2


class TestTutorialAdvancedExamples:
    def test_pass_and_capture(self):
        sim3 = mplang.Simulator.simple(3)
        x = mplang.evaluate(sim3, simp.prank)
        y = mplang.evaluate(sim3, lambda: simp.prandint(0, 100))

        def pass_and_capture(x):
            return simp.run(jnp.multiply)(x, y)

        z = mplang.evaluate(sim3, pass_and_capture, x)
        z_vals = mplang.fetch(sim3, z)
        x_vals = mplang.fetch(sim3, x)
        y_vals = mplang.fetch(sim3, y)
        for _i, (x_val, y_val, z_val) in enumerate(
            zip(x_vals, y_vals, z_vals, strict=False)
        ):
            assert z_val == x_val * y_val

    def test_jitted_function(self):
        sim3 = mplang.Simulator.simple(3)
        x = mplang.evaluate(sim3, simp.prank)
        y = mplang.evaluate(sim3, lambda: simp.prandint(0, 100))

        def pass_and_capture(x):
            return simp.run(jnp.multiply)(x, y)

        jitted = mplang.function(pass_and_capture)
        z1 = mplang.evaluate(sim3, jitted, x)
        z1_vals = mplang.fetch(sim3, z1)
        x_vals = mplang.fetch(sim3, x)
        y_vals = mplang.fetch(sim3, y)
        for _i, (x_val, y_val, z1_val) in enumerate(
            zip(x_vals, y_vals, z1_vals, strict=False)
        ):
            assert z1_val == x_val * y_val


class TestTutorialSimulationExamples:
    def test_simulation_millionaire(self):
        sim2 = mplang.Simulator.simple(2)

        @mplang.function
        def millionaire():
            x = simp.prandint(0, 10)
            y = simp.prandint(0, 10)
            x_ = simp.sealFrom(x, 0)
            y_ = simp.sealFrom(y, 1)
            z_ = simp.srun(lambda x, y: x < y)(x_, y_)
            z = simp.reveal(z_)
            return x, y, z

        x, y, z = mplang.evaluate(sim2, millionaire)
        x_vals, y_vals, z_vals = mplang.fetch(sim2, (x, y, z))
        expected = x_vals[0] < y_vals[1]
        assert all(result == expected for result in z_vals)


class TestTutorialErrorHandling:
    def test_invalid_party_mask(self):
        sim2 = mplang.Simulator.simple(2)

        @mplang.function
        def test_runAt_invalid_party():
            return simp.runAt(5, lambda: 42)()

        with pytest.raises(
            ValueError, match="Specified rmask 32 is not a subset of deduced pmask 3"
        ):
            mplang.evaluate(sim2, test_runAt_invalid_party)

    def test_mismatched_seal_operations(self):
        sim2 = mplang.Simulator.simple(2)

        @mplang.function
        def mismatched_seal():
            x = simp.prandint(0, 10)
            return simp.sealFrom(x, 5)

        with pytest.raises(Exception):
            mplang.evaluate(sim2, mismatched_seal)

    def test_empty_while_loop(self):
        sim2 = mplang.Simulator.simple(2)

        @mplang.function
        def empty_while():
            x = simp.constant(10)

            def cond(x):
                return simp.run(lambda x: x < 5)(x)

            def body(x):
                return simp.run(lambda x: x + 1)(x)

            return simp.while_loop(cond, body, x)

        result = mplang.evaluate(sim2, empty_while)
        result_vals = mplang.fetch(sim2, result)
        assert all(val == 10 for val in result_vals)


class TestTutorialDataStructures:
    def test_list_operations(self):
        sim2 = mplang.Simulator.simple(2)

        @mplang.function
        def simple_gather():
            val = simp.prank()
            # Build a source mask covering both parties explicitly for type clarity
            src_mask = Mask.from_ranks([0, 1])
            gathered = simp.gather_m(src_mask, 0, val)
            return gathered

        result = mplang.evaluate(sim2, simple_gather)
        result_vals = mplang.fetch(sim2, result)
        assert len(result_vals) == 2
        assert result_vals[0] is not None or result_vals[1] is not None

    def test_nested_data_structures(self):
        sim2 = mplang.Simulator.simple(2)

        @mplang.function
        def simple_data():
            x = simp.constant(5)
            y = simp.constant(10)
            return x, y

        result = mplang.evaluate(sim2, simple_data)
        x_vals, y_vals = mplang.fetch(sim2, result)
        assert all(val == 5 for val in x_vals)
        assert all(val == 10 for val in y_vals)
