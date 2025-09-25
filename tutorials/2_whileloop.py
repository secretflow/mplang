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

import jax
import jax.numpy as jnp

import mplang
import mplang.simp as simp


@mplang.function
def while_party_local():
    # Parties random a number privately
    x = simp.prandint(0, 10)

    def cond(x: simp.MPObject):
        assert isinstance(x, simp.MPObject), x
        return simp.run(lambda x: x < 15)(x)

    def body(x: simp.MPObject):
        return simp.run(lambda x: x + 1)(x)

    # if < 15, each party increase itself.
    r = simp.while_loop(cond, body, x)

    return x, r


@mplang.function
def while_sum_greater():
    # Parties random a number privately
    x = simp.prandint(0, 10)

    def cond(x: simp.MPObject):
        # Seal all parties private
        xs_ = simp.seal(x)
        # Sum them and reveal it.
        pred_ = simp.srun(lambda i: sum(i) < 15)(xs_)
        return simp.reveal(pred_)

    def body(x: simp.MPObject):
        return simp.run(lambda x: x + 1)(x)

    # if < 15, each party increase itself.
    r = simp.while_loop(cond, body, x)

    return x, r


@mplang.function
def while_until_ascending():
    x = simp.prandint(0, 10)

    def not_ascending(data):
        data = jnp.asarray(data)

        def body_fun(carry, idx):
            is_not_ascending = carry | (data[idx] >= data[idx + 1])
            return is_not_ascending, None

        result, _ = jax.lax.scan(body_fun, False, jnp.arange(len(data) - 1))

        return result

    def cond(x: simp.MPObject):
        # seal it, or we can not directly compare all parties numbers.
        xs_ = simp.seal(x)
        # check if parties' numbers are accending
        p_ = simp.srun(not_ascending)(xs_)
        # reveal the result, all parties agree on it.
        return simp.reveal(p_)

    def body(x: simp.MPObject):
        # randomize a new number
        return simp.prandint(0, 10)

    z = simp.while_loop(cond, body, x)

    return z


if __name__ == "__main__":
    # Three party computation.
    sim2 = mplang.Simulator.simple(2)
    sim3 = mplang.Simulator.simple(3)
    sim4 = mplang.Simulator.simple(4)

    print("all parties increase until 15")
    x = mplang.evaluate(sim2, while_party_local)
    print("x:", mplang.fetch(sim2, x))

    print("all parties increase until sum >= 15")
    y = mplang.evaluate(sim3, while_sum_greater)
    print("y:", mplang.fetch(sim3, y))

    print("random until all parties ascending")
    z = mplang.evaluate(sim4, while_until_ascending)
    print("z:", mplang.fetch(sim4, z))
