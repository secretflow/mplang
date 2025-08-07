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

import mplang
import mplang.random as mpr
import mplang.simp as simp
from mplang import smpc


@mplang.function
def negate_if_local_cond():
    # Parties random a number privately
    x = mpr.prandint(0, 20)

    # Check if local var is less than a given number.
    p = simp.run(lambda x: x <= 10)(x)

    # Compute positive and negative of the value.
    # Note: pos and neg will not be evaluated until parties launched it.
    pos = simp.run(lambda x: x)(x)
    neg = simp.run(lambda x: -x)(x)

    # Note: Each party evaluate only one of **pos or neg** code path according to
    # p's value, it's something like SIMT.
    # r = simp.select(p, pos, neg)
    # z = simp.run(lambda p, x, y: jnp.where(p, x, y))(p, pos, neg)
    z = simp.run(jnp.where)(p, pos, neg)

    return x, z


@mplang.function
def negate_if_shared_cond():
    # Seal all parties private
    x = mpr.prandint(0, 10)

    # seal all parties private variable.
    _xs = smpc.seal(x)
    # assert len(_xs) == 2  # Fixed from simp.cur_ctx().psize()

    # Sum it privately, and compare to 15
    _pred = smpc.srun(lambda xs: jnp.sum(jnp.stack(xs), axis=0) < 15)(_xs)
    # Reveal the comparison result.
    pred = smpc.reveal(_pred)

    # if the sum is greater than 10, return it, else return negate of it.
    pos = simp.run(lambda x: x)(x)
    neg = simp.run(lambda x: -x)(x)

    # Note: Since pred is a revealed value (all parties have same value), so
    # only one of the branch statement will run.
    # r = simp.select(pred, pos, neg)
    z = simp.run(jnp.where)(pred, pos, neg)
    return x, z


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


if __name__ == "__main__":
    WORLD_SIZE = 3
    mplang.set_ctx(mplang.Simulator(WORLD_SIZE))

    print("negate if x_i > 10")
    x, r = negate_if_local_cond()
    print(mplang.compile(mplang.cur_ctx(), negate_if_local_cond).compiler_ir())
    print(mplang.fetch(None, (x, r)))

    print("negate if sum(x) >= 15")
    x, r = negate_if_shared_cond()
    print(mplang.compile(mplang.cur_ctx(), negate_if_shared_cond).compiler_ir())
    print(mplang.fetch(None, (x, r)))

    print("party_branch_on_cond")
    x, z = party_branch_on_cond()
    print(mplang.compile(mplang.cur_ctx(), party_branch_on_cond).compiler_ir())
    print(mplang.fetch(None, (x, z)))
