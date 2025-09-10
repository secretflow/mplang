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

import time
from functools import partial

import jax.numpy as jnp
import jax.random as jr

import mplang
import mplang.simp as simp


def rand_from_host():
    # NOTE: THIS FUNCTION COULD NOT BE USED IN A REAL-WORLD SCENARIO,
    # this function demostrate that the random number is generated from host,
    # NOT from the parties.

    # host can use python's builtin random module.
    import random

    def func():
        # This generates different random number according to parties' runtime.
        # Note: parties that do not have python runtime can not evaluate this function.
        # Note: use partial to make jax trace happy.
        my_rand = partial(random.randint, 0, 10)

        # all parties will generate the same random number, which is normally not what we want.
        x = simp.run(my_rand)()

        def randint(lo, hi):
            # In this case, host make a 'public' random seed and use jax to generate a random number.
            # The random state is recored in PFunction, so all parties will generate the same number.
            # Note: this function also can not run without python runtime.
            key = time.time_ns() & 0x0FFFFFFF
            key = jr.PRNGKey(key)
            return jr.randint(key, shape=(), minval=lo, maxval=hi, dtype=jnp.int32)

        # P1 will generate a random number, with the currant random context.
        y = simp.run(randint)(0, 10)

        def randint2(key, lo, hi):
            # it's OK to use jax.random.split here.
            # But the root key is known to the host and also known to all parties.
            # So it's a 'public random', not a private random number for parties..
            cur_key, next_key = jr.split(key)
            ret = jr.randint(cur_key, shape=(), minval=lo, maxval=hi, dtype=jnp.int32)
            return ret, next_key

        key = int(time.time_ns()) & 0x0FFFFFFF
        key = jr.PRNGKey(key)
        z = simp.run(randint2)(key, 0, 10)

        # In short, all parties will generate the same random number if the random state is generated
        # from the host, which is not what we want in a real-world scenario.
        # return x, y, z
        return {
            "runtime py rand": x,
            "runtime py seed + jax rand": y,
            "comptime seed + jax rand": z[0],
        }

    # copts = simp.CompileOptions(2)
    # print(simp.compile(copts, func))

    sim5 = mplang.Simulator.simple(5)
    mplang.set_ctx(sim5)
    res = func()
    from pprint import pprint

    pprint(mplang.fetch(None, res))


def rand_from_parties():
    # this function demostrate that the random number is generated from parties,
    # NOT from the host.

    def jr_split(key):
        # TODO: since MPObject tensor does not implement slicing yet.
        # subkey, key = simp.run(jr.split)(key) does not work.
        # we workaround it by splitting inside tracer.
        subkey, key = jr.split(key)
        return subkey, key

    @mplang.function
    def gen_randoms(lo, hi):
        # use simp.prandint primitive to generate a random seed.
        seed = simp.prandint(lo, hi)

        # use jax random module to generate random number for different distribution.
        key = simp.run(jr.PRNGKey)(seed)
        subkey, key = simp.run(jr_split)(key)
        r_int = simp.run(jr.randint)(subkey, shape=(), minval=lo, maxval=hi)
        subkey, key = simp.run(jr_split)(key)
        jr_normal = partial(jr.normal, shape=(3), dtype=jnp.float32)
        std_normal = simp.run(jr_normal)(subkey)
        subkey, key = simp.run(jr_split)(key)
        bernoulli = simp.run(jr.bernoulli)(subkey, p=0.8)
        return {
            "randint": r_int,
            "normal": std_normal,
            "bernoulli": bernoulli,
        }

    sim8 = mplang.Simulator.simple(8)
    mplang.set_ctx(sim8)
    res = gen_randoms(0, 100)
    from pprint import pprint

    pprint(mplang.fetch(None, res))


if __name__ == "__main__":
    rand_from_host()
    rand_from_parties()
