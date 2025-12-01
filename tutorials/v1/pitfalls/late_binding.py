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

import mplang.v1 as mp


def pure_python_func():
    """
    In this function, we provided an example of late binding in closures that is inherent in Python.
    """
    print("======== pure python func with late binding begin: =========\n")
    funcs = []
    for i in range(3):
        # we may expect that the function will print 0, 1, 2, but it will print 2, 2, 2.
        # This is because the function is not executed immediately, but is deferred until the loop completes.
        # At that time, the value of i is 2, so all functions will print 2.
        funcs.append(lambda: print(i))

    print("call functions:")
    for f in funcs:
        f()
    print("======== pure python func with late binding end: =========\n")


@mp.function
def _run_wrong_func():
    values = []

    init = mp.constant(jnp.array(0))
    for i in range(3):

        def _update_offset(x):
            return x + i

        # Similar to the issue in pure Python, we expected to append 0, 1, and 2 in each iteration of the loop,
        # but we actually ended up with 2, 2, 2.
        offset = mp.run_jax(_update_offset, init)
        values.append(offset)

    return values


def wrong_func():
    print("======== wrong func, with late binding in mplang begin: =========\n")
    sim2 = mp.Simulator.simple(2)
    values = mp.evaluate(sim2, _run_wrong_func)
    print("fetch results: ", mp.fetch(sim2, values))

    # It's hard to distinguish the issue directly from the IR,
    # because the function `_update_offset` will be determined at runtime.
    print("compiled program: ", mp.compile(sim2, _run_wrong_func).compiler_ir())
    print("======== wrong func, with late binding in mplang end: =========\n")


@mp.function
def _run_correct_func():
    values = []
    values1 = []

    init = mp.constant(jnp.array(0))
    for i in range(3):
        # There are some tricks to avoid the issue
        def _update_offset(x, i=i):
            return x + i

        # 1. Use default argument to capture the value of i,
        # In Python, default parameters for functions are evaluated at the time of function definition,
        # not at the time of function call.
        # We get [0,1,2] here.
        offset = mp.run_jax(_update_offset, init)

        # 2. Just pass the value of i to the function directly
        # We get [0,10,20] here.
        offset1 = mp.run_jax(_update_offset, init, i * 10)

        values.append(offset)
        values1.append(offset1)

    return values, values1


def correct_func():
    print("======== correct func, with late binding in mplang begin: =========\n")
    sim2 = mp.Simulator.simple(2)
    values = mp.evaluate(sim2, _run_correct_func)
    print("fetch results: ", mp.fetch(sim2, values))
    print("compiled program: ", mp.compile(sim2, _run_correct_func).compiler_ir())
    print("======== correct func, with late binding in mplang end: =========\n")


if __name__ == "__main__":
    pure_python_func()
    wrong_func()
    correct_func()
