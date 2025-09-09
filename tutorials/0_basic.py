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

import random
from functools import partial

import mplang
import mplang.simp as simp


# This is a simple example of how to use the MPLang library to create a
# secure multi-party computation (MPC) program that simulates a millionaire
# problem, where two parties compare their private values without revealing them.
def millionaire():
    range_rand = partial(random.randint, 0, 10)

    # P0 make a random number.
    x = simp.runAt(0, range_rand)()
    # P1 make a random number too.
    y = simp.runAt(1, range_rand)()

    # both of them seal it
    x_ = simp.sealFrom(x, 0)
    y_ = simp.sealFrom(y, 1)

    # compare it securely.
    z_ = simp.srun(lambda x, y: x < y)(x_, y_)

    # reveal it to all.
    z = simp.reveal(z_)

    return x, y, z


# Now we create a simulator with 2 parties, which use two threads to simulate the two parties.
sim2 = mplang.Simulator.simple(2)
# Evaluate the millionaire function in the simulator.
x, y, z = mplang.evaluate(sim2, millionaire)
# x, y, z will be references to the values of the two parties.
print("millionaire result:", x, y, z)
# use mplang.fetchAt to get the values of the parties.
print(
    "millionaire fetch:",
    mplang.fetch(sim2, x),
    mplang.fetch(sim2, y),
    mplang.fetch(sim2, z),
)

# Of course, we can also run the same function in a different simulator with 3 parties.
sim3 = mplang.Simulator.simple(3)
# Evaluate the millionaire function in the simulator.
x, y, z = mplang.evaluate(sim3, millionaire)
# Uncomments to see the result.
# print(
#     "millionaire fetch (3PC):",
#     mplang.fetchAt(sim3, x),
#     mplang.fetchAt(sim3, y),
#     mplang.fetchAt(sim3, z),
# )


# Instead of programming towards a specific party, we can use a SIMP
# (Single Instruction Multiple Party) way to run the same function on all parties.
# This function is similar to millionaire, but in an SIMP way.
def millionaire_simp():
    range_rand = partial(random.randint, 0, 10)

    # Instead of runAt, all parties call randint(0, 10)
    x = simp.run(range_rand)()

    # all parties seal it, result a list of sealed values
    xs_ = simp.seal(x)
    # assert len(xs_) == 2  # Fixed from simp.cur_ctx().psize()

    # compare it securely.
    z_ = simp.srun(lambda x, y: x < y)(*xs_)

    # reveal it to all.
    z = simp.reveal(z_)

    return x, z


# Evaluate it.
x, r = mplang.evaluate(sim2, millionaire_simp)
print("millionaire_simp:", mplang.fetch(sim2, x), mplang.fetch(sim2, r))

# Why SIMP? it use less instructions and more close to SPMD (Single Program Multiple Data) programming model.
# To inspect the compiled code, we can use mplang.compile.
compiled = mplang.compile(sim2, millionaire)
simp_compiled = mplang.compile(sim2, millionaire_simp)
# SIMP compile will generate less code than the original function.
print("millionaire compiled:", compiled.compiler_ir())
print("millionaire_simp compiled:", simp_compiled.compiler_ir())

# Eager evaluation vs lazy evaluation.
# In the above example, we use `simp.runAt` to run a function at a specific party, it
# will be evaluated immediately when the function is called.
# In contrast, we can use 'mplang.function' to trace a whole function and send to parties
# once, which makes it more efficient in some cases. In secure computation setting, it also
# allows parties to audit and verify the function before running it.

# The following code is equivalent to add a decorator to the millionaire function,
# @mplang.function
# def millionaire(): ...
millionaire_jitted = mplang.function(millionaire)

# We can also evaluate or compile the jitted function.
x, y, z = mplang.evaluate(sim2, millionaire_jitted)
print(
    "millionaire_jitted result:",
    mplang.fetch(sim2, x),
    mplang.fetch(sim2, y),
    mplang.fetch(sim2, z),
)
# Or compile it.
print(
    "millionaire_jitted compiled:",
    mplang.compile(sim2, millionaire_jitted).compiler_ir(),
)


# Here is a more complicated example that shows how to use `mplang.function` to
# create a function that can be run at different parties, and how to use
# `simp` to securely compute the result.


@mplang.function
def sub_func():
    return simp.runAt(1, partial(random.randint, 0, 10))()


@mplang.function
def myfun(*args, **kwargs):
    # mplang.function can take both positional and keyword arguments.

    x = args[0]
    c0: int = args[1]
    y = kwargs["y"]
    c1: str = kwargs["s"]

    u = simp.runAt(0, partial(random.randint, 0, 10))()
    # Call a 'mplang.function' inside another 'mplang.function'.
    v = sub_func()

    c2 = c0 * 2
    c3 = c1 + " processed"

    a = simp.runAt(0, lambda v: v * 2)(u)
    b = simp.runAt(1, lambda v: v + 5)(v)

    x_ = simp.sealFrom(x, 0)
    y_ = simp.sealFrom(y, 1)
    z_ = simp.srun(lambda x, y: x < y)(x_, y_)
    c = simp.reveal(z_)

    # return complicated result.
    return a, [b, c2], {"c": c, "c3": c3}


# Set the global context, so we dont need to pass it around.
mplang.set_ctx(sim2)

# make a random number at P0
x = simp.runAt(0, partial(random.randint, 0, 10))()

# make a random number at P1
y = simp.runAt(1, partial(random.randint, 0, 10))()

# Call the myfun function with the random numbers.
z = myfun(x, 42, y=y, s="hello")
# Print the results.
print("z:", z)
print("fetch(z):", mplang.fetch(None, z))
