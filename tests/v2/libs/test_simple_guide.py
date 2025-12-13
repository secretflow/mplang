# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple verification test for func.call support."""

import mplang.v2 as mp
from mplang.v2.dialects import func, simp, tensor


def test_func_call_recursion():
    """Verify that func.call executes the target function recursively."""

    # 1. Define a helper function (pure python)
    def add_one(x):
        return tensor.run_jax(lambda a: a + 1, x)

    # 2. Define main function
    def main(x):
        # Use func.func to define 'add_one' in the graph, returns function handle
        fn_handle = func.func(add_one, x)

        # Call the function via handle
        return func.call(fn_handle, x)

    # 3. Setup Simulator
    sim = simp.make_simulator(2)
    from mplang.v2.edsl.context import push_context

    push_context(sim)

    # 4. Execute
    x_val = 10
    x_obj = tensor.constant(x_val)

    # Compile main
    traced_main = mp.compile(main, x_obj)

    # Evaluate
    result_obj = mp.evaluate(traced_main, x_obj)

    # Fetch
    result = mp.fetch(result_obj)

    assert result == 11
    print(f"Func Call Verified: {x_val} + 1 = {result}")


if __name__ == "__main__":
    test_func_call_recursion()
