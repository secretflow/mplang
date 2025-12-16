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

import concurrent.futures
import time
import unittest

from mplang.v2.edsl.graph import Graph, Operation
from mplang.v2.edsl.registry import register_impl
from mplang.v2.runtime.interpreter import Interpreter


# Dummy operations for testing
def add_impl(interpreter: Interpreter, op: Operation, x: int, y: int) -> int:
    return x + y


register_impl("test.add", add_impl)


def slow_add_impl(interpreter: Interpreter, op: Operation, x: int, y: int) -> int:
    time.sleep(0.1)
    return x + y


register_impl("test.slow_add", slow_add_impl)


class TestInterpreterAsync(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.x = self.graph.add_input("x", None)
        self.y = self.graph.add_input("y", None)

    def test_sync_execution(self):
        # Setup graph: z = x + y
        (z,) = self.graph.add_op("test.add", [self.x, self.y], output_types=[None])
        self.graph.add_output(z)

        interp = Interpreter()
        result = interp.evaluate_graph(self.graph, [1, 2])
        self.assertEqual(result[0], 3)

    def test_parallel_execution(self):
        # Setup graph:
        # a = slow_add(x, y)  (0.1s)
        # b = slow_add(x, y)  (0.1s)
        # c = add(a, b)
        (a,) = self.graph.add_op("test.slow_add", [self.x, self.y], output_types=[None])
        (b,) = self.graph.add_op("test.slow_add", [self.x, self.y], output_types=[None])
        (c,) = self.graph.add_op("test.add", [a, b], output_types=[None])
        self.graph.add_output(c)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            interp = Interpreter(executor=executor)
            interp.async_ops.add("test.slow_add")

            start_time = time.time()
            result = interp.evaluate_graph(self.graph, [1, 2])
            end_time = time.time()

            # Should take ~0.1s if parallel, ~0.2s if serial
            self.assertEqual(result[0], 6)
            self.assertLess(end_time - start_time, 0.18)

    def test_async_dependency(self):
        # Setup graph:
        # a = slow_add(x, y)  (async)
        # b = add(a, y)       (sync, depends on a)
        (a,) = self.graph.add_op("test.slow_add", [self.x, self.y], output_types=[None])
        (b,) = self.graph.add_op("test.add", [a, self.y], output_types=[None])
        self.graph.add_output(b)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            interp = Interpreter(executor=executor)
            interp.async_ops.add("test.slow_add")

            # Should block when executing 'test.add' because it needs 'a'
            result = interp.evaluate_graph(self.graph, [1, 2])
            self.assertEqual(result[0], 5)  # (1+2) + 2 = 5

    def test_mixed_async_sync(self):
        # Setup graph:
        # a = slow_add(x, y)  (async)
        # b = add(x, y)       (sync)
        # c = add(a, b)       (sync, depends on a and b)
        (a,) = self.graph.add_op("test.slow_add", [self.x, self.y], output_types=[None])
        (b,) = self.graph.add_op("test.add", [self.x, self.y], output_types=[None])
        (c,) = self.graph.add_op("test.add", [a, b], output_types=[None])
        self.graph.add_output(c)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            interp = Interpreter(executor=executor)
            interp.async_ops.add("test.slow_add")

            result = interp.evaluate_graph(self.graph, [10, 20])
            # a = 30 (async), b = 30 (sync), c = 60
            self.assertEqual(result[0], 60)


if __name__ == "__main__":
    unittest.main()
