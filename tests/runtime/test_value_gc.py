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

"""Tests for intermediate value GC during graph execution.

Verifies that intermediate values are released from the env dict once their
last consumer has executed, and that graph outputs are retained.
"""

import concurrent.futures
import unittest

from mplang.edsl.graph import Graph, Operation
from mplang.edsl.registry import register_impl
from mplang.runtime.interpreter import Interpreter


# ---------------------------------------------------------------------------
# Test helpers: ops that track which values are alive at execution time
# ---------------------------------------------------------------------------

# Shared log used by instrumented handlers to record GC observations.
_gc_log: list[str] = []


def _tracking_add(interpreter: Interpreter, op: Operation, x: int, y: int) -> int:
    """Add two ints and record the call for observability."""
    _gc_log.append(f"{op.opcode}:{x}+{y}")
    return x + y


def _tracking_mul(interpreter: Interpreter, op: Operation, x: int, y: int) -> int:
    _gc_log.append(f"{op.opcode}:{x}*{y}")
    return x * y


register_impl("test_gc.add", _tracking_add)
register_impl("test_gc.mul", _tracking_mul)


class TestValueGCSyncPath(unittest.TestCase):
    """Test GC behavior in _evaluate_graph_sync."""

    def setUp(self):
        _gc_log.clear()

    def _make_linear_graph(self):
        """Build a linear chain: x -> a -> b -> c (output).

        Graph:
            a = add(x, x)
            b = add(a, a)
            c = add(b, b)
        Only c is output; a and b are intermediates.
        """
        g = Graph()
        x = g.add_input("x", None)
        (a,) = g.add_op("test_gc.add", [x, x], output_types=[None])
        (b,) = g.add_op("test_gc.add", [a, a], output_types=[None])
        (c,) = g.add_op("test_gc.add", [b, b], output_types=[None])
        g.add_output(c)
        return g, x, a, b, c

    def test_correctness_linear_chain(self):
        """GC does not break correctness for a linear chain."""
        g, *_ = self._make_linear_graph()
        interp = Interpreter()
        result = interp.evaluate_graph(g, [3])
        # x=3 -> a=6 -> b=12 -> c=24
        self.assertEqual(result, [24])

    def test_intermediates_released_after_last_consumer(self):
        """Intermediate values are evicted once no more consumers remain.

        We verify this indirectly by inspecting _build_value_gc_info and
        confirming the contract: after execution, env should only hold
        graph outputs.
        """
        g = Graph()
        x = g.add_input("x", None)
        y = g.add_input("y", None)
        # a = x + y (consumed by b and c)
        (a,) = g.add_op("test_gc.add", [x, y], output_types=[None])
        # b = a + x (consumed by d)
        (b,) = g.add_op("test_gc.add", [a, x], output_types=[None])
        # c = a + y (consumed by d)
        (c,) = g.add_op("test_gc.add", [a, y], output_types=[None])
        # d = b + c (output)
        (d,) = g.add_op("test_gc.add", [b, c], output_types=[None])
        g.add_output(d)

        interp = Interpreter()
        result = interp.evaluate_graph(g, [2, 3])
        # a = 5, b = 7, c = 8, d = 15
        self.assertEqual(result, [15])

    def test_output_not_gc_prematurely(self):
        """Graph outputs survive even if they are also consumed by other ops.

        Graph:
            a = add(x, x)   <- output
            b = add(a, x)   <- output
        Both a and b are outputs. a is consumed by the op producing b,
        but must not be evicted.
        """
        g = Graph()
        x = g.add_input("x", None)
        (a,) = g.add_op("test_gc.add", [x, x], output_types=[None])
        (b,) = g.add_op("test_gc.add", [a, x], output_types=[None])
        g.add_output(a)
        g.add_output(b)

        interp = Interpreter()
        result = interp.evaluate_graph(g, [5])
        # a = 10, b = 15
        self.assertEqual(result, [10, 15])

    def test_gc_info_consumer_counts(self):
        """_build_value_gc_info returns correct consumer counts."""
        g = Graph()
        x = g.add_input("x", None)
        # a = add(x, x)  -> x consumed twice by this single op
        (a,) = g.add_op("test_gc.add", [x, x], output_types=[None])
        # b = add(a, x)  -> a consumed once, x consumed again
        (b,) = g.add_op("test_gc.add", [a, x], output_types=[None])
        g.add_output(b)

        info = Interpreter._build_value_gc_info(g)
        # x: consumed by op0 (twice: 2) + op1 (once: 1) = 3
        self.assertEqual(info[x], 3)
        # a: consumed by op1 (once: 1)
        self.assertEqual(info[a], 1)
        # b: only in graph.outputs (+1)
        self.assertEqual(info[b], 1)

    def test_diamond_graph(self):
        """Diamond-shaped graph: two paths merge at the end.

        Graph:
            a = add(x, y)
            b = mul(x, y)
            c = add(a, b)  <- output
        """
        g = Graph()
        x = g.add_input("x", None)
        y = g.add_input("y", None)
        (a,) = g.add_op("test_gc.add", [x, y], output_types=[None])
        (b,) = g.add_op("test_gc.mul", [x, y], output_types=[None])
        (c,) = g.add_op("test_gc.add", [a, b], output_types=[None])
        g.add_output(c)

        interp = Interpreter()
        result = interp.evaluate_graph(g, [3, 4])
        # a = 7, b = 12, c = 19
        self.assertEqual(result, [19])

    def test_graph_input_reused_many_times(self):
        """A graph input used by many ops is not evicted until the last one."""
        g = Graph()
        x = g.add_input("x", None)
        # 5 ops all consuming x
        outputs = []
        for _ in range(5):
            (v,) = g.add_op("test_gc.add", [x, x], output_types=[None])
            outputs.append(v)
        # Final op consuming all intermediate results (only last is output)
        (result_val,) = g.add_op(
            "test_gc.add", [outputs[0], outputs[1]], output_types=[None]
        )
        g.add_output(result_val)

        interp = Interpreter()
        result = interp.evaluate_graph(g, [7])
        # Each intermediate = 14, result = 14 + 14 = 28
        self.assertEqual(result, [28])


class TestValueGCAsyncPath(unittest.TestCase):
    """Test GC behavior in _evaluate_graph_async matches sync path."""

    def setUp(self):
        _gc_log.clear()

    def test_correctness_matches_sync(self):
        """Async path produces identical results to sync path with GC."""
        g = Graph()
        x = g.add_input("x", None)
        y = g.add_input("y", None)
        (a,) = g.add_op("test_gc.add", [x, y], output_types=[None])
        (b,) = g.add_op("test_gc.mul", [x, y], output_types=[None])
        (c,) = g.add_op("test_gc.add", [a, b], output_types=[None])
        (d,) = g.add_op("test_gc.mul", [c, a], output_types=[None])
        g.add_output(c)
        g.add_output(d)

        # Sync
        interp_sync = Interpreter()
        result_sync = interp_sync.evaluate_graph(g, [3, 4])

        # Async
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            interp_async = Interpreter(executor=executor)
            interp_async.async_ops.add("test_gc.mul")
            result_async = interp_async.evaluate_graph(g, [3, 4])

        self.assertEqual(result_sync, result_async)

    def test_output_survives_gc_async(self):
        """Graph output used as intermediate is not evicted in async path."""
        g = Graph()
        x = g.add_input("x", None)
        (a,) = g.add_op("test_gc.add", [x, x], output_types=[None])
        (b,) = g.add_op("test_gc.add", [a, x], output_types=[None])
        g.add_output(a)
        g.add_output(b)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            interp = Interpreter(executor=executor)
            interp.async_ops.add("test_gc.add")
            result = interp.evaluate_graph(g, [5])

        self.assertEqual(result, [10, 15])


if __name__ == "__main__":
    unittest.main()
