# Copyright 2026 Ant Group Co., Ltd.
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

"""Tests for concurrent graph execution with job_id isolation."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from typing import Any

from mplang.edsl.graph import Graph, Operation
from mplang.edsl.registry import register_impl
from mplang.runtime.interpreter import Interpreter


# ---------------------------------------------------------------------------
# Test helpers / dummy ops
# ---------------------------------------------------------------------------

_captured_job_ids: list[str | None] = []
_captured_job_ids_lock = threading.Lock()


def _capture_job_id_impl(interpreter: Interpreter, op: Operation, x: Any) -> Any:
    """Dummy op that captures current_job_id() for assertions."""
    jid = interpreter.current_job_id()
    with _captured_job_ids_lock:
        _captured_job_ids.append(jid)
    return x


register_impl("test.capture_job_id", _capture_job_id_impl)


def _slow_identity_impl(interpreter: Interpreter, op: Operation, x: Any) -> Any:
    time.sleep(0.1)
    return x


register_impl("test.slow_identity", _slow_identity_impl)


def _make_simple_graph() -> Graph:
    """Create a graph: out = capture_job_id(slow_identity(x))."""
    g = Graph()
    x = g.add_input("x", None)
    (y,) = g.add_op("test.slow_identity", [x], output_types=[None])
    (z,) = g.add_op("test.capture_job_id", [y], output_types=[None])
    g.add_output(z)
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_concurrent_same_graph_with_different_job_ids() -> None:
    """Two concurrent evaluate_graph calls with different job_ids must both succeed."""
    graph = _make_simple_graph()
    interp = Interpreter()

    results: list[Any] = [None, None]
    errors: list[Exception | None] = [None, None]

    def run(idx: int, job_id: str, input_val: int) -> None:
        try:
            results[idx] = interp.evaluate_graph(graph, [input_val], job_id=job_id)
        except Exception as e:
            errors[idx] = e

    t1 = threading.Thread(target=run, args=(0, "job-A", 10))
    t2 = threading.Thread(target=run, args=(1, "job-B", 20))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors[0] is None, f"job-A failed: {errors[0]}"
    assert errors[1] is None, f"job-B failed: {errors[1]}"
    assert results[0] == [10]
    assert results[1] == [20]


def test_concurrent_same_graph_without_job_id_raises() -> None:
    """Without job_id, concurrent execution of the same graph is still forbidden."""
    hold = threading.Event()

    def _blocking_impl(interpreter: Interpreter, op: Operation, x: Any) -> Any:
        hold.wait(timeout=5)
        return x

    register_impl("test.blocking_op", _blocking_impl)

    graph = Graph()
    x = graph.add_input("x", None)
    (y,) = graph.add_op("test.blocking_op", [x], output_types=[None])
    graph.add_output(y)

    interp = Interpreter()
    errors: list[Exception | None] = [None, None]
    started = threading.Event()

    def run_first() -> None:
        try:
            started.set()
            interp.evaluate_graph(graph, [1])  # No job_id, will block
        except Exception as e:
            errors[0] = e

    def run_second() -> None:
        try:
            started.wait(timeout=5)
            import time

            time.sleep(0.05)
            interp.evaluate_graph(graph, [2])  # No job_id, should fail
        except Exception as e:
            errors[1] = e
        finally:
            hold.set()

    t1 = threading.Thread(target=run_first)
    t2 = threading.Thread(target=run_second)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert isinstance(errors[1], RuntimeError), (
        f"Expected RuntimeError from thread 2, got: {errors[1]}"
    )
    assert errors[0] is None, f"Thread 1 should succeed, got: {errors[0]}"


def test_current_job_id_returns_correct_value() -> None:
    """current_job_id() should return the job_id set by evaluate_graph."""
    _captured_job_ids.clear()

    graph = _make_simple_graph()
    interp = Interpreter()

    interp.evaluate_graph(graph, [42], job_id="my-job-123")

    assert "my-job-123" in _captured_job_ids


def test_current_job_id_none_when_no_job() -> None:
    """current_job_id() should return None when evaluate_graph called without job_id."""
    _captured_job_ids.clear()

    graph = _make_simple_graph()
    interp = Interpreter()

    interp.evaluate_graph(graph, [42])

    assert None in _captured_job_ids


def test_current_job_id_outside_execution() -> None:
    """current_job_id() should return None outside of evaluate_graph."""
    interp = Interpreter()
    assert interp.current_job_id() is None


def test_exec_base_cleanup_after_job_completes() -> None:
    """Per-job exec_id counter entries should be cleaned up after top-level job completes."""
    graph = _make_simple_graph()
    interp = Interpreter()

    interp.evaluate_graph(graph, [1], job_id="cleanup-test")

    # After completion, no tuple keys with "cleanup-test" should remain
    with interp._exec_id_lock:
        remaining = [
            k
            for k in interp._graph_next_exec_base
            if isinstance(k, tuple) and k[0] == "cleanup-test"
        ]
    assert remaining == [], f"Expected cleanup, but found: {remaining}"


def test_exec_base_no_cleanup_without_job_id() -> None:
    """Without job_id, exec_id counters should persist (existing behavior)."""
    graph = _make_simple_graph()
    interp = Interpreter()

    interp.evaluate_graph(graph, [1])

    # String keys should remain (not cleaned up)
    with interp._exec_id_lock:
        string_keys = [k for k in interp._graph_next_exec_base if isinstance(k, str)]
    assert len(string_keys) > 0, "Expected persistent string-keyed exec_base entries"


def test_nested_graph_inherits_job_id() -> None:
    """When evaluate_graph is called recursively (e.g., region graphs),
    nested calls should see the job_id set by the top-level call."""
    _captured_job_ids.clear()

    # Build outer graph with a region that calls evaluate_graph on an inner graph
    inner_graph = Graph()
    ix = inner_graph.add_input("ix", None)
    (iy,) = inner_graph.add_op("test.capture_job_id", [ix], output_types=[None])
    inner_graph.add_output(iy)

    def region_impl(interpreter: Interpreter, op: Operation, x: Any) -> Any:
        """A higher-order op that evaluates an inner graph."""
        inner = op.attrs["inner_graph"]
        result = interpreter.evaluate_graph(inner, [x])
        return result[0]

    register_impl("test.region_call", region_impl)

    outer_graph = Graph()
    ox = outer_graph.add_input("ox", None)
    (oy,) = outer_graph.add_op(
        "test.region_call",
        [ox],
        output_types=[None],
        attrs={"inner_graph": inner_graph},
    )
    outer_graph.add_output(oy)

    interp = Interpreter()
    result = interp.evaluate_graph(outer_graph, [99], job_id="nested-job")

    assert result == [99]
    # The inner graph's capture_job_id op should have seen "nested-job"
    assert "nested-job" in _captured_job_ids


def test_job_id_isolation_of_exec_ids() -> None:
    """Different job_ids should produce independent exec_id sequences starting from 0."""
    exec_ids_by_job: dict[str, list[int]] = {}
    lock = threading.Lock()

    def capture_exec_id_impl(interpreter: Interpreter, op: Operation, x: Any) -> Any:
        jid = interpreter.current_job_id()
        eid = interpreter.current_op_exec_id()
        with lock:
            exec_ids_by_job.setdefault(jid, []).append(eid)  # type: ignore[arg-type]
        return x

    register_impl("test.capture_exec_id", capture_exec_id_impl)

    graph = Graph()
    x = graph.add_input("x", None)
    (y,) = graph.add_op("test.capture_exec_id", [x], output_types=[None])
    graph.add_output(y)

    interp = Interpreter()

    # Run two jobs sequentially — they should both get exec_id starting from 0
    interp.evaluate_graph(graph, [1], job_id="iso-A")
    interp.evaluate_graph(graph, [2], job_id="iso-B")

    # Both jobs should have the same exec_id pattern (starting from 0)
    assert exec_ids_by_job["iso-A"] == exec_ids_by_job["iso-B"], (
        f"Expected identical exec_id sequences, got "
        f"iso-A={exec_ids_by_job['iso-A']}, iso-B={exec_ids_by_job['iso-B']}"
    )
