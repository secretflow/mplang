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

"""Registry for primitive implementations.

This module decouples the Primitive definition from the Interpreter execution.
Primitives register their implementations here, and the Interpreter looks them up here.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Global registry for primitive implementations
# Key: opcode (str), Value: implementation function
_IMPL_REGISTRY: dict[str, Callable[..., Any]] = {}


# ==============================================================================
# Profiler for All Primitive Operations
# ==============================================================================


@dataclass
class OpProfiler:
    """Global profiler for tracking all primitive operation timing."""

    enabled: bool = False
    timings: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def reset(self) -> None:
        """Clear all timing data."""
        self.timings = defaultdict(list)

    def record(self, opcode: str, duration: float) -> None:
        """Record a timing measurement."""
        if self.enabled:
            self.timings[opcode].append(duration)

    def summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all operations."""
        result = {}
        for opcode, times in sorted(self.timings.items()):
            if times:
                result[opcode] = {
                    "count": len(times),
                    "total": sum(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }
        return result

    def print_summary(self, top_n: int = 20) -> None:
        """Print a formatted summary of timing statistics."""
        stats = self.summary()
        if not stats:
            print("No timing data collected.")
            return

        print("\n" + "=" * 80)
        print("PRIMITIVE OPERATION TIMING SUMMARY")
        print("=" * 80)
        print(
            f"{'Operation':<35} {'Count':>8} {'Total(s)':>10} "
            f"{'Mean(ms)':>10} {'Max(ms)':>10}"
        )
        print("-" * 80)

        total_time = sum(s["total"] for s in stats.values())

        # Sort by total time descending
        sorted_stats = sorted(stats.items(), key=lambda x: -x[1]["total"])

        for opcode, s in sorted_stats[:top_n]:
            pct = s["total"] / total_time * 100 if total_time > 0 else 0
            print(
                f"{opcode:<35} {s['count']:>8} {s['total']:>10.3f} "
                f"{s['mean'] * 1000:>10.3f} {s['max'] * 1000:>10.3f}  ({pct:>5.1f}%)"
            )

        if len(sorted_stats) > top_n:
            print(f"  ... and {len(sorted_stats) - top_n} more operations")

        print("-" * 80)
        print(f"{'TOTAL':<35} {'':<8} {total_time:>10.3f}s")

    def print_leaf_summary(self, top_n: int = 20) -> None:
        """Print summary excluding container ops (pcall, shuffle, etc.).

        This shows only 'leaf' operations that don't contain nested calls,
        giving accurate self-time without double-counting.
        """
        # Container ops that include nested operation time
        container_ops = {
            "simp.pcall_static",
            "simp.pcall_dynamic",
            "simp.shuffle_static",
            "simp.shuffle",
            "simp.uniform_cond",
            "simp.while_loop",
        }

        stats = self.summary()
        leaf_stats = {k: v for k, v in stats.items() if k not in container_ops}

        if not leaf_stats:
            print("No leaf timing data collected.")
            return

        print("\n" + "=" * 80)
        print("LEAF OPERATION TIMING SUMMARY (excludes container ops)")
        print("=" * 80)
        print(
            f"{'Operation':<35} {'Count':>8} {'Total(s)':>10} "
            f"{'Mean(ms)':>10} {'Max(ms)':>10}"
        )
        print("-" * 80)

        total_time = sum(s["total"] for s in leaf_stats.values())
        sorted_stats = sorted(leaf_stats.items(), key=lambda x: -x[1]["total"])

        for opcode, s in sorted_stats[:top_n]:
            pct = s["total"] / total_time * 100 if total_time > 0 else 0
            print(
                f"{opcode:<35} {s['count']:>8} {s['total']:>10.3f} "
                f"{s['mean'] * 1000:>10.3f} {s['max'] * 1000:>10.3f}  ({pct:>5.1f}%)"
            )

        if len(sorted_stats) > top_n:
            print(f"  ... and {len(sorted_stats) - top_n} more operations")

        print("-" * 80)
        print(f"{'TOTAL (leaf ops)':<35} {'':<8} {total_time:>10.3f}s")


# Global profiler instance
_profiler = OpProfiler()


def get_profiler() -> OpProfiler:
    """Get the global operation profiler instance."""
    return _profiler


def enable_profiling() -> None:
    """Enable primitive operation profiling."""
    _profiler.enabled = True
    _profiler.reset()


def disable_profiling() -> None:
    """Disable primitive operation profiling."""
    _profiler.enabled = False


# ==============================================================================
# Registry Functions
# ==============================================================================


def register_impl(opcode: str, fn: Callable[..., Any]) -> None:
    """Register an implementation for an opcode.

    Args:
        opcode: The unique name of the primitive (e.g. "add", "mul").
        fn: The function implementing the logic.
            Signature: (interpreter, op, *args) -> result
    """
    _IMPL_REGISTRY[opcode] = fn


def get_impl(opcode: str) -> Callable[..., Any] | None:
    """Get the registered implementation for an opcode.

    If profiling is enabled, returns a wrapped function that records timing.
    """
    fn = _IMPL_REGISTRY.get(opcode)
    if fn is None:
        return None

    if not _profiler.enabled:
        return fn

    # Return a profiling wrapper
    def profiled_fn(interpreter: Any, op: Any, *args: Any) -> Any:
        t0 = time.perf_counter()
        result = fn(interpreter, op, *args)
        _profiler.record(opcode, time.perf_counter() - t0)
        return result

    return profiled_fn
