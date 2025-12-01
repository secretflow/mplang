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

"""
Pure functional traversal utilities for MPLang expression graphs.

This module provides semantic-agnostic walkers over expression graphs. It exposes
both dataflow-only traversal (deps edges) and structural traversal (deps +
contained regions like function bodies, then/else, loop body). All traversals are
implemented iteratively to avoid Python recursion limits.

Notes
- These walkers never evaluate expressions nor decide runtime branches. For
  execution order, use an evaluator/driver that consults semantic rules.
- Topological order is produced w.r.t. deps edges, not region containment.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import cast

from mplang.v1.core.expr.ast import (
    AccessExpr,
    CallExpr,
    CondExpr,
    ConvExpr,
    EvalExpr,
    Expr,
    FuncDefExpr,
    ShflExpr,
    ShflSExpr,
    TupleExpr,
    VariableExpr,
    WhileExpr,
)

Node = Expr
GetDeps = Callable[[Node], Iterable[Node]]
YieldCond = Callable[[Node], bool]


def _identity_key(n: Node) -> int:
    """Identity-based key for hashing nodes that may not be hashable."""
    return id(n)


# ---------------------------- default dependency getters ----------------------------


def dataflow_deps(node: Node) -> Iterable[Node]:
    """Default dataflow dependencies for core Expr nodes.

    This includes only inputs that must be computed to evaluate the node itself.
    Contained regions (function/branch/loop bodies) are NOT traversed here.
    """
    if isinstance(node, EvalExpr):
        return node.args
    if isinstance(node, TupleExpr):
        return node.args
    if isinstance(node, CondExpr):
        # Pure dataflow: pred and actual args to branch functions
        return [node.pred, *node.args]
    if isinstance(node, WhileExpr):
        # Initial state args required; bodies are regions, not dataflow deps
        return list(node.args)
    if isinstance(node, ConvExpr):
        return node.vars
    if isinstance(node, ShflSExpr):
        return [node.src_val]
    if isinstance(node, ShflExpr):
        return [node.src, node.index]
    if isinstance(node, AccessExpr):
        return [node.src]
    if isinstance(node, VariableExpr):
        return []
    if isinstance(node, FuncDefExpr):
        # Definition is not a value-producing node in dataflow; no deps
        return []
    if isinstance(node, CallExpr):
        # Arguments are dataflow deps; function body is a region
        return node.args
    # Fallback: try best-effort empty
    return []


def _structural_region_roots(node: Node) -> Iterable[Node]:
    """Roots of contained regions for structural traversal.

    - Cond: then_fn.body, else_fn.body
    - While: cond_fn.body, body_fn.body
    - Call: fn.body
    - FuncDef: body
    """
    if isinstance(node, CondExpr):
        return [node.then_fn.body, node.else_fn.body]
    if isinstance(node, WhileExpr):
        return [node.cond_fn.body, node.body_fn.body]
    if isinstance(node, CallExpr):
        return [node.fn.body]
    if isinstance(node, FuncDefExpr):
        return [node.body]
    return []


# ---------------------------------- core walkers ----------------------------------


def walk(
    roots: Node | Sequence[Node],
    *,
    get_deps: GetDeps,
    traversal: str = "dfs_post_iter",
    yield_condition: YieldCond | None = None,
    detect_cycles: bool = True,
) -> Iterator[Node]:
    """Generic pure structural walker.

    Args:
        roots: Single root or a sequence of roots to start from.
        get_deps: Function mapping node -> iterable of dependency nodes.
        traversal: One of {'dfs_pre_iter','dfs_post_iter','bfs','topo'}.
        yield_condition: Optional predicate to filter yielded nodes.
        detect_cycles: If True, raises ValueError on cycles (for DFS/topo).

    Yields:
        Nodes in the chosen traversal order, filtered by yield_condition.
    """
    start: list[Node]
    if isinstance(roots, (list, tuple)):
        start = list(roots)
    else:
        start = [cast(Node, roots)]

    if traversal == "bfs":
        yield from _bfs(start, get_deps, yield_condition)
        return
    if traversal == "dfs_pre_iter":
        yield from _dfs_pre_iter(start, get_deps, yield_condition, detect_cycles)
        return
    if traversal == "dfs_post_iter":
        yield from _dfs_post_iter(start, get_deps, yield_condition, detect_cycles)
        return
    if traversal == "topo":
        yield from _topo_kahn(start, get_deps, yield_condition, detect_cycles)
        return

    raise ValueError(f"Invalid traversal type: {traversal}")


def walk_dataflow(
    roots: Node | Sequence[Node],
    *,
    traversal: str = "dfs_post_iter",
    yield_condition: YieldCond | None = None,
    detect_cycles: bool = True,
) -> Iterator[Node]:
    """Walk using default dataflow dependencies for Expr nodes."""
    return walk(
        roots,
        get_deps=dataflow_deps,
        traversal=traversal,
        yield_condition=yield_condition,
        detect_cycles=detect_cycles,
    )


def walk_structural(
    roots: Node | Sequence[Node],
    *,
    traversal: str = "dfs_post_iter",
    yield_condition: YieldCond | None = None,
    detect_cycles: bool = True,
) -> Iterator[Node]:
    """Walk including region containment (function bodies, branches, loop bodies).

    This augments dataflow dependencies with region roots once, so structure is
    fully traversed without runtime branch choices or loop iteration expansion.
    """

    def deps_plus_regions(n: Node) -> Iterable[Node]:
        yield from dataflow_deps(n)
        yield from _structural_region_roots(n)

    return walk(
        roots,
        get_deps=deps_plus_regions,
        traversal=traversal,
        yield_condition=yield_condition,
        detect_cycles=detect_cycles,
    )


# -------------------------------- traversal engines --------------------------------


def _maybe_yield(n: Node, pred: YieldCond | None) -> Iterator[Node]:
    if pred is None or pred(n):
        yield n


def _bfs(
    roots: Sequence[Node],
    get_deps: GetDeps,
    yield_condition: YieldCond | None,
) -> Iterator[Node]:
    seen: set[int] = set()
    q: deque[Node] = deque(roots)
    while q:
        n = q.popleft()
        k = _identity_key(n)
        if k in seen:
            continue
        seen.add(k)
        yield from _maybe_yield(n, yield_condition)
        for d in get_deps(n):
            q.append(d)


def _dfs_pre_iter(
    roots: Sequence[Node],
    get_deps: GetDeps,
    yield_condition: YieldCond | None,
    detect_cycles: bool,
) -> Iterator[Node]:
    seen: set[int] = set()
    onstack: set[int] = set()
    stack: list[tuple[Node, int]] = []  # (node, next_child_index)

    for root in roots:
        kroot = _identity_key(root)
        if kroot in seen:
            continue
        stack.append((root, 0))
        onstack.add(kroot)

        while stack:
            node, idx = stack[-1]
            k = _identity_key(node)
            if k not in seen:
                # Pre-order yield on first encounter
                seen.add(k)
                yield from _maybe_yield(node, yield_condition)

            deps = list(get_deps(node))
            if idx < len(deps):
                child = deps[idx]
                stack[-1] = (node, idx + 1)
                kc = _identity_key(child)
                if kc in onstack:
                    if detect_cycles:
                        raise ValueError("Cycle detected during dfs_pre_iter walk")
                    # skip on cycles if not detecting
                elif kc not in seen:
                    stack.append((child, 0))
                    onstack.add(kc)
                else:
                    # already processed child
                    pass
            else:
                stack.pop()
                onstack.discard(k)


def _dfs_post_iter(
    roots: Sequence[Node],
    get_deps: GetDeps,
    yield_condition: YieldCond | None,
    detect_cycles: bool,
) -> Iterator[Node]:
    seen: set[int] = set()
    onstack: set[int] = set()
    done: set[int] = set()
    stack: list[tuple[Node, int]] = []  # (node, next_child_index)

    for root in roots:
        kroot = _identity_key(root)
        if kroot in done:
            continue
        stack.append((root, 0))
        onstack.add(kroot)

        while stack:
            node, idx = stack[-1]
            k = _identity_key(node)
            deps = list(get_deps(node))
            if k not in seen:
                seen.add(k)

            if idx < len(deps):
                child = deps[idx]
                stack[-1] = (node, idx + 1)
                kc = _identity_key(child)
                if kc in done:
                    continue
                if kc in onstack:
                    if detect_cycles:
                        raise ValueError("Cycle detected during dfs_post_iter walk")
                    # else ignore to avoid infinite loop
                    continue
                stack.append((child, 0))
                onstack.add(kc)
            else:
                # all children processed
                stack.pop()
                onstack.discard(k)
                if k not in done:
                    done.add(k)
                    yield from _maybe_yield(node, yield_condition)


def _collect_closure(roots: Sequence[Node], get_deps: GetDeps) -> list[Node]:
    """Collect reachable nodes and adjacency (by identity) for topo sort.

    Returns (nodes_list, parents_map) where parents_map[v] is the set of keys of
    dependency nodes for v (edges dep -> v).
    """
    nodes: list[Node] = []
    seen: set[int] = set()

    q: deque[Node] = deque(roots)
    while q:
        n = q.popleft()
        kn = _identity_key(n)
        if kn in seen:
            continue
        seen.add(kn)
        nodes.append(n)
        for d in get_deps(n):
            q.append(d)
    return nodes


def _topo_kahn(
    roots: Sequence[Node],
    get_deps: GetDeps,
    yield_condition: YieldCond | None,
    detect_cycles: bool,
) -> Iterator[Node]:
    # Build closure and in-degree from deps edges (dep -> node)
    nodes = _collect_closure(roots, get_deps)

    # reverse map: parent -> set(children)
    children: dict[int, set[int]] = {}
    indeg: dict[int, int] = {}
    for n in nodes:
        kn = _identity_key(n)
        indeg.setdefault(kn, 0)
        for d in get_deps(n):
            kd = _identity_key(d)
            children.setdefault(kd, set()).add(kn)
            indeg[kn] = indeg.get(kn, 0) + 1
            indeg.setdefault(kd, indeg.get(kd, 0))

    # queue of zero in-degree nodes (ready after all deps)
    q: deque[int] = deque(k for k, v in indeg.items() if v == 0)
    key2node: dict[int, Node] = {_identity_key(n): n for n in nodes}
    produced = 0
    seen_keys: set[int] = set()

    while q:
        k = q.popleft()
        if k in seen_keys:
            continue
        seen_keys.add(k)
        n = key2node[k]
        produced += 1
        yield from _maybe_yield(n, yield_condition)
        for c in children.get(k, set()):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)

    if detect_cycles and produced < len(indeg):
        raise ValueError("Cycle detected during topo walk")
