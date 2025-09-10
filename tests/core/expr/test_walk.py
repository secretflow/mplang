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

import pytest

from mplang.core.dtype import FLOAT32, INT32
from mplang.core.expr import (
    AccessExpr,
    CondExpr,
    FuncDefExpr,
    TupleExpr,
    VariableExpr,
    WhileExpr,
    walk,
    walk_dataflow,
    walk_structural,
)
from mplang.core.mptype import MPType


def var(name: str):
    return VariableExpr(name, MPType.tensor(FLOAT32, (1,), None))


class TestWalkDataflow:
    def test_dfs_post_iter_order_respects_dependencies(self):
        # Build small DAG: a,b -> t(tuple) -> x=access(t,0)
        a = var("a")
        b = var("b")
        t = TupleExpr([a, b])
        x = AccessExpr(t, 0)

        order = list(walk_dataflow(x, traversal="dfs_post_iter"))

        # Helper: all deps must appear before node
        pos = {id(n): i for i, n in enumerate(order)}

        def get_deps(node):
            if node is t:
                return TupleExpr([a, b]).args
            elif node is x:
                return (t,)
            else:
                return ()

        def assert_deps_before(node):
            for d in get_deps(node):
                assert pos[id(d)] < pos[id(node)]

        # a and b before t, and t before x
        assert_deps_before(t)
        assert_deps_before(x)

    def test_bfs_and_topo_shapes(self):
        a = var("a")
        b = var("b")
        t = TupleExpr([a, b])
        x = AccessExpr(t, 0)

        bfs_list = list(walk_dataflow(x, traversal="bfs"))
        # root first for BFS
        assert bfs_list[0] is x
        assert t in bfs_list and a in bfs_list and b in bfs_list

        topo_list = list(walk_dataflow(x, traversal="topo"))
        # topo invariant: deps appear before consumer
        pos = {id(n): i for i, n in enumerate(topo_list)}
        assert pos[id(a)] < pos[id(t)]
        assert pos[id(b)] < pos[id(t)]
        assert pos[id(t)] < pos[id(x)]

    def test_yield_condition_filters(self):
        a = var("a")
        b = var("b")
        t = TupleExpr([a, b])
        x = AccessExpr(t, 0)

        only_vars = list(
            walk_dataflow(
                x,
                traversal="dfs_post_iter",
                yield_condition=lambda n: isinstance(n, VariableExpr),
            )
        )
        assert set(only_vars) == {a, b}

    def test_cycle_detection_on_self_cycle(self):
        t = TupleExpr([])
        # create a cycle: t depends on itself
        t.args.append(t)

        with pytest.raises(ValueError):
            _ = list(walk_dataflow(t, traversal="dfs_post_iter", detect_cycles=True))

        # Without detection, traversal should still terminate and yield t once
        nodes_no_detect = list(
            walk_dataflow(t, traversal="dfs_post_iter", detect_cycles=False)
        )
        assert nodes_no_detect.count(t) == 1


class TestWalkStructural:
    def test_cond_structural_includes_regions(self):
        # pred and arg
        pred = VariableExpr("pred", MPType.tensor(INT32, (), None))
        x = var("x")
        # then/else functions reference their parameters in body
        then_body = VariableExpr("x", x.mptype)
        else_body = VariableExpr("x", x.mptype)
        then_fn = FuncDefExpr(["x"], then_body)
        else_fn = FuncDefExpr(["x"], else_body)
        cond = CondExpr(pred, then_fn, else_fn, [x])

        # dataflow walk should NOT enter region bodies
        df_nodes = list(walk_dataflow(cond, traversal="dfs_post_iter"))
        assert then_body not in df_nodes and else_body not in df_nodes

        # structural walk SHOULD include region bodies (once)
        st_nodes = list(walk_structural(cond, traversal="dfs_post_iter"))
        assert then_body in st_nodes and else_body in st_nodes

    def test_while_structural_includes_body_once(self):
        # while with cond/body referencing param
        state = var("s")
        cond_body = VariableExpr("s", state.mptype)
        body_body = VariableExpr("s", state.mptype)
        cond_fn = FuncDefExpr(["s"], cond_body)
        body_fn = FuncDefExpr(["s"], body_body)
        w = WhileExpr(cond_fn, body_fn, [state])

        st_nodes = list(walk_structural(w, traversal="dfs_post_iter"))
        # include cond/body function bodies once (no runtime expansion)
        assert cond_body in st_nodes and body_body in st_nodes


def test_generic_walk_allows_custom_get_deps():
    # Define a tiny custom graph with manual deps function
    a = var("a")
    b = var("b")
    t = TupleExpr([a, b])

    def get_deps(n):
        if n is t:
            return [b, a]  # reversed order
        if isinstance(n, TupleExpr):
            return n.args
        if isinstance(n, VariableExpr):
            return []
        return []

    order = list(walk(t, get_deps=get_deps, traversal="dfs_post_iter"))
    # With reversed deps, a must still appear before t, same for b
    pos = {id(n): i for i, n in enumerate(order)}
    assert pos[id(a)] < pos[id(t)]
    assert pos[id(b)] < pos[id(t)]
