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

from functools import partial

import jax
import jax.numpy as jnp

import mplang.v1 as mp

# TL;DR / Quick Patterns:
# - Elementwise / per-party divergent predicate & both sides cheap -> jax.where
# - Pure local lazy control (no multi-party side-effects) -> jax.lax.cond (via mp.rjax)
# - Uniform predicate + expensive multi-party side-effects -> uniform_cond
# - Divergent structural branching you wish were uniform -> aggregate/reduce to uniform or fallback to elementwise where


@mp.function
def local_elementwise_select():
    """Case 1: Per-party / element-wise selection -> use jax.where.

    Each party privately samples x, and decides locally whether to negate.
    Both candidate values (pos/neg) are inexpensive and *both computed*.
    Predicate p may differ across parties.

    Correct primitive: jax.where (NOT uniform_cond).
    """
    x = mp.prandint(0, 20)
    p = mp.run_jax(lambda v: v <= 10, x)  # local predicate (can diverge per party)
    pos = mp.run_jax(lambda v: v, x)
    neg = mp.run_jax(lambda v: -v, x)
    z = mp.run_jax(jnp.where, p, pos, neg)
    return x, z


@mp.function
def uniform_multi_party_cond():
    """Case 2: Global uniform predicate with *expensive* multi-party branch.
    Contrast: if we used ``jax.lax.cond`` here we could only express *local lazy* control
    and we could NOT guarantee the non-selected branch's multi-party protocol is fully
    skipped (its subgraph would still be traced/captured early, potentially impacting
    backend optimization or resource usage).

    Steps:
        1. Each party generates a private ``x``.
        2. Perform ``seal -> srun_jax`` aggregation (derive a secret statistic) -> ``reveal`` to get a uniform bool.
        3. ``then`` branch simulates an expensive path (extra seal + reveal round).
        4. ``else`` branch performs a light-weight local transform.
    """
    x = mp.prandint(0, 10)
    xs_ = mp.seal(x)
    pred_secret = mp.srun_jax(lambda xs: jnp.sum(jnp.stack(xs), axis=0) < 15, xs_)
    pred = mp.reveal(pred_secret)  # public & uniform

    def then_branch(v):
        # Simulate expensive multi-party path: seal + aggregation + reveal
        sealed_v = mp.seal(v)
        agg = mp.srun_jax(lambda parts: jnp.sum(jnp.stack(parts), axis=0), sealed_v)
        return mp.reveal(agg)

    def else_branch(v):
        return mp.run_jax(lambda t: -t, v)

    # Runtime uniform verification here to catch accidental divergence. If you have a
    # provably uniform predicate and want to skip the O(P^2) tiny boolean gather, you
    # could pass verify_uniform=False.
    out = mp.uniform_cond(pred, then_branch, else_branch, x, verify_uniform=True)
    return x, out


@mp.function
def local_lazy_cond():
    """Supplement: Local (pure compute) lazy conditional using jax.lax.cond.
    Use case: branches contain no multi-party protocol / communication and you *only* want
    to avoid computing the non-selected pure local math. Even if the predicate ends up
    uniform you still do not need ``uniform_cond``; ``lax.cond`` is sufficient. If in the
    future you introduce seal/reveal (or other MPC side-effects) into the branches and you
    require truly skipping the other branch's communication, migrate to ``uniform_cond``.
    """
    x = mp.prandint(0, 20)
    # Compute predicate locally (pure value) so that we do not leak TraceVars into JAX tracing.
    pred = mp.run_jax(lambda v: v % 2 == 0, x)  # may diverge per party

    # Wrap the entire lax.cond invocation in a single local run so that the JAX tracer
    # only ever sees concrete numpy/jax arrays, not TraceVar wrappers.
    def _lazy_branch(pred_val, v_val):
        def t_fn(z):
            return z * 2

        def f_fn(z):
            return z + 3

        return jax.lax.cond(pred_val, t_fn, f_fn, v_val)

    res = mp.run_jax(_lazy_branch, pred, x)
    return x, res


@mp.function
def anti_pattern_uniform_cond_with_divergent_pred():
    """Case 3 (Anti-pattern): Divergent per-party predicate with uniform_cond.

    This demonstrates what *not* to do: uniform_cond expects a uniform predicate
    but here we build one that can diverge. For now (verify_uniform=False) the
    system will not detect it, but semantics are undefined. Use jax.where instead.
    """
    rank = mp.prank()
    pred = mp.run_jax(lambda r: r < 2, rank)  # may differ per party
    a = mp.constant(5)
    b = mp.constant(10)
    # DO NOT DO THIS IN REAL CODE (shown only for educational contrast)
    res = mp.uniform_cond(
        pred,
        partial(mp.run_jax, jnp.add),
        partial(mp.run_jax, jnp.subtract),
        a,
        b,
        verify_uniform=False,
    )
    return a, res


if __name__ == "__main__":
    WORLD_SIZE = 3
    mp.set_ctx(mp.Simulator.simple(WORLD_SIZE))

    print("Case 1: local_elementwise_select (use jax.where)")
    x1, r1 = local_elementwise_select()
    print(mp.compile(mp.cur_ctx(), local_elementwise_select).compiler_ir())
    print(mp.fetch(None, (x1, r1)))

    print(
        "Case 2: uniform_multi_party_cond (use uniform_cond for multi-party heavy branch)"
    )
    x2, r2 = uniform_multi_party_cond()
    print(mp.compile(mp.cur_ctx(), uniform_multi_party_cond).compiler_ir())
    print(mp.fetch(None, (x2, r2)))

    print(
        "Supplement: local_lazy_cond (pure local lazy via lax.cond, NOT uniform_cond)"
    )
    x_lazy, r_lazy = local_lazy_cond()
    print(mp.compile(mp.cur_ctx(), local_lazy_cond).compiler_ir())
    print(mp.fetch(None, (x_lazy, r_lazy)))

    print("Case 3: anti_pattern_uniform_cond_with_divergent_pred (DON'T DO THIS)")
    x3, r3 = anti_pattern_uniform_cond_with_divergent_pred()
    print(
        mp.compile(
            mp.cur_ctx(), anti_pattern_uniform_cond_with_divergent_pred
        ).compiler_ir()
    )
    print(mp.fetch(None, (x3, r3)))
