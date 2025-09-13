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

import jax
import jax.numpy as jnp

import mplang
import mplang.simp as simp

# Quick decision guide:
# 1) Per-element / per-party data mux (both sides cheap)      -> jax.where (local_elementwise_select)
# 2) Local lazy control (pure compute, may diverge by party)  -> jax.lax.cond (see local_lazy_cond example below)
# 3) Uniform predicate + expensive multi-party side effects   -> uniform_cond (uniform_multi_party_cond)
# Anti-pattern: using uniform_cond with divergent predicate (shown for contrast)


@mplang.function
def local_elementwise_select():
    """Case 1: Per-party / element-wise selection -> use jax.where.

    Each party privately samples x, and decides locally whether to negate.
    Both candidate values (pos/neg) are inexpensive and *both computed*.
    Predicate p may differ across parties.

    Correct primitive: jax.where (NOT uniform_cond).
    """
    x = simp.prandint(0, 20)
    p = simp.run(lambda v: v <= 10)(x)  # local predicate (can diverge per party)
    pos = simp.run(lambda v: v)(x)
    neg = simp.run(lambda v: -v)(x)
    z = simp.run(jnp.where)(p, pos, neg)
    return x, z


@mplang.function
def uniform_multi_party_cond():
    """Case 2: Global uniform predicate with *expensive* multi-party branch.
    Contrast: if we used ``jax.lax.cond`` here we could only express *local lazy* control
    and we could NOT guarantee the non-selected branch's multi-party protocol is fully
    skipped (its subgraph would still be traced/captured early, potentially impacting
    backend optimization or resource usage).

    Steps:
        1. Each party generates a private ``x``.
        2. Perform ``seal -> srun`` aggregation (derive a secret statistic) -> ``reveal`` to get a uniform bool.
        3. ``then`` branch simulates an expensive path (extra seal + reveal round).
        4. ``else`` branch performs a light-weight local transform.
    """
    x = simp.prandint(0, 10)
    xs_ = simp.seal(x)
    pred_secret = simp.srun(lambda xs: jnp.sum(jnp.stack(xs), axis=0) < 15)(xs_)
    pred = simp.reveal(pred_secret)  # public & uniform

    def then_branch(v):
        # Simulate expensive multi-party path: seal + aggregation + reveal
        sealed_v = simp.seal(v)
        agg = simp.srun(lambda parts: jnp.sum(jnp.stack(parts), axis=0))(sealed_v)
        return simp.reveal(agg)

    def else_branch(v):
        return simp.run(lambda t: -t)(v)

    # NOTE: verify_uniform placeholder not implemented yet → set False.
    out = simp.uniform_cond(pred, then_branch, else_branch, x, verify_uniform=False)
    return x, out


@mplang.function
def local_lazy_cond():
    """Supplement: Local (pure compute) lazy conditional using jax.lax.cond.
    Use case: branches contain no multi-party protocol / communication and you *only* want
    to avoid computing the non-selected pure local math. Even if the predicate ends up
    uniform you still do not need ``uniform_cond``; ``lax.cond`` is sufficient. If in the
    future you introduce seal/reveal (or other MPC side-effects) into the branches and you
    require truly skipping the other branch's communication, migrate to ``uniform_cond``.
    """
    x = simp.prandint(0, 20)
    pred = simp.run(lambda v: v % 2 == 0)(x)  # may diverge per party

    def t_fn(v):  # local pure compute
        return simp.run(lambda z: z * 2)(v)

    def f_fn(v):
        return simp.run(lambda z: z + 3)(v)

    # Express local lazy branching via jax.lax.cond compiled through peval.
    # Note: We stage the shape by passing v; both branches are local pure transforms.
    res = simp.run(jax.lax.cond)(pred, t_fn, f_fn, x)
    return x, res


@mplang.function
def anti_pattern_uniform_cond_with_divergent_pred():
    """Case 3 (Anti-pattern): Divergent per-party predicate with uniform_cond.

    This demonstrates what *not* to do: uniform_cond expects a uniform predicate
    but here we build one that can diverge. For now (verify_uniform=False) the
    system will not detect it, but semantics are undefined. Use jax.where instead.
    """
    rank = simp.prank()
    pred = simp.run(lambda r: r < 2)(rank)  # may differ per party
    a = simp.constant(5)
    b = simp.constant(10)
    # DO NOT DO THIS IN REAL CODE (shown only for educational contrast)
    res = simp.uniform_cond(
        pred, simp.run(jnp.add), simp.run(jnp.subtract), a, b, verify_uniform=False
    )
    return a, res


if __name__ == "__main__":
    WORLD_SIZE = 3
    mplang.set_ctx(mplang.Simulator.simple(WORLD_SIZE))

    print("Case 1: local_elementwise_select (use jax.where)")
    x1, r1 = local_elementwise_select()
    print(mplang.compile(mplang.cur_ctx(), local_elementwise_select).compiler_ir())
    print(mplang.fetch(None, (x1, r1)))

    print(
        "Case 2: uniform_multi_party_cond (use uniform_cond for multi-party heavy branch)"
    )
    x2, r2 = uniform_multi_party_cond()
    print(mplang.compile(mplang.cur_ctx(), uniform_multi_party_cond).compiler_ir())
    print(mplang.fetch(None, (x2, r2)))

    print(
        "Supplement: local_lazy_cond (pure local lazy via lax.cond, NOT uniform_cond)"
    )
    x_lazy, r_lazy = local_lazy_cond()
    print(mplang.compile(mplang.cur_ctx(), local_lazy_cond).compiler_ir())
    print(mplang.fetch(None, (x_lazy, r_lazy)))

    print("Case 3: anti_pattern_uniform_cond_with_divergent_pred (DON'T DO THIS)")
    x3, r3 = anti_pattern_uniform_cond_with_divergent_pred()
    print(
        mplang.compile(
            mplang.cur_ctx(), anti_pattern_uniform_cond_with_divergent_pred
        ).compiler_ir()
    )
    print(mplang.fetch(None, (x3, r3)))
