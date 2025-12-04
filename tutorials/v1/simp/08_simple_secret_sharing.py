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

"""Tutorial: SIMP-level u64 Secret Sharing with Beaver Multiplication (2PC, PHE triples)

This tutorial shows how to describe a simple secret sharing protocol over the ring
R = Z_{2^64} directly at the SIMP level (no built-in security semantics). We implement:

- Secret sharing and reveal over u64
- Local share addition
- Beaver multiplication using a triple generated with PHE (Paillier-like)

The point is to demonstrate the "kernel description" power at SIMP level: explicit
message passing (p2p, bcast, scatter), placement (run_at), and use of PHE ops
for offline triple generation.

Security note: This tutorial is for educational purposes only. The SIMP layer
has no security semantics; protocol and parameter choices here are not production-safe.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike

import mplang.v1 as mp
from mplang.v1.ops import phe

# PHE encoding bound (range encoding) as decimal string to bypass proto int64 limit.
# Choose B large enough to contain all homomorphic intermediate magnitudes.
# With 64-bit shares a,b and 64-bit r, C ≈ a0*b1 + b0*a1 + a1*b1 + r < 3*2^128 + 2^64.
# Pick B=2^132 for margin. Pass as string so backend parses big integer.
PHE_MAX_VALUE_STR = str(1 << 132)


# JAX-friendly u64 ops (avoid Python MASK64 constant inside jitted functions)
def to_u64_j(x: ArrayLike | jnp.ndarray) -> jnp.ndarray:
    """Convert input to a JAX uint64 array for Z_{2^64} arithmetic."""
    return jnp.asarray(x, dtype=jnp.uint64)


def rand_at(rank: int) -> mp.MPObject:
    """Return a random u64 generated at the specified party using mp.prand()."""
    var = mp.prand()
    mask = mp.Mask.from_ranks([rank])
    return mp.set_mask(var, mask)


# ----------------------------
# Share / Reveal (2PC)
# ----------------------------


def ss_share_from(owner: int, value: mp.MPObject) -> mp.MPObject:
    """Secret-share a u64 value owned by `owner` (0 or 1) into 2-party shares.

    Returns a distributed value `s` such that:
      - at P0 local value is s0
      - at P1 local value is s1
    and s0 + s1 ≡ value (mod 2^64).

    Implementation:
      - Owner samples random r, computes other = value - r (mod 2^64)
      - Create [s0, s1] list at owner and scatter to both parties
    """

    world_mask = mp.Mask.all(2)

    # sample r at owner (party-local random u64)
    r = rand_at(owner)
    # compute other share at owner (cast value to u64 to ensure Z_{2^64} semantics)
    other = mp.run_jax_at(owner, lambda v, r_: to_u64_j(v) - r_, value, r)

    s = mp.scatter_m(world_mask, owner, [r, other])
    return s


# ----------------------------
# Add shares (local)
# ----------------------------


def ss_add(x_: mp.MPObject, y_: mp.MPObject) -> mp.MPObject:
    """Compute z_ = x_ + y_ (mod 2^64) locally per party."""
    return mp.run_jax(lambda a, b: a + b, x_, y_)


# ----------------------------
# Beaver triples via PHE (2PC)
# ----------------------------


def beaver_gen() -> tuple[mp.MPObject, mp.MPObject, mp.MPObject]:
    """Generate a Beaver triple (a,b,c) over Z_{2^64} using PHE.

    Returns three distributed values a_share, b_share, c_share such that
      c = a * b (mod 2^64) and each is 2PC-shared across parties.

    Flow:
      - P0 keygen, broadcast pub
      - Each party locally samples its shares a_i, b_i
      - P0 encrypts a0, b0 and sends to P1
      - P1 computes T = Enc(a0)*b1 + Enc(b0)*a1 + a1*b1 + r and sends back
      - P0 decrypts T to get C = a0*b1 + b0*a1 + a1*b1 + r
      - Set shares for c = a*b: at P0 set c0 = a0*b0 + C, at P1 set c1 = -r (all mod 2^64)
    """
    world_mask = mp.Mask.all(2)

    # Keygen at P0 (set a large encoding bound to allow 64-bit shares)
    pk_p0, sk_p0 = mp.run_at(0, phe.keygen, max_value=PHE_MAX_VALUE_STR)
    pk_all = mp.bcast_m(world_mask, 0, pk_p0)

    # Sample random shares as distributed values (each party holds its local share)
    # Using prand keeps the code SIMP-style without manual packing.
    a_ = mp.prand()
    b_ = mp.prand()

    # Encrypt a0, b0 at P0
    A0_p0 = mp.run_at(0, phe.encrypt, a_, pk_all)
    B0_p0 = mp.run_at(0, phe.encrypt, b_, pk_all)

    # Send ciphertexts to P1
    A0_p1 = mp.p2p(0, 1, A0_p0)
    B0_p1 = mp.p2p(0, 1, B0_p0)

    # P1 computes T and sends back
    # T = A*b1 + B*a1 + Enc(a1*b1 + r)
    # r can also be 64-bit; with B=2^132, a1*b1+r remains well within range
    # r is only needed at P1; using a party-local random keeps it minimal.
    r = rand_at(1)

    # Homomorphic linear combination at P1
    a1b1_plus_r = mp.run_jax_at(1, lambda a1, b1, r_: a1 * b1 + r_, a_, b_, r)

    A0_mul_b1 = mp.run_at(1, phe.mul, A0_p1, b_)
    B0_mul_a1 = mp.run_at(1, phe.mul, B0_p1, a_)
    T_partial_ct = mp.run_at(1, phe.add, A0_mul_b1, B0_mul_a1)
    # Add plaintext term (a1*b1 + r) to ciphertext directly; kernel encrypts internally
    T_ct = mp.run_at(1, phe.add, T_partial_ct, a1b1_plus_r)

    # Send T back to P0 and decrypt
    T_ct_p0 = mp.p2p(1, 0, T_ct)
    c_dec = mp.run_at(0, phe.decrypt, T_ct_p0, sk_p0)

    # Set c shares directly such that c0 + c1 = a*b (mod 2^64)
    # Let P0 hold c0 = a0*b0 + C, and P1 hold c1 = -r
    a0b0 = mp.run_jax_at(0, lambda a, b: a * b, a_, b_)
    c0 = mp.run_jax_at(0, lambda a, b: a + b, a0b0, c_dec)
    c1 = mp.run_jax_at(1, lambda rr: jnp.uint64(0) - rr, r)
    c_ = mp.pconv([c0, c1])

    return a_, b_, c_


# ----------------------------
# Online mul using triple (2PC)
# ----------------------------


def ss_open(x_: mp.MPObject) -> mp.MPObject:
    """Open a distributed value x_ s.t. all parties learn x0 + x1 (mod 2^64).

    Implementation (symmetric, allgather-style using available primitives):
        - Each party extracts its local share (x0 at P0, x1 at P1)
        - Broadcast x0 from P0 and x1 from P1 to all parties
        - Each party sums the two broadcast values locally
    """
    world_mask = mp.Mask.all(2)
    # Broadcast both shares to all parties (emulates allgather for 2PC)
    # TODO(jint) allgather is not yet implemented in SIMP, use p2p+bcast as workaround
    x0_all = mp.bcast_m(world_mask, 0, x_)
    x1_all = mp.bcast_m(world_mask, 1, x_)
    # Each party can sum locally
    return mp.run_jax(lambda a, b: a + b, x0_all, x1_all)


def ss_mul(x_: mp.MPObject, y_: mp.MPObject) -> mp.MPObject:
    """Compute z_ = x_*y_ (mod 2^64) using a Beaver triple.

    If a_, b_, c_ are not provided, a PHE-based triple is generated internally.
    """
    # Generate Beaver triple (a, b, c) where c = a*b
    a_, b_, c_ = beaver_gen()
    # Local differences
    d_ = mp.run_jax(lambda x, a: x - a, x_, a_)
    e_ = mp.run_jax(lambda y, b: y - b, y_, b_)

    # Open d, e to all parties
    d_open = ss_open(d_)
    e_open = ss_open(e_)

    # Beaver reconstruction formulas:
    # z = c + d*b + e*a + d*e (where each party computes on local shares)
    # P0 computes: z0 = c0 + d*b0 + e*a0 + d*e
    def beaver_reconstruct_p0(c0, b0, a0, d, e):
        """P0's share: includes the constant term d*e."""
        return c0 + (d * b0 + e * a0) + d * e

    z0 = mp.run_jax_at(0, beaver_reconstruct_p0, c_, b_, a_, d_open, e_open)

    # P1 computes: z1 = c1 + d*b1 + e*a1
    def beaver_reconstruct_p1(c1, b1, a1, d, e):
        """P1's share: no constant term."""
        return c1 + (d * b1 + e * a1)

    z1 = mp.run_jax_at(1, beaver_reconstruct_p1, c_, b_, a_, d_open, e_open)

    # Package z0,z1 into a single distributed value
    z_ = mp.pconv([z0, z1])
    return z_


# ----------------------------
# Demo program
# ----------------------------


@mp.function
def demo_2pc_ss_u64() -> tuple[
    mp.MPObject, mp.MPObject, mp.MPObject, mp.MPObject, mp.MPObject, mp.MPObject
]:
    # Inputs: let x be owned by P0, y owned by P1
    # x_plain = mp.run_jax_at(0, random.getrandbits, 32)
    # y_plain = mp.run_jax_at(1, random.getrandbits, 32)
    x0 = mp.run_jax_at(0, lambda: 42)
    y1 = mp.run_jax_at(1, lambda: 17)

    # Share the inputs
    x_ = ss_share_from(0, x0)
    y_ = ss_share_from(1, y1)

    # Local add of shares and reveal/open
    z_add_ = ss_add(x_, y_)
    z_add = ss_open(z_add_)

    # Online mul (triple is generated internally via PHE for simplicity)
    z_mul_ = ss_mul(x_, y_)
    z_mul = ss_open(z_mul_)

    # For verification, also compute plaintext results (mod 2^64) at P0
    y_p0 = mp.p2p(1, 0, y1)
    add_plain = mp.run_jax_at(0, lambda x, y: to_u64_j(x) + to_u64_j(y), x0, y_p0)
    mul_plain = mp.run_jax_at(0, lambda x, y: to_u64_j(x) * to_u64_j(y), x0, y_p0)
    add_plain_all = mp.bcast_m(mp.Mask.all(2), 0, add_plain)
    mul_plain_all = mp.bcast_m(mp.Mask.all(2), 0, mul_plain)

    return (
        x0,
        y1,
        z_add,
        add_plain_all,
        z_mul,
        mul_plain_all,
    )


def run_simulation() -> None:
    sim = mp.Simulator.simple(2)
    results = mp.evaluate(sim, demo_2pc_ss_u64)
    x_plain, y_plain, z_add, add_plain, z_mul, mul_plain = mp.fetch(sim, results)

    print("Inputs:")
    print("  x (P0):", x_plain)
    print("  y (P1):", y_plain)

    print("\nAddition:")
    print("  reveal(x+y) :", z_add)
    print("  plaintext   :", add_plain)

    print("\nMultiplication:")
    print("  reveal(x*y) :", z_mul)
    print("  plaintext   :", mul_plain)

    compiled = mp.compile(sim, demo_2pc_ss_u64)
    print("\n=== Compilation IR (truncated) ===")
    print(compiled.compiler_ir())


if __name__ == "__main__":
    run_simulation()
