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

import mplang
import mplang.simp as simp
from mplang.frontend import crypto


def test_basic_callable_and_namespace():
    sim = mplang.Simulator.simple(3)

    @mplang.function
    def prog():
        # universal form
        a, b = simp.P0(crypto.kem_keygen, "x25519")
        # namespace form (tee side key, then quote)
        t_sk, t_pk = simp.P[2].crypto.kem_keygen("x25519")
        _ = simp.P[2].tee.quote(t_pk)
        # derive something simple at party 0 to ensure run path works
        _ = simp.P0(lambda x: x + 1, 41)
        return a, b, t_sk, t_pk

    a, b, t_sk, t_pk = mplang.evaluate(sim, prog)
    # Just basic shape checks: objects should be MPObjects with attrs; rely on existing crypto tests for deep correctness
    assert hasattr(a, "attrs") and hasattr(b, "attrs")
    assert hasattr(t_sk, "attrs") and hasattr(t_pk, "attrs")


def test_bound_method_style_lambda():
    sim = mplang.Simulator.simple(3)

    class Box:
        def __init__(self, v: int):
            self.v = v

        def inc(self, d: int) -> int:  # pure python op ok
            return self.v + d

    @mplang.function
    def prog():
        box = Box(10)
        # Pass bound method directly
        r1 = simp.P0(box.inc, 5)
        # Or via lambda exposing self
        r2 = simp.P0(lambda fn, d: fn(d), box.inc, 7)
        return r1, r2

    r1, r2 = mplang.evaluate(sim, prog)
    # Fetch the concrete per-party values (list per MPObject); we only need party0's view.
    r1_f, r2_f = mplang.fetch(sim, (r1, r2))
    assert int(r1_f[0]) == 15
    assert int(r2_f[0]) == 17
