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

from typing import Any

import jax.numpy as jnp
import numpy as np

import mplang as mp
from mplang.dialects import simp


def test_shuffle_key_unique_in_while_loop_slow_receiver() -> None:
    """Regression: shuffle inside while_loop must not overflow mailbox.

    Historically, simp.shuffle used key = f"shuffle_{op.name}_{tgt}", so repeated
    execution of the same shuffle op inside simp.while_loop could reuse the same
    key and overflow the receiver mailbox.

    The fix makes keys unique by incorporating a deterministic per-op exec_id.
    This test constructs a slow receiver (P0) and a fast sender-only (P1) to
    stress the mailbox.
    """

    sim = simp.make_simulator(world_size=2)

    def program() -> Any:
        recv_slot = mp.put("P0", np.int32(0))
        i0 = mp.put("P0", np.int32(0))

        payload = mp.put("P1", np.int32(123))
        i1 = mp.put("P1", np.int32(0))

        def slow_before_recv(x: jnp.ndarray) -> jnp.ndarray:
            # Keep it modest to avoid test slowness.
            m = jnp.ones((128, 128), dtype=jnp.float32)
            y = m @ m
            s = jnp.sum(y)
            dummy0 = jnp.where(
                s > 0,
                jnp.array(0, dtype=jnp.int32),
                jnp.array(0, dtype=jnp.int32),
            )
            return x + dummy0

        def cond_fn(carry: tuple[Any, Any, Any, Any]) -> Any:
            _, _, i0_, i1_ = carry
            pred0 = mp.device("P0").jax(lambda j: j < np.int32(5))(i0_)
            pred1 = mp.device("P1").jax(lambda j: j < np.int32(5))(i1_)
            return simp.converge(pred1, pred0)

        def body_fn(carry: tuple[Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any]:
            recv_slot_, payload_, i0_, i1_ = carry
            _ = mp.device("P0").jax(slow_before_recv)(recv_slot_)
            recv_slot2 = simp.shuffle_static(payload_, routing={0: 1})
            i0_2 = mp.device("P0").jax(lambda j: j + np.int32(1))(i0_)
            i1_2 = mp.device("P1").jax(lambda j: j + np.int32(1))(i1_)
            return recv_slot2, payload_, i0_2, i1_2

        return simp.while_loop(cond_fn, body_fn, (recv_slot, payload, i0, i1))

    with sim:
        # Should not raise Mailbox overflow.
        _ = mp.evaluate(program)
