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

"""Tests for SPU dialect."""

import jax.numpy as jnp
import pytest

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt
from mplang.v2.dialects import simp, spu


def test_spu_config():
    config = spu.SPUConfig()
    assert config.protocol == "SEMI2K"


def test_encrypt_decrypt_flow():
    # 1. Setup
    parties = (0, 1, 2)
    config = spu.SPUConfig()

    # 2. Define trace function
    def trace_fn(x):
        # Encrypt
        # Manual encrypt
        x_shares = simp.pcall_static(
            (0,), lambda x: spu.make_shares(config, x, count=3), x
        )
        x_dist = []
        for i, target in enumerate(parties):
            x_dist.append(simp.shuffle_static(x_shares[i], {target: 0}))
        x_enc = simp.converge(*x_dist)

        # Verify type
        assert isinstance(x_enc.type, elt.MPType)
        assert isinstance(x_enc.type.value_type, elt.SSType)
        assert x_enc.type.parties == (0, 1, 2)

        # Decrypt
        # Manual decrypt
        x_shares_back = []
        for source in parties:
            share = simp.pcall_static((source,), lambda x: x, x_enc)
            x_shares_back.append(simp.shuffle_static(share, {0: source}))

        x_dec = simp.pcall_static(
            (0,), lambda *s: spu.reconstruct(config, s), *x_shares_back
        )

        # Verify type
        assert isinstance(x_dec.type, elt.MPType)
        assert isinstance(x_dec.type.value_type, elt.TensorType)
        assert x_dec.type.parties == (0,)

        return x_dec

    # 3. Trace
    # Input x is on party 0
    x_tracer = el.Tracer()
    with x_tracer:
        # Create input value in the graph
        x_in = simp.constant((0,), jnp.array([1.0] * 10, dtype=jnp.float32))

        _ = trace_fn(x_in)

    # 4. Verify graph structure
    graph = x_tracer.graph
    # Check for primitives
    op_names = [op.opcode for op in graph.operations]
    assert "simp.pcall_static" in op_names
    assert "simp.shuffle" in op_names  # shuffle_static uses "simp.shuffle" opcode
    assert "simp.converge" in op_names


def test_jit_compilation():
    # 1. Setup
    parties = (0, 1, 2)
    config = spu.SPUConfig()

    # 2. Define JAX function
    def secure_add(x, y):
        return x + y

    # 3. Trace usage
    def trace_fn(x, y):
        # Assume x, y are already encrypted
        z = simp.pcall_static(
            parties, lambda x, y: spu.run_jax(config, secure_add, x, y), x, y
        )
        return z

    # Input types: MP[SS[Tensor], (0,1,2)]
    x_tracer = el.Tracer()
    with x_tracer:
        # Create encrypted inputs directly for testing
        # Note: In real usage, we would encrypt from plaintext.
        # Here we simulate having encrypted inputs by using graph inputs with correct types.
        ss_type = elt.SS(elt.Tensor(elt.f32, (10,)))
        mp_ss_type = elt.MPType(ss_type, (0, 1, 2))

        x_val = x_tracer.graph.add_input("x", mp_ss_type)
        y_val = x_tracer.graph.add_input("y", mp_ss_type)
        x_in = el.TraceObject(x_val, x_tracer)
        y_in = el.TraceObject(y_val, x_tracer)

        res = trace_fn(x_in, y_in)

        assert isinstance(res.type, elt.MPType)
        assert isinstance(res.type.value_type, elt.SSType)
        assert res.type.parties == (0, 1, 2)

    # Verify graph
    graph = x_tracer.graph
    # Should contain pcall_static for execution
    op_names = [op.opcode for op in graph.operations]
    assert "simp.pcall_static" in op_names

    # Inspect pcall region to see if it contains spu.exec
    # Find the pcall op
    pcall_ops = [op for op in graph.operations if op.opcode == "simp.pcall_static"]
    # The last one should be the execution (encrypt calls pcall too, but here we assume inputs are already encrypted)
    # Wait, we passed encrypted inputs directly. So only secure_add calls pcall.
    # But secure_add calls pcall multiple times?
    # 1. cast_from_ss (pcall)
    # 2. local_exec (pcall)
    # 3. cast_to_ss (pcall)

    # assert len(pcall_ops) >= 3

    # Check if one of them contains spu.exec
    found_exec = False
    for op in pcall_ops:
        region = op.regions[0]
        region_ops = [rop.opcode for rop in region.operations]
        if "spu.exec" in region_ops:
            found_exec = True
            break

    assert found_exec, "spu.exec not found in pcall regions"


def test_run_jax_mixed_pytree():
    # 1. Setup
    parties = (0, 1, 2)
    config = spu.SPUConfig()

    # 2. Define JAX function with mixed output
    def mixed_fn(x, y):
        return {
            "sum": x + y,
            "static": 42,
            "nested": (x * y, 100),
        }

    # 3. Trace usage
    def trace_fn(x, y):
        z = simp.pcall_static(
            parties, lambda x, y: spu.run_jax(config, mixed_fn, x, y), x, y
        )
        return z

    x_tracer = el.Tracer()
    with x_tracer:
        ss_type = elt.SS(elt.Tensor(elt.f32, (10,)))
        mp_ss_type = elt.MPType(ss_type, (0, 1, 2))

        x_val = x_tracer.graph.add_input("x", mp_ss_type)
        y_val = x_tracer.graph.add_input("y", mp_ss_type)
        x_in = el.TraceObject(x_val, x_tracer)
        y_in = el.TraceObject(y_val, x_tracer)

        res = trace_fn(x_in, y_in)

        # Verify result structure
        assert isinstance(res, dict)
        assert "sum" in res
        assert "static" in res
        assert "nested" in res

        # Check types
        # "sum" should be MPType[SSType]
        assert isinstance(res["sum"].type, elt.MPType)
        assert isinstance(res["sum"].type.value_type, elt.SSType)

        # "static" should be MPType[SSType] because JAX/SPU treats it as output
        # Note: spu_fe.compile returns OutInfo for static ints, so they become shares.
        assert isinstance(res["static"].type, elt.MPType)
        assert isinstance(res["static"].type.value_type, elt.SSType)

        # "nested" should be tuple (MPType[SSType], MPType[SSType])
        nested = res["nested"]
        assert isinstance(nested, tuple)
        assert len(nested) == 2
        assert isinstance(nested[0].type, elt.MPType)
        assert isinstance(nested[0].type.value_type, elt.SSType)
        assert isinstance(nested[1].type, elt.MPType)
        assert isinstance(nested[1].type.value_type, elt.SSType)


if __name__ == "__main__":
    pytest.main([__file__])
