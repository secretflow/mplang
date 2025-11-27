"""Tests for SPU dialect."""

import jax.numpy as jnp
import pytest

import mplang2.edsl as el
import mplang2.edsl.typing as elt
from mplang2.dialects import simp, spu


def test_spu_device():
    device = spu.SPUDevice(parties=(0, 1, 2))
    assert device.parties == (0, 1, 2)


def test_encrypt_decrypt_flow():
    # 1. Setup
    device = spu.SPUDevice(parties=(0, 1, 2))

    # 2. Define trace function
    def trace_fn(x):
        # Encrypt
        # Manual encrypt
        x_shares = simp.pcall_static((0,), lambda x: spu.make_shares(x, count=3), x)
        x_dist = []
        for i, target in enumerate(device.parties):
            x_dist.append(simp.shuffle_static(x_shares[i], {target: 0}))
        x_enc = simp.converge(*x_dist)

        # Verify type
        assert isinstance(x_enc.type, elt.MPType)
        assert isinstance(x_enc.type.value_type, elt.SSType)
        assert x_enc.type.parties == (0, 1, 2)

        # Decrypt
        # Manual decrypt
        x_shares_back = []
        for source in device.parties:
            share = simp.pcall_static((source,), lambda x: x, x_enc)
            x_shares_back.append(simp.shuffle_static(share, {0: source}))

        x_dec = simp.pcall_static((0,), lambda *s: spu.reconstruct(s), *x_shares_back)

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
    device = spu.SPUDevice(parties=(0, 1, 2))

    # 2. Define JAX function
    def secure_add(x, y):
        return x + y

    # 3. Trace usage
    def trace_fn(x, y):
        # Assume x, y are already encrypted
        z = simp.pcall_static(
            device.parties, lambda x, y: spu.run_jax(secure_add, x, y), x, y
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


if __name__ == "__main__":
    pytest.main([__file__])
