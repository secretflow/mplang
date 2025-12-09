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

import jax.numpy as jnp
import pytest
import spu.libspu as libspu

from mplang.v1.ops import spu
from tests.v1.ops.dummy import DummyTensor


def test_make_shares_basic() -> None:
    t = DummyTensor(jnp.float32, (3,))
    pfunc, ins, _out_tree = spu.makeshares(t, world_size=3)
    assert pfunc.fn_type == "spu.makeshares"
    assert len(ins) == 1
    assert len(pfunc.outs_info) == 3
    assert pfunc.attrs["world_size"] == 3


def test_make_shares_private_validation() -> None:
    t = DummyTensor(jnp.int32, (1,))
    pfunc, _, _ = spu.makeshares(
        t,
        world_size=3,
        visibility=spu.Visibility.PRIVATE,
        owner_rank=2,
        enable_private=True,
    )
    assert pfunc.attrs["owner_rank"] == 2
    assert pfunc.attrs["visibility"] == libspu.Visibility.VIS_PRIVATE


def test_reconstruct_basic() -> None:
    s1 = DummyTensor(jnp.int64, (2,))
    s2 = DummyTensor(jnp.int64, (2,))
    pfunc, ins, _tree = spu.reconstruct(s1, s2)
    assert pfunc.fn_type == "spu.reconstruct"
    assert len(ins) == 2
    assert len(pfunc.outs_info) == 1


def test_reconstruct_world_size_mismatch() -> None:
    with pytest.raises(ValueError):
        spu.reconstruct()  # no shares provided


def test_jax_compile_simple_add() -> None:
    def fn(a, b):  # type: ignore[no-untyped-def]
        return a + b

    a = DummyTensor(jnp.float32, (2,))
    b = DummyTensor(jnp.float32, (2,))
    pfunc, ins, _out_tree = spu.jax_compile(fn, a, b)
    assert pfunc.fn_type == "spu.run_pphlo"
    assert len(ins) == 2
    assert len(pfunc.outs_info) == 1
    assert pfunc.fn_text is not None and "func" in pfunc.fn_text


def test_jax_compile_multiple_outputs() -> None:
    def fn(a):  # type: ignore[no-untyped-def]
        return a + 1, a * 2

    a = DummyTensor(jnp.int32, (3,))
    pfunc, _ins, _out_tree = spu.jax_compile(fn, a)
    assert len(pfunc.outs_info) == 2


def test_jax_compile_visibility_metadata_secret() -> None:
    def fn(a):  # type: ignore[no-untyped-def]
        return a * 2

    a = DummyTensor(jnp.float32, (2,))
    pfunc, _ins, _ = spu.jax_compile(fn, a)
    assert pfunc.attrs["input_visibilities"] == [libspu.Visibility.VIS_SECRET]
