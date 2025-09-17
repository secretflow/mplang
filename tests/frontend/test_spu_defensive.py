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
Defensive tests for the SPU frontend (contract-level checks).

Scope:
- Validate MPLang's SPU frontend operators (e.g., SpuJaxCompile) without relying on a
    real SPU runtime: shapes/dtypes, number of outputs, visibility metadata, and
    determinism of compilation results.
- Do NOT test SPU runtime/protocol correctness or performance; those are covered by
    the SPU library itself.
"""

import jax.numpy as jnp
import pytest
import spu.libspu as libspu

from mplang.core.cluster import ClusterSpec, Device, Node, RuntimeInfo
from mplang.core.dtype import DType
from mplang.core.mpobject import MPContext, MPObject
from mplang.core.mptype import MPType
from mplang.core.tensor import TensorType
from mplang.frontend.spu import SpuConfig, SpuJaxCompile


class DummyContext(MPContext):
    def __init__(self) -> None:
        runtime = RuntimeInfo(version="dev", platform="local", backends=[])
        node = Node(name="p0", rank=0, endpoint="local", runtime_info=runtime)
        device = Device(name="p0_local", kind="local", members=[node])
        spec = ClusterSpec(nodes={node.name: node}, devices={device.name: device})
        super().__init__(spec)


class Tensor(MPObject):
    def __init__(self, arr: jnp.ndarray) -> None:  # type: ignore[valid-type]
        self._arr = arr
        self._mptype = MPType.tensor(DType.from_any(str(arr.dtype)), tuple(arr.shape))
        self._ctx = DummyContext()

    @property
    def mptype(self) -> MPType:
        return self._mptype

    @property
    def ctx(self) -> MPContext:
        return self._ctx

    def runtime_obj(self):
        return self._arr


@pytest.mark.parametrize(
    "func,input_shapes,expected_outputs",
    [
        (lambda x, y: x + y, [(2, 3), (2, 3)], 1),
        (lambda x, y: x * y, [(3,), (3,)], 1),
        (lambda x: x + 1.0, [(2,)], 1),
        (lambda x: (x + 1, x * 2), [(3,)], 2),
        (
            lambda x: (jnp.mean(x), jnp.sum(x), jnp.max(x)),
            [(3,)],
            3,
        ),
    ],
)
def test_basic_and_multi_outputs(func, input_shapes, expected_outputs):
    cfg = SpuConfig(world_size=2)
    compiler = SpuJaxCompile(cfg)
    args = [Tensor(jnp.zeros(shape, dtype=jnp.float32)) for shape in input_shapes]
    pfunc, _ins, _tree = compiler(func, *args)
    assert pfunc.fn_type == "mlir.pphlo"
    assert len(pfunc.outs_info) == expected_outputs


@pytest.mark.parametrize(
    "dtype1,dtype2",
    [
        (jnp.float32, jnp.float32),
        (jnp.float64, jnp.float64),
        (jnp.int32, jnp.int32),
    ],
)
def test_different_dtypes(dtype1, dtype2):
    def fn(x, y):
        return x + y

    cfg = SpuConfig(world_size=2)
    compiler = SpuJaxCompile(cfg)
    args = [
        Tensor(jnp.zeros((2,), dtype=dtype1)),
        Tensor(jnp.zeros((2,), dtype=dtype2)),
    ]
    pfunc, _ins, _tree = compiler(fn, *args)
    assert isinstance(pfunc.ins_info[0], TensorType)
    assert isinstance(pfunc.ins_info[1], TensorType)
    assert pfunc.ins_info[0].dtype.name == dtype1.__name__
    assert pfunc.ins_info[1].dtype.name == dtype2.__name__


def test_complex_function_deterministic():
    def complex_fn(x, y, z):
        temp = x + y
        result = temp * z
        return jnp.sum(result, axis=0)

    cfg = SpuConfig(world_size=2)
    compiler = SpuJaxCompile(cfg)
    args = [Tensor(jnp.zeros((3, 4), dtype=jnp.float32)) for _ in range(3)]
    p1, _i1, _t1 = compiler(complex_fn, *args)
    p2, _i2, _t2 = compiler(complex_fn, *args)
    assert p1.fn_type == p2.fn_type
    assert p1.fn_name == p2.fn_name
    assert p1.ins_info == p2.ins_info
    assert p1.outs_info == p2.outs_info


@pytest.mark.parametrize(
    "test_name,func_def,input_shape,expected_output_shape",
    [
        (
            "high_dimensional_tensors",
            lambda x: jnp.sum(x, axis=(1, 3)),
            (2, 3, 4, 5),
            (2, 4),
        ),
        (
            "matrix_multiplication",
            lambda x, y: jnp.dot(x, y),
            [(100, 50), (50, 75)],
            (100, 75),
        ),
        ("basic_reshape", lambda x: jnp.reshape(x, (-1,)), (2, 3), (6,)),
        ("transpose_operation", lambda x: jnp.transpose(x), (3, 4), (4, 3)),
    ],
)
def test_tensor_operations_parametrized(
    test_name, func_def, input_shape, expected_output_shape
):
    cfg = SpuConfig(world_size=2)
    compiler = SpuJaxCompile(cfg)
    if isinstance(input_shape[0], tuple):
        args = [Tensor(jnp.zeros(shape, dtype=jnp.float32)) for shape in input_shape]
    else:
        args = [Tensor(jnp.zeros(input_shape, dtype=jnp.float32))]
    pfunc, _ins, _ = compiler(func_def, *args)
    assert len(pfunc.ins_info) == len(args)
    assert len(pfunc.outs_info) >= 1


@pytest.mark.parametrize("n_inputs", [1, 2, 3])
def test_visibility_settings_all_secret(n_inputs):
    def multi_input_fn(*args):
        result = args[0]
        for i in range(1, len(args)):
            result = result + args[i]
        return result

    cfg = SpuConfig(world_size=2)
    compiler = SpuJaxCompile(cfg)
    args = [Tensor(jnp.zeros((2,), dtype=jnp.float32)) for _ in range(n_inputs)]
    pfunc, _ins, _ = compiler(multi_input_fn, *args)
    vis = pfunc.attrs["input_visibilities"]
    assert len(vis) == n_inputs
    assert all(v == libspu.Visibility.VIS_SECRET for v in vis)
