# Copyright 2026 Ant Group Co., Ltd.
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


from types import SimpleNamespace

import numpy as np

from mplang.backends.util import _make_tensor_type


def test_make_tensor_type():
    """Test _make_tensor_type with various shapes and dtypes."""

    # Test scalar tensors
    arr = np.array(5, dtype=np.float32)
    assert _make_tensor_type(arr) == "tensor<f32>"

    arr = np.array(True, dtype=np.bool_)
    assert _make_tensor_type(arr) == "tensor<i1>"

    # Test static shapes
    arr = np.ones((10,), dtype=np.float32)
    assert _make_tensor_type(arr) == "tensor<10xf32>"

    arr = np.ones((3, 4), dtype=np.int64)
    assert _make_tensor_type(arr) == "tensor<3x4xi64>"

    arr = np.ones((1, 224, 224), dtype=np.float16)
    assert _make_tensor_type(arr) == "tensor<1x224x224xf16>"

    # Test dtype mappings
    test_cases = [
        (np.float16, "f16"),
        (np.float32, "f32"),
        (np.float64, "f64"),
        (np.int8, "i8"),
        (np.int16, "i16"),
        (np.int32, "i32"),
        (np.int64, "i64"),
        (np.uint8, "ui8"),
        (np.uint16, "ui16"),
        (np.uint32, "ui32"),
        (np.uint64, "ui64"),
        (np.bool_, "i1"),
    ]

    for dtype, expected_suffix in test_cases:
        arr = np.ones((2, 3), dtype=dtype)
        expected = f"tensor<2x3x{expected_suffix}>"
        assert _make_tensor_type(arr) == expected

    # Test custom dtype fallback
    tensor = SimpleNamespace(shape=(2,), dtype=np.dtype("float128"))
    assert _make_tensor_type(tensor) == "tensor<2xf128>"
