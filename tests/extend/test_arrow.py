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

import numpy as np
import numpy.testing as npt
import pyarrow as pa
import pytest

from mplang.extend.arrow import (
    OneHotVectorArray,
    dense_vector,
    onehot_vector,
    sparse_vector,
)


class TestVectorArray:
    def test_dense_array(self):
        # Test DenseVectorArray.to_numpy
        dense_dtype = dense_vector(size=3, value_type=pa.float32())
        dense_storage = pa.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type=dense_dtype.storage_type
        )
        dense_array = pa.ExtensionArray.from_storage(dense_dtype, dense_storage)
        dense_result = dense_array.to_numpy(zero_copy_only=False)
        assert dense_result.shape == (2, 3)
        assert dense_result.dtype == np.float32
        npt.assert_array_equal(
            dense_result, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )

        # Test DenseVectorArray.to_pylist
        assert dense_array.to_pylist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        # Test DenseVectorArray with int32
        int_dtype = dense_vector(size=2, value_type=pa.int32())
        int_storage = pa.array([[10, 20], [30, 40]], type=int_dtype.storage_type)
        int_array = pa.ExtensionArray.from_storage(int_dtype, int_storage)
        int_result = int_array.to_numpy(zero_copy_only=False)
        assert int_result.shape == (2, 2)
        assert int_result.dtype == np.int32
        npt.assert_array_equal(
            int_result, np.array([[10, 20], [30, 40]], dtype=np.int32)
        )

    def test_sparse_array(self):
        # Test SparseVectorArray.to_numpy
        sparse_dtype = sparse_vector(size=5, value_type=pa.float32())
        sparse_storage = pa.array(
            [
                {"indices": [0, 2], "values": [1.0, 3.0]},
                {"indices": [1, 4], "values": [2.0, 5.0]},
                {"indices": [3], "values": [4.0]},
            ],
            type=sparse_dtype.storage_type,
        )
        sparse_array = pa.ExtensionArray.from_storage(sparse_dtype, sparse_storage)
        sparse_result = sparse_array.to_numpy()
        assert sparse_result.shape == (3, 5)
        assert sparse_result.dtype == np.float32
        npt.assert_array_equal(
            sparse_result,
            np.array(
                [
                    [1.0, 0.0, 3.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0, 5.0],
                    [0.0, 0.0, 0.0, 4.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )

        # Test SparseVectorArray with int64
        int_sparse_dtype = sparse_vector(size=4, value_type=pa.int64())
        int_sparse_storage = pa.array(
            [
                {"indices": [0, 3], "values": [100, 400]},
                {"indices": [1], "values": [200]},
            ],
            type=int_sparse_dtype.storage_type,
        )
        int_sparse_array = pa.ExtensionArray.from_storage(
            int_sparse_dtype, int_sparse_storage
        )
        int_sparse_result = int_sparse_array.to_numpy()
        assert int_sparse_result.shape == (2, 4)
        assert int_sparse_result.dtype == np.int64
        npt.assert_array_equal(
            int_sparse_result,
            np.array(
                [[100, 0, 0, 400], [0, 200, 0, 0]],
                dtype=np.int64,
            ),
        )

        # Test SparseVectorArray.to_pylist
        assert sparse_array.to_pylist() == [
            [1.0, 0.0, 3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 4.0, 0.0],
        ]

        # Test __getitem__
        from mplang.extend.arrow.vector import SparseVectorItem

        item = sparse_array[0]
        assert isinstance(item, SparseVectorItem)
        assert item[2] == 3.0
        assert item.as_py() == [1.0, 0.0, 3.0, 0.0, 0.0]

        # Test slice
        assert len(sparse_array[::1]) == 3

    def test_onehot_array(self):
        # Test OneHotVectorArray.to_numpy
        onehot_dtype = onehot_vector(size=4, value_type=pa.float32())
        onehot_storage = pa.array([1, 2, 0, 3], type=onehot_dtype.storage_type)
        onehot_array = pa.ExtensionArray.from_storage(onehot_dtype, onehot_storage)
        onehot_result = onehot_array.to_numpy()
        assert onehot_result.shape == (4, 4)
        assert onehot_result.dtype == np.float32
        npt.assert_array_equal(
            onehot_result,
            np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )

        # Test OneHotVectorArray with int64
        int_onehot_dtype = onehot_vector(size=3, value_type=pa.int64())
        int_onehot_storage = pa.array([0, 1, 2], type=int_onehot_dtype.storage_type)
        int_onehot_array = pa.ExtensionArray.from_storage(
            int_onehot_dtype, int_onehot_storage
        )
        int_onehot_result = int_onehot_array.to_numpy()
        assert int_onehot_result.shape == (3, 3)
        assert int_onehot_result.dtype == np.int64
        npt.assert_array_equal(
            int_onehot_result,
            np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                dtype=np.int64,
            ),
        )

        # Test OneHotVectorArray with None/null values
        onehot_dtype = onehot_vector(size=3, value_type=pa.float32())
        onehot_storage_with_null = pa.array(
            [1, None, 2], type=onehot_dtype.storage_type
        )
        onehot_array_with_null = pa.ExtensionArray.from_storage(
            onehot_dtype, onehot_storage_with_null
        )
        onehot_result_with_null = onehot_array_with_null.to_numpy()
        assert onehot_result_with_null.shape == (3, 3)
        assert onehot_result_with_null.dtype == np.float32
        npt.assert_array_equal(
            onehot_result_with_null,
            np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],  # None becomes all zeros
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )

        # Test __getitem__
        from mplang.extend.arrow.vector import OneHotVectorItem

        item = onehot_array[0]
        assert isinstance(item, OneHotVectorItem)
        assert item[0] == 0
        assert item[1] == 1
        assert item.as_py() == [0.0, 1.0, 0.0, 0.0]

        # Test slice
        assert len(onehot_array[::1]) == 4

    def test_onehot_array_cast(self):
        # Test basic cast from int32 array to one-hot vector array
        ints = pa.array([0, 1, 2, 1])
        vt = onehot_vector(size=3, value_type=pa.float32(), index_type=pa.int32())
        result = OneHotVectorArray.from_array(ints, vt)

        assert isinstance(result, OneHotVectorArray)
        assert result.type == vt
        npt.assert_array_equal(
            result.to_numpy(),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32),
        )

        # Test cast to_pylist
        assert result.to_pylist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]

        # Test cast with different value types
        vt_int64 = onehot_vector(size=3, value_type=pa.int64(), index_type=pa.int64())
        result_int64 = OneHotVectorArray.from_array(
            pa.array([2, 0, 1], type=pa.int64()), vt_int64
        )
        assert result_int64.to_numpy().dtype == np.int64
        npt.assert_array_equal(
            result_int64.to_numpy(),
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.int64),
        )

        # Test cast with null values
        ints_with_null = pa.array([1, None, 2], type=pa.int32())
        vt = onehot_vector(size=3)
        result_with_null = OneHotVectorArray.from_array(ints_with_null, vt)
        npt.assert_array_equal(
            result_with_null.to_numpy(),
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float32),
        )

        # Test cast invalid type (float)
        with pytest.raises(TypeError, match="Input array must be of integer type"):
            OneHotVectorArray.from_array(
                pa.array([1.0, 2.0, 3.0]),
                onehot_vector(size=3),
            )


def test_vector_type():
    # Test dense vector
    dtype = dense_vector(size=10, value_type=pa.int32())
    assert dtype.size == 10
    assert str(dtype) == "dense_vector[int32, size=10]"

    # Test sparse vector
    dtype = sparse_vector(size=100, value_type=pa.float64())
    assert dtype.size == 100
    # PyArrow represents float64 as 'double' in string representation
    assert str(dtype) == "sparse_vector[double, size=100]"

    # Test onehot vector
    dtype = onehot_vector(size=10, value_type=pa.float32())
    assert dtype.size == 10
    # PyArrow represents float32 as 'float' in string representation
    assert str(dtype) == "onehot_vector[float, size=10]"

    # Test equality
    assert dense_vector(5) == dense_vector(5)
    assert sparse_vector(10) != sparse_vector(20)
    assert onehot_vector(10) != dense_vector(10)

    # Test invalid size
    with pytest.raises(ValueError, match="size must be > 0"):
        dense_vector(size=0)
    with pytest.raises(ValueError, match="size must be > 0"):
        sparse_vector(size=-1)

    # Test invalid value type
    with pytest.raises(TypeError, match="value_type must be a numeric type"):
        dense_vector(size=10, value_type=pa.string())


def test_table_to_numpy():
    from mplang.extend.arrow.vector import table_to_numpy

    # Test 1: No VectorType - regular columns remain as-is
    table = pa.table({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = table_to_numpy(table)
    expected = np.array([[1, 4.0], [2, 5.0], [3, 6.0]])
    npt.assert_array_equal(result, expected)
    assert result.shape == (3, 2)

    # Test 2: Mixed with all VectorType types and regular columns
    # Data:
    #   a:   [10, 20]
    #   dense:   [[1.0, 2.0], [3.0, 4.0]]  -> expands to 2 columns
    #   sparse:  [[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]] -> expands to 3 columns
    #   onehot:  [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] -> expands to 3 columns
    #   b:   [3.0, 4.0]
    # Result shape: (2, 1+2+3+3+1) = (2, 10)
    dense_dtype = dense_vector(size=2, value_type=pa.float32())
    dense_storage = pa.array([[1.0, 2.0], [3.0, 4.0]], type=dense_dtype.storage_type)
    dense_array = pa.ExtensionArray.from_storage(dense_dtype, dense_storage)

    sparse_dtype = sparse_vector(size=3, value_type=pa.float32())
    sparse_storage = pa.array(
        [{"indices": [0, 2], "values": [1.0, 3.0]}, {"indices": [1], "values": [2.0]}],
        type=sparse_dtype.storage_type,
    )
    sparse_array = pa.ExtensionArray.from_storage(sparse_dtype, sparse_storage)

    onehot_dtype = onehot_vector(size=3, value_type=pa.float32())
    onehot_storage = pa.array([0, 1], type=onehot_dtype.storage_type)
    onehot_array = pa.ExtensionArray.from_storage(onehot_dtype, onehot_storage)

    # Create schema to preserve extension types
    schema = pa.schema([
        pa.field("a", pa.int64()),
        pa.field("dense", dense_dtype),
        pa.field("sparse", sparse_dtype),
        pa.field("onehot", onehot_dtype),
        pa.field("b", pa.float64()),
    ])
    table = pa.table(
        {
            "a": [10, 20],
            "dense": dense_array,
            "sparse": sparse_array,
            "onehot": onehot_array,
            "b": [3.0, 4.0],
        },
        schema=schema,
    )

    result = table_to_numpy(table)
    # Row 0: 10, 1.0, 2.0, 1.0, 0.0, 3.0, 1.0, 0.0, 0.0, 3.0
    # Row 1: 20, 3.0, 4.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 4.0
    expected = np.array(
        [
            [10, 1.0, 2.0, 1.0, 0.0, 3.0, 1.0, 0.0, 0.0, 3.0],
            [20, 3.0, 4.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 4.0],
        ],
        dtype=np.float32,
    )
    npt.assert_array_equal(result, expected)
    assert result.shape == (2, 10)
