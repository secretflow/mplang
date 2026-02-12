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
import pyarrow.compute as pc
import pytest

from mplang.extend.arrow import (
    OneHotVectorArray,
    dense_vector,
    one_hot_vector,
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
        from mplang.extend.arrow import SparseVectorItem

        item = sparse_array[0]
        assert isinstance(item, SparseVectorItem)
        assert item[2] == 3.0
        assert item.as_py() == [1.0, 0.0, 3.0, 0.0, 0.0]

        # Test slice
        assert len(sparse_array[::1]) == 3

    def test_one_hot_array(self):
        # Test OneHotVectorArray.to_numpy
        onehot_dtype = one_hot_vector(size=4, value_type=pa.float32())
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
        int_onehot_dtype = one_hot_vector(size=3, value_type=pa.int64())
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
        onehot_dtype = one_hot_vector(size=3, value_type=pa.float32())
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
        from mplang.extend.arrow import OneHotVectorItem

        item = onehot_array[0]
        assert isinstance(item, OneHotVectorItem)
        assert item[0] == 0
        assert item[1] == 1
        assert item.as_py() == [0.0, 1.0, 0.0, 0.0]

        # Test slice
        assert len(onehot_array[::1]) == 4


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
    dtype = one_hot_vector(size=10, value_type=pa.float32())
    assert dtype.size == 10
    # PyArrow represents float32 as 'float' in string representation
    assert str(dtype) == "one_hot_vector[float, size=10]"

    # Test equality
    assert dense_vector(5) == dense_vector(5)
    assert sparse_vector(10) != sparse_vector(20)
    assert one_hot_vector(10) != dense_vector(10)

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

    onehot_dtype = one_hot_vector(size=3, value_type=pa.float32())
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


class TestCompute:
    def test_one_hot(self):
        from mplang.extend.arrow.compute import one_hot

        # Test basiccast with default array type (int64) - itype="" keeps array type
        # pa.array([0, 1, 2, 1]) creates int64 by default
        ints = pa.array([0, 1, 2, 1])
        result = one_hot(ints, 3)

        assert isinstance(result, OneHotVectorArray)
        vt = one_hot_vector(size=3, value_type=pa.float32(), index_type=pa.int64())
        assert result.type == vt
        npt.assert_array_equal(
            result.to_numpy(),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32),
        )

        # Test cast to_pylist
        assert result.to_pylist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]

        # Test with explicit int32 input type - itype="" keeps int32
        ints_int32 = pa.array([0, 1, 2, 1], type=pa.int32())
        result_int32 = one_hot(ints_int32, 3)
        vt_int32 = one_hot_vector(
            size=3, value_type=pa.float32(), index_type=pa.int32()
        )
        assert result_int32.type == vt_int32

        # Test cast with different value types
        result_int64 = one_hot(pa.array([2, 0, 1], type=pa.int64()), 3, dtype="int64")
        assert result_int64.to_numpy().dtype == np.int64
        npt.assert_array_equal(
            result_int64.to_numpy(),
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.int64),
        )

        # Test cast with explicit itype - override default behavior
        ints_int32 = pa.array([0, 1, 2], type=pa.int32())
        result_custom_itype = one_hot(ints_int32, 3, itype="int8")
        vt_custom = one_hot_vector(
            size=3, value_type=pa.float32(), index_type=pa.int8()
        )
        assert result_custom_itype.type == vt_custom

        # Test cast with null values
        ints_with_null = pa.array([1, None, 2], type=pa.int32())
        result_with_null = one_hot(ints_with_null, 3)
        npt.assert_array_equal(
            result_with_null.to_numpy(),
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float32),
        )

        # Test cast invalid type (float)
        with pytest.raises(TypeError, match="Input array must be of integer type"):
            one_hot(pa.array([1.0, 2.0, 3.0]), 3)

    def test_call_one_hot(self):
        """Test call one_hot func via pc.call_function.

        Covers:
        1. Multiple input types (int8, int16, int32, int64, uint8, uint16, uint32, uint64)
        2. 2 parameters: (array, num_classes) - uses default dtype="float32", itype=""
        3. 3 parameters: (array, num_classes, dtype) - uses default itype=""
        4. 4 parameters: (array, num_classes, dtype, itype) - full specification
        """
        import mplang.extend.arrow.compute as mpc

        mpc.register_functions()

        # ========== Helper function for testing ==========
        def test_arity_for_type(
            input_type: pa.DataType, suffix: str, dtype: str | None, itype: str | None
        ):
            """Test one_hot function with given parameters.

            - dtype=None, itype=None: test 2 params (array, num_classes)
            - dtype!=None, itype=None: test 3 params (array, num_classes, dtype)
            - dtype!=None, itype!=None: test 4 params (array, num_classes, dtype, itype)

            The storage result has the same values as test_data, but the type varies.
            For 2 and 3 param functions, storage type is int32 (registered out_type).
            For 4 param functions, storage type is the specified itype.
            """
            test_data = [0, 1, 2, 1]
            ints = pa.array(test_data, type=input_type)

            if not dtype and not itype:
                func_name = f"one_hot_{suffix}"
                args = [ints, pa.scalar(3)]
                output_type = input_type
            elif dtype and not itype:
                func_name = f"one_hot_{suffix}_3"
                args = [ints, pa.scalar(3), pa.scalar(dtype)]
                output_type = input_type
            elif dtype and itype:
                func_name = f"one_hot_{suffix}_4"
                args = [ints, pa.scalar(3), pa.scalar(dtype), pa.scalar(itype)]
                output_type = pa.int32()

            storage = pc.call_function(func_name, args)

            # Check: storage type is always int32 (registered out_type)
            assert storage.type == output_type

            # Check: storage values match input test_data
            npt.assert_array_equal(storage.to_pylist(), test_data)

        test_cases = [
            # Test each integer type with default params
            (pa.int8(), "int8", None, None),
            (pa.int16(), "int16", None, None),
            (pa.int32(), "int32", None, None),
            (pa.int64(), "int64", None, None),
            (pa.uint8(), "uint8", None, None),
            (pa.uint16(), "uint16", None, None),
            (pa.uint32(), "uint32", None, None),
            (pa.uint64(), "uint64", None, None),
            # Test different dtypes
            (pa.int32(), "int32", "float64", None),
            (pa.int32(), "int32", "int32", None),
            (pa.int32(), "int32", "int64", None),
            (pa.int64(), "int64", "float64", None),
            # # Test explicit itype
            (pa.int32(), "int32", "float32", "int8"),
            (pa.int32(), "int32", "float32", "int16"),
            (pa.int32(), "int32", "float32", "int64"),
            (pa.int64(), "int64", "float32", "int32"),
        ]

        for input_type, suffix, dtype, itype in test_cases:
            test_arity_for_type(input_type, suffix, dtype, itype)
