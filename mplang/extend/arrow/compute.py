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

import pyarrow as pa
import pyarrow.compute as pc

from mplang.extend.arrow import OneHotVectorArray, one_hot_vector
from mplang.extend.arrow.util import _str_to_type


def one_hot(
    array: pa.Array, num_classes: int, *, dtype: str = "float32", itype: str = ""
) -> OneHotVectorArray:
    """Create a one-hot vector array from an integer array.

    Each integer value represents the active position in the one-hot vector.

    Args:
        array: PyArrow array of integer values representing the active position
        num_classes: The size/length of each one-hot vector
        dtype: The data type for the vector values (default: "float32")
        itype: The index type for storing the active position. When empty
            (default), uses the same type as the input array without casting.
            When specified, must be an integer type (e.g., "int32", "int64").

    Returns:
        OneHotVectorArray where each element is a one-hot vector

    Raises:
        TypeError: If the input array is not an integer type, or if itype
            is specified but not an integer type.

    Examples:
        >>> import pyarrow as pa
        >>> from mplang.extend.arrow.compute import one_hot
        >>> # itype empty - keeps input array type
        >>> ints = pa.array([0, 1, 2, 1], type=pa.int32())
        >>> result = one_hot(ints, num_classes=3, dtype="float32")
        >>> # itype specified - uses user-provided type
        >>> result = one_hot(ints, num_classes=3, dtype="float32", itype="int8")
        >>> result.to_pylist()
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]
    """
    if not pa.types.is_integer(array.type):
        raise TypeError(f"Input array must be of integer type, got {array.type}")
    value_type = _str_to_type(dtype)

    # Handle itype: empty means keep array type, otherwise validate integer type
    if itype == "":
        index_type = array.type
    else:
        index_type = _str_to_type(itype)
        if not pa.types.is_integer(index_type):
            raise TypeError(f"itype must be an integer type, got {index_type}")

    vector_type = one_hot_vector(num_classes, value_type, index_type)
    storage = pa.array(array.cast(vector_type.storage_type))
    result: OneHotVectorArray = pa.ExtensionArray.from_storage(vector_type, storage)
    return result


def register_functions() -> None:
    """Register all custom compute functions with PyArrow.

    For each integer type, registers three functions (with same name, different arity):
    - 2 args: one_hot_<type>(array, num_classes) - defaults dtype="float32", itype=""
    - 3 args: one_hot_<type>(array, num_classes, dtype) - default itype=""
    - 4 args: one_hot_<type>(array, num_classes, dtype, itype) - full specification

    Uses *args in the wrapper to handle optional parameters flexibly.
    """
    int_types = [
        (pa.int8(), "int8"),
        (pa.int16(), "int16"),
        (pa.int32(), "int32"),
        (pa.int64(), "int64"),
        (pa.uint8(), "uint8"),
        (pa.uint16(), "uint16"),
        (pa.uint32(), "uint32"),
        (pa.uint64(), "uint64"),
    ]
    for int_type, suffix in int_types:
        _register_one_hot_arity(int_type, f"one_hot_{suffix}", 2)
        _register_one_hot_arity(int_type, f"one_hot_{suffix}_3", 3)
        _register_one_hot_arity(int_type, f"one_hot_{suffix}_4", 4)


def _register_one_hot_arity(
    input_type: pa.DataType, function_name: str, arity: int
) -> None:
    """Register a one_hot function for a specific integer type and parameter arity.

    Args:
        input_type: The input integer type (e.g., pa.int32())
        function_name: The name for the registered function (e.g., "one_hot_int32")
        arity: Number of parameters supported (2, 3, or 4)
    """
    if arity == 2:
        out_type = input_type
        in_types = {"array": input_type, "num_classes": pa.int64()}
        param_desc = "array, num_classes (defaults: dtype='float32', itype='')"

        def wrapper(_ctx: Any, *args: object) -> pa.Array:
            """Wrapper for one_hot with 2 params."""
            array = args[0]  # type: ignore[index]
            num_classes = args[1].as_py()  # type: ignore[attr-defined, index]
            result = one_hot(array, num_classes, dtype="float32", itype="")
            return result.storage

    elif arity == 3:
        out_type = input_type
        in_types = {
            "array": input_type,
            "num_classes": pa.int64(),
            "dtype": pa.string(),
        }
        param_desc = "array, num_classes, dtype (default: itype='')"

        def wrapper(_ctx: Any, *args: object) -> pa.Array:
            """Wrapper for one_hot with 3 params."""
            array = args[0]  # type: ignore[index]
            num_classes = args[1].as_py()  # type: ignore[attr-defined, index]
            dtype_val = args[2].as_py()  # type: ignore[attr-defined, index]
            result = one_hot(array, num_classes, dtype=dtype_val, itype="")
            return result.storage

    elif arity == 4:
        out_type = pa.int32()
        in_types = {
            "array": input_type,
            "num_classes": pa.int64(),
            "dtype": pa.string(),
            "itype": pa.string(),
        }
        param_desc = "array, num_classes, dtype, itype"

        def wrapper(_ctx: Any, *args: object) -> pa.Array:
            """Wrapper for one_hot with 4 params."""
            array = args[0]  # type: ignore[index]
            num_classes = args[1].as_py()  # type: ignore[attr-defined, index]
            dtype_val = args[2].as_py()  # type: ignore[attr-defined, index]
            itype_val = args[3].as_py()  # type: ignore[attr-defined, index]
            result = one_hot(array, num_classes, dtype=dtype_val, itype=itype_val)
            return result.storage.cast(out_type)

    else:
        raise ValueError(f"Invalid arity: {arity}, must be 2, 3, or 4")

    func_doc = {
        "summary": f"One-hot encoding for {input_type}",
        "description": f"Convert {input_type} values to one-hot vectors. "
        f"Signature: function({param_desc}). Parameters:\n"
        f"  - array: Input integer array of type {input_type}\n"
        "  - num_classes: Number of classes (vector dimension)\n"
        "  - dtype: Value type for vector elements (default: 'float32')\n"
        "  - itype: Index type for storing active position (default: '' keeps input array type)",
    }

    try:
        pc.register_scalar_function(
            func=wrapper,
            function_name=function_name,
            function_doc=func_doc,
            in_types=in_types,
            out_type=out_type,
        )
    except Exception:
        pass  # Function might already be registered, skip silently
