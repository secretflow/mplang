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

import os
import tempfile

import numpy as np
import pytest

from mplang.core.base import TensorInfo
from mplang.core.pfunc import PFunction
from mplang.plib.stdio_handler import StdioHandler


class TestStdioHandler:
    """Test cases for StdioHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = StdioHandler()

    def test_list_fn_names(self):
        """Test that handler lists correct function names."""
        fn_names = self.handler.list_fn_names()
        assert "Read" in fn_names
        assert "Write" in fn_names
        assert len(fn_names) == 2

    def test_write_and_read_numpy_array(self):
        """Test writing and reading a numpy array."""
        # Create test data
        test_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Test write
            write_pfunc = PFunction(
                fn_name="Write",
                fn_type="Write",
                fn_text="",
                ins_info=(TensorInfo.from_obj(test_data),),
                outs_info=(),
                attrs={"path": tmp_path},
            )

            result = self.handler.execute(write_pfunc, [test_data])
            assert result == [test_data]

            # Test read
            read_pfunc = PFunction(
                fn_name="Read",
                fn_type="Read",
                fn_text="",
                ins_info=(),
                outs_info=(TensorInfo.from_obj(test_data),),
                attrs={"path": tmp_path},
            )

            result = self.handler.execute(read_pfunc, [])
            assert len(result) == 1

            loaded_data = result[0]
            assert isinstance(loaded_data, np.ndarray)
            np.testing.assert_array_equal(test_data, loaded_data)
            assert loaded_data.dtype == test_data.dtype

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_write_different_dtypes(self):
        """Test writing arrays with different data types."""
        test_cases = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([1.5, 2.5, 3.5], dtype=np.float64),
            np.array([True, False, True], dtype=np.bool_),
            np.array([1 + 2j, 3 + 4j], dtype=np.complex64),
        ]

        for test_data in test_cases:
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                # Write
                write_pfunc = PFunction(
                    fn_name="Write",
                    fn_type="Write",
                    fn_text="",
                    ins_info=(TensorInfo.from_obj(test_data),),
                    outs_info=(),
                    attrs={"path": tmp_path},
                )
                result = self.handler.execute(write_pfunc, [test_data])
                assert result == [test_data]

                # Read
                read_pfunc = PFunction(
                    fn_name="Read",
                    fn_type="Read",
                    fn_text="",
                    ins_info=(),
                    outs_info=(TensorInfo.from_obj(test_data),),
                    attrs={"path": tmp_path},
                )
                result = self.handler.execute(read_pfunc, [])
                loaded_data = result[0]

                # Cast to numpy array for comparison to handle TensorLike protocol
                np.testing.assert_array_equal(
                    np.asarray(test_data), np.asarray(loaded_data)
                )
                assert loaded_data.dtype == test_data.dtype

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    def test_read_missing_path(self):
        """Test read operation with missing path attribute."""
        read_pfunc = PFunction(
            fn_name="Read",
            fn_type="Read",
            fn_text="",
            ins_info=(),
            outs_info=(),
            attrs={},  # No path
        )

        with pytest.raises(ValueError, match="Read function requires 'path' attribute"):
            self.handler.execute(read_pfunc, [])

    def test_write_missing_path(self):
        """Test write operation with missing path attribute."""
        test_data = np.array([1, 2, 3])
        write_pfunc = PFunction(
            fn_name="Write",
            fn_type="Write",
            fn_text="",
            ins_info=(TensorInfo.from_obj(test_data),),
            outs_info=(),
            attrs={},  # No path
        )

        with pytest.raises(
            ValueError, match="Write function requires 'path' attribute"
        ):
            self.handler.execute(write_pfunc, [test_data])

    def test_read_wrong_number_of_args(self):
        """Test read operation with wrong number of arguments."""
        read_pfunc = PFunction(
            fn_name="Read",
            fn_type="Read",
            fn_text="",
            ins_info=(),
            outs_info=(),
            attrs={"path": "dummy.npy"},
        )

        with pytest.raises(ValueError, match="Read expects no arguments"):
            self.handler.execute(read_pfunc, [np.array([1, 2, 3])])

    def test_write_wrong_number_of_args(self):
        """Test write operation with wrong number of arguments."""
        write_pfunc = PFunction(
            fn_name="Write",
            fn_type="Write",
            fn_text="",
            ins_info=(),
            outs_info=(),
            attrs={"path": "dummy.npy"},
        )

        with pytest.raises(ValueError, match="Write expects exactly one argument"):
            self.handler.execute(write_pfunc, [])

        with pytest.raises(ValueError, match="Write expects exactly one argument"):
            self.handler.execute(write_pfunc, [np.array([1]), np.array([2])])

    def test_read_nonexistent_file(self):
        """Test reading from a non-existent file."""
        read_pfunc = PFunction(
            fn_name="Read",
            fn_type="Read",
            fn_text="",
            ins_info=(),
            outs_info=(),
            attrs={"path": "/nonexistent/path/file.npy"},
        )

        with pytest.raises(RuntimeError, match="Failed to read from"):
            self.handler.execute(read_pfunc, [])

    def test_unsupported_function_type(self):
        """Test unsupported function type."""
        pfunc = PFunction(
            fn_name="Unknown",
            fn_type="Unknown",
            fn_text="",
            ins_info=(),
            outs_info=(),
            attrs={},
        )

        with pytest.raises(ValueError, match="Unsupported function type: Unknown"):
            self.handler.execute(pfunc, [])

    def test_write_scalar_value(self):
        """Test writing scalar values."""
        test_data = np.array(42.0)  # Convert to numpy array

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Write scalar
            write_pfunc = PFunction(
                fn_name="Write",
                fn_type="Write",
                fn_text="",
                ins_info=(TensorInfo.from_obj(test_data),),
                outs_info=(),
                attrs={"path": tmp_path},
            )
            result = self.handler.execute(write_pfunc, [test_data])
            assert result == [test_data]

            # Read back
            read_pfunc = PFunction(
                fn_name="Read",
                fn_type="Read",
                fn_text="",
                ins_info=(),
                outs_info=(TensorInfo.from_obj(test_data),),
                attrs={"path": tmp_path},
            )
            result = self.handler.execute(read_pfunc, [])
            loaded_data = result[0]

            # Scalar values are loaded as 0-d arrays
            assert np.isscalar(loaded_data) or loaded_data.shape == ()
            np.testing.assert_equal(test_data, loaded_data)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
