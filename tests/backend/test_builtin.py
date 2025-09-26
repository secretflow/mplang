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

from mplang.backend.context import RuntimeContext
from mplang.core.pfunc import PFunction
from mplang.core.tensor import TensorType


class TestBuiltinHandler:
    """Tests for flat builtin kernels (legacy name retained)."""

    def setup_method(self):
        # initialize backend context for rank 0 (world_size=1) once per test
        self.runtime = RuntimeContext(rank=0, world_size=1)

    @staticmethod
    def _exec_static(runtime, pfunc: PFunction, args: list):
        return runtime.run_kernel(pfunc, args)

    def _exec(self, pfunc: PFunction, args: list):  # instance helper
        return self._exec_static(self.runtime, pfunc, args)

    def test_list_fn_names(self):
        """Test that handler lists correct function names."""
        # list_registered_kernels lives in backend.base; import lazily
        from mplang.backend.base import list_kernels

        kernels = list_kernels()
        for expected in [
            "builtin.identity",
            "builtin.read",
            "builtin.write",
            "builtin.constant",
            "builtin.rank",
            "builtin.prand",
            "builtin.table_to_tensor",
            "builtin.tensor_to_table",
            "builtin.debug_print",
        ]:
            assert expected in kernels

    def test_identity(self):
        """Test identity operation."""
        # Create test data
        test_data = np.array([1, 2, 3], dtype=np.float32)

        # Test identity
        identity_pfunc = PFunction(
            fn_type="builtin.identity",
            ins_info=(TensorType.from_obj(test_data),),
            outs_info=(TensorType.from_obj(test_data),),
            fn_name="Identity",
        )
        result = self._exec(identity_pfunc, [test_data])
        assert len(result) == 1
        assert np.array_equal(result[0], test_data)
        assert result[0] is test_data  # Should be the same object

    def test_write_and_read_numpy_array(self):
        """Test writing and reading a numpy array."""
        # Create test data
        test_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Test write
            write_pfunc = PFunction(
                fn_type="builtin.write",
                ins_info=(TensorType.from_obj(test_data),),
                outs_info=(TensorType.from_obj(test_data),),
                fn_name="Write",
                path=tmp_path,
            )

            write_result = self._exec(write_pfunc, [test_data])
            assert len(write_result) == 1
            assert np.array_equal(write_result[0], test_data)

            # Verify file was created
            assert os.path.exists(tmp_path)

            # Test read
            read_pfunc = PFunction(
                fn_type="builtin.read",
                ins_info=(),
                outs_info=(TensorType.from_obj(test_data),),
                fn_name="Read",
                path=tmp_path,
            )

            read_result = self._exec(read_pfunc, [])
            assert len(read_result) == 1
            assert np.array_equal(read_result[0], test_data)
            assert read_result[0].dtype == test_data.dtype

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_write_creates_directory(self):
        """Test that write creates directory if it doesn't exist."""
        test_data = np.array([1, 2, 3], dtype=np.int32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_path = os.path.join(tmp_dir, "nested", "deep", "file.npy")

            write_pfunc = PFunction(
                fn_type="builtin.write",
                ins_info=(TensorType.from_obj(test_data),),
                outs_info=(TensorType.from_obj(test_data),),
                fn_name="Write",
                path=nested_path,
            )

            result = self._exec(write_pfunc, [test_data])
            assert len(result) == 1
            assert np.array_equal(result[0], test_data)
            assert os.path.exists(nested_path)

            # Verify the file content
            loaded = np.load(nested_path)
            assert np.array_equal(loaded, test_data)

    def test_write_different_tensor_types(self):
        """Test writing different types of tensor-like objects."""
        test_cases = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([True, False, True], dtype=bool),
            42,  # scalar
            np.array([1, 2, 3]),  # converted list to numpy array
        ]

        for i, test_data in enumerate(test_cases):
            with tempfile.NamedTemporaryFile(
                suffix=f"_{i}.npy", delete=False
            ) as tmp_file:
                tmp_path = tmp_file.name

            try:
                tensor_info = TensorType.from_obj(test_data)
                write_pfunc = PFunction(
                    fn_type="builtin.write",
                    ins_info=(tensor_info,),
                    outs_info=(tensor_info,),
                    fn_name="Write",
                    path=tmp_path,
                )

                result = self._exec(write_pfunc, [test_data])
                assert len(result) == 1

                # Read back and verify
                read_pfunc = PFunction(
                    fn_type="builtin.read",
                    ins_info=(),
                    outs_info=(tensor_info,),
                    fn_name="Read",
                    path=tmp_path,
                )

                read_result = self._exec(read_pfunc, [])
                assert len(read_result) == 1
                assert np.array_equal(read_result[0], np.array(test_data))

            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    def test_identity_wrong_args(self):
        """Test identity with wrong number of arguments."""
        # Declare expected single input so runtime enforces arity directly.
        identity_pfunc = PFunction(
            fn_type="builtin.identity",
            ins_info=(TensorType.from_obj(np.array(1, dtype=np.int32)),),
            outs_info=(TensorType.from_obj(np.array(1, dtype=np.int32)),),
            fn_name="Identity",
        )
        with pytest.raises(ValueError, match=r"arg count mismatch: got 0, expect 1"):
            self._exec(identity_pfunc, [])
        with pytest.raises(ValueError, match=r"arg count mismatch: got 2, expect 1"):
            self._exec(identity_pfunc, [1, 2])

    def test_read_missing_path(self):
        """Test read operation without path attribute."""
        read_pfunc = PFunction(
            fn_type="builtin.read",
            ins_info=(),
            outs_info=(TensorType.from_obj(np.array([1])),),
            fn_name="Read",
        )
        with pytest.raises(ValueError, match=r"missing path attr for builtin.read"):
            self._exec(read_pfunc, [])

    def test_read_wrong_args(self):
        """Test read with wrong number of arguments."""
        read_pfunc = PFunction(
            fn_type="builtin.read",
            ins_info=(),
            outs_info=(TensorType.from_obj(np.array([1])),),
            fn_name="Read",
            path="dummy.npy",
        )
        with pytest.raises(ValueError, match=r"arg count mismatch: got 1, expect 0"):
            self._exec(read_pfunc, [np.array([1])])

    def test_write_missing_path(self):
        """Test write operation without path attribute."""
        test_data = np.array([1, 2, 3])
        write_pfunc = PFunction(
            fn_type="builtin.write",
            ins_info=(TensorType.from_obj(test_data),),
            outs_info=(TensorType.from_obj(test_data),),
            fn_name="Write",
        )
        with pytest.raises(ValueError, match=r"missing path attr for builtin.write"):
            self._exec(write_pfunc, [test_data])

    def test_write_wrong_args(self):
        """Test write with wrong number of arguments."""
        write_pfunc = PFunction(
            fn_type="builtin.write",
            ins_info=(TensorType.from_obj(np.array(1, dtype=np.int32)),),
            outs_info=(TensorType.from_obj(np.array(1, dtype=np.int32)),),
            fn_name="Write",
            path="dummy.npy",
        )
        with pytest.raises(ValueError, match=r"arg count mismatch: got 0, expect 1"):
            self._exec(write_pfunc, [])
        with pytest.raises(ValueError, match=r"arg count mismatch: got 2, expect 1"):
            self._exec(write_pfunc, [1, 2])

    def test_read_nonexistent_file(self):
        """Test reading from a nonexistent file."""
        read_pfunc = PFunction(
            fn_type="builtin.read",
            ins_info=(),
            outs_info=(TensorType.from_obj(np.array([1])),),
            fn_name="Read",
            path="nonexistent_file.npy",
        )
        with pytest.raises(RuntimeError, match=r"builtin.read failed:"):
            self._exec(read_pfunc, [])

    def test_unsupported_function_type(self):
        """Test unsupported function type."""
        pfunc = PFunction(
            fn_type="unsupported.operation",
            ins_info=(),
            outs_info=(),
            fn_name="Unsupported",
        )

        with pytest.raises(NotImplementedError, match="no backend kernel registered"):
            self._exec(pfunc, [])
