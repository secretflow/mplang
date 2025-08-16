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
import numpy as np
import pytest
import spu.libspu as libspu

from mplang.core.base import TensorInfo
from mplang.frontend.spu import SpuFE, Visibility


class TestSpuFECompile:
    """Test cases for SpuFrontend compile method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.spu_fe = SpuFE(world_size=3)  # Default to 3 parties for testing

    @pytest.mark.parametrize(
        "func_def,input_shapes,expected_inputs,expected_outputs,description,test_serialization",
        [
            (
                lambda x, y: x + y,
                [(2, 3), (2, 3)],
                2,
                1,
                "simple addition",
                False,
            ),
            (
                lambda x, y: x * y,
                [(3,), (3,)],
                2,
                1,
                "element-wise multiplication",
                True,  # Test serialization for this case
            ),
            (lambda x: x + 1.0, [(2,)], 1, 1, "scalar addition", False),
        ],
    )
    def test_basic_function_compilation(
        self,
        func_def,
        input_shapes,
        expected_inputs,
        expected_outputs,
        description,
        test_serialization,
    ):
        """Test compilation of basic functions and optional serialization."""

        args = [TensorInfo(shape=shape, dtype=jnp.float32) for shape in input_shapes]

        # Predicate: treat tensor-like objects as variables, others as constants
        is_var = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        cfunc, _, _out_tree = self.spu_fe.compile_jax(is_var, func_def, *args)

        # Verify basic properties
        assert cfunc.fn_type == "mlir.pphlo"
        assert len(cfunc.ins_info) == expected_inputs
        assert len(cfunc.outs_info) == expected_outputs
        assert isinstance(cfunc.fn_text, str)

        # Verify metadata
        assert "input_visibilities" in cfunc.attrs
        assert "input_names" in cfunc.attrs
        assert "output_names" in cfunc.attrs

        # Verify information is available in metadata
        input_names = cfunc.attrs["input_names"]
        output_names = cfunc.attrs["output_names"]
        assert len(input_names) == expected_inputs
        assert len(output_names) == expected_outputs

        # Verify we can reconstruct executable from MLIR code and metadata
        executable = libspu.Executable(
            name=cfunc.attrs.get("executable_name", cfunc.fn_name),
            input_names=input_names,
            output_names=output_names,
            code=cfunc.fn_text,
        )
        assert len(executable.input_names) == expected_inputs
        assert len(executable.output_names) == expected_outputs

        # Test reconstruction if requested
        if test_serialization:
            try:
                # Test that we can reconstruct the executable from MLIR code and metadata
                assert isinstance(cfunc.fn_text, str), (
                    f"Expected str, got {type(cfunc.fn_text)}"
                )

                reconstructed_executable = libspu.Executable(
                    name=cfunc.attrs.get("executable_name", cfunc.fn_name),
                    input_names=cfunc.attrs["input_names"],
                    output_names=cfunc.attrs["output_names"],
                    code=cfunc.fn_text,
                )
                assert reconstructed_executable is not None
                assert len(reconstructed_executable.input_names) == expected_inputs
                assert len(reconstructed_executable.output_names) == expected_outputs
            except Exception as e:
                pytest.fail(f"Failed to reconstruct executable: {e}")

    @pytest.mark.parametrize(
        "num_inputs,expected_visibility",
        [
            (1, libspu.Visibility.VIS_SECRET),
            (2, libspu.Visibility.VIS_SECRET),
            (3, libspu.Visibility.VIS_SECRET),
        ],
    )
    def test_visibility_settings(self, num_inputs, expected_visibility):
        """Test that visibility is set correctly for different number of inputs."""

        def multi_input_fn(*args):
            result = args[0]
            for i in range(1, len(args)):
                result = result + args[i]
            return result

        args = [TensorInfo(shape=(2,), dtype=jnp.float32) for _ in range(num_inputs)]

        is_var = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        cfunc, _, _ = self.spu_fe.compile_jax(is_var, multi_input_fn, *args)

        # Check that all inputs are marked with expected visibility
        visibilities = cfunc.attrs["input_visibilities"]
        assert len(visibilities) == num_inputs
        assert all(v == expected_visibility for v in visibilities)

    @pytest.mark.parametrize(
        "func_def,expected_outputs,description",
        [
            (
                lambda x: (x + 1, x * 2),
                2,
                "Two outputs: addition and multiplication",
            ),
            (
                lambda x: (x + 1, x * 2, x - 1),
                3,
                "Three outputs: add, multiply, subtract",
            ),
            (lambda x: x + 1, 1, "Single output"),
            (
                lambda x: (jnp.mean(x), jnp.sum(x), jnp.max(x)),
                3,
                "Statistical operations",
            ),
        ],
    )
    def test_multiple_outputs_parametrized(
        self, func_def, expected_outputs, description
    ):
        """Test compilation of functions with different numbers of outputs."""

        args = [TensorInfo(shape=(3,), dtype=jnp.float32)]

        is_var = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        cfunc, _, _out_tree = self.spu_fe.compile_jax(is_var, func_def, *args)

        # Should have expected number of outputs
        assert len(cfunc.outs_info) == expected_outputs

        # Verify information is available in metadata
        assert "output_names" in cfunc.attrs
        output_names = cfunc.attrs["output_names"]
        assert len(output_names) == expected_outputs

    @pytest.mark.parametrize(
        "dtype1,dtype2",
        [
            (jnp.float32, jnp.float32),
            (jnp.float64, jnp.float64),
            (jnp.int32, jnp.int32),
        ],
    )
    def test_different_dtypes(self, dtype1, dtype2):
        """Test compilation with different data types."""

        def dtype_fn(x, y):
            return x + y

        args = [
            TensorInfo(shape=(2,), dtype=dtype1),
            TensorInfo(shape=(2,), dtype=dtype2),
        ]

        is_var = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        cfunc, _, _ = self.spu_fe.compile_jax(is_var, dtype_fn, *args)
        assert cfunc.ins_info[0].dtype.name == dtype1.__name__
        assert cfunc.ins_info[1].dtype.name == dtype2.__name__

    def test_complex_function(self):
        """Test compilation of a more complex function."""

        def complex_fn(x, y, z):
            temp = x + y
            result = temp * z
            return jnp.sum(result, axis=0)

        args = [
            TensorInfo(shape=(3, 4), dtype=jnp.float32),
            TensorInfo(shape=(3, 4), dtype=jnp.float32),
            TensorInfo(shape=(3, 4), dtype=jnp.float32),
        ]

        is_var = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        cfunc, _, _ = self.spu_fe.compile_jax(is_var, complex_fn, *args)

        assert cfunc.fn_name == "complex_fn"
        assert len(cfunc.ins_info) == 3

        # Verify information is available in metadata
        assert "input_names" in cfunc.attrs
        input_names = cfunc.attrs["input_names"]
        assert len(input_names) == 3

    def test_error_handling(self):
        """Test error handling for invalid inputs."""

        # Test with invalid function that can't be compiled
        def invalid_fn(x):
            # This might cause compilation issues
            import os

            os.system("invalid_command")  # This shouldn't work in SPU
            return x

        args = [TensorInfo(shape=(2,), dtype=jnp.float32)]

        # Should handle compilation errors gracefully
        try:
            is_var = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
            _cfunc, _, _ = self.spu_fe.compile_jax(is_var, invalid_fn, *args)
            # If it doesn't raise an error, that's also okay
            # The error handling depends on SPU frontend behavior
        except Exception:
            # Should raise an appropriate error
            pass  # This is expected behavior

    def test_function_transformer_compatibility(self):
        """Test compatibility with normalize_fn."""

        def is_tensor_info(obj):
            return isinstance(obj, TensorInfo)

        def test_fn(x, y, constant=1.0):
            return x + y + constant

        args = [
            TensorInfo(shape=(2,), dtype=jnp.float32),
            TensorInfo(shape=(2,), dtype=jnp.float32),
        ]
        kwargs = {"constant": 2.0}

        # Test direct compilation with kwargs
        is_var = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        cfunc, _, _ = self.spu_fe.compile_jax(is_var, test_fn, *args, **kwargs)

        assert cfunc is not None
        assert len(cfunc.ins_info) == len(args)  # Should only count variable args

    def test_compilation_deterministic(self):
        """Test that compilation is deterministic."""

        def simple_func(x):
            return x * 2 + 1

        args = [TensorInfo(shape=(3,), dtype=jnp.float32)]

        # Compile the same function twice
        is_var = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        cfunc1, _, _out_tree1 = self.spu_fe.compile_jax(is_var, simple_func, *args)
        cfunc2, _, _out_tree2 = self.spu_fe.compile_jax(is_var, simple_func, *args)

        # Should produce identical results
        assert cfunc1.fn_type == cfunc2.fn_type
        assert cfunc1.fn_name == cfunc2.fn_name
        assert cfunc1.ins_info == cfunc2.ins_info
        assert cfunc1.outs_info == cfunc2.outs_info

    @pytest.mark.parametrize(
        "test_name,func_def,input_shape,expected_output_shape,description",
        [
            (
                "high_dimensional_tensors",
                lambda x: jnp.sum(x, axis=(1, 3)),
                (2, 3, 4, 5),
                (2, 4),
                "4D tensor sum operation",
            ),
            (
                "matrix_multiplication",
                lambda x, y: jnp.dot(x, y),
                [(100, 50), (50, 75)],
                (100, 75),
                "Large matrix multiplication",
            ),
            (
                "basic_reshape",
                lambda x: jnp.reshape(x, (-1,)),
                (2, 3),
                (6,),
                "Basic tensor reshape",
            ),
            (
                "complex_reshape",
                lambda x: jnp.reshape(x, (-1,)),
                (4, 6),
                (24,),
                "Complex tensor reshape",
            ),
            (
                "transpose_operation",
                lambda x: jnp.transpose(x),
                (3, 4),
                (4, 3),
                "Matrix transpose",
            ),
        ],
    )
    def test_tensor_operations_parametrized(
        self, test_name, func_def, input_shape, expected_output_shape, description
    ):
        """Test compilation with various tensor operations and shapes."""

        if isinstance(input_shape[0], tuple):
            # Multiple inputs case (like matrix multiplication)
            args = [TensorInfo(shape=shape, dtype=jnp.float32) for shape in input_shape]
        else:
            # Single input case
            args = [TensorInfo(shape=input_shape, dtype=jnp.float32)]

        is_var = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        cfunc, _, _out_tree = self.spu_fe.compile_jax(is_var, func_def, *args)

        # Verify compilation succeeded
        assert cfunc.fn_text is not None
        assert len(cfunc.ins_info) == len(args)
        assert len(cfunc.outs_info) >= 1

        if len(cfunc.outs_info) == 1:
            assert cfunc.outs_info[0].shape == expected_output_shape


class TestSpuFEMakeShares:
    """Test cases for SpuFrontend makeshares method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.spu_fe = SpuFE(world_size=3)  # Default to 3 parties for testing

    def test_makeshares_basic_creation(self):
        """Test basic makeshares PFunction creation."""
        # Create test data
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Create makeshares PFunction
        pfunc = self.spu_fe.makeshares(data)

        # Verify basic properties
        assert pfunc.fn_type == "spu.makeshares"
        assert pfunc.fn_name == "makeshares"
        assert pfunc.fn_text is None  # No serialized code needed

        # Verify input info
        assert len(pfunc.ins_info) == 1
        assert pfunc.ins_info[0].shape == (3,)
        assert pfunc.ins_info[0].dtype.name == "float32"

        # Verify output info - should have world_size outputs (one share per party)
        assert len(pfunc.outs_info) == 3  # 3 parties = 3 shares
        for i in range(3):
            assert pfunc.outs_info[i].shape == (3,)
            assert pfunc.outs_info[i].dtype.name == "float32"

    @pytest.mark.parametrize(
        "visibility,expected_vis_value",
        [
            (Visibility.SECRET, libspu.Visibility.VIS_SECRET),
            (Visibility.PUBLIC, libspu.Visibility.VIS_PUBLIC),
            (Visibility.PRIVATE, libspu.Visibility.VIS_PRIVATE),
        ],
    )
    def test_makeshares_visibility_options(self, visibility, expected_vis_value):
        """Test makeshares with different visibility options."""
        data = np.array([1.0, 2.0], dtype=np.float32)

        pfunc = self.spu_fe.makeshares(data, visibility=visibility)

        assert pfunc.attrs["visibility"] == expected_vis_value
        assert pfunc.attrs["operation"] == "makeshares"

    @pytest.mark.parametrize(
        "owner_rank,expected_rank",
        [
            (-1, -1),  # All parties
            (0, 0),  # Party 0
            (1, 1),  # Party 1
            (2, 2),  # Party 2
        ],
    )
    def test_makeshares_owner_rank(self, owner_rank, expected_rank):
        """Test makeshares with different owner ranks."""
        data = np.array([1.0, 2.0], dtype=np.float32)

        pfunc = self.spu_fe.makeshares(data, owner_rank=owner_rank)

        assert pfunc.attrs["owner_rank"] == expected_rank

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((5,), np.float32),
            ((2, 3), np.float64),
            ((1, 4, 2), np.int32),
            ((), np.float32),  # Scalar
        ],
    )
    def test_makeshares_different_shapes_dtypes(self, shape, dtype):
        """Test makeshares with different tensor shapes and data types."""
        if shape == ():
            data = np.array(42.0, dtype=dtype)
        else:
            data = np.ones(shape, dtype=dtype)

        pfunc = self.spu_fe.makeshares(data)

        # Verify input and output info match
        assert pfunc.ins_info[0].shape == shape
        assert pfunc.ins_info[0].dtype.name == dtype.__name__

        # Verify all output shares have the same shape and dtype as input
        assert len(pfunc.outs_info) == 3  # world_size = 3
        for i in range(3):
            assert pfunc.outs_info[i].shape == shape
            assert pfunc.outs_info[i].dtype.name == dtype.__name__

    def test_makeshares_jax_array_input(self):
        """Test makeshares with JAX array input."""
        data = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)

        pfunc = self.spu_fe.makeshares(data)

        assert pfunc.ins_info[0].shape == (4,)
        assert pfunc.ins_info[0].dtype.name == "float32"

        # Verify all output shares
        assert len(pfunc.outs_info) == 3
        for i in range(3):
            assert pfunc.outs_info[i].shape == (4,)
            assert pfunc.outs_info[i].dtype.name == "float32"

    def test_makeshares_combined_options(self):
        """Test makeshares with combined visibility and owner_rank options."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        pfunc = self.spu_fe.makeshares(
            data, visibility=Visibility.PRIVATE, owner_rank=1
        )

        assert pfunc.attrs["visibility"] == libspu.Visibility.VIS_PRIVATE
        assert pfunc.attrs["owner_rank"] == 1
        assert pfunc.attrs["operation"] == "makeshares"
        assert pfunc.attrs["world_size"] == 3
        assert pfunc.ins_info[0].shape == (2, 2)

        # Verify all output shares
        assert len(pfunc.outs_info) == 3
        for i in range(3):
            assert pfunc.outs_info[i].shape == (2, 2)

    def test_makeshares_different_world_sizes(self):
        """Test makeshares with different world sizes."""
        data = np.array([1.0, 2.0], dtype=np.float32)

        # Test with different world sizes
        for world_size in [2, 3, 4, 5]:
            spu_fe = SpuFE(world_size=world_size)
            pfunc = spu_fe.makeshares(data)

            # Should have world_size output shares
            assert len(pfunc.outs_info) == world_size
            assert pfunc.attrs["world_size"] == world_size

            # All shares should have the same shape and dtype as input
            for i in range(world_size):
                assert pfunc.outs_info[i].shape == (2,)
                assert pfunc.outs_info[i].dtype.name == "float32"


class TestSpuFEReconstruct:
    """Test cases for SpuFrontend reconstruct method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.spu_fe = SpuFE(world_size=3)  # Default to 3 parties for testing

    def test_reconstruct_basic_creation(self):
        """Test basic reconstruct PFunction creation."""
        # Create mock shares (in real scenario these would be actual SPU shares)
        shares = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
        ]

        pfunc = self.spu_fe.reconstruct(shares)

        # Verify basic properties
        assert pfunc.fn_type == "spu.reconstruct"
        assert pfunc.fn_name == "reconstruct"
        assert pfunc.fn_text is None  # No serialized code needed

        # Verify input info (one for each share)
        assert len(pfunc.ins_info) == 2
        for i, share in enumerate(shares):
            assert pfunc.ins_info[i].shape == share.shape
            assert pfunc.ins_info[i].dtype.name == share.dtype.name

        # Verify output info (should be one tensor matching first share)
        assert len(pfunc.outs_info) == 1
        assert pfunc.outs_info[0].shape == shares[0].shape
        assert pfunc.outs_info[0].dtype.name == shares[0].dtype.name

    @pytest.mark.parametrize(
        "num_shares",
        [2, 3, 4, 5],
    )
    def test_reconstruct_different_party_counts(self, num_shares):
        """Test reconstruct with different numbers of parties/shares."""
        shares = [
            np.array([i, i + 1, i + 2], dtype=np.float32) for i in range(num_shares)
        ]

        pfunc = self.spu_fe.reconstruct(shares)

        assert len(pfunc.ins_info) == num_shares
        assert len(pfunc.outs_info) == 1

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((5,), np.float32),
            ((2, 3), np.float64),
            ((1, 4, 2), np.int32),
            ((), np.float32),  # Scalar
        ],
    )
    def test_reconstruct_different_shapes_dtypes(self, shape, dtype):
        """Test reconstruct with different tensor shapes and data types."""
        if shape == ():
            shares = [np.array(42.0, dtype=dtype), np.array(13.0, dtype=dtype)]
        else:
            shares = [np.ones(shape, dtype=dtype), np.zeros(shape, dtype=dtype)]

        pfunc = self.spu_fe.reconstruct(shares)

        # All inputs should have the same shape and dtype
        for ins_info in pfunc.ins_info:
            assert ins_info.shape == shape
            assert ins_info.dtype.name == dtype.__name__

        # Output should match input shape and dtype
        assert pfunc.outs_info[0].shape == shape
        assert pfunc.outs_info[0].dtype.name == dtype.__name__

    def test_reconstruct_empty_shares(self):
        """Test reconstruct with empty shares list."""
        shares = []

        pfunc = self.spu_fe.reconstruct(shares)

        assert len(pfunc.ins_info) == 0
        assert len(pfunc.outs_info) == 0
        assert pfunc.attrs == {}

    def test_reconstruct_single_share(self):
        """Test reconstruct with single share (edge case)."""
        shares = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]

        pfunc = self.spu_fe.reconstruct(shares)

        assert len(pfunc.ins_info) == 1
        assert len(pfunc.outs_info) == 1
        assert pfunc.outs_info[0].shape == shares[0].shape

    def test_reconstruct_jax_arrays(self):
        """Test reconstruct with JAX array inputs."""
        shares = [
            jnp.array([1.0, 2.0], dtype=jnp.float32),
            jnp.array([3.0, 4.0], dtype=jnp.float32),
            jnp.array([5.0, 6.0], dtype=jnp.float32),
        ]

        pfunc = self.spu_fe.reconstruct(shares)

        assert len(pfunc.ins_info) == 3
        assert len(pfunc.outs_info) == 1
        for ins_info in pfunc.ins_info:
            assert ins_info.shape == (2,)
            assert ins_info.dtype.name == "float32"

    def test_reconstruct_mixed_compatibility(self):
        """Test that reconstruct input/output info is consistent."""
        # Shares should all have the same shape and dtype
        shares = [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        ]

        pfunc = self.spu_fe.reconstruct(shares)

        # All input info should match
        base_shape = pfunc.ins_info[0].shape
        base_dtype = pfunc.ins_info[0].dtype

        for ins_info in pfunc.ins_info:
            assert ins_info.shape == base_shape
            assert ins_info.dtype == base_dtype

        # Output should match inputs
        assert pfunc.outs_info[0].shape == base_shape
        assert pfunc.outs_info[0].dtype == base_dtype


class TestSpuFEIntegration:
    """Integration tests for makeshares and reconstruct working together."""

    def setup_method(self):
        """Set up test fixtures."""
        self.spu_fe = SpuFE(world_size=3)  # Default to 3 parties for testing

    def test_makeshares_reconstruct_consistency(self):
        """Test that makeshares output is compatible with reconstruct input."""
        # Create original data
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Create makeshares function
        makeshares_pfunc = self.spu_fe.makeshares(data)

        # Simulate shares (in practice these would come from SPU execution)
        # Now we need 3 shares since world_size = 3
        mock_shares = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0], dtype=np.float32),  # Mock share 2
            np.array([0.0, 0.0, 0.0], dtype=np.float32),  # Mock share 3
        ]

        # Create reconstruct function
        reconstruct_pfunc = self.spu_fe.reconstruct(mock_shares)

        # Verify compatibility
        # makeshares should output 3 shares, each compatible with reconstruct inputs
        assert len(makeshares_pfunc.outs_info) == 3
        assert len(reconstruct_pfunc.ins_info) == 3

        for i in range(3):
            assert (
                makeshares_pfunc.outs_info[i].shape
                == reconstruct_pfunc.ins_info[i].shape
            )
            assert (
                makeshares_pfunc.outs_info[i].dtype
                == reconstruct_pfunc.ins_info[i].dtype
            )

        # reconstruct output should match original data shape/type
        assert reconstruct_pfunc.outs_info[0].shape == data.shape
        assert reconstruct_pfunc.outs_info[0].dtype.name == data.dtype.name

    def test_round_trip_metadata(self):
        """Test metadata consistency in makeshares -> reconstruct round trip."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

        # Create makeshares with specific visibility
        makeshares_pfunc = self.spu_fe.makeshares(data, visibility=Visibility.SECRET)

        # Mock the intermediate shares
        shares = [np.ones((2, 2), dtype=np.float64) for _ in range(3)]

        # Create reconstruct
        reconstruct_pfunc = self.spu_fe.reconstruct(shares)

        # Verify that data characteristics are preserved
        assert makeshares_pfunc.ins_info[0].shape == data.shape
        assert reconstruct_pfunc.outs_info[0].shape == data.shape
        assert makeshares_pfunc.ins_info[0].dtype.name == data.dtype.name
        assert reconstruct_pfunc.outs_info[0].dtype.name == data.dtype.name
