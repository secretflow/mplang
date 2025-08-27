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

from concurrent.futures import ThreadPoolExecutor

import jax.numpy as jnp
import numpy as np
import spu.api as spu_api
import spu.libspu as libspu

from mplang.backend.spu import SpuHandler, SpuValue
from mplang.core.tensor import TensorType
from mplang.frontend.spu import SpuFE
from mplang.runtime.grpc_comm import LinkCommunicator


def create_mem_link_contexts(world_size: int = 2):
    """Create real SPU MemLink contexts for testing."""
    # Create fake addresses for MemLink (not used in memory mode)
    addrs = [f"P{i}" for i in range(world_size)]

    link_contexts = []
    for rank in range(world_size):
        link_comm = LinkCommunicator(rank, addrs, mem_link=True)
        link_contexts.append(link_comm)

    return link_contexts


def create_spu_runtimes(world_size: int = 2):
    """Create real SPU runtimes for testing."""
    # Create SPU config
    spu_config = libspu.RuntimeConfig()
    spu_config.protocol = libspu.ProtocolKind.SEMI2K
    spu_config.field = libspu.FieldType.FM128

    # Create link contexts
    link_contexts = create_mem_link_contexts(world_size)

    # Create SPU runtimes
    spu_runtimes = []
    for link_ctx in link_contexts:
        spu_rt = spu_api.Runtime(link_ctx, spu_config)
        spu_runtimes.append(spu_rt)

    return spu_runtimes, spu_config


class TestSpuHandler:
    """Test cases for SpuExecutableHandler focusing on core functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create SPU config
        self.spu_config = libspu.RuntimeConfig()
        self.spu_config.protocol = libspu.ProtocolKind.SEMI2K
        self.spu_config.field = libspu.FieldType.FM128

    def test_compilation_produces_valid_executable(self):
        """Test that SpuFrontend.compile produces executables that SpuRT can parse."""

        def add_fn(x, y):
            return x + y

        args = [
            TensorType(shape=(3,), dtype=jnp.float32),
            TensorType(shape=(3,), dtype=jnp.float32),
        ]

        # Compile the function
        is_var = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        cfunc, _, _ = SpuFE(world_size=2).compile_jax(is_var, add_fn, *args)

        # Verify the compiled function format
        assert cfunc.fn_type == "mlir.pphlo"
        assert isinstance(cfunc.fn_text, str)

        # Verify runtime can parse the metadata
        assert "input_visibilities" in cfunc.attrs
        assert "input_names" in cfunc.attrs
        assert "output_names" in cfunc.attrs
        assert "executable_name" in cfunc.attrs

        # Verify we can reconstruct the executable from MLIR code and metadata
        executable = libspu.Executable(
            name=cfunc.attrs["executable_name"],
            input_names=cfunc.attrs["input_names"],
            output_names=cfunc.attrs["output_names"],
            code=cfunc.fn_text,
        )
        assert executable.name == "add_fn"
        assert len(executable.input_names) == 2

    def test_real_spu_execution_single_party(self):
        """Test execution with real SPU runtime in single-party mode.

        Uses REF2K (mock/reference) protocol which supports single-party execution,
        unlike multi-party protocols like SEMI2K which require >= 2 parties.
        """

        def add_fn(x, y):
            return x + y

        args = [
            TensorType(shape=(3,), dtype=jnp.float32),
            TensorType(shape=(3,), dtype=jnp.float32),
        ]

        # Compile the function (this should always work)
        is_var = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        cfunc, _, _ = SpuFE(world_size=1).compile_jax(is_var, add_fn, *args)

        # Create test data
        x_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_data = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        expected = x_data + y_data

        # Test compilation and metadata parsing (these work without runtime)
        assert cfunc.fn_type == "mlir.pphlo"
        assert isinstance(cfunc.fn_text, str)
        assert "input_visibilities" in cfunc.attrs

        # Verify we can reconstruct the executable and it contains the expected names
        executable = libspu.Executable(
            name=cfunc.attrs.get("executable_name", cfunc.fn_name),
            input_names=cfunc.attrs["input_names"],
            output_names=cfunc.attrs["output_names"],
            code=cfunc.fn_text,
        )
        assert len(executable.input_names) == 2

        # For single-party testing, we use REF2K protocol which is a mock/reference
        # implementation that doesn't require multi-party synchronization

        # Create SPU config for single-party (mock) execution
        spu_config = libspu.RuntimeConfig(
            protocol=libspu.ProtocolKind.REF2K,  # Mock protocol
            field=libspu.FieldType.FM128,
        )

        # Create SPU IO for single party
        spu_io = spu_api.Io(world_size=1, config=spu_config)

        # Create secret shares for inputs (REF2K supports single party)
        x_shares = spu_io.make_shares(x_data, libspu.Visibility.VIS_SECRET)
        y_shares = spu_io.make_shares(y_data, libspu.Visibility.VIS_SECRET)

        # Verify that share creation works (should have 1 share each for single party)
        assert len(x_shares) == 1
        assert len(y_shares) == 1

        # Create SpuValue objects
        spu_args = [
            SpuValue(
                shape=x_data.shape,
                dtype=x_data.dtype,
                vtype=libspu.Visibility.VIS_SECRET,
                share=x_shares[0],
            ),
            SpuValue(
                shape=y_data.shape,
                dtype=y_data.dtype,
                vtype=libspu.Visibility.VIS_SECRET,
                share=y_shares[0],
            ),
        ]

        # Verify SpuValue structure
        for spu_arg in spu_args:
            assert isinstance(spu_arg, SpuValue)
            assert hasattr(spu_arg, "shape")
            assert hasattr(spu_arg, "dtype")
            assert hasattr(spu_arg, "vtype")
            assert hasattr(spu_arg, "share")

        # Now attempt runtime creation and execution with single-party setup
        link_contexts = create_mem_link_contexts(world_size=1)
        runtime = SpuHandler(1, spu_config)
        runtime.set_link_context(link_contexts[0])

        # Runtime setup with REF2K should work without synchronization issues
        runtime.setup(0)  # rank 0 for single party

        # Execute the compiled function
        results = runtime.execute(cfunc, spu_args)  # type: ignore[arg-type]

        # Verify results
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SpuValue)
        assert result.shape == expected.shape
        assert result.dtype == np.float32
        assert result.vtype == libspu.Visibility.VIS_SECRET

        # Reconstruct and verify result (single party can reconstruct directly)
        result_shares = [result.share]
        reconstructed = spu_io.reconstruct(result_shares)
        np.testing.assert_allclose(reconstructed, expected, rtol=1e-5)

        print("SPU single-party execution completed successfully!")
        print(f"Expected: {expected}")
        print(f"Got: {reconstructed}")

        runtime.teardown()

    def test_real_spu_execution_multiparty(self):
        """Test execution with real SPU runtime in multi-party mode.

        Uses SEMI2K protocol with proper multi-threaded synchronization to avoid
        deadlock issues that occur when SPU runtimes are initialized sequentially.
        """

        def add_fn(x, y):
            return x + y

        args = [
            TensorType(shape=(3,), dtype=jnp.float32),
            TensorType(shape=(3,), dtype=jnp.float32),
        ]

        # Compile the function
        is_var = lambda obj: hasattr(obj, "dtype") and hasattr(obj, "shape")
        cfunc, _, _ = SpuFE(world_size=2).compile_jax(is_var, add_fn, *args)

        # Create test data
        x_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_data = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        expected = x_data + y_data

        # Create SPU IO for multi-party (SEMI2K requires >= 2 parties)
        spu_io = spu_api.Io(world_size=2, config=self.spu_config)

        # Create secret shares for inputs (will have 2 shares each)
        x_shares = spu_io.make_shares(x_data, libspu.Visibility.VIS_SECRET)
        y_shares = spu_io.make_shares(y_data, libspu.Visibility.VIS_SECRET)

        # Create link contexts once for all parties - CRITICAL: shared between all threads
        world_size = 2
        link_contexts = create_mem_link_contexts(world_size=world_size)

        def party_worker(party_id):
            """Worker function for each party in multi-party computation."""
            # Each party gets their respective share
            spu_args = [
                SpuValue(
                    shape=x_data.shape,
                    dtype=x_data.dtype,
                    vtype=libspu.Visibility.VIS_SECRET,
                    share=x_shares[party_id],
                ),
                SpuValue(
                    shape=y_data.shape,
                    dtype=y_data.dtype,
                    vtype=libspu.Visibility.VIS_SECRET,
                    share=y_shares[party_id],
                ),
            ]

            # Create runtime for this party using shared link contexts
            runtime = SpuHandler(world_size, self.spu_config)
            runtime.set_link_context(link_contexts[party_id])
            runtime.setup(party_id)

            # Execute the compiled function
            results = runtime.execute(cfunc, spu_args)  # type: ignore[arg-type]

            # Verify results
            assert len(results) == 1
            result = results[0]
            assert isinstance(result, SpuValue)

            runtime.teardown()
            return party_id, result.share

        # Execute all party workers simultaneously using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=world_size) as executor:
            futures = [
                executor.submit(party_worker, party_id)
                for party_id in range(world_size)
            ]

            # Wait for all futures to complete and collect results
            party_results = {}
            result_shares = []

            for i, future in enumerate(futures):
                try:
                    party_id, share = future.result(timeout=30)
                    party_results[party_id] = "success"
                    result_shares.append(share)
                except Exception as e:
                    raise AssertionError(f"Party {i} failed with error: {e}") from e

        # Check that all parties succeeded
        success_count = sum(
            1 for status in party_results.values() if status == "success"
        )
        assert success_count == world_size, (
            f"Only {success_count}/{world_size} parties succeeded: {party_results}"
        )

        # All parties succeeded - reconstruct the final result
        spu_io = spu_api.Io(world_size=2, config=self.spu_config)
        reconstructed = spu_io.reconstruct(result_shares)
        np.testing.assert_allclose(reconstructed, expected, rtol=1e-5)

        print("SPU multi-party execution completed successfully!")
        print(f"Expected: {expected}")
        print(f"Got: {reconstructed}")
