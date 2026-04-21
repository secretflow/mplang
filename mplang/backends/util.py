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

from typing import Any, Protocol

import numpy as np


class TensorLike(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype[Any]: ...


def _refine_stablehlo(code: str, args: tuple[TensorLike, ...]) -> str:
    """Convert dynamic shapes in StableHLO MLIR to static shapes using stablehlo-opt.

    Args:
        code: StableHLO MLIR code string with potentially dynamic shapes
        args: Input tensor values to determine static shapes

    Returns:
        StableHLO MLIR code with static shapes
    """
    import os
    import shutil
    import subprocess
    import tempfile

    # Check if stablehlo-opt exists
    if not shutil.which("stablehlo-opt"):
        raise RuntimeError(
            "stablehlo-opt not found. Please install the stablehlo-opt standalone tool."
        )

    # Build type strings for each input tensor
    type_strings = []
    for arg in args:
        # Get dtype name (e.g., f32, i64)
        dtype_str = (
            str(arg.dtype)
            .replace("float", "f")
            .replace("int", "i")
            .replace("uint", "u")
            .replace("bool", "i1")
        )

        # Build tensor type with static shape
        shape_part = "x".join(str(dim) for dim in arg.shape)
        tensor_type = f"tensor<{shape_part}x{dtype_str}>"
        type_strings.append(tensor_type)

    # Build the refine arguments string
    refine_args = "types=" + ",".join(type_strings)

    # Write the input MLIR to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
        f.write(code)
        input_file = f.name

    try:
        # Run stablehlo-opt with the refine pipeline
        cmd = [
            "stablehlo-opt",
            input_file,
            f"--stablehlo-refine-arguments={refine_args}",
            "--stablehlo-refine-shapes",
            "--stablehlo-canonicalize-dynamism",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Filter out shape_assertion custom calls that cause compilation errors
        filtered_output = "\n".join(
            line
            for line in result.stdout.split("\n")
            if "stablehlo.custom_call @shape_assertion" not in line
        )

        return filtered_output

    except subprocess.CalledProcessError as e:
        # Re-throw with additional information
        raise RuntimeError(
            f"Failed to refine StableHLO shapes. Command: {' '.join(cmd)}. "
            f"Error: {e.stderr}. Return code: {e.returncode}"
        ) from e
    finally:
        # Clean up temporary file
        os.unlink(input_file)
