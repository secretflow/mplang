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

"""Test suite for README.md Python examples."""

import subprocess
import sys
from pathlib import Path

import pytest


class ReadmeExampleTest:
    """Test class to execute and verify Python code blocks from README.md."""

    def __init__(self, readme_path: str = "README.md"):
        self.readme_path = Path(readme_path)
        self.code_blocks = self._extract_python_blocks()

    def _extract_python_blocks(self) -> list[tuple[int, str]]:
        """Extract Python code blocks from README.md with their line numbers."""
        if not self.readme_path.exists():
            raise FileNotFoundError(f"README.md not found at {self.readme_path}")

        content = self.readme_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        python_blocks = []
        in_python_block = False
        block_start_line = 0
        current_block = []

        for i, line in enumerate(lines, 1):
            if line.strip() == "```python":
                in_python_block = True
                block_start_line = i
                current_block = []
            elif line.strip() == "```" and in_python_block:
                in_python_block = False
                if current_block:
                    python_blocks.append((block_start_line, "\n".join(current_block)))
            elif in_python_block:
                current_block.append(line)

        return python_blocks

    @staticmethod
    def _execute_code(code: str) -> tuple[str, str, int]:
        """Execute Python code and return stdout, stderr, and return code."""
        try:
            # Create a temporary Python script
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=Path(__file__).resolve().parent.parent,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Timeout: Code execution took too long", 1
        except Exception as e:
            return "", str(e), 1


# Initialize the test instance
readme_test = ReadmeExampleTest(Path(__file__).resolve().parent.parent / "README.md")


@pytest.mark.parametrize("block_start_line,python_code", readme_test.code_blocks)
def test_readme_python_blocks(block_start_line: int, python_code: str):
    """Test each Python code block from README.md."""
    print(f"\n--- Testing Python block starting at line {block_start_line} ---")
    print("Code to execute:")
    print(python_code)
    print("--- Execution output ---")

    stdout, stderr, returncode = readme_test._execute_code(python_code)

    if stdout:
        print(stdout)
    if stderr:
        print(f"STDERR: {stderr}")
    print(f"Return code: {returncode}")
    print("--- End execution ---")

    # Assert that the code executed successfully
    assert returncode == 0, (
        f"Python block at line {block_start_line} failed with return code {returncode}\nSTDERR: {stderr}"
    )


if __name__ == "__main__":
    # Allow running the tests directly
    pytest.main([__file__, "-v"])
