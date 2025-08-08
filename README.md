# MPLang: Multi-Party Programming Language

MPLang enables writing programs for multi-device execution (PPU/SPU) with explicit security annotations.

## 1. Explicit Security with Device Annotation

MPLang allows you to assign computations to specific devices like "P0", "P1" (parties), and "SP0" (secure computation units).
This approach makes the security model clear directly from the program syntax, abstracting away underlying cryptographic complexities.

### Example: Millionaire's Problem (Device-Annotated)

```python
import random
import mplang.device as mpd

# Define device configurations
device_conf = {
    "P0": {"type": "PPU", "node_ids": ["node:0"]},
    "P1": {"type": "PPU", "node_ids": ["node:1"]},
    "SP0": {"type": "SPU", "node_ids": ["node:0", "node:1"]},
}
# Initialize for simulation mode (no real network)
mpd.init(device_conf, {})

def millionaire():
    # Alice's wealth, computed on P0
    alice = mpd.device("P0")(random.randint)(0, 1000)
    # Bob's wealth, computed on P1
    bob = mpd.device("P1")(random.randint)(0, 1000)

    # Comparison happens securely on SP0
    who_is_richer = mpd.device("SP0")(lambda x, y: x < y)(alice, bob)

    # Result can be moved to a specific party, e.g., P0
    mpd.put("P0", who_is_richer)

# To run:
# result = millionaire()
# print(f"Result (device-annotated): {mpd.fetch(result)}")
```

However, executing Python functions remotely in this manner can introduce system security vulnerabilities and RPC overhead.

## 2. Compilation with `@mpd.function`

To address these concerns, MPLang introduces the `@mpd.function` decorator. Inspired by TensorFlow's `@tf.function`,
this decorator compiles a Python function containing multiple device calls into a Directed Acyclic Graph (DAG).

**Benefits of Compilation:**

- **Enhanced Security:** Restricts the instruction set for the backend, significantly reducing the Remote Code Execution (RCE) attack surface.
- **Performance Optimization:** Opens up opportunities for graph analysis and transformation, leading to potential performance gains.

### Example: Compiling the Millionaire's Problem

The previous `millionaire` function can be compiled by simply adding the `@mpd.function` decorator. The core logic and device calls remain unchanged:

```python

@mpd.function # The key change point
def millionaire():
    alice = mpd.device("P0")(random.randint)(0, 1000)
    bob = mpd.device("P1")(random.randint)(0, 1000)
    # nothing change in the function body.
    ...
```

This compiled graph can then be executed by the MPLang runtime, offering better security and potential performance improvements.

## Installation

### For Users

Install the package directly from source:

```bash
uv pip install .
```

### For Developers

For development, install in editable mode with all development dependencies:

```bash
uv pip install -e .
```

This will install:

- The package itself in editable mode (changes to source code take effect immediately)
- All runtime dependencies
- Development tools: ruff, mypy, pytest, pytest-cov, sphinx, sphinx_rtd_theme

## Development

### Setting up Development Environment

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd mplang
   ```

2. Install development dependencies:

   ```bash
   # using uv (recommended)
   # install uv if not installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   uv venv
   source .venv/bin/activate

   uv pip install -e .
   ```

3. Verify installation:

   ```bash
   python -c 'import mplang; print("Installation successful!")'
   ```

### Running Tests

```bash
uv sync --group dev
# Run tests with pytest
uv run pytest
```

### Code Formatting and Linting

```bash
# install dev dependencies
uv sync --group dev
# Format and lint code (ruff replaces black, isort, and flake8)
uv run ruff check . --fix
uv run ruff format .

# Type checking
uv run mypy mplang/
```

## Getting Started

To learn more and see practical examples, please start by exploring the [tutorials/](tutorials/) directory.

To install `mplang` and run the tutorials:

1. Install `mplang` in editable mode (from the root of this `mplang` repo):

    ```bash
    uv pip install -e .
    ```

2. Run a specific tutorial, for example `0_basic.py`:

    ```bash
    uv run tutorials/0_basic.py
    ```

    You can replace `0_basic.py` with other tutorial files like `1_condition.py`, `2_whileloop.py`, etc.
    (see the [tutorials/run.sh](tutorials/run.sh) script for more examples).
