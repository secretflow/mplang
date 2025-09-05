# MPLang: A Programming Language for Multi-Party Computation

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/secretflow/mplang/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/secretflow/mplang/tree/main)
[![Lint](https://github.com/secretflow/mplang/actions/workflows/lint.yml/badge.svg)](https://github.com/secretflow/mplang/actions/workflows/lint.yml)
[![Mypy](https://github.com/secretflow/mplang/actions/workflows/mypy.yml/badge.svg)](https://github.com/secretflow/mplang/actions/workflows/mypy.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

MPLang is a Python-native library for building and executing multi-party and multi-device programs.
It simplifies secure computation by allowing developers to write a single program that orchestrates
multiple parties in a synchronous, SPMD (Single Program, Multiple Data) fashion.

## Features

- **Single-Controller SPMD**: Write one program that runs across multiple parties in lockstep.
- **Explicit Device Placement**: Clearly annotate and control where data lives and computation happens (e.g., on party `P0`, `P1`, or a secure `SPU`).
- **Function-Level Compilation**: Use the `@mplang.function` decorator to compile Python functions into an auditable, optimizable graph representation.
- **Pluggable Architecture**: Easily extend MPLang with new frontends (like JAX, Ibis) and backends (like StableHLO, SPU).

## Getting Started

### Installation

You'll need a modern Python environment (3.10+). We recommend using `uv` for fast installation.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install MPLang from PyPI
uv pip install mplang
```

### Quick Example

Here's a taste of what MPLang looks like. This example shows a "millionaire's problem" where two parties compare their wealth without revealing it.

```python
import mplang
import mplang.device as mpd
from numpy.random import randint

# Use a decorator to compile this function for multi-party execution
@mplang.function
def millionaire():
    # Alice's value, placed on device P0
    x = mpd.device("P0")(randint)(0, 1000000)
    # Bob's value, placed on device P1
    y = mpd.device("P1")(randint)(0, 1000000)
    # The comparison happens on a secure device (SPU)
    z = mpd.device("SPU")(lambda a, b: a < b)(x, y)
    return z

# Set up a local simulator with 2 parties
sim = mplang.Simulator(2)

# Evaluate the compiled function
result = mplang.eval(sim, millionaire)

# Securely fetch the result
print("Is Alice poorer than Bob?", mplang.fetch(sim, result))
```

## Learn More

- **Tutorials**: Check out the `tutorials/` directory for in-depth, runnable examples covering conditions, loops, and more.
- **Contributing**: We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) to get started with the development setup.

## License

MPLang is licensed under the Apache 2.0 License.
