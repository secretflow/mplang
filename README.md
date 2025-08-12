# MPLang: Multi-Party Programming Language

[![Lint](https://github.com/secretflow/mplang/actions/workflows/lint.yml/badge.svg)](https://github.com/secretflow/mplang/actions/workflows/lint.yml)
[![MyPy](https://github.com/secretflow/mplang/actions/workflows/mypy.yml/badge.svg)](https://github.com/secretflow/mplang/actions/workflows/mypy.yml)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/secretflow/mplang/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/secretflow/mplang/tree/main)


MPLang (Multi-Party Language) is a single-controller programming library for multi-party and multi-device workloads. It follows the SPMD (Single Program, Multiple Data) model, where one Python program orchestrates multiple parties and devices (e.g., P0/P1/SPU) with explicit security domains. Programs are compilable and auditable, and can run in local simulation or on secure-computation backends.

## Highlights

- Single-controller SPMD: one program, multiple parties in lockstep
- Explicit devices and security domains: clear annotations for P0/P1/SPU
- Function-level compilation (@mplang.function): narrow instruction surface, reduce RCE risk, enable graph optimizations and audit
- Pluggable frontends and backends: not tied to specific FE/BE technologies
    - Frontends (FE): JAX, Ibis, and other computation frameworks
    - Backends (BE): StableHLO IR, Substrait IR, SPU PPHlo IR, and other intermediate representations
    - Execution: in-memory simulation, gRPC-based executors, or your custom engines

## Quick start

Writing multi-party secure computation programs is easy:

Note: The snippet below is illustrative and not directly runnable; for a complete runnable example of the device API, see `tutorials/3_device.py`.

```python
import mplang.device as mpd

def millionaire():
    # Alice's value on P0
    x = mpd.device("P0")(randint)(0, 1000000)
    # Bob's value on P1
    y = mpd.device("P1")(randint)(0, 1000000)
    # Compare values on SPU
    z = mpd.device("SPU")(lambda a, b: a < b)(x, y)
    return z
```

Add one decorator to get the "compiled version":

```python
@mplang.function
def millionaire():
    x = mpd.device("P0")(randint)(0, 1000000)
    y = mpd.device("P1")(randint)(0, 1000000)
    z = mpd.device("SPU")(lambda a, b: a < b)(x, y)
    return z

# Run it
sim = mplang.Simulator(2)
result = mplang.eval(sim, millionaire)
print("result:", mplang.fetch(sim, result))
```


## Installation and setup

- Install uv (if not installed):

    Linux/macOS:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

    Or with pipx:

    ```bash
    pipx install uv
    ```

- Install from source:

    ```bash
    uv pip install .
    ```

- Editable install for development:

    ```bash
    uv pip install -e .
    ```

- Run tutorials (complete examples and explanations):

    ```bash
    uv run tutorials/0_basic.py
    ```

See more examples in `tutorials/` (e.g., `1_condition.py`, `2_whileloop.py`).

## Beyond the basics

- SPMD for all-party execution: describe once, execute on all parties
- `mplang.compile(...)`: inspect compiler IR for understanding and optimization
- SMPC primitives: `smpc.seal`, `smpc.reveal`, `smpc.srun` to express secure operators

## Contributing and development

We welcome PRs and issues. Common dev commands:

```bash
uv sync --group dev
uv run pytest
uv run ruff check . --fix && uv run ruff format .
uv run mypy mplang/
```

## License

Apache-2.0. See `LICENSE` for details.
