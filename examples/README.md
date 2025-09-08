# MPLang Examples

This directory contains extended examples demonstrating how to use MPLang with various frontend/backends and workloads.

## Install Example Dependencies

Example dependencies are intentionally kept out of the core package to keep the base install and lockfile small.

```bash
uv pip install -r examples/requirements.txt
```

## Run Examples

A few entry points:

```bash
# Stax neural network demo (JAX based)
uv run python examples/stax_nn/stax_nn.py

# XGBoost style histogram example (NumPy/JAX hybrid)
uv run python examples/xgboost/hist_jax.py
```

## Notes

- Keep heavyweight ML dependencies here; do not reintroduce them into `pyproject.toml`.
- Add any new example-only dependency to `examples/requirements.txt`.
- Prefer small, focused scripts.
