# Contributing to MPLang

First off, thank you for considering contributing to MPLang! It's people like you that make MPLang such a great tool. We welcome any form of contribution, from documentation and bug reports to new features.

## Setting up the Development Environment

To get started, you'll need to set up your local environment for development.

### 1. Prerequisites

You'll need the following tools installed:

- **[uv](https://github.com/astral-sh/uv)**: A fast Python package installer and resolver.

  ```bash
  # Install uv on Linux/macOS
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **[buf](https://buf.build/docs/installation)**: For Protobuf code generation.

  ```bash
  # Install buf by following the official guide: https://buf.build/docs/installation
  ```

### 2. Install Dependencies

Once the prerequisites are installed, clone the repository and install the required Python packages for development in editable mode.

```bash
# Clone the repository
git clone https://github.com/secretflow/mplang.git
cd mplang

# Install dependencies, including development tools
uv sync --group dev

# Install the project in editable mode
uv pip install -e .
```

## Development Workflow

### 1. Protobuf/Buf workflow

If you touch `.proto` files under `protos/`, use the following workflow:

```bash
# Optional: format .proto files in place
buf format -w

# Lint proto style and naming
buf lint

# Validate schema compiles and imports resolve
buf build

# Generate Python stubs and gRPC code
buf generate
```

When you introduce new imports (for example, `google/api/*.proto`), ensure dependencies are up to date:

```bash
# After updating deps in buf.yaml if needed, refresh the lockfile
buf dep update
```

Notes:

- Protos live in `protos/`.
- Buf configs live at repo root: `buf.yaml` and `buf.gen.yaml`.
- Generated Python files are written into the repo (out: `.`), e.g. under `mplang/protos/v1alpha1/`.
- Commit updated generated files together with your `.proto` changes.

### 2. Running Tests

We use `pytest` for testing.

```bash
# Run all tests
uv run pytest

# Run tests in parallel for speed
uv run pytest -n auto

# Run a specific test file
uv run pytest tests/core/test_primitive.py
```

### 3. Code Quality

We use `ruff` for linting and formatting, and `mypy` for static type checking. Please run these tools before submitting your changes.

```bash
# Format code
uv run ruff format .

# Lint and automatically fix issues
uv run ruff check . --fix

# Run the type checker
uv run mypy mplang/
```

## Submitting Pull Requests

1. Fork the repository and create a new branch for your feature or bug fix.
2. Make your changes and ensure all tests and code quality checks pass.
3. Push your branch and open a pull request against the `main` branch.
4. Provide a clear description of your changes in the pull request.

Thank you for your contribution!
