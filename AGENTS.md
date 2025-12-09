# AGENTS.md

Concise guidance for AI coding agents working in this repo. Keep answers specific to MPLang and its current code, not generic.

## What this project is

- MPLang is a Python SPMD framework to orchestrate multi-party/multi-device programs from a single controller.
- Core idea: trace Python functions into an IR, then interpret/execute across parties/devices with explicit security domains.

## Architecture map (key directories/files)

- Core SPMD/IR: `mplang/core/`
  - `primitive.py` (defines `@primitive`/`function`, control-flow like `cond`, `while_loop`, `peval`),
  - `tracer.py` (TraceContext/TraceVar), `interp.py` (InterpContext/InterpVar, `apply`),
  - `mpobject.py`, `mptype.py`, `dtype.py`, `tensor.py`, `table.py`, `pfunc.py`, `mask.py`.
- Expression AST: `mplang/core/expr/` (Expr nodes used by tracing; see imports in `primitive.py`).
- Runtime: `mplang/runtime/` (`simulation.py` for local multi-threaded runs, `driver.py`, `server.py`, `client.py`, `communicator.py`).
- Frontends/Backends: `mplang/ops/*` (basic, jax_cc, etc.) and `mplang/kernels/*` (std, spu, phe, sql_duckdb, stablehlo).
- Devices API: `mplang/device.py` (device placement/transforms).
- Low-level party API: `mplang/simp/*` (MPI-style ops, random, smpc) — used when you need rank-level control.
- Public API surface: `mplang/__init__.py` (re-exports: `function`, `compile`, `evaluate`, `fetch`, `Simulator`, etc.).

## Project-specific patterns and conventions

- Contexts are king: code must work under both `TraceContext` and `InterpContext`.
  - Inside `@primitive` functions, use `cur_ctx()` and do NOT cross-share `TraceVar`/`InterpVar`; capture into the current context (see `_switch_ctx` in `primitive.py`).
- New primitives: decorate with `@primitive` (or `function` alias). For backend-evaluable ops, route through `mplang/ops/basic.py` and call `peval(pfunc, eval_args, rmask?)` to create IR; return the unflattened tree.
- Masks: `Mask` models which parties hold/execute values. `set_mask` enforces runtime execution mask; static/dynamic pmask rules are documented in its docstring.
- Control flow: use `cond`, `while_loop`, `peval`, `ConvExpr`/`ShflExpr` via helpers in `primitive.py` instead of ad-hoc Python control flow inside traced code.
- Devices: prefer `mplang.device` for placement and `@mplang.function` for graph capture; avoid leaking ranks into device-level code (rank-level ops live in `mplang/simp`).
- Types/Style: Python 3.10+, type hints everywhere. Use `ruff` for lint/format and `mypy` on `mplang/` before PRs.

## Dev workflow (commands)

- Install: `uv sync --group dev` then `uv pip install -e .`
- Tests: `uv run pytest` (parallel: `-n auto`), focused: e.g. `tests/core/test_primitive.py`.
- Lint/Format: `uv run ruff check . --fix` and `uv run ruff format .`
- Types: `uv run mypy mplang/`
- Tutorials: `uv run tutorials/0_basic.py`, `1_condition.py`, `2_whileloop.py`, etc.

## Protobuf (Buf) specifics

- Protos live under `protos/` (root) and generate into the repo (e.g., `mplang/protos/v1alpha1/`).
- Workflow: `buf format -w`, `buf lint`, `buf generate`, `buf breaking --against '.git#branch=main'`, `buf dep update` when deps change. Commit generated files with the proto changes.

## Good starting references (read these before changes)

- How primitives build IR: `mplang/core/primitive.py` (see `peval`, `prand`, `constant`, `cond`).
- Context behavior: `mplang/core/tracer.py`, `mplang/core/interp.py`, and `_switch_ctx` in `primitive.py`.
- Runtime simulation and E2E: mplang/runtime/simulation.py, tutorials/\*.py, tests in tests/\*\* (e.g., tests/kernels/test_basic.py).

## Do/Don't (project-specific)

- Do add new high-level APIs by composing existing primitives instead of bypassing trace/IR.
- Do keep device-level code rank-agnostic; if you need ranks, implement in `mplang/simp`.
- Don't move `TraceVar`/`InterpVar` across contexts directly; always capture/switch via current context.
- Don't introduce hidden global state for masks/devices; pass masks explicitly where supported (e.g., `peval(..., rmask=...)`).

---

## MPLang2 (next-gen EDSL, replacing mplang.core)

Location: `mplang2/`. Status: ~95% feature parity with v1 (2025-12-01).

### Why rewrite?

1. **Dialect extensibility**: v1 treats `simp` as first-class dialect; other ops (`tensor`, `table`, `spu`, etc.) are second-class, must embed in `@primitive`, can't be traced/extended independently. v2 makes all dialects equal citizens.
2. **Type system**: v1 only has `TensorType`/`TableType`; other types (ciphertexts, keys) are "simulated" via Tensor, losing compile-time checks. v2 has extensible `ScalarType`, `TensorType`, `TableType`, `VectorType`, `SSType`, `CustomType`.

Neither v1 nor v2 focuses on optimization passes yet; v2's goals are **typing** and **dialect extensibility**.

### mplang2 structure

- `mplang2/edsl/`: typing.py (unified types), graph.py (Op List + SSA IR), primitive.py, tracer.py, interpreter.py, jit.py.
- `mplang2/dialects/`: simp, tensor, table, spu, tee, bfv, phe, dtypes.
- `mplang2/backends/`: simp_simulator, simp_http_driver, simp_http_worker, cli, *_impl.
- `mplang2/libs/device.py`: Device API (v1-compatible).

### Key API diffs (v1 → v2)

- Import: `import mplang2 as mp`
- JAX on PPU: `mp.device("P0").jax(fn)` (explicit frontend via .jax property).
- Constants: `mp.put("P0", 42)` instead of `mp.device("P0")(lambda: 42)()`.
- IR inspection: `mp.jit(fn).compiler_ir()`.

### Migration status

✅ device/put, Simulator/Driver, evaluate/fetch, jit/trace, TensorType/TableType/ScalarType, dialects (simp/tensor/table/spu/tee/bfv/phe), table I/O, CLI.
❌ analysis/diagrams not yet ported.

---

## RFC process (for new modules)

Before significant changes, write `design/<feature>.md` with: Summary, Motivation, API Surface, Implementation (modules, TraceContext vs InterpContext interaction), Alternatives, Migration/Compat, Test Plan. See `design/architecture_v2.md` as example.

Questions or unclear areas? Pin them to specific files/APIs and I'll refine this doc.
