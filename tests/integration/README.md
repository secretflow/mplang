# Integration Tests

This directory contains integration (end-to-end / cross-module) tests.

## Definition

Integration tests exercise multiple layers of the MPLang stack together:

- Frontend APIs (`mplang.frontend.*` / `simp.*` helpers)
- Runtime scheduling and Simulator coordination
- Backend handlers (TEE, crypto, SPU, PHE, etc.)
- Cross‑party or cross‑device interactions (e.g. encryption handshake, gather / scatter, conditional SPMD control flow)

They differ from unit tests which should isolate a single function or module with minimal dependencies.

## Naming Guidelines

- Files live directly under `tests/integration/`.
- No mandatory `_integration` suffix; choose descriptive names (e.g. `test_crypto_roundtrip.py`, `test_tutorials_integration.py`).
- Prefer grouping related scenarios instead of one huge catch‑all file.

## Pytest Marker

All tests here are implicitly marked with the `integration` marker (see `pyproject.toml`).
Add at file top if you create a new file:

```python
import pytest
pytestmark = pytest.mark.integration
```

## When to Add a Test Here

Add to `tests/integration/` if a test:

1. Requires multiple simulated parties (`Simulator.simple(n)`)
2. Spans frontend → runtime → backend
3. Validates protocol/session state (e.g. TEE session cache, key agreement)
4. Depends on side effects or ordering (e.g. gather / conditional branching across parties)
5. Is derived from a tutorial as an executable regression example

Keep it in module/unit tests if it:

- Touches only pure functions or a single backend primitive
- Can be validated without Simulator / multi-party orchestration

## Running

Run only unit (exclude integration):

```bash
pytest -m 'not integration'
```

Run only integration:

```bash
pytest -m integration
```

Run everything:

```bash
pytest
```

## Best Practices

- Keep each test focused (one core behavior / invariant)
- Avoid excessive randomness; use deterministic seeds where practical
- Document intentionally insecure/mock behaviors (e.g. mock crypto) in the test docstring
- Prefer small payload sizes for speed

## Future Ideas

- Add SPU / PHE realistic interoperability smoke tests
- Property-based fuzz (hypothesis) for masking / conditional control flow
- Performance sanity thresholds (optional, skipped by default)
