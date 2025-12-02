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

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest

import mplang
import mplang.v1.simp.party as party
from mplang.v1.ops import crypto


def test_basic_callable_and_namespace():
    sim = mplang.Simulator.simple(3)

    @mplang.function
    def prog():
        # universal form
        a, b = party.P0(crypto.kem_keygen, "x25519")
        # namespace form (tee side key, then quote)
        t_sk, t_pk = party.P[2].crypto.kem_keygen("x25519")
        _ = party.P[2].tee.quote_gen(t_pk)
        # derive something simple at party 0 to ensure run path works
        _ = party.P0(lambda x: x + 1, 41)
        return a, b, t_sk, t_pk

    a, b, t_sk, t_pk = mplang.evaluate(sim, prog)
    # Just basic shape checks: objects should be MPObjects with attrs; rely on existing crypto tests for deep correctness
    assert hasattr(a, "attrs") and hasattr(b, "attrs")
    assert hasattr(t_sk, "attrs") and hasattr(t_pk, "attrs")


def test_bound_method_style_lambda():
    sim = mplang.Simulator.simple(3)

    class Box:
        def __init__(self, v: int):
            self.v = v

        def inc(self, d: int) -> int:  # pure python op ok
            return self.v + d

    @mplang.function
    def prog():
        box = Box(10)
        # Pass bound method directly
        r1 = party.P0(box.inc, 5)
        # Or via lambda exposing self
        r2 = party.P0(lambda fn, d: fn(d), box.inc, 7)
        return r1, r2

    r1, r2 = mplang.evaluate(sim, prog)
    # Fetch the concrete per-party values (list per MPObject); we only need party0's view.
    r1_f, r2_f = mplang.fetch(sim, (r1, r2))
    assert int(r1_f[0]) == 15
    assert int(r2_f[0]) == 17


def test_load_module_conflict():
    # First registration should succeed
    party.load_module("mplang.ops.crypto", alias="crypto_alias")
    # Re-register same alias -> idempotent
    party.load_module("mplang.ops.crypto", alias="crypto_alias")
    # Different target with same alias should raise
    with pytest.raises(ValueError):
        party.load_module("mplang.ops.tee", alias="crypto_alias")


def test_non_callable_attribute_raises(monkeypatch: pytest.MonkeyPatch):
    # Create a fake module with a non-callable attr
    module_name = "mplang.ops._fake_const_mod"
    fake_mod = types.ModuleType(module_name)
    fake_mod.VALUE = 123  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, fake_mod)

    party.load_module(module_name, alias="fakec")
    with pytest.raises(AttributeError):
        _ = party.P0.fakec.VALUE  # type: ignore[attr-defined]


def test_frontend_import_failure_graceful(monkeypatch: pytest.MonkeyPatch):
    # Simulate frontend package missing by removing from sys.modules and making import fail.
    def _raise_import(*_a: Any, **_k: Any):  # pragma: no cover - executed in test
        raise ImportError("frontend missing")

    monkeypatch.setattr(importlib, "import_module", _raise_import)
    # Re-trigger prelude load; should not raise
    from importlib import reload

    reload(party)  # type: ignore
    # No assertion needed; success == no exception
