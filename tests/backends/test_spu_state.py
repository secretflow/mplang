# Copyright 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for SPU dialect state."""

from __future__ import annotations

from types import SimpleNamespace

from mplang.backends import spu_state
from mplang.backends.spu_state import BrpcLinkConfig, SPUState
from mplang.dialects import spu


def test_effective_brpc_config_merges_spu_link_desc() -> None:
    base = BrpcLinkConfig(
        protocol="h2",
        connection_type="single",
        recv_timeout_ms=1000,
        http_max_payload_size=1024,
        http_timeout_ms=None,
        connect_retry_times=3,
        connect_retry_interval_ms=4,
    )
    state = SPUState(brpc_config=base)
    config = spu.SPUConfig(
        link_desc=spu.SPULinkDesc(
            brpc_channel_protocol="http",
            brpc_channel_connection_type="pooled",
            recv_timeout_ms=7200000,
            http_timeout_ms=7200000,
        )
    )

    effective = state._effective_brpc_config(config)

    assert effective.protocol == "http"
    assert effective.connection_type == "pooled"
    assert effective.recv_timeout_ms == 7200000
    assert effective.http_timeout_ms == 7200000
    assert effective.http_max_payload_size == 1024
    assert effective.connect_retry_times == 3
    assert effective.connect_retry_interval_ms == 4


def test_create_brpc_link_applies_explicit_config(monkeypatch) -> None:
    class FakeDesc:
        def __init__(self) -> None:
            self.parties: list[tuple[str, str]] = []

        def add_party(self, name: str, endpoint: str) -> None:
            self.parties.append((name, endpoint))

    captured = {}

    def fake_create_brpc(desc: FakeDesc, local_rank: int) -> str:
        captured["desc"] = desc
        captured["local_rank"] = local_rank
        return "fake-link"

    fake_link = SimpleNamespace(Desc=FakeDesc, create_brpc=fake_create_brpc)
    monkeypatch.setattr(spu_state.libspu, "link", fake_link)

    cfg = BrpcLinkConfig(
        protocol="http",
        connection_type="pooled",
        recv_timeout_ms=7200000,
        http_max_payload_size=67108864,
        http_timeout_ms=7200000,
        connect_retry_times=9,
        connect_retry_interval_ms=10,
    )

    link = SPUState()._create_brpc_link(1, ["127.0.0.1:9000", "127.0.0.1:9001"], cfg)

    assert link == "fake-link"
    assert captured["local_rank"] == 1
    desc = captured["desc"]
    assert desc.recv_timeout_ms == 7200000
    assert desc.http_max_payload_size == 67108864
    assert desc.http_timeout_ms == 7200000
    assert desc.brpc_channel_protocol == "http"
    assert desc.brpc_channel_connection_type == "pooled"
    assert desc.connect_retry_times == 9
    assert desc.connect_retry_interval_ms == 10
    assert desc.parties == [("P0", "127.0.0.1:9000"), ("P1", "127.0.0.1:9001")]
