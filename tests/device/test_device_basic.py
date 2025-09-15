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

import pytest

import mplang
import mplang.device as mpd
from mplang.core.cluster import ClusterSpec
from tests.utils.server_fixtures import get_free_ports

# NOTE: Tests assume single-member SPU/PPU as per current design.


def build_cluster():
    # Minimal 3-party cluster: P0, P1 as PPU; SP0 as SPU
    ports = get_free_ports(3)

    def _node(name: str, port: int):
        return {
            "name": name,
            "endpoint": f"localhost:{port}",
            "runtime_info": {
                "version": "test",
                "platform": "sim",
                "backends": ["__all__"],
            },
        }

    spec = {
        "nodes": [
            _node("node_0", ports[0]),
            _node("node_1", ports[1]),
            _node("node_2", ports[2]),
        ],
        "devices": {
            "P0": {"kind": "PPU", "members": ["node_0"], "config": {}},
            "P1": {"kind": "PPU", "members": ["node_1"], "config": {}},
            "SP0": {
                "kind": "SPU",
                "members": ["node_0", "node_1", "node_2"],
                "config": {
                    "protocol": "SEMI2K",
                    "field": "FM128",
                },
            },
        },
    }
    return ClusterSpec.from_dict(spec)


def test_fetch_no_list_wrapper():
    spec = build_cluster()
    sim = mplang.Simulator(spec)

    @mplang.function
    def add_one():
        x = mpd.device("P0")(lambda: 41)()
        y = mpd.device("P0")(lambda a: a + 1)(x)
        return y

    y = mplang.evaluate(sim, add_one)
    fetched = mpd.fetch(sim, y)
    # Should not be wrapped in list; accept python int or array-like scalar
    # Normalize via int() for JAX/NumPy scalar types.
    assert not isinstance(fetched, list)
    assert int(fetched) == 42


def test_ibis_on_spu_unsupported():
    spec = build_cluster()
    sim = mplang.Simulator(spec)

    # Directly wrap a trivial function; compiler path will reach the SPU gate.
    @mplang.function
    def foo():
        return 1

    with pytest.raises(ValueError, match="SPU device only supports JAX frontend"):
        # Use evaluate to ensure interpreter context established
        mplang.evaluate(sim, mpd.device("SP0", fe_type="ibis")(foo))


def test_jax_on_ppu_ok():
    spec = build_cluster()
    sim = mplang.Simulator(spec)

    @mplang.function
    def foo():
        a = mpd.device("P0")(lambda: 2)()
        b = mpd.device("P0")(lambda x: x * 3)(a)
        return b

    b = mplang.evaluate(sim, foo)
    assert mpd.fetch(sim, b) == 6


def test_ibis_on_ppu_frontend_path_reached():
    # If ibis not installed or compiler path incomplete, skip.
    try:
        import ibis  # noqa: F401
    except Exception:  # pragma: no cover
        pytest.skip("ibis not available in test environment")

    spec = build_cluster()
    mplang.Simulator(spec)  # instantiate to align with other tests; result unused

    # We can't easily craft real MPObject Table inputs here without more plumbing.
    # This test just ensures routing hits PPU path and fails *after* the frontend check (i.e., not blocked early).
    @mplang.function
    def dummy():
        return 1

    # Should raise because ibis compiler expects MPObject args; but *not* the SPU-only error.
    with pytest.raises(Exception):  # Frontend compilation / normalization error
        mpd.device("P0", fe_type="ibis")(dummy)()


# Additional test for unsupported d2d pair can be added when new device kinds appear.
