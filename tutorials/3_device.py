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

import random

import mplang
import mplang.device as mpd
from mplang.core.cluster import ClusterSpec
from mplang.core.context_mgr import set_ctx
from mplang.runtime.simulation import Simulator

cluster_spec = ClusterSpec.from_dict({
    "nodes": [
        {"name": "node_0", "endpoint": "127.0.0.1:61920"},
        {"name": "node_1", "endpoint": "127.0.0.1:61921"},
        {"name": "node_2", "endpoint": "127.0.0.1:61922"},
        {"name": "node_3", "endpoint": "127.0.0.1:61923"},
        {"name": "node_4", "endpoint": "127.0.0.1:61924"},
    ],
    "devices": {
        "SP0": {
            "kind": "SPU",
            "members": ["node_1", "node_2", "node_3"],
            "config": {
                "protocol": "SEMI2K",
                "field": "FM128",
                "enable_pphlo_profile": True,
            },
        },
        "P0": {"kind": "PPU", "members": ["node_0"], "config": {}},
        "P1": {"kind": "PPU", "members": ["node_4"], "config": {}},
    },
})


@mpd.function
def alice_input(a, b):
    return mpd.device("P0")(random.randint)(a, b)


@mpd.function
def bob_input(a, b):
    return mpd.device("P1")(random.randint)(a, b)


@mpd.function
def myfun(x, y):
    c0 = 10
    c1 = "hello"

    x = mpd.device("P0")(lambda x: x + 1)(x)
    y = mpd.device("P1")(lambda y: y * 2)(y)
    z = mpd.device("SP0")(lambda x, y: x < y)(x, y)

    return x, [y, c0], {"z": z, "s": c1}


@mpd.function
def millionaire():
    x = mpd.device("P0")(random.randint)(0, 10)
    assert mpd.Utils.get_devid(x) == "P0", x

    y = mpd.device("P1")(random.randint)(0, 10)
    assert mpd.Utils.get_devid(y) == "P1", y

    z = mpd.device("SP0")(lambda x, y: x < y)(x, y)
    assert mpd.Utils.get_devid(z) == "SP0", z

    r = mpd.put("P0", z)
    return x, y, z, r


def main(ctx):
    compiled = mplang.compile(ctx, millionaire)
    print("millionaire compiled:", compiled.compiler_ir())

    x, y, z, r = mplang.evaluate(ctx, millionaire)
    print("x:", x, mplang.fetch(ctx, x))
    print("y:", y, mplang.fetch(ctx, y))
    print("z:", z, mplang.fetch(ctx, z))
    print("r:", r, mplang.fetch(ctx, r))

    print("-" * 10, "myfun", "-" * 10)
    xx, [yy, _c0], res_dict = mplang.evaluate(ctx, myfun, x, y)
    print("xx:", xx)
    print("yy:", yy)
    print("res_dict", res_dict)


def main2(ctx):
    set_ctx(ctx)

    # the function is evaluated immediately, on P0
    x = mpd.device("P0")(random.randint)(0, 10)
    assert mpd.Utils.get_devid(x) == "P0", x

    # the function is evaluated immediately, on P1
    y = mpd.device("P1")(random.randint)(0, 10)
    assert mpd.Utils.get_devid(y) == "P1", y

    # compare the two numbers, on SP0
    z = mpd.device("SP0")(lambda x, y: x < y)(x, y)
    assert mpd.Utils.get_devid(z) == "SP0", z

    print("x:", x)
    print("y:", y)
    print("z:", z)

    # reveal the result to P0
    u = mpd.put("P0", z)
    print("w:", u)

    # direct reveal a variable from P0 to P1
    v = mpd.put("P1", x)
    print("v:", v)


if __name__ == "__main__":
    # Create a simple simulator with cluster_spec directly
    simulator = Simulator(cluster_spec)

    main(simulator)
    main2(simulator)
