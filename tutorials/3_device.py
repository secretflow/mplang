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

cluster_spec = mplang.ClusterSpec.from_dict({
    "nodes": [
        {"name": "node_0", "endpoint": "127.0.0.1:61920"},
        {"name": "node_1", "endpoint": "127.0.0.1:61921"},
        {"name": "node_2", "endpoint": "127.0.0.1:61922"},
        {"name": "node_3", "endpoint": "127.0.0.1:61923"},
        {"name": "node_4", "endpoint": "127.0.0.1:61924"},
        {"name": "node_5", "endpoint": "127.0.0.1:61925"},
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
        "TEE0": {"kind": "TEE", "members": ["node_5"], "config": {}},
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
def millionaire(dev_name):
    x = mpd.device("P0")(random.randint)(0, 10)
    y = mpd.device("P1")(random.randint)(0, 10)
    # Run comparison inside secure device.
    z = mpd.device(dev_name)(lambda x, y: x < y)(x, y)
    # Bring result back to P0
    r = mpd.put("P0", z)
    return x, y, z, r


def run_spu():
    print("-" * 10, "millionaire (SPU)", "-" * 10)

    sim = mplang.Simulator(cluster_spec)
    x, y, z, r = mplang.evaluate(sim, millionaire, "SP0")
    print("x:", x, mpd.fetch(sim, x))
    print("y:", y, mpd.fetch(sim, y))
    print("z:", z, mpd.fetch(sim, z))
    print("r:", r, mpd.fetch(sim, r))

    # compiled = mplang.compile(sim, millionaire, "SP0")
    # print("SPU compiled:", compiled.compiler_ir())

    # ofcourse we can run other funcs in the same sim instance
    # print("-" * 10, "myfun", "-" * 10)
    # xx, [yy, _c0], res_dict = mplang.evaluate(sim, myfun, x, y)
    # print("xx:", xx)
    # print("yy:", yy)
    # print("res_dict", res_dict)


def run_tee():
    print("-" * 10, "millionaire (TEE)", "-" * 10)

    sim = mplang.Simulator(cluster_spec)
    x_p0, y_p1, z_t, r_p0 = mplang.evaluate(sim, millionaire, "TEE0")
    print("x_p0:", x_p0, mpd.fetch(sim, x_p0))
    print("y_p1:", y_p1, mpd.fetch(sim, y_p1))
    print("z_t:", z_t, mpd.fetch(sim, z_t))
    print("r_p0:", r_p0, mpd.fetch(sim, r_p0))

    # copts = mplang.CompileOptions(cluster_spec)
    # compiled = mplang.compile(copts, millionaire, "TEE0")
    # print("TEE compiled:", compiled.compiler_ir())


if __name__ == "__main__":
    # SPU version
    run_spu()
    # TEE version
    run_tee()
