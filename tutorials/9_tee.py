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

device_conf = {
    "TEE0": {
        "type": "TEEU",
        "node_ids": ["node:0", "node:1", "node:2"],
        "configs": {
            "tee_node": "node:2",
            "tee_mode": "sim",
        },
    },
    "P0": {"type": "PPU", "node_ids": ["node:0"]},
    "P1": {"type": "PPU", "node_ids": ["node:1"]},
}
node_def = {
    "node:0": "127.0.0.1:61920",
    "node:1": "127.0.0.1:61921",
    "node:2": "127.0.0.1:61922",
}


@mpd.function
def myfun(x, y):
    c0 = 10
    c1 = "hello"

    x = mpd.device("P0")(lambda x: x + 1)(x)
    y = mpd.device("P1")(lambda y: y * 2)(y)
    z = mpd.device("TEE0")(lambda x, y: x < y)(x, y)

    return x, [y, c0], {"z": z, "s": c1}


@mpd.function
def millionaire():
    x = mpd.device("P0")(random.randint)(0, 10)
    assert mpd.Utils.get_devid(x) == "P0", x

    y = mpd.device("P1")(random.randint)(0, 10)
    assert mpd.Utils.get_devid(y) == "P1", y

    z = mpd.device("TEE0")(lambda x, y: x < y)(x, y)
    assert mpd.Utils.get_devid(z) == "TEE0", z

    r = mpd.put("P0", z)
    return x, y, z, r


def lazy_eval() -> None:
    compiled = mplang.compile(mplang.cur_ctx(), millionaire)
    print("millionaire compiled:", compiled.compiler_ir())

    x, y, z, r = millionaire()
    print("x:", x, mplang.fetch(None, x))
    print("y:", y, mplang.fetch(None, y))
    print("z:", z, mplang.fetch(None, z))
    print("r:", r, mplang.fetch(None, r))

    print("-" * 10, "myfun", "-" * 10)
    xx, [yy, _c0], res_dict = myfun(x, y)
    print("xx:", xx)
    print("yy:", yy)
    print("res_dict", res_dict)


def eager_eval() -> None:
    # the function is evaluated immediately, on P0
    x = mpd.device("P0")(random.randint)(0, 10)
    assert mpd.Utils.get_devid(x) == "P0", x

    # the function is evaluated immediately, on P1
    y = mpd.device("P1")(random.randint)(0, 10)
    assert mpd.Utils.get_devid(y) == "P1", y

    # compare the two numbers, on TEE0
    z = mpd.device("TEE0")(lambda x, y: x < y)(x, y)
    assert mpd.Utils.get_devid(z) == "TEE0", z

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
    mpd.init(device_conf, {})
    print("-" * 10, "lazy_eval", "-" * 10)
    lazy_eval()
    print("-" * 10, "eager_eval", "-" * 10)
    eager_eval()
