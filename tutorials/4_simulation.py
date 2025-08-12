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
import mplang.random as mpr
import mplang.runtime as mprt
import mplang.smpc as smpc


def randint(lo, hi):
    rng = random.Random()
    rng.seed()
    return rng.randint(lo, hi)


@mplang.function
def millionaire():
    # Note: mpl.run(random.randint) will not work, because
    # the random number generator's state is captured and will always
    # return the same number on both parties.
    x = mpr.prandint(0, 10)
    y = mpr.prandint(0, 10)

    # both of them seal it
    x_ = smpc.sealFrom(x, 0)
    y_ = smpc.sealFrom(y, 1)

    # compare it seally.
    z_ = smpc.srun(lambda x, y: x < y)(x_, y_)

    # reveal it to all.
    z = smpc.reveal(z_)

    return x, y, z


def simp_main(driver):
    mplang.set_ctx(driver)

    # run the simp function on a given executor
    x, y, z = millionaire()

    # result a reference to the resource on the executor
    hx, hy, hz = mplang.fetch(None, (x, y, z))
    print("x:", hx)
    print("y:", hy)
    print("z:", hz)


@mpd.function
def alice_input(a, b):
    return mpd.device("P0")(randint)(a, b)


@mpd.function
def bob_input(a, b):
    return mpd.device("P1")(randint)(a, b)


@mpd.function
def myfun(x, y):
    c0 = 10
    c1 = "hello"

    x = mpd.device("P0")(lambda x: x + 1)(x)
    y = mpd.device("P1")(lambda y: y * 2)(y)
    z = mpd.device("SP0")(lambda x, y: x < y)(x, y)

    return x, [y, c0], {"z": z, "s": c1}


device_conf = {
    "SP0": {
        "type": "SPU",
        "node_ids": ["node:0", "node:1", "node:2"],
        "configs": {
            "protocol": "SEMI2K",
            "field": "FM128",
            "enable_pphlo_profile": True,
        },
    },
    "P0": {"type": "PPU", "node_ids": ["node:0"]},
    "P1": {"type": "PPU", "node_ids": ["node:1"]},
}


# def device_lazy(ectx):
#     assert ectx.world_size == 3
#     simp.WORLD_SIZE = 3
#     driver = mpd.DeviceDriver(mpd.parse_device_conf(device_conf), ectx)

#     x = alice_input(driver, 0, 10)
#     assert isinstance(x, mpd.DeviceObject) and x.owner() == "P0", x

#     y = bob_input(driver, 0, 10)
#     assert isinstance(y, mpd.DeviceObject) and y.owner() == "P1", y

#     xx, [yy, c0], res_dict = myfun(driver, x, y)
#     # print("xx:", xx)
#     # print("yy:", yy)
#     # print("z:", res_dict["z"])

#     x, y, xx, yy, z = mpd.fetch(driver, (x, y, xx, yy, res_dict["z"]))
#     print("x:", x)
#     print("y:", y)
#     print("xx:", xx)
#     print("yy:", yy)
#     print("z:", z)


@mpd.function
def device_func():
    # Use the custom randint function instead of random.randint
    # to avoid the random state capture issue
    x = mpd.device("P0")(randint)(0, 10)
    assert mpd.Utils.get_devid(x) == "P0", x

    y = mpd.device("P1")(randint)(0, 10)
    assert mpd.Utils.get_devid(y) == "P1", y

    z = mpd.device("SP0")(lambda x, y: x < y)(x, y)
    assert mpd.Utils.get_devid(z) == "SP0", z

    return x, y, z


def device_eager(ectx):
    # Initialize device configuration for simulation
    device_conf = {
        "SP0": {
            "type": "SPU",
            "node_ids": ["node:0", "node:1", "node:2"],
            "configs": {
                "protocol": "SEMI2K",
                "field": "FM128",
                "enable_pphlo_profile": True,
            },
        },
        "P0": {"type": "PPU", "node_ids": ["node:0"]},
        "P1": {"type": "PPU", "node_ids": ["node:1"]},
    }

    # Initialize device configuration
    mpd.init(device_conf, {})
    mplang.set_ctx(ectx)

    x, y, z = device_func()
    hx, hy, hz = mplang.fetch(None, (x, y, z))
    print("hx:", hx)
    print("hy:", hy)
    print("hz:", hz)


def aot_compilation(ectx):
    # This function illustrates the AOT compilation process
    copts = mplang.CompileOptions(ectx.psize(), spu_mask=ectx.attr("spu_mask"))
    compiled = mplang.compile(copts, millionaire)
    print(compiled.compiler_ir())


def main(driver):
    print("-" * 10, "simp", "-" * 10)
    simp_main(driver)

    # TODO(jint): not working for now
    # device_eager(driver)

    print("-" * 10, "aot example", "-" * 10)
    aot_compilation(driver)


def cmd_main(main_func) -> None:
    """Simple command line interface for simulation."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="MPLang simulation and execution.")
    parser.add_argument(
        "-c", "--config", default="examples/conf/3pc.json", help="the config"
    )
    subparsers = parser.add_subparsers(dest="command")
    parser_start = subparsers.add_parser("start", help="to start a single node")
    parser_start.add_argument("-n", "--node_id", default="node:0", help="the node id")
    subparsers.add_parser("up", help="to bring up all nodes")
    subparsers.add_parser("sim", help="simulate the test code")
    subparsers.add_parser("run", help="run the test code")
    args = parser.parse_args()

    # Load config file
    with open(args.config) as file:
        conf = json.load(file)
    nodes_def = conf["nodes"]

    devices_conf = mpd.parse_device_conf(conf["devices"])
    all_node_ids = sorted(nodes_def.keys())
    spu_conf = [dev for dev in devices_conf.values() if dev.type == "SPU"]
    spu_mask = 0  # Default mask
    if len(spu_conf) == 1:
        spu_mask = 0
        for nid in spu_conf[0].node_ids:
            spu_mask |= 1 << all_node_ids.index(nid)

    if args.command == "start":
        node_id = args.node_id
        if node_id not in nodes_def:
            print(f"Error: Node ID '{node_id}' not found in nodes definition")
            print(f"Available nodes: {list(nodes_def.keys())}")
            return
        mprt.serve(node_id, nodes_def[node_id], debug_execution=True)
    elif args.command == "up":
        mprt.start_cluster(nodes_def, debug_execution=True)
    elif args.command == "sim":
        simulator = mplang.Simulator(
            len(nodes_def),
            spu_mask=spu_mask,
        )
        main_func(simulator)
    elif args.command == "run":
        driver = mprt.ExecutorDriver(
            nodes_def,
            spu_mask=spu_mask,
        )
        main_func(driver)
    else:
        parser.print_help()


if __name__ == "__main__":
    # run the function on simulator or executor
    # To run on simulator, use command:
    #   python tutorials/4_simulation.py sim
    #
    # To run on executor, first start the executor with command:
    #   python tutorials/4_simulation.py up  (start all nodes)
    # Or start a single node:
    #   python tutorials/4_simulation.py start -n node:0
    # Then run the command:
    #   python tutorials/4_simulation.py run
    cmd_main(main)
