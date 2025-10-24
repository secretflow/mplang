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

import yaml

import mplang as mp


def randint(lo, hi):
    rng = random.Random()
    rng.seed()
    return rng.randint(lo, hi)


@mp.function
def millionaire():
    # Note: mpl.run(random.randint) will not work, because
    # the random number generator's state is captured and will always
    # return the same number on both parties.
    x = mp.prandint(0, 10)
    y = mp.prandint(0, 10)

    # both of them seal it
    x_ = mp.seal_from(0, x)
    y_ = mp.seal_from(3, y)

    # compare it seally.
    z_ = mp.srun_jax(lambda x, y: x < y, x_, y_)

    # reveal it to all.
    z = mp.reveal(z_)

    return x, y, z


def simp_main(driver):
    mp.set_ctx(driver)

    # run the simp function on a given executor
    x, y, z = millionaire()

    # result a reference to the resource on the executor
    hx, hy, hz = mp.fetch(None, (x, y, z))
    print("x:", hx)
    print("y:", hy)
    print("z:", hz)


@mp.function
def alice_input(a, b):
    return mp.device("P0")(randint)(a, b)


@mp.function
def bob_input(a, b):
    return mp.device("P1")(randint)(a, b)


@mp.function
def myfun(x, y):
    c0 = 10
    c1 = "hello"

    x = mp.device("P0")(lambda x: x + 1)(x)
    y = mp.device("P1")(lambda y: y * 2)(y)
    z = mp.device("SP0")(lambda x, y: x < y)(x, y)

    return x, [y, c0], {"z": z, "s": c1}


@mp.function
def device_func():
    # Use the custom randint function instead of random.randint
    # to avoid the random state capture issue
    x = mp.device("P0")(randint)(0, 10)
    assert mp.get_dev_attr(x) == "P0", x

    y = mp.device("P1")(randint)(0, 10)
    assert mp.get_dev_attr(y) == "P1", y

    z = mp.device("SP0")(lambda x, y: x < y)(x, y)
    assert mp.get_dev_attr(z) == "SP0", z

    return x, y, z


def device_eager(ctx):
    x, y, z = mp.evaluate(ctx, device_func)
    hx, hy, hz = mp.fetch(ctx, (x, y, z))
    print("hx:", hx)
    print("hy:", hy)
    print("hz:", hz)


def aot_compilation(ectx):
    # This function illustrates the AOT compilation process
    copts = mp.CompileOptions(ectx.cluster_spec)
    compiled = mp.compile(copts, millionaire)
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

    parser = argparse.ArgumentParser(description="MPLang simulation and execution.")
    parser.add_argument(
        "-c", "--config", default="examples/conf/3pc.yaml", help="the config"
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("sim", help="simulate the test code")
    subparsers.add_parser("run", help="run the test code")
    args = parser.parse_args()

    # load ClusterSpec from yaml file
    with open(args.config) as file:
        conf = yaml.safe_load(file)
    cluster_spec = mp.ClusterSpec.from_dict(conf)

    if args.command == "sim":
        sim = mp.Simulator(cluster_spec)
        main_func(sim)
    elif args.command == "run":
        driver = mp.Driver(cluster_spec)
        main_func(driver)
    else:
        parser.print_help()


if __name__ == "__main__":
    # Run the function on simulator or executor
    #
    # To run on simulator, use command:
    #   uv run python tutorials/4_simulation.py sim
    #
    # To run on real multi-party execution:
    # 1. First start the cluster with:
    #    uv run python -m mplang.runtime.cli up -c examples/conf/3pc.yaml
    # 2. Then run the computation:
    #    uv run python tutorials/4_simulation.py run
    cmd_main(main)
