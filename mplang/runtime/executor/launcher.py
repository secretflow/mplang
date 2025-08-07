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

"""
Legacy executor module - re-exports separated server and client components.

This module maintains backward compatibility by re-exporting the separated
executor components. New code should import directly from executor_server
and executor_client modules.
"""

# Legacy compatibility imports
import functools
import operator

from mplang.device import DeviceContext, parse_device_conf

# Re-export client components
from mplang.runtime.driver import ExecutorDriver

# Re-export path utilities for backward compatibility
# Re-export server components
from mplang.runtime.executor.server import (
    serve,
    start_cluster,
)
from mplang.runtime.simulation import Simulator


def cmd_main(main, nodes_def):
    """Legacy command line interface - maintained for backward compatibility."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="SPU executor service.")
    parser.add_argument(
        "-c", "--config", default="examples/conf/3pc.json", help="the config"
    )
    subparsers = parser.add_subparsers(dest="command")
    parser_start = subparsers.add_parser("start", help="to start a single node")
    parser_start.add_argument("-n", "--node_id", default="node:0", help="the node id")
    subparsers.add_parser("up", help="to bring up all nodes")
    subparsers.add_parser("run", help="run the test code")
    subparsers.add_parser("sim", help="simulate the test code")
    args = parser.parse_args()

    spu_mask = 0  # Default mask

    if args.config:
        with open(args.config) as file:
            conf = json.load(file)
        nodes_def = conf["nodes"]
        all_node_ids = sorted(nodes_def.keys())
        devices_conf = parse_device_conf(conf["devices"])
        DeviceContext(devices_conf)
        used_node_ids = list(
            set(functools.reduce(operator.iadd, [info.node_ids for info in devices_conf.values()], []))
        )
        assert all(nid in nodes_def for nid in used_node_ids), (
            "Some node ids are not defined in the config."
        )
        spu_conf = [dev for dev in devices_conf.values() if dev.type == "SPU"]
        if len(spu_conf) == 1:
            spu_mask = 0
            for nid in spu_conf[0].node_ids:
                spu_mask |= 1 << all_node_ids.index(nid)

    if args.command == "start":
        serve(args.node_id, nodes_def[args.node_id])
    elif args.command == "up":
        start_cluster(nodes_def)
    elif args.command == "run":
        driver = ExecutorDriver(
            nodes_def,
            spu_mask=spu_mask,
        )
        main(driver)
    elif args.command == "sim":
        simulator = Simulator(
            len(nodes_def),
            spu_mask=spu_mask,
        )
        main(simulator)
    else:
        parser.print_help()


# Sample configuration for testing
SAMPLE_NODES_DEF = {
    "node:0": "127.0.0.1:61920",
    "node:1": "127.0.0.1:61921",
    "node:2": "127.0.0.1:61922",
}
