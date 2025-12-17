#!/usr/bin/env python3
# Copyright 2025 Ant Group Co., Ltd.
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

"""
Command-line interface for MPLang2 clusters and jobs.

Examples:
    # Generate a cluster config file
    python -m mplang.v2.cli config gen -w 3 -p 8100 -o cluster.yaml

    # Start a single worker (production usage)
    python -m mplang.v2.cli worker --rank 0 -c cluster.yaml

    # Start 3 local workers (development usage)
    python -m mplang.v2.cli up -c cluster.yaml

    # Check cluster status
    python -m mplang.v2.cli status -c cluster.yaml

    # Run a job
    python -m mplang.v2.cli run -c cluster.yaml -f my_job.py
"""

import argparse
import glob
import importlib.util
import json
import multiprocessing
import os
import re
import signal
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any, cast

import uvicorn
import yaml


def run_worker(
    rank: int,
    world_size: int,
    port: int,
    endpoints: list[str],
    spu_endpoints: dict[int, str] | None = None,
) -> None:
    """Run a single worker server."""
    # Reset signal handlers to default in child process to avoid conflict with parent's shutdown handler
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    from mplang.v2.backends.simp_worker.http import create_worker_app

    app = create_worker_app(rank, world_size, endpoints, spu_endpoints)

    log_config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": f"[Worker {rank}] %(levelname)s: %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "WARNING"},
            "uvicorn.error": {"handlers": ["default"], "level": "WARNING"},
            "uvicorn.access": {"handlers": ["default"], "level": "WARNING"},
        },
    }

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_config=log_config,
        log_level="warning",
    )


def build_endpoints(
    args: argparse.Namespace,
) -> tuple[list[str], list[int], int, dict[int, str] | None]:
    """Build endpoints and SPU endpoints from config or CLI flags."""

    def normalize_endpoint(ep: str) -> str:
        return ep if ep.startswith("http") else f"http://{ep}"

    spu_endpoints: dict[int, str] | None = None

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            conf = yaml.safe_load(f)
        nodes = conf.get("nodes", [])
        if not nodes:
            raise ValueError("Config must contain nodes")
        world_size = len(nodes)
        endpoints = []
        ports = []
        for node in nodes:
            endpoint = normalize_endpoint(node["endpoint"])
            endpoints.append(endpoint)
            ports.append(int(endpoint.split(":")[-1]))

        devices = conf.get("devices", {})
        for _dev_name, dev_conf in devices.items():
            if dev_conf.get("kind", "").upper() == "SPU":
                spu_endpoints = {}
                spu_base_port = args.spu_base_port or (ports[0] + 1000)
                for i, node in enumerate(nodes):
                    host = node["endpoint"].split(":")[0]
                    spu_endpoints[i] = f"{host}:{spu_base_port + i}"
                break
    else:
        world_size = args.world_size
        base_port = getattr(args, "base_port", 5000)
        ports = [base_port + i for i in range(world_size)]
        endpoints = [f"http://127.0.0.1:{p}" for p in ports]

        if args.spu_base_port:
            spu_endpoints = {
                i: f"127.0.0.1:{args.spu_base_port + i}" for i in range(world_size)
            }

    if args.endpoints:
        endpoints = [normalize_endpoint(ep.strip()) for ep in args.endpoints.split(",")]
        ports = [int(ep.split(":")[-1]) for ep in endpoints]
        world_size = len(endpoints)

    return endpoints, ports, world_size, spu_endpoints


def add_cluster_args(
    parser: argparse.ArgumentParser, *, include_ports: bool = True
) -> None:
    """Add common cluster arguments to subparsers."""

    parser.add_argument("-c", "--config", type=str, help="Cluster config YAML")
    parser.add_argument("--endpoints", type=str, help="Comma-separated HTTP endpoints")
    parser.add_argument(
        "--spu-endpoints", type=str, help="Comma-separated SPU BRPC endpoints"
    )
    parser.add_argument(
        "--spu-base-port",
        type=int,
        default=None,
        help="Base port for SPU BRPC (default: http_port + 1000)",
    )
    parser.add_argument(
        "-w",
        "--world-size",
        type=int,
        default=3,
        help="Number of workers (default: 3)",
    )
    if include_ports:
        parser.add_argument(
            "-p",
            "--base-port",
            type=int,
            default=8100,
            help="Base port (default: 8100)",
        )


def cmd_config_gen(args: argparse.Namespace) -> None:
    """Generate cluster configuration."""
    world_size = args.world_size
    base_port = args.base_port

    nodes = []
    for i in range(world_size):
        nodes.append({"name": f"node_{i}", "endpoint": f"127.0.0.1:{base_port + i}"})

    config: dict[str, Any] = {"nodes": nodes}

    # Add default PPU devices
    devices: dict[str, Any] = {}
    for i in range(world_size):
        devices[f"P{i}"] = {
            "kind": "ppu",
            "members": [f"node_{i}"],
        }

    if args.spu_base_port:
        devices["SPU0"] = {
            "kind": "SPU",
            "members": [n["name"] for n in nodes],
            "config": {"protocol": "ABY3", "field": "FM64"},
        }
    config["devices"] = devices

    yaml_content = yaml.dump(config, sort_keys=False)

    if args.output:
        with open(args.output, "w") as f:
            f.write(yaml_content)
        print(f"Config written to {args.output}")
    else:
        print(yaml_content)


def cmd_worker(args: argparse.Namespace) -> None:
    """Start a single worker process."""
    endpoints, ports, world_size, spu_endpoints = build_endpoints(args)
    rank = args.rank

    if rank < 0 or rank >= world_size:
        raise ValueError(f"Rank {rank} is out of range [0, {world_size - 1}]")

    print(f"Starting Worker {rank} on {endpoints[rank]}...")
    if spu_endpoints and rank in spu_endpoints:
        print(f"  SPU BRPC: {spu_endpoints[rank]}")

    run_worker(rank, world_size, ports[rank], endpoints, spu_endpoints)


def cmd_up(args: argparse.Namespace) -> None:
    """Start worker servers locally."""
    endpoints, ports, world_size, spu_endpoints = build_endpoints(args)

    print(f"Starting {world_size} workers...")
    for i, endpoint in enumerate(endpoints):
        print(f"  Worker {i}: {endpoint}")
    if spu_endpoints:
        print("SPU BRPC endpoints:")
        for rank, ep in spu_endpoints.items():
            print(f"  Rank {rank}: {ep}")

    processes: list[multiprocessing.Process] = []

    def shutdown(signum: int, frame: Any) -> None:
        print("\nShutting down workers...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for rank in range(world_size):
        p = multiprocessing.Process(
            target=run_worker,
            args=(rank, world_size, ports[rank], endpoints, spu_endpoints),
        )
        p.start()
        processes.append(p)

    print("\nWorkers started. Press Ctrl+C to stop.")

    for p in processes:
        p.join()


def cmd_status(args: argparse.Namespace) -> None:
    """Check /health of workers."""
    import httpx

    endpoints, _, world_size, _ = build_endpoints(args)

    print(f"Checking {len(endpoints)} endpoints (world_size={world_size})...")
    for ep in endpoints:
        url = f"{ep}/health"
        try:
            resp = httpx.get(url, timeout=3.0)
            resp.raise_for_status()
            print(f"OK  {url} -> {resp.json()}")
        except Exception as exc:
            print(f"ERR {url} -> {exc}")


def load_user_module(path: str) -> ModuleType:
    """Load a Python module from file path."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("mp_user_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_entry(module: ModuleType, name: str) -> Callable[..., Any]:
    entry = getattr(module, name, None)
    if entry is None or not callable(entry):
        raise AttributeError(f"Entry function '{name}' not found or not callable")
    return cast(Callable[..., Any], entry)


def parse_spu_endpoints(
    raw: str | None, world_size: int, default: dict[int, str] | None
) -> dict[int, str] | None:
    if raw is None:
        return default
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != world_size:
        raise ValueError("spu-endpoints count must match world size")
    return {i: parts[i] for i in range(world_size)}


def cmd_run(args: argparse.Namespace) -> None:
    """Run a user job via HTTP cluster or local simulator."""
    from mplang.v2 import make_driver, make_simulator
    from mplang.v2.edsl.context import pop_context, push_context
    from mplang.v2.libs.device import ClusterSpec

    cluster: ClusterSpec

    if args.config:
        # Load cluster from config file
        with open(args.config, encoding="utf-8") as f:
            conf = yaml.safe_load(f)
        cluster = ClusterSpec.from_dict(conf)
    else:
        # Build cluster from CLI arguments
        endpoints, _, world_size, _ = build_endpoints(args)
        cluster = ClusterSpec.simple(world_size, endpoints=endpoints)

    driver: Any
    if args.backend == "sim":
        enable_tracing = getattr(args, "profile", False)
        driver = make_simulator(
            cluster.world_size,
            cluster_spec=cluster,
            enable_tracing=enable_tracing,
        )
    else:
        driver = make_driver(cluster.endpoints, cluster_spec=cluster)

    # Set up context: push driver and set global cluster
    push_context(driver)
    # REMOVED: set_global_cluster(cluster)

    module = load_user_module(args.file)
    entry = resolve_entry(module, args.entry)

    try:
        # Entry function doesn't need driver parameter - it uses context
        result = entry(*args.args)
        if result is not None:
            print(result)
    finally:
        pop_context()
        if hasattr(driver, "shutdown"):
            driver.shutdown()


def cmd_trace_merge(args: argparse.Namespace) -> None:
    """Merge multiple Chrome Trace JSON files into a single file."""
    pattern = args.pattern
    output_file = args.output

    files = glob.glob(pattern)
    if not files:
        print(f"No files found matching pattern: {pattern}")
        sys.exit(1)

    print(f"Found {len(files)} trace files.")

    merged_events = []

    # Regex to extract rank from filename if present (e.g., trace_..._rank_0.json)
    rank_pattern = re.compile(r"_rank_(\d+)\.json$")

    for fname in files:
        print(f"Processing {fname}...")
        try:
            with open(fname) as f:
                data = json.load(f)
                events = data.get("traceEvents", [])

                # Determine rank/pid offset
                match = rank_pattern.search(fname)
                if match:
                    rank = int(match.group(1))
                    # Remap PID: rank 0 -> 1000, rank 1 -> 2000, etc.
                    # Or just use rank as PID if it's small enough, but Perfetto likes PIDs
                    pid_offset = (rank + 1) * 10000
                else:
                    pid_offset = 0

                for event in events:
                    # Remap PID
                    if "pid" in event:
                        # If original PID is present, we shift it to avoid collision
                        # between different processes on different machines that might have same PID
                        original_pid = event["pid"]
                        # Simple remapping: new_pid = offset + (original_pid % 10000)
                        # This preserves thread grouping within a rank
                        event["pid"] = pid_offset + (original_pid % 10000)

                    # Add rank info to args if not present
                    if match:
                        event_args = event.get("args", {})
                        event_args["rank"] = rank
                        event["args"] = event_args

                    merged_events.append(event)

        except Exception as e:
            print(f"Error processing {fname}: {e}")

    # Write merged file
    with open(output_file, "w") as f:
        json.dump({"traceEvents": merged_events}, f)

    print(f"Successfully merged {len(merged_events)} events into {output_file}")


def cmd_objects(args: argparse.Namespace) -> None:
    """List objects on workers."""
    endpoints, _, world_size, _ = build_endpoints(args)

    print(f"Listing objects on {world_size} workers...")
    print("-" * 80)
    print(f"{'Rank':<6} | {'Endpoint':<25} | {'Count':<6} | {'Objects'}")
    print("-" * 80)

    import httpx

    for rank in range(world_size):
        url = f"{endpoints[rank]}/objects"
        try:
            resp = httpx.get(url, timeout=2.0)
            if resp.status_code == 200:
                objects = resp.json()["objects"]
                count = len(objects)
                # Truncate list if too long
                obj_str = ", ".join(objects[:3])
                if count > 3:
                    obj_str += ", ..."
                print(f"{rank:<6} | {endpoints[rank]:<25} | {count:<6} | {obj_str}")
            else:
                print(
                    f"{rank:<6} | {endpoints[rank]:<25} | {'Err':<6} | Status {resp.status_code}"
                )
        except Exception as e:
            print(f"{rank:<6} | {endpoints[rank]:<25} | {'Err':<6} | {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MPLang2 cluster and job CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'config' subcommand
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_subparsers = config_parser.add_subparsers(
        dest="config_command", help="Config commands"
    )

    # 'config gen'
    gen_parser = config_subparsers.add_parser("gen", help="Generate cluster config")
    gen_parser.add_argument(
        "-w", "--world-size", type=int, default=3, help="Number of workers"
    )
    gen_parser.add_argument(
        "-p", "--base-port", type=int, default=8100, help="Base port"
    )
    gen_parser.add_argument(
        "--spu-base-port", type=int, default=None, help="Base port for SPU"
    )
    gen_parser.add_argument("-o", "--output", type=str, help="Output file path")

    # 'worker' subcommand
    worker_parser = subparsers.add_parser("worker", help="Start a single worker")
    add_cluster_args(worker_parser)
    worker_parser.add_argument(
        "--rank", type=int, required=True, help="Rank of this worker"
    )

    # 'up' subcommand
    up_parser = subparsers.add_parser("up", help="Start local cluster (all workers)")
    add_cluster_args(up_parser)

    # 'status' subcommand
    status_parser = subparsers.add_parser("status", help="Check worker health")
    add_cluster_args(status_parser, include_ports=True)

    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Run a user job")
    add_cluster_args(run_parser, include_ports=True)
    run_parser.add_argument("-f", "--file", required=True, help="Path to user script")
    run_parser.add_argument(
        "--entry",
        default="__mp_main__",
        help="Entry function name in the user script (default: __mp_main__)",
    )
    run_parser.add_argument(
        "--backend",
        choices=["http", "sim"],
        default="http",
        help="Execution backend: http (cluster) or sim (local simulator)",
    )
    run_parser.add_argument(
        "--args",
        nargs="*",
        default=[],
        help="Arguments passed to the entry function",
    )
    run_parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling (only for sim backend)",
    )

    # 'sim' subcommand
    sim_parser = subparsers.add_parser("sim", help="Run a user job in local simulator")
    add_cluster_args(sim_parser, include_ports=False)
    sim_parser.add_argument("-f", "--file", required=True, help="Path to user script")
    sim_parser.add_argument(
        "--entry",
        default="__mp_main__",
        help="Entry function name in the user script (default: __mp_main__)",
    )
    sim_parser.add_argument(
        "--args",
        nargs="*",
        default=[],
        help="Arguments passed to the entry function",
    )
    sim_parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling",
    )

    # 'objects' subcommand
    objects_parser = subparsers.add_parser("objects", help="List objects on workers")
    add_cluster_args(objects_parser, include_ports=False)

    # 'trace' subcommand
    trace_parser = subparsers.add_parser("trace", help="Trace utilities")
    trace_subparsers = trace_parser.add_subparsers(
        dest="trace_command", help="Trace commands"
    )

    # 'trace merge'
    merge_parser = trace_subparsers.add_parser("merge", help="Merge trace files")
    merge_parser.add_argument(
        "pattern", help="Glob pattern for trace files (e.g. 'trace_*.json')"
    )
    merge_parser.add_argument(
        "-o", "--output", default="merged_trace.json", help="Output filename"
    )

    args = parser.parse_args()

    if args.command == "config":
        if args.config_command == "gen":
            cmd_config_gen(args)
        else:
            config_parser.print_help()
    elif args.command == "trace":
        if args.trace_command == "merge":
            cmd_trace_merge(args)
        else:
            trace_parser.print_help()
    elif args.command == "worker":
        cmd_worker(args)
    elif args.command == "up":
        cmd_up(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "sim":
        args.backend = "sim"
        cmd_run(args)
    elif args.command == "objects":
        cmd_objects(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
