#!/usr/bin/env python3
"""
Command-line interface for managing MPLang2 workers.

Usage:
    # Start workers for a 3-party cluster
    python -m mplang2.backends.cli up --world-size 3 --base-port 8100

    # Or with a config file
    python -m mplang2.backends.cli up -c cluster.yaml
"""

import argparse
import multiprocessing
import signal
import sys
from typing import Any

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
    from mplang2.backends.simp_http import create_worker_app

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


def cmd_up(args: argparse.Namespace) -> None:
    """Start worker servers."""
    spu_endpoints: dict[int, str] | None = None

    if args.config:
        # Load from config file
        with open(args.config) as f:
            conf = yaml.safe_load(f)
        nodes = conf.get("nodes", [])
        world_size = len(nodes)
        endpoints = []
        ports = []
        for node in nodes:
            endpoint = node["endpoint"]
            endpoints.append(f"http://{endpoint}")
            # Extract port from endpoint
            port = int(endpoint.split(":")[-1])
            ports.append(port)

        # Check for SPU device and extract BRPC endpoints
        devices = conf.get("devices", {})
        for _dev_name, dev_conf in devices.items():
            if dev_conf.get("kind", "").upper() == "SPU":
                # Build SPU endpoints from node endpoints
                # SPU BRPC typically uses a different port (e.g., base_port + 1000)
                spu_endpoints = {}
                spu_base_port = args.spu_base_port or (ports[0] + 1000)
                for i, node in enumerate(nodes):
                    # Use hostname from endpoint but different port for BRPC
                    host = node["endpoint"].split(":")[0]
                    spu_endpoints[i] = f"{host}:{spu_base_port + i}"
                break
    else:
        # Use command line args
        world_size = args.world_size
        base_port = args.base_port
        ports = [base_port + i for i in range(world_size)]
        endpoints = [f"http://127.0.0.1:{p}" for p in ports]

        # Generate SPU endpoints if requested
        if args.spu_base_port:
            spu_endpoints = {
                i: f"127.0.0.1:{args.spu_base_port + i}" for i in range(world_size)
            }

    print(f"Starting {world_size} workers...")
    for i, endpoint in enumerate(endpoints):
        print(f"  Worker {i}: {endpoint}")
    if spu_endpoints:
        print("SPU BRPC endpoints:")
        for rank, ep in spu_endpoints.items():
            print(f"  Rank {rank}: {ep}")

    # Start workers in separate processes
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

    # Wait for all processes
    for p in processes:
        p.join()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MPLang2 Worker CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start 3 workers on ports 8100, 8101, 8102
  python -m mplang2.backends.cli up --world-size 3 --base-port 8100

  # Start workers with SPU BRPC enabled
  python -m mplang2.backends.cli up -w 3 -p 8100 --spu-base-port 9100

  # Start workers from config file
  python -m mplang2.backends.cli up -c examples/conf/3pc.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 'up' subcommand
    up_parser = subparsers.add_parser("up", help="Start worker servers")
    up_parser.add_argument(
        "-c", "--config", type=str, help="Path to cluster config YAML file"
    )
    up_parser.add_argument(
        "-w", "--world-size", type=int, default=3, help="Number of workers (default: 3)"
    )
    up_parser.add_argument(
        "-p", "--base-port", type=int, default=8100, help="Base port (default: 8100)"
    )
    up_parser.add_argument(
        "--spu-base-port",
        type=int,
        default=None,
        help="Base port for SPU BRPC link. If set, enables SPU BRPC mode.",
    )

    args = parser.parse_args()

    if args.command == "up":
        cmd_up(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
