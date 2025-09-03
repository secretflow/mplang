#!/usr/bin/env python3
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
Command-line interface for managing MPLang clusters.
"""

import argparse
import json
import multiprocessing
import sys
from typing import Any

import uvicorn

from mplang.runtime.server import app


def load_config(config_path: str) -> dict[Any, Any]:
    """Load configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Dictionary containing the configuration
    """
    with open(config_path) as f:
        return dict[Any, Any](json.load(f))


def run_server(port: int, node_id: str) -> None:
    """Run a uvicorn server on a specific port.

    Args:
        port: The port to run the server on
        node_id: The ID of the node
    """
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": f"%(levelname)s: [{node_id}] %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": f'%(levelname)s: [{node_id}] %(client_addr)s - "%(request_line)s" %(status_code)s',
                "use_colors": None,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_config=log_config,
        ws="none",  # Disable websockets
    )
    server = uvicorn.Server(config)
    server.run()


def start_command(args: argparse.Namespace) -> int:
    """Handle the start command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Load configuration
        config = load_config(args.config)
        nodes = config.get("nodes", {})

        if not nodes:
            print("No nodes defined in configuration")
            return 1

        # Find the endpoint for the specified node
        node_id = args.node_id
        if node_id not in nodes:
            print(f"Node {node_id} not found in configuration")
            return 1

        endpoint = nodes[node_id]
        # Extract port from endpoint (format: host:port)
        port = int(endpoint.split(":")[-1])

        print(f"Starting node {node_id} on port {port}...")
        # Run the server directly (blocking)
        run_server(port, node_id)

        return 0
    except Exception as e:
        print(f"Error starting node: {e}")
        return 1


def start_cluster_command(args: argparse.Namespace) -> int:
    """Handle the start-cluster command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Load configuration
        config = load_config(args.config)
        nodes = config.get("nodes", {})

        if not nodes:
            print("No nodes defined in configuration")
            return 1

        # Start a process for each node
        processes = []
        for node_id, endpoint in nodes.items():
            # Extract port from endpoint (format: host:port)
            port = int(endpoint.split(":")[-1])

            # Create and start process
            process = multiprocessing.Process(target=run_server, args=(port, node_id))
            process.start()
            processes.append((node_id, process))
            print(f"Started node {node_id} on port {port} (PID: {process.pid})")

        print(f"Started {len(processes)} nodes in cluster")
        print("Press Ctrl+C to stop all nodes")

        # Wait for all processes to complete or for interruption
        try:
            for _, process in processes:
                process.join()
        except KeyboardInterrupt:
            print("\nStopping all nodes...")
            for _, process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
            print("All nodes stopped")

        return 0
    except Exception as e:
        print(f"Error starting cluster: {e}")
        return 1


def status_command(args: argparse.Namespace) -> int:
    """Handle the status command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Load configuration
        config = load_config(args.config)
        nodes = config.get("nodes", {})

        if not nodes:
            print("No nodes defined in configuration")
            return 1

        # Import Driver here to avoid unnecessary imports
        from mplang.runtime.driver import Driver

        # Create driver instance
        driver = Driver(nodes)

        # Check status of each node
        print("Node Status:")
        print("-" * 40)
        all_healthy = True

        for node_id, endpoint in nodes.items():
            try:
                is_healthy = driver.ping(node_id)
                status = "HEALTHY" if is_healthy else "UNHEALTHY"
                if not is_healthy:
                    all_healthy = False
                print(f"{node_id:<15} {endpoint:<20} {status}")
            except Exception as e:
                print(f"{node_id:<15} {endpoint:<20} ERROR - {e}")
                all_healthy = False

        return 0 if all_healthy else 1
    except Exception as e:
        print(f"Error checking status: {e}")
        return 1


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="mplang-cli",
        description="Command-line interface for managing MPLang clusters",
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Help command
    subparsers.add_parser("help", help="Show this help message and exit")

    # Status command
    status_parser = subparsers.add_parser(
        "status", help="Check status of nodes in the cluster"
    )
    status_parser.add_argument(
        "--config", "-c", required=True, help="Path to the JSON configuration file"
    )
    status_parser.set_defaults(func=status_command)

    # Start command
    start_parser = subparsers.add_parser("start", help="Start a single MPC node")
    start_parser.add_argument(
        "--config", "-c", required=True, help="Path to the JSON configuration file"
    )
    start_parser.add_argument(
        "--node-id", "-n", required=True, type=str, help="ID of the node to start"
    )
    start_parser.set_defaults(func=start_command)

    # Start cluster command
    start_cluster_parser = subparsers.add_parser(
        "start-cluster", help="Start all MPC nodes in the cluster"
    )
    start_cluster_parser.add_argument(
        "--config", "-c", required=True, help="Path to the JSON configuration file"
    )
    start_cluster_parser.set_defaults(func=start_cluster_command)

    # Up command (alias for start-cluster)
    up_parser = subparsers.add_parser(
        "up", help="Start all MPC nodes in the cluster (alias for start-cluster)"
    )
    up_parser.add_argument(
        "--config", "-c", required=True, help="Path to the JSON configuration file"
    )
    up_parser.set_defaults(func=start_cluster_command)

    # Parse arguments
    args = parser.parse_args()

    # Handle help command
    if args.command == "help" or args.command is None:
        parser.print_help()
        return 0

    # Handle subcommands
    if hasattr(args, "func"):
        result = args.func(args)
        return int(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
