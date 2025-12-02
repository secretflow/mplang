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
import asyncio
import multiprocessing
import sys
from typing import Any

import uvicorn
import yaml

from mplang.v1.core import ClusterSpec
from mplang.v1.runtime.client import HttpExecutorClient
from mplang.v1.runtime.server import app


def load_config(config_path: str) -> ClusterSpec:
    """Load configuration from a YAML file."""
    with open(config_path) as file:
        conf = yaml.safe_load(file)
        return ClusterSpec.from_dict(conf)


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
        cluster_spec = load_config(args.config)
        nodes = cluster_spec.nodes

        if not nodes:
            print("No nodes defined in configuration")
            return 1

        # Find the endpoint for the specified node
        node_id = args.node_id
        if node_id not in nodes:
            print(f"Node {node_id} not found in configuration")
            return 1

        endpoint = nodes[node_id].endpoint
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
        cluster_spec = load_config(args.config)
        nodes = cluster_spec.nodes

        if not nodes:
            print("No nodes defined in configuration")
            return 1

        # Start a process for each node
        processes = []
        for node_id, node in nodes.items():
            # Extract port from endpoint (format: host:port)
            port = int(node.endpoint.split(":")[-1])

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

    async def _get_node_status(
        node_id: str, endpoint: str, details: int = 0, timeout: int = 60
    ) -> dict[str, Any]:
        """Get status information for a single node.

        Args:
            node_id: Identifier for the node
            endpoint: HTTP endpoint of the node
            details: Verbosity level (0=basic, 1=-v, 2=-vv)
            timeout: HTTP request timeout in seconds (default: 60)
        """

        client = HttpExecutorClient(endpoint, timeout)
        status: dict[str, Any] = {
            "node_id": node_id,
            "endpoint": client.endpoint,  # Use the normalized endpoint from client
            "healthy": False,
            "sessions": [],
            "error": None,
        }

        try:
            # Check node health
            status["healthy"] = await client.health_check()

            if status["healthy"]:
                # Get sessions on this node
                sessions = await client.list_sessions()
                status["sessions"] = sessions

                # Get detailed session info based on verbosity level
                # details=1 (-v): show session names and basic counts
                # details=2 (-vv): show full computation and symbol lists
                if details >= 1:
                    session_details = []
                    for session_name in sessions:
                        try:
                            # Get computations and symbols for each session
                            computations = await client.list_computations(session_name)
                            symbols = await client.list_symbols(session_name)
                            session_info = {
                                "name": session_name,
                                "computations": len(computations),
                                "symbols": len(symbols),
                            }
                            # Include full lists only at -vv level
                            if details >= 2:
                                session_info["computation_list"] = computations
                                session_info["symbol_list"] = symbols
                            session_details.append(session_info)
                        except Exception as e:
                            session_details.append({
                                "name": session_name,
                                "error": str(e),
                            })
                    status["session_details"] = session_details

        except Exception as e:
            status["error"] = str(e)

        finally:
            await client.close()

        return status

    async def _collect_cluster_status(
        nodes: dict[str, str], details: int = 0
    ) -> list[dict[str, Any] | BaseException]:
        """Collect status from all nodes concurrently.

        Args:
            nodes: Dictionary mapping node IDs to their HTTP endpoints
            details: Verbosity level (0=basic, 1=-v, 2=-vv)

        Returns:
            List of status dictionaries or exceptions for each node
        """
        tasks = [
            _get_node_status(node_id, endpoint, details)
            for node_id, endpoint in nodes.items()
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    try:
        # Load configuration
        cluster_spec = load_config(args.config)
        nodes = cluster_spec.nodes

        if not nodes:
            print("No nodes defined in configuration")
            return 1

        node_addrs = {node_id: node.endpoint for node_id, node in nodes.items()}

        # Collect status from all nodes
        verbosity = getattr(args, "verbose", 0)
        cluster_status = asyncio.run(_collect_cluster_status(node_addrs, verbosity))

        # Basic node health check
        print("Node Status:")
        print("-" * 50)
        all_healthy = True

        valid_statuses = []
        for status in cluster_status:
            if isinstance(status, BaseException):
                print(f"{'UNKNOWN':<15} {'UNKNOWN':<20} ERROR - {status}")
                all_healthy = False
                continue

            valid_statuses.append(status)
            node_id = status["node_id"]
            endpoint = status["endpoint"]

            if status["error"]:
                print(f"{node_id:<15} {endpoint:<20} ERROR - {status['error']}")
                all_healthy = False
            elif status["healthy"]:
                session_count = len(status["sessions"])
                print(
                    f"{node_id:<15} {endpoint:<20} HEALTHY ({session_count} sessions)"
                )
            else:
                print(f"{node_id:<15} {endpoint:<20} UNHEALTHY")
                all_healthy = False

        # If verbose mode is enabled, show detailed information
        if verbosity >= 1 and valid_statuses:
            print("\nDetailed Runtime Status:")
            print("-" * 50)

            for status in valid_statuses:
                node_id = status["node_id"]

                if status["error"]:
                    print(f"{node_id}: Error - {status['error']}")
                    continue

                if not status["healthy"]:
                    print(f"{node_id}: Node is unhealthy")
                    continue

                sessions = status.get("session_details", status["sessions"])
                print(f"{node_id}: {len(sessions)} session(s)")

                if isinstance(sessions, list) and sessions:
                    for session in sessions:
                        if isinstance(session, str):
                            # Simple session name only
                            print(f"  - Session '{session}'")
                        elif isinstance(session, dict):
                            # Detailed session info
                            session_name = session["name"]
                            if "error" in session:
                                print(
                                    f"  - Session '{session_name}': Error - {session['error']}"
                                )
                            else:
                                computations = session.get("computations", 0)
                                symbols = session.get("symbols", 0)
                                print(
                                    f"  - Session '{session_name}': {computations} computations, {symbols} symbols"
                                )
                                # At -vv level, show the actual lists
                                if verbosity >= 2:
                                    comp_list = session.get("computation_list", [])
                                    symbol_list = session.get("symbol_list", [])
                                    if comp_list:
                                        print(f"    Computations: {comp_list}")
                                    if symbol_list:
                                        print(f"    Symbols: {symbol_list}")
                elif not sessions:
                    print("  - No active sessions")

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
        "--config", "-c", required=True, help="Path to the YAML configuration file"
    )
    status_parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity: -v for session details, -vv for full lists",
    )
    status_parser.set_defaults(func=status_command)

    # Start command
    start_parser = subparsers.add_parser("start", help="Start a single MPC node")
    start_parser.add_argument(
        "--config", "-c", required=True, help="Path to the YAML configuration file"
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
        "--config", "-c", required=True, help="Path to the YAML configuration file"
    )
    start_cluster_parser.set_defaults(func=start_cluster_command)

    # Up command (alias for start-cluster)
    up_parser = subparsers.add_parser(
        "up", help="Start all MPC nodes in the cluster (alias for start-cluster)"
    )
    up_parser.add_argument(
        "--config", "-c", required=True, help="Path to the YAML configuration file"
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
