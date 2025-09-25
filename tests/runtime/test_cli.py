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
Tests for the CLI module.
"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from mplang.runtime.cli import (
    load_config,
    start_cluster_command,
    start_command,
    status_command,
)


def create_test_config():
    """Create a valid test configuration."""
    return {
        "nodes": [
            {"name": "P0", "endpoint": "127.0.0.1:9530"},
            {"name": "P1", "endpoint": "127.0.0.1:9531"},
        ],
        "devices": {
            "SPU": {
                "kind": "SPU",
                "members": ["P0", "P1"],
                "config": {"protocol": "ABY3", "field": "FM64"},
            },
            "P0": {"kind": "PPU", "members": ["P0"], "config": {}},
            "P1": {"kind": "PPU", "members": ["P1"], "config": {}},
        },
    }


def test_load_config():
    """Test loading configuration from a YAML file."""
    # Create a temporary config file with correct format
    config_data = create_test_config()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        # Load the config
        loaded_config = load_config(config_path)
        # Check that the loaded config has the expected structure
        assert len(loaded_config.nodes) == 2
        assert "P0" in loaded_config.nodes
        assert "P1" in loaded_config.nodes
        assert loaded_config.nodes["P0"].endpoint == "127.0.0.1:9530"
    finally:
        # Clean up
        os.unlink(config_path)


def test_load_config_invalid_file():
    """Test loading configuration from an invalid file."""
    # Try to load from a non-existent file
    with pytest.raises(FileNotFoundError):
        load_config("/path/that/does/not/exist.yaml")


def test_status_command_success():
    """Test the status command when all nodes are healthy."""
    # Create a temporary config file
    config_data = create_test_config()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        # Mock the HttpExecutorClient
        mock_client = AsyncMock()
        mock_client.endpoint = "http://127.0.0.1:9530"  # Add endpoint attribute
        mock_client.health_check.return_value = True
        mock_client.list_sessions.return_value = ["session1"]
        mock_client.list_computations.return_value = ["comp1", "comp2"]
        mock_client.list_symbols.return_value = ["sym1", "sym2", "sym3"]
        mock_client.close.return_value = None

        with patch("mplang.runtime.cli.HttpExecutorClient", return_value=mock_client):
            # Create mock args
            args = MagicMock()
            args.config = config_path
            args.details = True

            # Run the status command
            result = status_command(args)

            # Check that it returns 0 (success)
            assert result == 0
    finally:
        # Clean up
        os.unlink(config_path)


def test_status_command_unhealthy_node():
    """Test the status command when a node is unhealthy."""
    # Create a temporary config file
    config_data = create_test_config()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        # Mock different health states for different nodes
        def mock_client_side_effect(endpoint, timeout=60):
            mock_client = AsyncMock()
            mock_client.endpoint = f"http://{endpoint}"  # Add endpoint attribute
            if "9530" in endpoint:
                # P0 is healthy
                mock_client.health_check.return_value = True
                mock_client.list_sessions.return_value = ["session1"]
            else:
                # P1 is unhealthy
                mock_client.health_check.return_value = False
                mock_client.list_sessions.return_value = []
            mock_client.close.return_value = None
            return mock_client

        with patch(
            "mplang.runtime.cli.HttpExecutorClient",
            side_effect=mock_client_side_effect,
        ):
            # Create mock args
            args = MagicMock()
            args.config = config_path
            args.details = False

            # Run the status command
            result = status_command(args)

            # Check that it returns 1 (failure due to unhealthy node)
            assert result == 1
    finally:
        # Clean up
        os.unlink(config_path)


def test_status_command_exception():
    """Test the status command when an exception occurs."""
    # Create a temporary config file
    config_data = create_test_config()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        # Mock the HttpExecutorClient to raise an exception
        mock_client = AsyncMock()
        mock_client.endpoint = "http://127.0.0.1:9530"  # Add endpoint attribute
        mock_client.health_check.side_effect = Exception("Connection failed")
        mock_client.close.return_value = None

        with patch("mplang.runtime.cli.HttpExecutorClient", return_value=mock_client):
            # Create mock args
            args = MagicMock()
            args.config = config_path
            args.details = False

            # Run the status command
            result = status_command(args)

            # Check that it returns 1 (failure due to exception)
            assert result == 1
    finally:
        # Clean up
        os.unlink(config_path)


def test_status_command_empty_config():
    """Test the status command with an empty configuration."""
    # Create a temporary config file with no nodes
    config_data = {"nodes": [], "devices": {}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        # Create mock args
        args = MagicMock()
        args.config = config_path

        # Run the status command
        result = status_command(args)

        # Check that it returns 1 (failure due to no nodes)
        assert result == 1
    finally:
        # Clean up
        os.unlink(config_path)


def test_start_command_success():
    """Test the start command when successful."""
    # Create a temporary config file
    config_data = create_test_config()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        # Mock the run_server function to avoid actually starting a server
        with patch("mplang.runtime.cli.run_server") as mock_run_server:
            # Create mock args
            args = MagicMock()
            args.config = config_path
            args.node_id = "P0"

            # Run the start command
            result = start_command(args)

            # Check that it returns 0 (success)
            assert result == 0

            # Check that run_server was called with the correct port and node_id
            mock_run_server.assert_called_once_with(9530, "P0")
    finally:
        # Clean up
        os.unlink(config_path)


def test_start_command_node_not_found():
    """Test the start command when the specified node is not found."""
    # Create a temporary config file
    config_data = create_test_config()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        # Create mock args
        args = MagicMock()
        args.config = config_path
        args.node_id = "P2"  # This node doesn't exist in the config

        # Run the start command
        result = start_command(args)

        # Check that it returns 1 (failure due to node not found)
        assert result == 1
    finally:
        # Clean up
        os.unlink(config_path)


def test_start_command_empty_config():
    """Test the start command with an empty configuration."""
    # Create a temporary config file with no nodes
    config_data = {"nodes": [], "devices": {}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        # Create mock args
        args = MagicMock()
        args.config = config_path
        args.node_id = "P0"

        # Run the start command
        result = start_command(args)

        # Check that it returns 1 (failure due to no nodes)
        assert result == 1
    finally:
        # Clean up
        os.unlink(config_path)


def test_start_cluster_command_success():
    """Test the start-cluster command when successful."""
    # Create a temporary config file
    config_data = create_test_config()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        # Mock the multiprocessing.Process class to avoid actually starting processes
        with patch("mplang.runtime.cli.multiprocessing.Process") as mock_process_class:
            # Create a mock process instance
            mock_process = MagicMock()
            mock_process.pid = 12345
            mock_process_class.return_value = mock_process

            # Create mock args
            args = MagicMock()
            args.config = config_path

            # Run the start_cluster command
            result = start_cluster_command(args)

            # Check that it returns 0 (success)
            assert result == 0

            # Check that Process was called twice (once for each node)
            assert mock_process_class.call_count == 2

            # Check that start was called on each process
            assert mock_process.start.call_count == 2
    finally:
        # Clean up
        os.unlink(config_path)


def test_start_cluster_command_empty_config():
    """Test the start-cluster command with an empty configuration."""
    # Create a temporary config file with no nodes
    config_data = {"nodes": [], "devices": {}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        # Create mock args
        args = MagicMock()
        args.config = config_path

        # Run the start_cluster command
        result = start_cluster_command(args)

        # Check that it returns 1 (failure due to no nodes)
        assert result == 1
    finally:
        # Clean up
        os.unlink(config_path)
