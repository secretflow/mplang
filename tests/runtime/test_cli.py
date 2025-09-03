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

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from mplang.runtime.cli import (
    load_config,
    start_cluster_command,
    start_command,
    status_command,
)
from mplang.runtime.driver import Driver


def test_load_config():
    """Test loading configuration from a JSON file."""
    # Create a temporary config file
    config_data = {
        "nodes": {
            "node:0": "127.0.0.1:9530",
            "node:1": "127.0.0.1:9531",
            "node:2": "127.0.0.1:9532",
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name

    try:
        # Load the config
        loaded_config = load_config(config_path)
        assert loaded_config == config_data
    finally:
        # Clean up
        os.unlink(config_path)


def test_load_config_invalid_file():
    """Test loading configuration from an invalid file."""
    # Try to load from a non-existent file
    with pytest.raises(FileNotFoundError):
        load_config("/path/that/does/not/exist.json")


def test_status_command_success():
    """Test the status command when all nodes are healthy."""
    # Create a temporary config file
    config_data = {"nodes": {"node:0": "127.0.0.1:9530", "node:1": "127.0.0.1:9531"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name

    try:
        # Mock the Driver.ping method to return True (healthy)
        with patch.object(Driver, "ping", return_value=True):
            # Create mock args
            args = MagicMock()
            args.config = config_path

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
    config_data = {"nodes": {"node:0": "127.0.0.1:9530", "node:1": "127.0.0.1:9531"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name

    try:
        # Mock the Driver.ping method to return False for one node
        def mock_ping(node_id):
            return node_id == 0  # Node 0 is healthy, node 1 is not

        with patch.object(Driver, "ping", side_effect=mock_ping):
            # Create mock args
            args = MagicMock()
            args.config = config_path

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
    config_data = {"nodes": {"node:0": "127.0.0.1:9530", "node:1": "127.0.0.1:9531"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name

    try:
        # Mock the Driver.ping method to raise an exception
        with patch.object(Driver, "ping", side_effect=Exception("Connection failed")):
            # Create mock args
            args = MagicMock()
            args.config = config_path

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
    config_data = {"nodes": {}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
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
    config_data = {"nodes": {"node:0": "127.0.0.1:9530", "node:1": "127.0.0.1:9531"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name

    try:
        # Mock the run_server function to avoid actually starting a server
        with patch("mplang.runtime.cli.run_server") as mock_run_server:
            # Create mock args
            args = MagicMock()
            args.config = config_path
            args.node_id = 0

            # Run the start command
            result = start_command(args)

            # Check that it returns 0 (success)
            assert result == 0

            # Check that run_server was called with the correct port
            mock_run_server.assert_called_once_with(9530)
    finally:
        # Clean up
        os.unlink(config_path)


def test_start_command_node_not_found():
    """Test the start command when the specified node is not found."""
    # Create a temporary config file
    config_data = {"nodes": {"node:0": "127.0.0.1:9530", "node:1": "127.0.0.1:9531"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name

    try:
        # Create mock args
        args = MagicMock()
        args.config = config_path
        args.node_id = 2  # This node doesn't exist in the config

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
    config_data = {"nodes": {}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name

    try:
        # Create mock args
        args = MagicMock()
        args.config = config_path
        args.node_id = 0

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
    config_data = {"nodes": {"node:0": "127.0.0.1:9530", "node:1": "127.0.0.1:9531"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
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
    config_data = {"nodes": {}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
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
