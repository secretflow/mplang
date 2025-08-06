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

"""Unit tests for the executor resource module."""

import pytest

from mplang.runtime.executor.resource import (
    ExecutionName,
    MessageName,
    ResourceName,
    ResourceParser,
    SessionName,
    SymbolName,
    SymbolScope,
    execution,
    parse_resource,
    session,
    symbol,
)


class TestSessionName:
    """Test SessionName resource"""

    def test_create_session_name(self):
        """Test creating a session name"""
        session_name = SessionName("test-session")
        assert session_name.session_id == "test-session"
        assert session_name.to_string() == "sessions/test-session"

    def test_parse_valid_session_path(self):
        """Test parsing a valid session path"""
        path = "sessions/my-session"
        session_name = SessionName.parse(path)
        assert session_name is not None
        assert session_name.session_id == "my-session"

    def test_parse_invalid_session_path(self):
        """Test parsing an invalid session path"""
        invalid_paths = [
            "invalid/path",
            "sessions/",
            "",
            "executions/test",
        ]
        for path in invalid_paths:
            result = SessionName.parse(path)
            assert result is None

    def test_session_equality(self):
        """Test session name equality"""
        session1 = SessionName("test")
        session2 = SessionName("test")
        session3 = SessionName("different")

        assert session1 == session2
        assert session1 != session3
        assert hash(session1) == hash(session2)
        assert hash(session1) != hash(session3)


class TestExecutionName:
    """Test ExecutionName resource"""

    def test_create_execution_name(self):
        """Test creating an execution name"""
        exec_name = ExecutionName("session-1", "exec-1")
        assert exec_name.session_id == "session-1"
        assert exec_name.execution_id == "exec-1"
        assert exec_name.to_string() == "sessions/session-1/executions/exec-1"

    def test_parse_valid_execution_path(self):
        """Test parsing a valid execution path"""
        path = "sessions/my-session/executions/my-execution"
        exec_name = ExecutionName.parse(path)
        assert exec_name is not None
        assert exec_name.session_id == "my-session"
        assert exec_name.execution_id == "my-execution"

    def test_parse_invalid_execution_path(self):
        """Test parsing an invalid execution path"""
        invalid_paths = [
            "sessions/test",
            "sessions/test/executions/",
            "invalid/path",
            "",
        ]
        for path in invalid_paths:
            result = ExecutionName.parse(path)
            assert result is None

    def test_execution_session_parent(self):
        """Test getting parent session from execution"""
        exec_name = ExecutionName("session-1", "exec-1")
        session = exec_name.session()
        assert isinstance(session, SessionName)
        assert session.session_id == "session-1"

    def test_execution_equality(self):
        """Test execution name equality"""
        exec1 = ExecutionName("session", "exec")
        exec2 = ExecutionName("session", "exec")
        exec3 = ExecutionName("session", "different")

        assert exec1 == exec2
        assert exec1 != exec3


class TestSymbol:
    """Test Symbol resource"""

    def test_create_global_symbol(self):
        """Test creating a global symbol"""
        global_sym = SymbolName.global_symbol("my-symbol")
        assert global_sym.symbol_id == "my-symbol"
        assert global_sym.scope == SymbolScope.GLOBAL
        assert global_sym.session_id is None
        assert global_sym.execution_id is None
        assert global_sym.to_string() == "symbols/my-symbol"

    def test_create_session_symbol(self):
        """Test creating a session-scoped symbol"""
        session_sym = SymbolName.session_symbol("session-1", "my-symbol")
        assert session_sym.symbol_id == "my-symbol"
        assert session_sym.scope == SymbolScope.SESSION
        assert session_sym.session_id == "session-1"
        assert session_sym.execution_id is None
        assert session_sym.to_string() == "sessions/session-1/symbols/my-symbol"

    def test_create_execution_symbol(self):
        """Test creating an execution-scoped symbol"""
        exec_sym = SymbolName.execution_symbol("session-1", "exec-1", "my-symbol")
        assert exec_sym.symbol_id == "my-symbol"
        assert exec_sym.scope == SymbolScope.EXECUTION
        assert exec_sym.session_id == "session-1"
        assert exec_sym.execution_id == "exec-1"
        assert (
            exec_sym.to_string()
            == "sessions/session-1/executions/exec-1/symbols/my-symbol"
        )

    def test_parse_global_symbol(self):
        """Test parsing a global symbol path"""
        path = "symbols/my-global-symbol"
        symbol = SymbolName.parse(path)
        assert symbol is not None
        assert symbol.symbol_id == "my-global-symbol"
        assert symbol.scope == SymbolScope.GLOBAL

    def test_parse_session_symbol(self):
        """Test parsing a session symbol path"""
        path = "sessions/session-1/symbols/my-session-symbol"
        symbol = SymbolName.parse(path)
        assert symbol is not None
        assert symbol.symbol_id == "my-session-symbol"
        assert symbol.scope == SymbolScope.SESSION
        assert symbol.session_id == "session-1"

    def test_parse_execution_symbol(self):
        """Test parsing an execution symbol path"""
        path = "sessions/session-1/executions/exec-1/symbols/my-exec-symbol"
        symbol = SymbolName.parse(path)
        assert symbol is not None
        assert symbol.symbol_id == "my-exec-symbol"
        assert symbol.scope == SymbolScope.EXECUTION
        assert symbol.session_id == "session-1"
        assert symbol.execution_id == "exec-1"

    def test_parse_invalid_symbol_path(self):
        """Test parsing invalid symbol paths"""
        invalid_paths = [
            "symbols/",
            "sessions/test/symbols/",
            "sessions/test/executions/exec/symbols/",
            "invalid/path",
            "",
        ]
        for path in invalid_paths:
            result = SymbolName.parse(path)
            assert result is None

    def test_symbol_scope_conversion(self):
        """Test converting between symbol scopes"""
        # Start with execution-scoped symbol
        exec_sym = SymbolName.execution_symbol("session-1", "exec-1", "test-symbol")

        # Convert to session scope
        session_sym = exec_sym.to_session_scope()
        assert session_sym.scope == SymbolScope.SESSION
        assert session_sym.session_id == "session-1"
        assert session_sym.execution_id is None

        # Convert to global scope
        global_sym = exec_sym.to_global_scope()
        assert global_sym.scope == SymbolScope.GLOBAL
        assert global_sym.session_id is None
        assert global_sym.execution_id is None

    def test_symbol_parent_resources(self):
        """Test getting parent resources from symbols"""
        exec_sym = SymbolName.execution_symbol("session-1", "exec-1", "test-symbol")

        # Get parent session
        session = exec_sym.session()
        assert session is not None
        assert session.session_id == "session-1"

        # Get parent execution
        execution = exec_sym.execution()
        assert execution is not None
        assert execution.session_id == "session-1"
        assert execution.execution_id == "exec-1"

        # Global symbol has no parents
        global_sym = SymbolName.global_symbol("global-symbol")
        assert global_sym.session() is None
        assert global_sym.execution() is None

    def test_symbol_equality(self):
        """Test symbol equality"""
        sym1 = SymbolName.global_symbol("test")
        sym2 = SymbolName.global_symbol("test")
        sym3 = SymbolName.session_symbol("session", "test")

        assert sym1 == sym2
        assert sym1 != sym3


class TestMessageName:
    """Test MessageName resource"""

    def test_create_message_name(self):
        """Test creating a message name"""
        msg = MessageName("session-1", "exec-1", "msg-1", 0)
        assert msg.session_id == "session-1"
        assert msg.execution_id == "exec-1"
        assert msg.msg_id == "msg-1"
        assert msg.frm_rank == 0
        assert (
            msg.to_string() == "sessions/session-1/executions/exec-1/msgs/msg-1/frm/0"
        )

    def test_parse_valid_message_path(self):
        """Test parsing a valid message path"""
        path = "sessions/my-session/executions/my-exec/msgs/my-msg/frm/2"
        msg = MessageName.parse(path)
        assert msg is not None
        assert msg.session_id == "my-session"
        assert msg.execution_id == "my-exec"
        assert msg.msg_id == "my-msg"
        assert msg.frm_rank == 2

    def test_parse_invalid_message_path(self):
        """Test parsing invalid message paths"""
        invalid_paths = [
            "sessions/test/executions/exec/msgs/msg/frm/",
            "sessions/test/executions/exec/msgs/msg/frm/abc",  # non-numeric rank
            "sessions/test/executions/exec/msgs/",
            "invalid/path",
            "",
        ]
        for path in invalid_paths:
            result = MessageName.parse(path)
            assert result is None

    def test_message_parent_resources(self):
        """Test getting parent resources from message"""
        msg = MessageName("session-1", "exec-1", "msg-1", 0)

        # Get parent session
        session = msg.session()
        assert session.session_id == "session-1"

        # Get parent execution
        execution = msg.execution()
        assert execution.session_id == "session-1"
        assert execution.execution_id == "exec-1"

    def test_message_equality(self):
        """Test message name equality"""
        msg1 = MessageName("session", "exec", "msg", 0)
        msg2 = MessageName("session", "exec", "msg", 0)
        msg3 = MessageName("session", "exec", "msg", 1)

        assert msg1 == msg2
        assert msg1 != msg3


class TestResourceParser:
    """Test ResourceParser"""

    def test_parse_various_resource_types(self):
        """Test parsing different resource types"""
        test_cases = [
            ("sessions/test", SessionName),
            ("sessions/test/executions/exec", ExecutionName),
            ("symbols/global-sym", SymbolName),
            ("sessions/test/symbols/session-sym", SymbolName),
            ("sessions/test/executions/exec/symbols/exec-sym", SymbolName),
            ("sessions/test/executions/exec/msgs/msg/frm/0", MessageName),
        ]

        for path, expected_type in test_cases:
            result = ResourceParser.parse(path)
            assert result is not None
            assert isinstance(result, expected_type)

    def test_parse_invalid_resource(self):
        """Test parsing invalid resource paths"""
        invalid_paths = [
            "invalid/path",
            "",
            "sessions/",
            "unknown/resource/type",
        ]

        for path in invalid_paths:
            result = ResourceParser.parse(path)
            assert result is None

    def test_get_resource_type(self):
        """Test getting resource type names"""
        test_cases = [
            ("sessions/test", "SessionName"),
            ("sessions/test/executions/exec", "ExecutionName"),
            ("symbols/global-sym", "SymbolName"),
            ("sessions/test/executions/exec/msgs/msg/frm/0", "MessageName"),
        ]

        for path, expected_type_name in test_cases:
            type_name = ResourceParser.get_resource_type(path)
            assert type_name == expected_type_name

        # Test invalid path
        assert ResourceParser.get_resource_type("invalid/path") is None


class TestFactoryFunctions:
    """Test convenience factory functions"""

    def test_session_factory(self):
        """Test session factory function"""
        session_name = session("test-session")
        assert isinstance(session_name, SessionName)
        assert session_name.session_id == "test-session"

    def test_execution_factory(self):
        """Test execution factory function"""
        exec_name = execution("test-session", "test-execution")
        assert isinstance(exec_name, ExecutionName)
        assert exec_name.session_id == "test-session"
        assert exec_name.execution_id == "test-execution"

    def test_symbol_factory(self):
        """Test symbol factory function"""
        sym = symbol("test-symbol")
        assert isinstance(sym, SymbolName)
        assert sym.symbol_id == "test-symbol"
        assert sym.scope == SymbolScope.GLOBAL

    def test_parse_resource_factory(self):
        """Test parse_resource factory function"""
        # Test valid resource
        result = parse_resource("sessions/test")
        assert isinstance(result, SessionName)

        # Test invalid resource
        result = parse_resource("invalid/path")
        assert result is None


class TestResourceCaching:
    """Test resource path caching"""

    def test_path_caching(self):
        """Test that resource paths are cached"""
        session_name = SessionName("test")

        # First call should compute the path
        path1 = session_name.to_string()

        # Second call should use cached path
        path2 = session_name.to_string()

        assert path1 == path2
        assert path1 == "sessions/test"

    def test_cache_invalidation(self):
        """Test that cache is properly managed"""
        symbol = SymbolName.global_symbol("test")
        path1 = symbol.to_string()

        # Create another symbol - should have its own cache
        symbol2 = SymbolName.global_symbol("test2")
        path2 = symbol2.to_string()

        assert path1 != path2
        assert path1 == "symbols/test"
        assert path2 == "symbols/test2"
