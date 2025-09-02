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
Object-oriented resource name implementation for better user experience.

This version provides strongly-typed resource objects with support for method chaining,
type safety, and convenient parsing/building functionality. Symbol resources are unified
to handle different scopes (global, session, execution) through attributes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar, TypeVar

from mplang.runtime.path_template import PathTemplate

# Type variable for generic resource types
T = TypeVar("T", bound="ResourceName")

# =============================================================================
# Scope Definitions
# =============================================================================


class SymbolScope(Enum):
    """Symbol scope types"""

    GLOBAL = "global"
    SESSION = "session"
    EXECUTION = "execution"


# =============================================================================
# Base Abstract Classes
# =============================================================================


class ResourceName(ABC):
    """Base abstract class for resources"""

    def __init__(self) -> None:
        self._path_cache: str | None = None

    @abstractmethod
    def to_string(self) -> str:
        """Convert to resource path string"""

    @classmethod
    @abstractmethod
    def parse(cls, path: str) -> ResourceName | None:
        """Parse resource object from path string"""

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.to_string()}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResourceName):
            return False
        return self.to_string() == other.to_string()

    def __hash__(self) -> int:
        return hash(self.to_string())


# =============================================================================
# Concrete Resource Implementations
# =============================================================================


class SessionName(ResourceName):
    """Session resource name

    Examples:
        session = SessionName("session_123")
        print(session)  # sessions/session_123

        # Parsing
        parsed = SessionName.parse("sessions/session_123")
        print(parsed.session_id)  # session_123
    """

    _TEMPLATE = PathTemplate("sessions/{session_id}")

    def __init__(self, session_id: str) -> None:
        super().__init__()
        self.session_id = session_id

    def to_string(self) -> str:
        if self._path_cache is None:
            self._path_cache = self._TEMPLATE.render(session_id=self.session_id)
        return self._path_cache

    @classmethod
    def parse(cls, path: str) -> SessionName | None:
        match = cls._TEMPLATE.match(path)
        if match:
            return cls(match["session_id"])
        return None

    def execution(self, execution_id: str) -> ExecutionName:
        """Create an execution resource under this session"""
        return ExecutionName(self.session_id, execution_id)

    def symbol(self, symbol_id: str) -> SymbolName:
        """Create a session-scoped symbol resource"""
        return SymbolName.session_symbol(self.session_id, symbol_id)


class ExecutionName(ResourceName):
    """Execution resource name

    Examples:
        execution = ExecutionName("session_123", "exec_456")
        print(execution)  # sessions/session_123/executions/exec_456

        # Create from session
        session = SessionName("session_123")
        execution = session.execution("exec_456")
    """

    _TEMPLATE = PathTemplate("sessions/{session_id}/executions/{execution_id}")

    def __init__(self, session_id: str, execution_id: str) -> None:
        super().__init__()
        self.session_id = session_id
        self.execution_id = execution_id

    def to_string(self) -> str:
        if self._path_cache is None:
            self._path_cache = self._TEMPLATE.render(
                session_id=self.session_id, execution_id=self.execution_id
            )
        return self._path_cache

    @classmethod
    def parse(cls, path: str) -> ExecutionName | None:
        match = cls._TEMPLATE.match(path)
        if match:
            return cls(match["session_id"], match["execution_id"])
        return None

    def session(self) -> SessionName:
        """Get the parent session resource"""
        return SessionName(self.session_id)

    def symbol(self, symbol_id: str) -> SymbolName:
        """Create an execution-scoped symbol resource"""
        return SymbolName.execution_symbol(
            self.session_id, self.execution_id, symbol_id
        )

    def message(self, msg_id: str, frm_rank: int) -> MessageName:
        """Create a message resource under this execution"""
        return MessageName(self.session_id, self.execution_id, msg_id, frm_rank)


class SymbolName(ResourceName):
    """Unified symbol resource name

    Supports symbols with different scopes: global, session, execution.
    Different scope types are distinguished through attributes and methods.

    Examples:
        # Global symbol
        global_symbol = Symbol("global_sym_123")
        print(global_symbol.is_global())  # True

        # Session-scoped symbol
        session_symbol = Symbol("session_sym_456", session_id="session_123")
        print(session_symbol.is_session_scoped())  # True
        print(session_symbol.session_id)  # "session_123"

        # Execution-scoped symbol
        exec_symbol = Symbol("exec_sym_789", session_id="session_123", execution_id="exec_456")
        print(exec_symbol.is_execution_scoped())  # True
        print(exec_symbol.scope)  # SymbolScope.EXECUTION
    """

    _GLOBAL_TEMPLATE = PathTemplate("symbols/{symbol_id}")
    _SESSION_TEMPLATE = PathTemplate("sessions/{session_id}/symbols/{symbol_id}")
    _EXECUTION_TEMPLATE = PathTemplate(
        "sessions/{session_id}/executions/{execution_id}/symbols/{symbol_id}"
    )

    def __init__(
        self,
        symbol_id: str,
        session_id: str | None = None,
        execution_id: str | None = None,
    ) -> None:
        super().__init__()
        self.symbol_id = symbol_id
        self.session_id = session_id
        self.execution_id = execution_id

        # Validate parameter logic
        if execution_id and not session_id:
            raise ValueError("execution_id requires session_id")

    @property
    def scope(self) -> SymbolScope:
        """Get the symbol's scope type"""
        if self.execution_id:
            return SymbolScope.EXECUTION
        elif self.session_id:
            return SymbolScope.SESSION
        else:
            return SymbolScope.GLOBAL

    def is_global(self) -> bool:
        """Check if this is a global scope symbol"""
        return self.scope == SymbolScope.GLOBAL

    def is_session_scoped(self) -> bool:
        """Check if this is a session scope symbol"""
        return self.scope == SymbolScope.SESSION

    def is_execution_scoped(self) -> bool:
        """Check if this is an execution scope symbol"""
        return self.scope == SymbolScope.EXECUTION

    def to_string(self) -> str:
        if self._path_cache is None:
            if self.is_execution_scoped():
                assert self.session_id is not None, (
                    "session_id should not be None for execution scoped symbols"
                )
                assert self.execution_id is not None, (
                    "execution_id should not be None for execution scoped symbols"
                )
                self._path_cache = self._EXECUTION_TEMPLATE.render(
                    session_id=self.session_id,
                    execution_id=self.execution_id,
                    symbol_id=self.symbol_id,
                )
            elif self.is_session_scoped():
                assert self.session_id is not None, (
                    "session_id should not be None for session scoped symbols"
                )
                self._path_cache = self._SESSION_TEMPLATE.render(
                    session_id=self.session_id, symbol_id=self.symbol_id
                )
            else:  # global
                self._path_cache = self._GLOBAL_TEMPLATE.render(
                    symbol_id=self.symbol_id
                )
        return self._path_cache

    @classmethod
    def parse(cls, path: str) -> SymbolName | None:
        """Parse symbol from path, automatically detecting scope"""
        # Try execution scope
        if (match := cls._EXECUTION_TEMPLATE.match(path)) is not None:
            return cls(
                symbol_id=match["symbol_id"],
                session_id=match["session_id"],
                execution_id=match["execution_id"],
            )

        # Try session scope
        if (match := cls._SESSION_TEMPLATE.match(path)) is not None:
            return cls(symbol_id=match["symbol_id"], session_id=match["session_id"])

        # Try global scope
        if (match := cls._GLOBAL_TEMPLATE.match(path)) is not None:
            return cls(symbol_id=match["symbol_id"])

        return None

    @classmethod
    def global_symbol(cls, symbol_id: str) -> SymbolName:
        """Create a global scope symbol"""
        return cls(symbol_id)

    @classmethod
    def session_symbol(cls, session_id: str, symbol_id: str) -> SymbolName:
        """Create a session scope symbol"""
        return cls(symbol_id, session_id=session_id)

    @classmethod
    def execution_symbol(
        cls, session_id: str, execution_id: str, symbol_id: str
    ) -> SymbolName:
        """Create an execution scope symbol"""
        return cls(symbol_id, session_id=session_id, execution_id=execution_id)

    def session(self) -> SessionName | None:
        """Get the parent session resource (if any)"""
        if self.session_id:
            return SessionName(self.session_id)
        return None

    def execution(self) -> ExecutionName | None:
        """Get the parent execution resource (if any)"""
        if self.session_id and self.execution_id:
            return ExecutionName(self.session_id, self.execution_id)
        return None

    def to_session_scope(self) -> SymbolName:
        """Convert to session scope symbol (if currently execution scope)"""
        if not self.is_execution_scoped():
            raise ValueError(
                "Only execution-scoped symbols can be converted to session scope"
            )
        return SymbolName(self.symbol_id, session_id=self.session_id)

    def to_global_scope(self) -> SymbolName:
        """Convert to global scope symbol"""
        return SymbolName(self.symbol_id)


class MessageName(ResourceName):
    """Message resource name"""

    _TEMPLATE = PathTemplate(
        "sessions/{session_id}/executions/{execution_id}/msgs/{msg_id}/frm/{frm_rank}"
    )

    def __init__(
        self, session_id: str, execution_id: str, msg_id: str, frm_rank: int
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self.execution_id = execution_id
        self.msg_id = msg_id
        self.frm_rank = frm_rank

    def to_string(self) -> str:
        if self._path_cache is None:
            self._path_cache = self._TEMPLATE.render(
                session_id=self.session_id,
                execution_id=self.execution_id,
                msg_id=self.msg_id,
                frm_rank=str(self.frm_rank),
            )
        return self._path_cache

    @classmethod
    def parse(cls, path: str) -> MessageName | None:
        match = cls._TEMPLATE.match(path)
        if match:
            try:
                frm_rank = int(match["frm_rank"])
                return cls(
                    match["session_id"],
                    match["execution_id"],
                    match["msg_id"],
                    frm_rank,
                )
            except (ValueError, TypeError):
                return None
        return None

    def session(self) -> SessionName:
        """Get the parent session resource"""
        return SessionName(self.session_id)

    def execution(self) -> ExecutionName:
        """Get the parent execution resource"""
        return ExecutionName(self.session_id, self.execution_id)


# =============================================================================
# Generic Resource Parser
# =============================================================================


class ResourceParser:
    """Generic resource parser that can parse any type of resource path"""

    _RESOURCE_TYPES: ClassVar[list[type[ResourceName]]] = [
        MessageName,
        SymbolName,
        ExecutionName,
        SessionName,
    ]

    @classmethod
    def parse(cls, path: str) -> ResourceName | None:
        """Try to parse the path into the corresponding resource object"""
        for resource_type in cls._RESOURCE_TYPES:
            result = resource_type.parse(path)
            if result:
                return result
        return None

    @classmethod
    def get_resource_type(cls, path: str) -> str | None:
        """Get the resource type name"""
        resource = cls.parse(path)
        if resource:
            return resource.__class__.__name__
        return None


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def session(session_id: str) -> SessionName:
    """Create a session resource"""
    return SessionName(session_id)


def execution(session_id: str, execution_id: str) -> ExecutionName:
    """Create an execution resource"""
    return ExecutionName(session_id, execution_id)


def symbol(symbol_id: str) -> SymbolName:
    """Create a global symbol resource"""
    return SymbolName.global_symbol(symbol_id)


def parse_resource(path: str) -> ResourceName | None:
    """Parse any resource path"""
    return ResourceParser.parse(path)
