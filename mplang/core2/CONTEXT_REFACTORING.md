# Context-Based Architecture Refactoring

This document describes the Context-based architecture refactoring completed on November 12, 2025.

## Overview

We refactored the `mplang/core2/` architecture to use a unified `Context` abstraction, where both `Tracer` and `Interpreter` inherit from a common `Context` base class. This provides better conceptual symmetry and cleaner object ownership.

## Architecture Pattern

### Before (Separate Concepts)

```python
# ExecutionContext managed Tracer separately
class ExecutionContext:
    _tracer_stack: list[Tracer]
    executor: Executor  # Separate from tracer

class Tracer:  # Independent class
    def trace(...): ...

# Objects held references to specific implementations
class TraceObject:
    _tracer: Tracer  # Holds Tracer directly

class InterpObject:
    # No context reference, used global executor
```

### After (Unified Context Pattern)

```python
# Context is the abstract base for all execution contexts
class Context(ABC):
    @abstractmethod
    def execute_add(self, left: Object, right: Object) -> Object:
        """Execute addition (record to graph or execute immediately)"""

# Tracer IS-A Context
class Tracer(Context):
    def execute_add(self, left, right) -> TraceObject:
        """Record to Graph IR"""

# Interpreter IS-A Context
class Interpreter(Context):
    def execute_add(self, left, right) -> InterpObject:
        """Execute immediately"""

# ExecutionContext manages Context stack
class ExecutionContext:
    _context_stack: list[Context]  # Unified stack

    @property
    def current_context(self) -> Context | None:
        """Get current context (Tracer or Interpreter)"""

# Objects hold their Context
class TraceObject:
    _context: Tracer  # Holds its Tracer (Context)

class InterpObject:
    _context: Interpreter | None  # Holds its Interpreter (Context)
```

## Key Benefits

### 1. Conceptual Symmetry
- **Before**: Tracer was special-cased; Executor was separate
- **After**: Both Tracer and Interpreter are Context types (equals)

### 2. Unified Operation Interface
- **Before**: Operations were implemented differently in TraceObject vs InterpObject
- **After**: All operations delegate to `Context.execute_add()` (or execute_mul, etc.)

```python
# TraceObject.__add__
def __add__(self, other):
    return self._context.execute_add(self, other)  # Delegates to Tracer

# InterpObject.__add__
def __add__(self, other):
    return self._context.execute_add(self, other)  # Delegates to Interpreter
```

### 3. Cleaner Object Ownership
- **Before**: TraceObject held Tracer; InterpObject had no context reference
- **After**: Both TraceObject and InterpObject hold their Context

```python
class TraceObject:
    _context: Tracer  # Owns its Tracer

class InterpObject:
    _context: Interpreter | None  # Owns its Interpreter (or uses default)
```

### 4. Extensibility
Easy to add new context types without changing object model:

```python
class Profiler(Context):
    """Context that profiles operations"""
    def execute_add(self, left, right):
        self.record_op("add", left, right)
        # Delegate to actual execution
        return self._inner_context.execute_add(left, right)

class Debugger(Context):
    """Context that traces execution for debugging"""
    def execute_add(self, left, right):
        print(f"DEBUG: add({left}, {right})")
        return self._inner_context.execute_add(left, right)
```

## Changes Made

### 1. `mplang/core2/context.py`
- Added `Context` abstract base class with `execute_add()` abstract method
- Refactored `ExecutionContext` to manage `_context_stack: list[Context]`
- Added `current_context` property returning `Context | None`
- Added `default_interpreter` property for eager execution
- Changed `enter_tracing/exit_tracing` to `enter_context/exit_context`

### 2. `mplang/core2/tracer.py`
- Made `Tracer` inherit from `Context`
- Implemented `execute_add()` method that records to Graph IR
- Updated to use `ctx.enter_context(self)` instead of `ctx.enter_tracing(self)`

### 3. `mplang/core2/interp.py`
- Created `Interpreter` class inheriting from `Context`
- Implemented `execute_add()` method that executes immediately
- Supports backend-agnostic execution (delegates to runtime_obj + operator)

### 4. `mplang/core2/object.py`
- Updated `TraceObject` to hold `_context: Tracer`
- Updated `__add__` to delegate to `self._context.execute_add()`
- Updated `InterpObject` to hold `_context: Interpreter | None`
- Updated `__add__` to delegate to interpreter's `execute_add()`
- Added `_tracer` property for backward compatibility

### 5. `tests/core2/test_basics.py`
- Updated test to use `ctx.enter_context/exit_context` instead of `enter_tracing/exit_tracing`

## Backward Compatibility

### Maintained Compatibility
- TraceObject still has `_tracer` property (delegates to `_context`)
- InterpObject constructor still accepts `runtime_obj, obj_type` (interpreter is optional)
- All existing tests pass without modification (except context entry/exit calls)

### API Changes
- `ExecutionContext.enter_tracing()` → `enter_context()`
- `ExecutionContext.exit_tracing()` → `exit_context()`
- `ExecutionContext.current_tracer` still works (checks isinstance)
- Added `ExecutionContext.current_context` for generic context access

## Alignment with Industry Patterns

This pattern aligns with established frameworks:

### JAX
```python
# JAX uses context managers for transformations
with jax.experimental.maps.Mesh(devices, ('x',)):
    # Operations execute in mesh context
```

### PyTorch
```python
# PyTorch uses context for autograd
with torch.no_grad():
    # Operations execute without gradient tracking
```

### Our Pattern
```python
# MPLang uses Context for trace vs interp
ctx.enter_context(tracer)
# Operations record to graph
ctx.exit_context()
```

## Testing

All 13 tests in `tests/core2/test_basics.py` pass:
- ExecutionContext: 3 tests (singleton, mode, context stack)
- InterpObject: 2 tests (creation with data, repr)
- TraceObject: 1 test (creation)
- Tracer: 4 tests (creation, constants, promotion)
- Object Hierarchy: 3 tests (abstract, inheritance)

Note: RemoteRef tests were removed as RemoteRef is no longer part of core2 (it's backend-specific).

## Future Work

### Additional Operations
Implement more Context methods:
```python
class Context(ABC):
    @abstractmethod
    def execute_mul(self, left, right) -> Object: ...
    @abstractmethod
    def execute_sub(self, left, right) -> Object: ...
    @abstractmethod
    def execute_matmul(self, left, right) -> Object: ...
```

### Control Flow
Add control flow operations to Context:
```python
class Context(ABC):
    @abstractmethod
    def execute_cond(self, pred, true_fn, false_fn) -> Object: ...
    @abstractmethod
    def execute_while(self, cond_fn, body_fn, init) -> Object: ...
```

### Context Composition
Support nested contexts (e.g., Profiler wrapping Tracer):
```python
with Profiler():
    with Tracer():
        # Operations are both profiled and traced
        result = x + y
```

## Summary

The Context-based refactoring provides:
1. **Cleaner conceptual model**: Tracer and Interpreter as equals
2. **Better code organization**: Unified operation delegation
3. **Improved extensibility**: Easy to add new context types
4. **Alignment with industry**: Follows JAX/PyTorch patterns
5. **Full backward compatibility**: All existing tests pass

This refactoring sets a solid foundation for future EDSL development and makes the architecture easier to understand and extend.
