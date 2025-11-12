"""Interpreter: Execute Graph IR and Eager Operations.

Interpreter is a Context that executes operations immediately.
It can execute both:
1. Graph IR (via GraphInterpreter)
2. Eager operations on InterpObject (via backend executors)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mplang.core2.context import Context
from mplang.edsl.graph import Graph

if TYPE_CHECKING:
    from mplang.core2.object import InterpObject, Object


class Interpreter(Context):
    """Execution context for eager execution.

    Inherits from Context and implements execute_add() by executing immediately.

    Responsibilities:
    1. Execute operations on InterpObject immediately
    2. Delegate to backend-specific executors
    3. Execute Graph IR (via GraphInterpreter)

    Example:
        >>> interp = Interpreter()
        >>> x = InterpObject(np.array([1, 2, 3]), Tensor[f32, (3,)])
        >>> y = InterpObject(np.array([4, 5, 6]), Tensor[f32, (3,)])
        >>> z = interp.execute_add(x, y)  # Executes immediately
    """

    def __init__(self):
        # TODO: Backend executor registry
        self._executors = {}

    def execute_add(self, left: Object, right: Object) -> InterpObject:
        """Execute addition immediately.

        This is the Context.execute_add() implementation for Interpreter.
        Called by InterpObject.__add__() during eager execution.

        Args:
            left: Left operand (must be InterpObject)
            right: Right operand (must be InterpObject)

        Returns:
            InterpObject containing the result

        Raises:
            TypeError: If operands are not InterpObject
        """
        from mplang.core2.object import InterpObject

        if not isinstance(left, InterpObject) or not isinstance(right, InterpObject):
            raise TypeError("Both operands must be InterpObject for eager execution")

        # TODO: Dispatch to appropriate backend executor based on type
        # For now, simple numpy addition (assumes runtime_obj supports +)
        result_data = left.runtime_obj + right.runtime_obj
        return InterpObject(result_data, left.type)


class GraphInterpreter:
    """Interpreter for Graph IR.

    Traverses the Graph's operations and executes them one by one.

    TODO: Implement complete interpretation logic.
    """

    def __init__(self):
        pass

    def run(self, graph: Graph, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute a Graph.

        Args:
            graph: Graph IR to execute
            inputs: Input values (name → value)

        Returns:
            Output values (name → value)
        """
        # TODO: Implement
        raise NotImplementedError("GraphInterpreter not yet implemented")


def interpret(graph: Graph, args: tuple) -> Any:
    """Convenience function: Interpret and execute a Graph.

    Args:
        graph: Graph IR
        args: Input arguments

    Returns:
        Execution result
    """
    GraphInterpreter()
    # TODO: Convert args to inputs dict
    # TODO: Call interpreter.run()
    raise NotImplementedError("interpret() not yet implemented")
