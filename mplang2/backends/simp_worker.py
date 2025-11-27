"""SIMP worker module."""

from __future__ import annotations

from typing import Any

# Register implementations
# Note: Implicit import of def_impl classes is currently required for the worker
# to correctly interpret and execute operations.
# TODO: Consider moving to an explicit registration pattern in the future.
from mplang2.dialects import simp
from mplang2.edsl.graph import Operation
from mplang2.edsl.interpreter import Interpreter



class WorkerInterpreter(Interpreter):
    """Interpreter running on a single party (worker)."""

    def __init__(self, rank: int, world_size: int, communicator: Any):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.communicator: Any = communicator


@simp.pcall_static_p.def_impl
def pcall_static_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Implementation of simp.pcall_static (Worker side)."""
    if not isinstance(interpreter, WorkerInterpreter):
        # Fallback for backward compatibility or mixed mode
        # If called from HostInterpreter directly (e.g. nested pcall?), we might need to handle it.
        # But HostInterpreter.evaluate_graph intercepts execution.
        # If we are here, we are likely in a WorkerInterpreter.
        raise RuntimeError(
            f"pcall_static_impl must be run in WorkerInterpreter, got {type(interpreter)}"
        )

    parties = op.attrs.get("parties")
    if parties is None:
        # Should not happen for static pcall
        raise ValueError("pcall_static requires 'parties' attribute")

    if interpreter.rank in parties:
        # Execute region
        fn_graph = op.regions[0]
        return interpreter.evaluate_graph(fn_graph, list(args))
    else:
        # Return dummy values (None) for each output
        # This ensures downstream operations (like shuffle) receive the correct number of inputs
        if len(op.outputs) == 1:
            return None
        else:
            return [None] * len(op.outputs)




@simp.pcall_dynamic_p.def_impl
def pcall_dynamic_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Implementation of simp.pcall_dynamic (Worker side)."""
    # Dynamic pcall runs on all parties (or based on input availability).
    # For now, run on all parties.
    fn_graph = op.regions[0]
    return interpreter.evaluate_graph(fn_graph, list(args))


@simp.shuffle_static_p.def_impl
def shuffle_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Implementation of simp.shuffle (Worker side)."""
    if not isinstance(interpreter, WorkerInterpreter):
        raise RuntimeError(
            f"shuffle_impl must be run in WorkerInterpreter, got {type(interpreter)}"
        )

    routing = op.attrs.get("routing")
    if routing is None:
        # Identity if no routing specified?
        return args[0]

    comm = interpreter.communicator
    my_rank = interpreter.rank
    data = args[0]

    # Send phase
    # routing: target_rank -> source_rank
    # If I am source_rank, I send to target_rank
    for tgt, src in routing.items():
        if src == my_rank:
            if tgt == my_rank:
                continue
            # I am source, send to target
            # Key needs to be unique per op execution.
            # We use op.name as unique identifier for this op instance in the graph
            key = f"shuffle_{op.name}_{tgt}"
            comm.send(tgt, key, data)

    # Recv phase
    if my_rank in routing:
        src = routing[my_rank]
        if src == my_rank:
            return data  # Local copy
        key = f"shuffle_{op.name}_{my_rank}"
        return comm.recv(src, key)
    else:
        return None  # Not a target


@simp.converge_p.def_impl
def converge_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Implementation of simp.converge (Worker side)."""
    # In SPMD, converge(x, y) means we have a value that is logically the union.
    # Locally, I have either x (if I am P0) or y (if I am P1).
    # The other one is likely None or dummy.
    # We just return the one that is present.
    # If multiple are present, it's a conflict (or we pick first).
    for arg in args:
        if arg is not None:
            return arg
    return None


@simp.uniform_cond_p.def_impl
def uniform_cond_impl(
    interpreter: Interpreter, op: Operation, pred: Any, *args: Any
) -> Any:
    """Implementation of simp.uniform_cond (Worker side)."""
    # pred is local boolean.
    # We assume it's uniform across parties.
    # In a real system, we might want to verify this with AllReduce.
    if op.attrs.get("verify_uniform", True):
        # TODO: Implement AllReduce verification
        pass

    if pred:
        return interpreter.evaluate_graph(op.regions[0], list(args))
    else:
        return interpreter.evaluate_graph(op.regions[1], list(args))


@simp.while_loop_p.def_impl
def while_loop_impl(interpreter: Interpreter, op: Operation, *args: Any) -> Any:
    """Implementation of simp.while_loop (Worker side)."""
    cond_graph = op.regions[0]
    body_graph = op.regions[1]

    num_state = len(op.outputs)
    current_state = list(args[:num_state])
    captures = list(args[num_state:])

    while True:
        region_inputs_list = current_state + captures

        # Execute condition locally
        cond_res = interpreter.evaluate_graph(cond_graph, region_inputs_list)

        # cond_res is local boolean
        if not cond_res:
            break

        body_res = interpreter.evaluate_graph(body_graph, region_inputs_list)

        if isinstance(body_res, list):
            current_state = body_res
        else:
            current_state = [body_res]

    if len(current_state) == 1:
        return current_state[0]
    return current_state
