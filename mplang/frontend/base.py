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

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from jax.tree_util import PyTreeDef, tree_flatten

from mplang.core.mpobject import MPContext, MPObject
from mplang.core.pfunc import PFunction
from mplang.core.table import TableType
from mplang.core.tensor import TensorType

# -----------------------------------------------------------------------------
# Triad ABI
# The standard return contract for frontend operations (FeOperation.trace).
#
# Triad := (PFunction, list[MPObject], PyTreeDef)
# - PFunction: Captures fn_type (routing key, e.g., "mlir.stablehlo", "sql.duckdb"),
#              input/output MPTypes and optional attributes.
# - list[MPObject]: The flat positional MPObjects captured under the current
#                   context (Trace/Interp). Order matches pfunc.ins_info.
# - PyTreeDef: The output pytree structure to unflatten results after execution.
#
# Error modes:
# - Type errors if non-MPObject positional args provided to simple ops.
# - Kernel/type builder must produce TensorType/TableType leaves for outs.
# - Context errors propagate from cur_ctx() usage if called outside capture.
# -----------------------------------------------------------------------------
Triad = tuple[PFunction, list[MPObject], PyTreeDef]


# -----------------------------------------------------------------------------
# Lightweight fe module/feop system (new FeOperation based)
# -----------------------------------------------------------------------------


# Global registry for frontend modules and operations
class FeRegistry:
    """Registry for FeModules and FeOperations.

    Maintains:
    - modules: name -> FeModule
    - ops: (module, name) -> FeOperation (callable) returning Triad
    """

    __slots__ = ("_modules", "_ops")

    def __init__(self) -> None:
        self._modules = {}
        self._ops = {}

    # ----------------------------- Modules -----------------------------
    def register_module(self, mod: FeModule, *, replace: bool = False) -> None:
        if not replace and mod.name in self._modules:
            raise ValueError(f"Module already registered: {mod.name}")
        self._modules[mod.name] = mod

    def get_module(self, name: str) -> FeModule:
        if name not in self._modules:
            raise KeyError(f"Unknown module: {name}")
        return self._modules[name]

    def has_module(self, name: str) -> bool:
        return name in self._modules

    def list_modules(self) -> dict[str, FeModule]:
        return dict(self._modules)

    # ------------------------------ Ops -------------------------------
    def register_op(
        self, module: str, name: str, op: FeOperation, *, replace: bool = False
    ) -> None:
        key = (module, name)
        if not replace and key in self._ops:
            raise ValueError(f"Op already registered: {module}.{name}")
        self._ops[key] = op

    def get_op(self, module: str, name: str) -> FeOperation:
        key = (module, name)
        if key not in self._ops:
            raise KeyError(f"Unknown op: {module}.{name}")
        return self._ops[key]

    def list_ops(self, module: str | None = None) -> dict[tuple[str, str], FeOperation]:
        if module is None:
            return dict(self._ops)
        return {k: v for k, v in self._ops.items() if k[0] == module}


_REGISTRY = FeRegistry()


def get_registry() -> FeRegistry:
    return _REGISTRY


def is_feop(x: Any) -> bool:
    """Return True if x is a frontend operation instance."""
    return isinstance(x, FeOperation)


class FeModule(ABC):
    """Frontend module handle with ctx-level state, init lifecycle, and feop/typed_op decorators.

    Choosing between typed_op and feop:
    - typed_op: Use when your kernel can determine output types purely from input MPObjects and simple attrs.
        - Kernel returns TensorType/TableType or a PyTree thereof.
        - Positional MPObject args and any MPObject kwargs become inputs to PFunction (order: positionals then kwargs).
        - Non-MPObject kwargs become attributes on PFunction.
        - PFunction is assembled automatically with fn_type = pfunc_name and ins/outs inferred.
    - feop: Use when you need full control over PFunction construction (multiple outputs, special attrs, custom packing).
        - Kernel directly returns (PFunction, list[MPObject], PyTreeDef).
    - Subclass FeOperation if you need persistent state or complex compilation flows.

    Tips:
    - Keep routing information in PFunction.fn_type (e.g., "builtin.read", "sql[duckdb]", "mlir.stablehlo").
    - Avoid backend-specific logic in kernels; only validate and shape types.
    - For deterministic input ordering with many MPObject kwargs, prefer passing them positionally or sort by key upstream.
    """

    def __init__(self, name: str):
        self.name = name
        get_registry().register_module(self)

    @abstractmethod
    def initialize(self, ctx: MPContext) -> None: ...

    def feop(self, name: str):
        """Decorator for inline/complex ops which already return a Triad.

        Usage:
            @mymod.feop(name="scale")
            def scale(x: MPObject, factor: int) -> Triad:
                # build PFunction and return triad directly
                ...
                return pfunc, [x], out_tree
        """

        def _decorator(trace_fn: Callable[..., Triad]) -> FeOperation:
            op = InlineFeOperation(self, name, trace_fn)
            get_registry().register_op(self.name, name, op)
            return op

        return _decorator

    def typed_op(self, name: str, pfunc_name: str):
        """Decorator for type-driven ops that return only types/schemas.

        The decorated kernel should compute and return a TensorType/TableType (or PyTree thereof).
        Positional MPObject args and MPObject kwargs become inputs. Non-MPObject kwargs are attributes.

        Example:
            @mymod.typed_op(name="add", pfunc_name="builtin.add")
            def add_kernel(x: MPObject, y: MPObject) -> TensorType:
                return x.mptype._type  # same shape/type as x
        """

        def _decorator(ret_type_builder: Callable[..., Any]) -> FeOperation:
            op = SimpleFeOperation(self, name, pfunc_name, ret_type_builder)
            get_registry().register_op(self.name, name, op)
            return op

        return _decorator

    # Backward-compatible alias
    def simple(self, name: str, pfunc_name: str):
        return self.typed_op(name, pfunc_name)


class StatelessFeModule(FeModule):
    """Stateless frontend module with no ctx-level state."""

    def initialize(self, ctx: MPContext) -> None:
        pass


# -----------------------------------------------------------------------------
# Class-based contracts and adapters
# -----------------------------------------------------------------------------


class FeOperation(ABC):
    """Class-based frontend operation contract.

    Subclasses implement trace() to produce a standard triad. __call__ delegates to trace().
    """

    module: FeModule
    name: str

    def __init__(self, module: FeModule, name: str):
        self.module = module
        self.name = name

    @abstractmethod
    def trace(self, *args: Any, **kwargs: Any) -> Triad:
        """Produce a standard triad for this operation."""

    # Convenience: allow calling an FeOperation like a function.
    def __call__(self, *args: Any, **kwargs: Any) -> Triad:
        return self.trace(*args, **kwargs)


class InlineFeOperation(FeOperation):
    """FeOperation that delegates tracing to a provided triad-returning function."""

    def __init__(self, module: FeModule, name: str, trace_fn: Callable[..., Triad]):
        super().__init__(module, name)
        self._trace_fn = trace_fn

    # override
    def trace(self, *args: Any, **kwargs: Any) -> Triad:
        return self._trace_fn(*args, **kwargs)


class SimpleFeOperation(FeOperation):
    """FeOperation that builds Triad from a kernel returning type tree + attrs."""

    def __init__(
        self,
        module: FeModule,
        name: str,
        pfunc_name: str,
        kernel: Callable[..., Any],
    ):
        super().__init__(module, name)
        self.pfunc_name = pfunc_name
        self.ret_type_builder = kernel

    # override
    def trace(self, *args: MPObject, **kwargs: Any) -> Triad:
        # Split inputs: positional/keyword MPObjects as inputs; others as attributes
        pos_mp_inputs: list[MPObject] = [a for a in args if isinstance(a, MPObject)]
        kw_mp_inputs: list[MPObject] = [
            v for v in kwargs.values() if isinstance(v, MPObject)
        ]

        # Prepare kernel call: map non-MPObject positional args to keyword-only params by order
        import inspect

        sig = inspect.signature(self.ret_type_builder)

        non_mp_positional = [a for a in args if not isinstance(a, MPObject)]
        kwargs_for_kernel = dict(kwargs)

        kwonly_names = [
            p.name
            for p in sig.parameters.values()
            if p.kind == inspect.Parameter.KEYWORD_ONLY
        ]
        for i, val in enumerate(non_mp_positional):
            if i < len(kwonly_names) and kwonly_names[i] not in kwargs_for_kernel:
                kwargs_for_kernel[kwonly_names[i]] = val

        # Attributes for PFunction: non-MPObject kwargs, excluding TensorType/TableType
        from mplang.core.table import TableType as _TableType
        from mplang.core.tensor import TensorType as _TensorType

        attr_kwargs: dict[str, Any] = {
            k: v
            for k, v in kwargs_for_kernel.items()
            if not isinstance(v, MPObject)
            and not isinstance(v, (_TensorType, _TableType))
        }

        # Decide call strategy: if original args bind to the signature, use them; else use mapped kwargs
        binding_ok = True
        try:
            sig.bind_partial(*args, **kwargs)
        except TypeError:
            binding_ok = False

        if binding_ok:
            result = self.ret_type_builder(*args, **kwargs)
        else:
            # Call kernel with positional MPObjects and merged kwargs
            result = self.ret_type_builder(*pos_mp_inputs, **kwargs_for_kernel)

        outs_info, out_tree = tree_flatten(result)

        # ensure all out_vars are TensorType or TableType.
        # TODO(jint), theoretically we can also python constants here.
        for o in outs_info:
            if not isinstance(o, (TensorType, TableType)):
                raise TypeError(
                    f"simple op kernel must return TensorType or TableType, got {type(o).__name__}"
                )
        # Build input types from positional MPObjects followed by keyword MPObject values
        all_inputs: list[MPObject] = pos_mp_inputs + kw_mp_inputs
        ins_info = [a.mptype._type for a in all_inputs]

        # Compose PFunction and return triad
        pfunc = PFunction(
            fn_type=self.pfunc_name,
            ins_info=tuple(ins_info),
            outs_info=tuple(outs_info),
            **attr_kwargs,
        )
        return pfunc, all_inputs, out_tree


def femod(mod_name: str) -> FeModule:
    return StatelessFeModule(mod_name)


def list_feops(module: str | None = None) -> dict[tuple[str, str], FeOperation]:
    """Return a view of registered feops, optionally filtered by module name."""
    return get_registry().list_ops(module)


# -----------------------------------------------------------------------------
# Guidance: complex ops via subclassing
# -----------------------------------------------------------------------------

# Example pattern (non-executable) showing how a complex op (e.g., jax_cc) could
# capture a Python callable and compile it to a Triad by subclassing FeOperation.
#
# class JaxCompileOp(FeOperation):
#     def __init__(self, module: FeModule, name: str, func: Callable[..., Any], *,
#                  fn_type: str = "mlir.stablehlo", **options: Any) -> None:
#         super().__init__(module, name)
#         self.func = func
#         self.fn_type = fn_type
#         self.options = dict(options)
#
#     def trace(self, *args: MPObject, **kwargs: Any) -> Triad:
#         # 1) Infer output types from func and args, respecting current ctx/masks.
#         # 2) Build PFunction with fn_type=self.fn_type and any attributes.
#         # 3) Return (pfunc, list(args), out_tree)
#         raise NotImplementedError


# -----------------------------------------------------------------------------
# Migration notes (checklist)
# -----------------------------------------------------------------------------

# - Replace any isinstance(FEOp)/metadata checks with isinstance(x, FeOperation).
# - Define a FeModule via femod("module_name") and register it in FeRegistry automatically.
# - For inline ops that already produce a triad, use @module.feop(name)(trace_fn).
# - For simple type-only kernels, use @module.simple(name, pfunc_name)(kernel).
# - For complex ops (with Python callables/closures), subclass FeOperation and register
#   using get_registry().register_op(module, name, op_instance) or use @module.feop with InlineFeOperation.
# - Ensure PFunction.fn_type is set as the routing key (e.g., "mlir.stablehlo", "sql.duckdb").
# - Keep device selection/routing out of frontend code; only set fn_type and attributes.
# - Avoid moving MPObjects across contexts directly; capture within current ctx in trace().
