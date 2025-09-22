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
        # Typed registries
        self._modules: dict[str, FeModule] = {}
        self._ops: dict[tuple[str, str], FeOperation] = {}

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
    """Frontend module with feop/typed_op decorators.

    When to use which:
    - Use typed_op (SimpleFeOperation) when:
        - You know the backend routing key up front via pfunc_name, and the kernel is pure type logic.
        - Inputs are MPObjects (positional/kwargs). Attributes are simple Python values (int/float/str/bytes/tuples/lists of primitives) passed as keywords.
        - The kernel returns TensorType/TableType (or a PyTree thereof); no IR construction inside.
    - Use feop (InlineFeOperation) when:
        - You already build and return the Triad explicitly, or need custom packing/attrs/multi-output composition.
    - Subclass FeOperation when:
        - You need compilation/stateful behavior/dynamic routing, multiple PFunctions, or complex capture flows.

    Tips:
    - Keep routing information in PFunction.fn_type (e.g., "builtin.read", "sql[duckdb]", "mlir.stablehlo").
    - Avoid backend-specific logic in kernels; only validate and shape types.
    - Prefer keyword-only attributes in typed_op kernels for clarity (def op(x: MPObject, *, attr: int)).
    """

    def __init__(self, name: str):
        self.name = name
        get_registry().register_module(self)

    @abstractmethod
    def initialize(self, ctx: MPContext) -> None: ...

    def op_def(self) -> Callable[[Callable[..., Triad]], FeOperation]:
        """Decorator for inline/complex ops which already return a Triad.

        Usage:
            @mymod.feop()
            def scale(x: MPObject, factor: int) -> Triad:
                # build PFunction and return triad directly
                ...
                return pfunc, [x], out_tree
        """

        def _decorator(trace_fn: Callable[..., Triad]) -> FeOperation:
            name = trace_fn.__name__
            op = InlineFeOperation(self, name, trace_fn)
            get_registry().register_op(self.name, name, op)
            return op

        return _decorator

    def simple_op(
        self, pfunc_name: str | None = None
    ) -> Callable[[Callable[..., Any]], FeOperation]:
        """Decorator for type-driven ops that return only types/schemas.

        The decorated kernel should compute and return a TensorType/TableType (or PyTree thereof).
        Positional inputs may be MPObjects (captured as inputs) or data-like values (TableLike/TensorLike)
        used for type inference/validation. Keyword arguments are PFunction attributes and must be plain
        Python values (int/float/str/bytes/tuples/lists of primitives). Passing MPObjects via kwargs is not allowed.

        SSOT naming: The operation name is derived from the kernel function name (kernel.__name__),
        ensuring there's a single source of truth and improving readability. Use clear, concise
        function names to define the public op names.

        Example:
            @mymod.typed_op(pfunc_name="builtin.add")
            def add_kernel(x: MPObject, y: MPObject) -> TensorType:
                return x.mptype._type  # same shape/type as x

        Bad vs Good (signatures and calls):
        - Bad:  def op(x: MPObject, **kwargs): ...               # disallowed: **kwargs
          Good: def op(x: MPObject, *, attr: int): ...

        - Bad:  def op(*args, **kwargs): ...                     # disallowed: *args/**kwargs
          Good: def op(x: MPObject, y: MPObject, *, k: str): ...

        - Bad:  enc(plaintext=pt, key=mp_key)                    # MPObject via kwargs (disallowed)
          Good: enc(pt, mp_key)                                  # pass MPObjects positionally

        - Good: hkdf(secret, "info")                             # data-like positional mapped to kw-only attr
          Also good: hkdf(secret, info="info")

        - Good: phe.mul(jnp.array(...), jnp.array(...))          # data-like positionals allowed for type inference
        """

        def _decorator(kernel: Callable[..., Any]) -> FeOperation:
            # Default PFunction routing when not provided: "<module>.<kernel_name>"
            final_pfunc_name = pfunc_name or f"{self.name}.{kernel.__name__}"
            op = SimpleFeOperation(self, final_pfunc_name, kernel)
            # Use kernel function name as SSOT for op name
            get_registry().register_op(self.name, op.name, op)
            return op

        return _decorator


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
    """FeOperation that builds Triad from a type-only kernel.

    Contract (keep it simple):
    - Kernel computes and returns TensorType/TableType or a PyTree thereof.
    - Positional inputs may be MPObjects (captured as inputs) or data-like values (TableLike/TensorLike)
        used for type inference/validation. Keyword arguments are attributes and must be plain Python
        values (TensorType/TableType are also excluded from attrs). MPObject kwargs are disallowed.
    - Prefer keyword-only attributes in the kernel signature for explicitness. For convenience, non-MPObject
        positional values that are not data-like will be mapped to keyword-only parameters by order when possible.
    - No IR building inside the kernel; PFunction is assembled here with fn_type=pfunc_name.
    """

    def __init__(
        self,
        module: FeModule,
        pfunc_name: str,
        kernel: Callable[..., Any],
    ):
        # Derive operation name from kernel function name for SSOT
        super().__init__(module, kernel.__name__)
        self.pfunc_name = pfunc_name
        self._kernel = kernel

        # Validate kernel signature: typed_op kernels must not use *args/**kwargs.
        import inspect

        sig = inspect.signature(kernel)
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                raise TypeError(
                    f"typed_op kernel '{module.name}.{kernel.__name__}' must not use **kwargs; define explicit keywords instead"
                )
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                raise TypeError(
                    f"typed_op kernel '{module.name}.{kernel.__name__}' must not use *args; define explicit parameters instead"
                )

        # Cache signature and kw-only parameter names for fast path in trace
        self._kernel_sig = sig
        self._kwonly_names = [
            p.name
            for p in sig.parameters.values()
            if p.kind == inspect.Parameter.KEYWORD_ONLY
        ]

    # override
    def trace(self, *args: MPObject, **kwargs: Any) -> Triad:
        # Actual params may not match kernel signature exactly, so we do flexible binding.
        sig = self._kernel_sig

        # Inputs at PFunction layer are MPObjects captured from positional args only.
        pos_mp_inputs: list[MPObject] = [a for a in args if isinstance(a, MPObject)]

        # Enforce: no MPObject kwargs per simplified contract
        for k, v in kwargs.items():
            if isinstance(v, MPObject):
                raise TypeError(
                    f"typed_op does not accept MPObject kwargs: {k}; pass MPObjects positionally"
                )

        # Try original call; if it binds, keep it as-is to support data-like positionals
        try:
            sig.bind_partial(*args, **kwargs)
            call_pos = args
            call_kwargs = kwargs
        except TypeError as _bind_err:
            # Fallback: For convenience, map non-MPObject positional arguments to
            # keyword-only parameters by order. This allows ergonomic calls like
            # `crypto.keygen(32)` where the kernel is `def keygen(*, length: int)`.
            # The direct binding `sig.bind_partial(32)` would fail, so we manually
            # map the positional `32` to the `length` keyword.
            non_mp_positional = [a for a in args if not isinstance(a, MPObject)]
            call_kwargs = dict(kwargs)
            filled = 0
            for _i, name in enumerate(self._kwonly_names):
                if filled < len(non_mp_positional) and name not in call_kwargs:
                    call_kwargs[name] = non_mp_positional[filled]
                    filled += 1
            if filled < len(non_mp_positional):
                leftover = non_mp_positional[filled:]
                raise TypeError(
                    f"too many non-MPObject positional values for typed_op '{self.module.name}.{self.name}': {leftover}. "
                    "Pass attributes explicitly by keyword (e.g., foo(x, *, attr=...))."
                ) from None
            call_pos = tuple(pos_mp_inputs)

        # Compute PFunction attrs from the call kwargs (exclude MPObject and type objects)
        attr_kwargs: dict[str, Any] = {
            k: v
            for k, v in call_kwargs.items()
            if not isinstance(v, MPObject)
            and not isinstance(v, (TensorType, TableType))
        }

        # Prepare kernel positional arguments: replace MPObject with its underlying type so
        # the kernel always sees TensorType/TableType (never TraceVar/InterpVar).
        call_pos_types = tuple(a.mptype._type for a in call_pos)

        # Sanity: no MPObject should appear in kwargs (enforced earlier), but be safe.
        if any(isinstance(v, MPObject) for v in call_kwargs.values()):
            raise TypeError("kernel kwargs should not be MPObject")

        # Execute kernel to compute return types
        result = self._kernel(*call_pos_types, **call_kwargs)

        outs_info, out_tree = tree_flatten(result)

        # ensure all out_vars are TensorType or TableType.
        # TODO(jint), theoretically we can also python constants here.
        for o in outs_info:
            if not isinstance(o, (TensorType, TableType)):
                raise TypeError(
                    f"simple op kernel must return TensorType or TableType, got {type(o).__name__}"
                )

        # Build input types from positional MPObjects only
        ins_info = [a.mptype._type for a in pos_mp_inputs]

        # Compose PFunction and return triad
        pfunc = PFunction(
            fn_type=self.pfunc_name,
            ins_info=tuple(ins_info),
            outs_info=tuple(outs_info),
            **attr_kwargs,
        )
        return pfunc, pos_mp_inputs, out_tree


def stateless_mod(mod_name: str) -> FeModule:
    return StatelessFeModule(mod_name)


def list_ops(module: str | None = None) -> dict[tuple[str, str], FeOperation]:
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
# - For inline ops that already produce a triad, use @module.feop()(trace_fn). The op name is derived from the function name.
# - For type-only kernels, use @module.typed_op(pfunc_name)(kernel). The op name is derived from the kernel function name.
# - For complex ops (with Python callables/closures), subclass FeOperation and register
#   using get_registry().register_op(module, name, op_instance) or use @module.feop with InlineFeOperation.
# - Ensure PFunction.fn_type is set as the routing key (e.g., "mlir.stablehlo", "sql.duckdb").
# - Keep device selection/routing out of frontend code; only set fn_type and attributes.
# - Avoid moving MPObjects across contexts directly; capture within current ctx in trace().
