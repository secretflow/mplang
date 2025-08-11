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

from typing import Any

import numpy as np
import spu.libspu as spu_api

from mplang.core.base import MPType, TensorInfo
from mplang.core.dtype import DType
from mplang.core.pfunc import PFunction
from mplang.expr import Expr, ExprVisitor, FuncDefExpr
from mplang.expr.ast import (
    AccessExpr,
    CallExpr,
    CondExpr,
    ConstExpr,
    ConvExpr,
    EvalExpr,
    RandExpr,
    RankExpr,
    ShflExpr,
    ShflSExpr,
    TupleExpr,
    VariableExpr,
    WhileExpr,
)
from mplang.protos import mpir_pb2

# Single mapping table for dtype conversion
DTYPE_MAPPING = {
    np.float32: mpir_pb2.MPTypeProto.DataType.F32,
    np.uint8: mpir_pb2.MPTypeProto.DataType.U8,
    np.int8: mpir_pb2.MPTypeProto.DataType.I8,
    np.uint16: mpir_pb2.MPTypeProto.DataType.U16,
    np.int16: mpir_pb2.MPTypeProto.DataType.I16,
    np.int32: mpir_pb2.MPTypeProto.DataType.I32,
    np.int64: mpir_pb2.MPTypeProto.DataType.I64,
    np.str_: mpir_pb2.MPTypeProto.DataType.STRING,
    np.bool_: mpir_pb2.MPTypeProto.DataType.BOOL,
    np.float16: mpir_pb2.MPTypeProto.DataType.F16,
    np.float64: mpir_pb2.MPTypeProto.DataType.F64,
    np.uint32: mpir_pb2.MPTypeProto.DataType.U32,
    np.uint64: mpir_pb2.MPTypeProto.DataType.U64,
    np.complex64: mpir_pb2.MPTypeProto.DataType.COMPLEX64,
    np.complex128: mpir_pb2.MPTypeProto.DataType.COMPLEX128,
}


def dtype_to_proto(dtype_like):
    """Convert dtype (DType, NumPy dtype, or type) to protobuf DataType."""
    # If it's already a DType, convert to numpy for protobuf mapping
    if isinstance(dtype_like, DType):
        numpy_dtype = dtype_like.to_numpy()
        key_type = numpy_dtype.type
    else:
        # Handle NumPy dtypes and other types
        try:
            key_type = np.dtype(dtype_like).type
        except TypeError:
            # Handle cases where dtype_like might already be a type object
            # that np.dtype() can't process but is a valid key.
            if isinstance(dtype_like, type) and issubclass(dtype_like, np.generic):
                key_type = dtype_like
            else:
                raise ValueError(f"Invalid dtype: {dtype_like}") from None

    if key_type in DTYPE_MAPPING:
        return DTYPE_MAPPING[key_type]
    else:
        raise ValueError(f"Unsupported dtype: {dtype_like}")


def proto_to_dtype(dtype_enum: int) -> DType:
    """Convert protobuf DataType enum to DType."""
    # Find the numpy type for the given enum by searching the mapping
    for numpy_type, proto_enum in DTYPE_MAPPING.items():
        if proto_enum == dtype_enum:
            # Convert numpy type to dtype
            np_dtype = np.dtype(numpy_type)

            # Special handling for string types since DType.from_numpy doesn't support them
            if np_dtype.kind == "U":  # Unicode string
                # Create a string DType manually
                return DType("str", 0, False, False, False)
            else:
                return DType.from_numpy(np_dtype)

    # If we get here, the enum was not found
    raise ValueError(f"Unsupported dtype enum: {dtype_enum}")


def attr_to_proto(py_value: Any) -> mpir_pb2.AttrProto:
    """Convert a Python attribute value to an AttrProto."""
    attr_proto = mpir_pb2.AttrProto()
    if isinstance(py_value, int):
        attr_proto.type = mpir_pb2.AttrProto.INT
        attr_proto.i = py_value
    elif isinstance(py_value, float):
        attr_proto.type = mpir_pb2.AttrProto.FLOAT
        attr_proto.f = py_value
    elif isinstance(py_value, str):
        attr_proto.type = mpir_pb2.AttrProto.STRING
        attr_proto.s = py_value
    elif isinstance(py_value, bytes):
        attr_proto.type = mpir_pb2.AttrProto.BYTES
        attr_proto.raw_bytes = py_value
    elif isinstance(py_value, tuple | list):
        if all(isinstance(item, int) for item in py_value):
            attr_proto.type = mpir_pb2.AttrProto.INTS
            attr_proto.ints.extend(list(py_value))
        elif all(isinstance(item, float) for item in py_value):
            attr_proto.type = mpir_pb2.AttrProto.FLOATS
            attr_proto.floats.extend(list(py_value))
        elif all(isinstance(item, str) for item in py_value):
            attr_proto.type = mpir_pb2.AttrProto.STRINGS
            attr_proto.strs.extend(list(py_value))
        elif all(isinstance(item, spu_api.Visibility) for item in py_value):
            # Handle list of enum types (like [Visibility.VIS_SECRET, Visibility.VIS_SECRET])
            attr_proto.type = mpir_pb2.AttrProto.INTS
            attr_proto.ints.extend([int(item) for item in py_value])
        else:
            raise TypeError(f"Unsupported tuple/list type: {type(py_value)}")
    elif isinstance(py_value, FuncDefExpr):
        # Convert FuncDefExpr to GraphProto
        graph = Writer().dumps(py_value)
        attr_proto.type = mpir_pb2.AttrProto.GRAPH
        attr_proto.graph.CopyFrom(graph)
    elif isinstance(py_value, PFunction):
        attr_proto.type = mpir_pb2.AttrProto.FUNCTION
        attr_proto.func.type = py_value.fn_type
        attr_proto.func.name = py_value.fn_name
        if py_value.fn_text is not None:
            attr_proto.func.body = str(py_value.fn_text)

        # Serialize attrs dictionary
        if py_value.attrs:
            for attr_name, attr_value in py_value.attrs.items():
                attr_proto.func.attrs[attr_name].CopyFrom(attr_to_proto(attr_value))

        # Note: We don't serialize ins_info and outs_info since they can be
        # inferred from the input expressions during deserialization
    elif isinstance(py_value, spu_api.Visibility):
        # Handle enum types (like spu.libspu.Visibility) by storing as int
        attr_proto.type = mpir_pb2.AttrProto.INT
        attr_proto.i = int(py_value)
    else:
        raise TypeError(f"Unsupported attribute type: {type(py_value)}")
    return attr_proto


class Writer(ExprVisitor):
    """Writer for serializing Expr-based expressions to GraphProto."""

    def __init__(self, var_name_mapping: dict[str, str] | None = None):
        self._counter = 0
        self._expr_ids: dict[int, str] = {}  # Use expr id instead of Node
        self._nodes: list[mpir_pb2.NodeProto] = []
        self._var_name_mapping = var_name_mapping or {}

    def expr_name(self, expr: Expr) -> str:
        """Get or create a name for an expression."""
        expr_id = id(expr)
        if expr_id not in self._expr_ids:
            self._expr_ids[expr_id] = f"%{self._counter}"
            self._counter += 1
        return self._expr_ids[expr_id]

    def value_name(self, expr: Expr, out_idx: int = 0) -> str:
        """Get value name for expression output."""
        if len(expr.mptypes) == 1:
            return self.expr_name(expr)
        else:
            return f"{self.expr_name(expr)}:{out_idx}"

    def _ensure_visited(self, *exprs: Expr) -> None:
        """Ensure expressions are visited."""
        for expr in exprs:
            expr_id = id(expr)
            if expr_id not in self._expr_ids:
                expr.accept(self)

    def reset(self):
        """Reset writer state."""
        self._counter = 0
        self._expr_ids.clear()
        self._nodes.clear()

    def _create_node_proto(self, expr: Expr, op_type: str) -> mpir_pb2.NodeProto:
        """Helper: Create a basic NodeProto with common fields set."""
        op = mpir_pb2.NodeProto()
        op.op_type = op_type
        op.name = self.expr_name(expr)
        return op

    def _add_output_info(self, op: mpir_pb2.NodeProto, expr: Expr) -> None:
        """Helper: Add output type information to a NodeProto."""
        for out_info in expr.mptypes:
            out_proto = op.outs_info.add()
            out_proto.dtype = dtype_to_proto(out_info.dtype)
            out_proto.shape_dims.extend(list(out_info.shape))
            if out_info.pmask is not None:
                out_proto.pmask = out_info.pmask.to_bytes(8, byteorder="big")

    def _add_expr_inputs(self, op: mpir_pb2.NodeProto, *exprs: Expr) -> None:
        """Helper: Add expression inputs to NodeProto."""
        for expr in exprs:
            op.inputs.extend([
                self.value_name(expr, i) for i in range(len(expr.mptypes))
            ])

    def _add_single_expr_inputs(self, op: mpir_pb2.NodeProto, *exprs: Expr) -> None:
        """Helper: Add single-output expression inputs to NodeProto."""
        for expr in exprs:
            op.inputs.append(self.value_name(expr, 0))

    def _add_attrs(self, op: mpir_pb2.NodeProto, **attrs: Any) -> None:
        """Helper: Add attributes to NodeProto."""
        for key, value in attrs.items():
            if value is not None:  # Skip None values
                op.attrs[key].CopyFrom(attr_to_proto(value))

    def _finalize_node(self, op: mpir_pb2.NodeProto, expr: Expr) -> str:
        """Helper: Add output info, append to nodes, and return expr name."""
        self._add_output_info(op, expr)
        self._nodes.append(op)
        return self.expr_name(expr)

    def dumps(self, expr: Expr) -> mpir_pb2.GraphProto:
        """Dump an expression to GraphProto."""
        self.reset()

        # Visit the expression tree
        self._ensure_visited(expr)

        # Create graph metadata
        graph_attrs = {}
        if isinstance(expr, FuncDefExpr):
            graph_attrs["name"] = attr_to_proto(f"function_{id(expr)}")
            # For function definitions, the outputs should be the FuncDefExpr itself
            outputs = [self.value_name(expr, i) for i in range(len(expr.mptypes))]
        else:
            # For regular expressions, outputs are the expression outputs
            outputs = [self.value_name(expr, i) for i in range(len(expr.mptypes))]

        return mpir_pb2.GraphProto(
            version=mpir_pb2.VersionInfo(major=1, minor=0, patch=0),
            nodes=self._nodes,
            outputs=outputs,
            attrs=graph_attrs,
        )

    def visit_rank(self, expr) -> Any:
        """Visit rank expression."""
        op = self._create_node_proto(expr, "rank")
        return self._finalize_node(op, expr)

    def visit_const(self, expr) -> Any:
        """Visit constant expression."""
        op = self._create_node_proto(expr, "const")
        self._add_attrs(op, data=expr.data_bytes)
        return self._finalize_node(op, expr)

    def visit_rand(self, expr) -> Any:
        """Visit random expression."""
        op = self._create_node_proto(expr, "rand")
        return self._finalize_node(op, expr)

    def visit_eval(self, expr) -> Any:
        """Visit evaluation expression."""
        # Visit all argument expressions
        self._ensure_visited(*expr.args)

        op = self._create_node_proto(expr, "eval")
        self._add_expr_inputs(op, *expr.args)
        self._add_attrs(op, pfunc=expr.pfunc, rmask=expr.rmask)
        return self._finalize_node(op, expr)

    def visit_variable(self, expr) -> Any:
        """Visit variable expression."""
        op = self._create_node_proto(expr, "variable")
        # Use mapped name if available, otherwise use original name
        mapped_name = self._var_name_mapping.get(expr.name, expr.name)
        self._add_attrs(op, name=mapped_name)
        return self._finalize_node(op, expr)

    def visit_tuple(self, expr) -> Any:
        """Visit tuple expression."""
        # Visit all argument expressions
        self._ensure_visited(*expr.args)

        op = self._create_node_proto(expr, "tuple")
        self._add_single_expr_inputs(op, *expr.args)
        return self._finalize_node(op, expr)

    def visit_cond(self, expr) -> Any:
        """Visit conditional expression."""
        # Visit predicate and all argument expressions
        self._ensure_visited(expr.pred, *expr.args)

        op = self._create_node_proto(expr, "cond")
        self._add_single_expr_inputs(op, expr.pred)
        self._add_expr_inputs(op, *expr.args)
        self._add_attrs(op, then_fn=expr.then_fn, else_fn=expr.else_fn)
        return self._finalize_node(op, expr)

    def visit_call(self, expr) -> Any:
        """Visit function call expression."""
        # Visit function definition and all argument expressions
        self._ensure_visited(expr.fn, *expr.args)

        op = self._create_node_proto(expr, "call")
        self._add_single_expr_inputs(op, expr.fn)
        self._add_expr_inputs(op, *expr.args)
        return self._finalize_node(op, expr)

    def visit_while(self, expr) -> Any:
        """Visit while loop expression."""
        # Visit all argument expressions
        self._ensure_visited(*expr.args)

        op = self._create_node_proto(expr, "while")
        self._add_expr_inputs(op, *expr.args)
        self._add_attrs(op, cond_fn=expr.cond_fn, body_fn=expr.body_fn)
        return self._finalize_node(op, expr)

    def visit_conv(self, expr) -> Any:
        """Visit convergence expression."""
        # Visit all variable expressions
        self._ensure_visited(*expr.vars)

        op = self._create_node_proto(expr, "conv")
        self._add_expr_inputs(op, *expr.vars)
        return self._finalize_node(op, expr)

    def visit_shfl_s(self, expr) -> Any:
        """Visit static shuffle expression."""
        # Visit source value expression
        self._ensure_visited(expr.src_val)

        op = self._create_node_proto(expr, "shfl_s")
        self._add_single_expr_inputs(op, expr.src_val)
        self._add_attrs(op, pmask=expr.pmask, src_ranks=expr.src_ranks)
        return self._finalize_node(op, expr)

    def visit_shfl(self, expr) -> Any:
        """Visit dynamic shuffle expression."""
        # Visit source and index expressions
        self._ensure_visited(expr.src, expr.index)

        op = self._create_node_proto(expr, "shfl")
        self._add_single_expr_inputs(op, expr.src, expr.index)
        return self._finalize_node(op, expr)

    def visit_access(self, expr) -> Any:
        """Visit access expression."""
        # Visit source expression
        self._ensure_visited(expr.src)

        op = self._create_node_proto(expr, "access")
        # For access, we use the specific output index
        op.inputs.append(self.value_name(expr.src, expr.index))
        self._add_attrs(op, index=expr.index)
        return self._finalize_node(op, expr)

    def visit_func_def(self, expr) -> Any:
        """Visit function definition expression."""
        # Visit body expression
        self._ensure_visited(expr.body)

        op = self._create_node_proto(expr, "func_def")
        self._add_expr_inputs(op, expr.body)
        self._add_attrs(op, params=expr.params)
        return self._finalize_node(op, expr)


class Reader:
    """Reader for deserializing GraphProto back to Expr-based expressions."""

    def __init__(self):
        self._value_cache: dict[str, Expr] = {}

    def loads(self, graph_proto: mpir_pb2.GraphProto) -> Expr | None:
        """Load an expression from a GraphProto.

        Args:
            graph_proto: The protobuf graph to deserialize

        Returns:
            The deserialized expression or None if empty
        """
        self._value_cache.clear()

        # Process nodes in topological order
        processed_nodes = set()

        def process_node(node_proto: mpir_pb2.NodeProto):
            if node_proto.name in processed_nodes:
                return

            # First process all dependencies
            for input_name in node_proto.inputs:
                dep_node_name = input_name.split(":")[0]
                for dep_proto in graph_proto.nodes:
                    if (
                        dep_proto.name == dep_node_name
                        and dep_proto.name not in processed_nodes
                    ):
                        process_node(dep_proto)

            # Now process this node
            expr = self._create_expr_from_proto(node_proto)
            processed_nodes.add(node_proto.name)

            # Cache the expression
            self._value_cache[node_proto.name] = expr

        # Process all nodes
        for node_proto in graph_proto.nodes:
            process_node(node_proto)

        # Extract outputs - for now, just return the first output expression
        if graph_proto.outputs:
            output_name = graph_proto.outputs[0].split(":")[0]
            if output_name in self._value_cache:
                return self._value_cache[output_name]
            else:
                raise ValueError(f"Output {output_name} not found in processed nodes")

        return None

    def _create_expr_from_proto(self, node_proto: mpir_pb2.NodeProto) -> Expr:
        """Create an Expression from a NodeProto."""
        if node_proto.op_type == "rank":
            # Parse pmask from output info
            pmask = 0  # Default to party 0 if no pmask
            if node_proto.outs_info:
                pmask_bytes = node_proto.outs_info[0].pmask
                if pmask_bytes:
                    pmask = int.from_bytes(pmask_bytes, byteorder="big")
            return RankExpr(pmask)

        elif node_proto.op_type == "const":
            # Parse constant data
            data_bytes = self._proto_to_attr(node_proto.attrs["data"])

            # Parse type info from output info
            if not node_proto.outs_info:
                raise ValueError("Const node missing output info")
            out_info = node_proto.outs_info[0]
            dtype = proto_to_dtype(out_info.dtype)
            shape = tuple(out_info.shape_dims)
            pmask = 0  # Default to party 0 if no pmask
            if out_info.pmask:
                pmask = int.from_bytes(out_info.pmask, byteorder="big")

            tensor_info = TensorInfo(dtype, shape)
            return ConstExpr(tensor_info, data_bytes, pmask)

        elif node_proto.op_type == "rand":
            # Parse type info from output info
            if not node_proto.outs_info:
                raise ValueError("Rand node missing output info")
            out_info = node_proto.outs_info[0]
            dtype = proto_to_dtype(out_info.dtype)
            shape = tuple(out_info.shape_dims)
            pmask = 0  # Default to party 0 if no pmask
            if out_info.pmask:
                pmask = int.from_bytes(out_info.pmask, byteorder="big")

            tensor_info = TensorInfo(dtype, shape)
            return RandExpr(tensor_info, pmask)

        elif node_proto.op_type == "eval":
            # Parse inputs
            input_exprs = []
            for input_name in node_proto.inputs:
                dep_name = input_name.split(":")[0]
                if dep_name in self._value_cache:
                    input_exprs.append(self._value_cache[dep_name])
                else:
                    raise ValueError(f"Input {input_name} not found")

            # Parse function
            pfunc = self._proto_to_attr(node_proto.attrs["pfunc"])
            rmask = None
            if "rmask" in node_proto.attrs:
                rmask = self._proto_to_attr(node_proto.attrs["rmask"])

            # Fill in ins_info and outs_info for PFunction
            # ins_info from input expressions (use mptype for single type per value)
            ins_info = []
            for input_expr in input_exprs:
                # Use mptype directly for single MPType
                mptype = input_expr.mptype
                ins_info.append(TensorInfo(mptype.dtype, mptype.shape))

            # outs_info from NodeProto.outs_info
            outs_info = []
            for out_proto in node_proto.outs_info:
                dtype = proto_to_dtype(out_proto.dtype)
                shape = tuple(out_proto.shape_dims)
                outs_info.append(TensorInfo(dtype, shape))

            # Create a complete PFunction with proper type information
            complete_pfunc = PFunction(
                fn_type=pfunc.fn_type,
                fn_name=pfunc.fn_name,
                fn_body=pfunc.fn_body,
                fn_text=pfunc.fn_text,
                ins_info=ins_info,
                outs_info=outs_info,
                attrs=pfunc.attrs,  # Restore attributes
            )

            return EvalExpr(complete_pfunc, input_exprs, rmask)

        elif node_proto.op_type == "variable":
            # Parse variable name
            name = self._proto_to_attr(node_proto.attrs["name"])

            # Parse type info from output info (VariableExpr needs a single MPType)
            if not node_proto.outs_info:
                raise ValueError("Variable node missing output info")

            mptype = self._proto_to_mptype(node_proto.outs_info[0])
            return VariableExpr(name, mptype)

        elif node_proto.op_type == "tuple":
            # Parse inputs
            input_exprs = []
            for input_name in node_proto.inputs:
                dep_name = input_name.split(":")[0]
                if dep_name in self._value_cache:
                    input_exprs.append(self._value_cache[dep_name])
                else:
                    raise ValueError(f"Input {input_name} not found")

            return TupleExpr(input_exprs)

        elif node_proto.op_type == "cond":
            # Parse predicate and arguments
            pred_name = node_proto.inputs[0].split(":")[0]
            pred_expr = self._value_cache[pred_name]

            arg_exprs = []
            for input_name in node_proto.inputs[1:]:
                dep_name = input_name.split(":")[0]
                if dep_name in self._value_cache:
                    arg_exprs.append(self._value_cache[dep_name])
                else:
                    raise ValueError(f"Input {input_name} not found")

            # Parse functions
            then_fn = self._proto_to_attr(node_proto.attrs["then_fn"])
            else_fn = self._proto_to_attr(node_proto.attrs["else_fn"])

            return CondExpr(pred_expr, then_fn, else_fn, arg_exprs)

        elif node_proto.op_type == "while":
            # Parse arguments
            arg_exprs = []
            for input_name in node_proto.inputs:
                dep_name = input_name.split(":")[0]
                if dep_name in self._value_cache:
                    arg_exprs.append(self._value_cache[dep_name])
                else:
                    raise ValueError(f"Input {input_name} not found")

            # Parse functions
            cond_fn = self._proto_to_attr(node_proto.attrs["cond_fn"])
            body_fn = self._proto_to_attr(node_proto.attrs["body_fn"])

            return WhileExpr(cond_fn, body_fn, arg_exprs)

        elif node_proto.op_type == "access":
            # Parse source expression
            input_name = node_proto.inputs[0]
            dep_name = input_name.split(":")[0]
            src_expr = self._value_cache[dep_name]

            # Parse index
            index = self._proto_to_attr(node_proto.attrs["index"])

            return AccessExpr(src_expr, index)

        elif node_proto.op_type == "func_def":
            # Parse body expression
            input_names = node_proto.inputs
            if not input_names:
                raise ValueError("FuncDef node missing body input")

            body_name = input_names[0].split(":")[0]
            body_expr = self._value_cache[body_name]

            # Parse parameters
            params = self._proto_to_attr(node_proto.attrs["params"])

            return FuncDefExpr(params, body_expr)

        elif node_proto.op_type == "shfl_s":
            # Parse source expression
            input_name = node_proto.inputs[0]
            dep_name = input_name.split(":")[0]
            src_val = self._value_cache[dep_name]

            # Parse attributes
            pmask = self._proto_to_attr(node_proto.attrs["pmask"])
            src_ranks = self._proto_to_attr(node_proto.attrs["src_ranks"])

            return ShflSExpr(src_val, pmask, src_ranks)

        elif node_proto.op_type == "shfl":
            # Parse source and index expressions
            src_name = node_proto.inputs[0].split(":")[0]
            index_name = node_proto.inputs[1].split(":")[0]
            src_expr = self._value_cache[src_name]
            index_expr = self._value_cache[index_name]

            return ShflExpr(src_expr, index_expr)

        elif node_proto.op_type == "conv":
            # Parse variable expressions
            var_exprs = []
            for input_name in node_proto.inputs:
                dep_name = input_name.split(":")[0]
                if dep_name in self._value_cache:
                    var_exprs.append(self._value_cache[dep_name])
                else:
                    raise ValueError(f"Input {input_name} not found")

            return ConvExpr(var_exprs)

        elif node_proto.op_type == "call":
            # Parse function and arguments
            fn_name = node_proto.inputs[0].split(":")[0]
            fn_expr = self._value_cache[fn_name]

            # Ensure function is FuncDefExpr
            if not isinstance(fn_expr, FuncDefExpr):
                raise ValueError(
                    f"Call function must be FuncDefExpr, got {type(fn_expr)}"
                )

            arg_exprs = []
            for input_name in node_proto.inputs[1:]:
                dep_name = input_name.split(":")[0]
                if dep_name in self._value_cache:
                    arg_exprs.append(self._value_cache[dep_name])
                else:
                    raise ValueError(f"Input {input_name} not found")

            return CallExpr(fn_expr, arg_exprs)

        else:
            raise ValueError(f"Unsupported node type: {node_proto.op_type}")

    def _proto_to_mptype(self, type_proto: mpir_pb2.MPTypeProto) -> MPType:
        """Convert MPTypeProto to MPType."""
        # Convert datatype
        dtype = proto_to_dtype(type_proto.dtype)

        # Convert shape
        shape = tuple(type_proto.shape_dims)

        # Convert pmask
        pmask = None
        if type_proto.pmask:
            pmask = int.from_bytes(type_proto.pmask, byteorder="big")

        # Convert attributes
        attrs = {}
        for attr_name, attr_proto in type_proto.attrs.items():
            attrs[attr_name] = self._proto_to_attr(attr_proto)

        return MPType(dtype, shape, pmask, attrs)

    def _proto_to_attr(self, attr_proto: mpir_pb2.AttrProto) -> Any:
        """Convert AttrProto to Python value."""
        if attr_proto.type == mpir_pb2.AttrProto.INT:
            return attr_proto.i
        elif attr_proto.type == mpir_pb2.AttrProto.FLOAT:
            return attr_proto.f
        elif attr_proto.type == mpir_pb2.AttrProto.STRING:
            return attr_proto.s
        elif attr_proto.type == mpir_pb2.AttrProto.BYTES:
            return attr_proto.raw_bytes
        elif attr_proto.type == mpir_pb2.AttrProto.INTS:
            return list(attr_proto.ints)
        elif attr_proto.type == mpir_pb2.AttrProto.FLOATS:
            return list(attr_proto.floats)
        elif attr_proto.type == mpir_pb2.AttrProto.STRINGS:
            return list(attr_proto.strs)
        elif attr_proto.type == mpir_pb2.AttrProto.FUNCTION:
            # Reconstruct PFunction - since Expr already contains MPType information,
            # we don't need to reconstruct ins_info and outs_info from serialized data.
            # The type information will be inferred from the actual input expressions.

            # Deserialize attrs dictionary
            attrs = {}
            for attr_name, attr_value_proto in attr_proto.func.attrs.items():
                attrs[attr_name] = self._proto_to_attr(attr_value_proto)

            return PFunction(
                fn_type=attr_proto.func.type,
                fn_name=attr_proto.func.name,
                fn_body=None,  # Cannot reconstruct callable from text
                fn_text=attr_proto.func.body if attr_proto.func.body else None,
                ins_info=[],  # Will be inferred from input expressions
                outs_info=[],  # Will be inferred from context
                attrs=attrs,  # Restore serialized attributes
            )
        elif attr_proto.type == mpir_pb2.AttrProto.GRAPH:
            # Handle nested expressions (for control flow)
            reader = Reader()
            return reader.loads(attr_proto.graph)
        else:
            raise TypeError(f"Unsupported attribute type: {attr_proto.type}")


def get_graph_statistics(graph_proto: mpir_pb2.GraphProto) -> str:
    """Get statistics about a GraphProto structure.

    Args:
        graph_proto: The protobuf GraphProto to analyze

    Returns:
        A formatted string with:
        - Graph version information
        - Node count and breakdown by operation type
        - Output variable information
        - Graph attributes count
    """
    # Build statistics string
    lines = []
    lines.append("GraphProto structure analysis:")

    # Version information
    version = graph_proto.version
    lines.append(f"- Version: {version.major}.{version.minor}.{version.patch}")

    # Node and output counts
    lines.append(f"- Number of nodes: {len(graph_proto.nodes)}")
    lines.append(f"- Number of outputs: {len(graph_proto.outputs)}")
    lines.append(f"- Graph attributes: {len(graph_proto.attrs)}")
    lines.append("")

    # Node breakdown by operation type
    lines.append("Node breakdown by operation type:")
    op_counts = {}
    for node in graph_proto.nodes:
        op_type = node.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1

    for op_type, count in sorted(op_counts.items()):
        lines.append(f"- {op_type}: {count} nodes")
    lines.append("")

    # Output variables
    lines.append("Output variables:")
    for i, output in enumerate(graph_proto.outputs):
        lines.append(f"- Output {i}: {output}")

    return "\n".join(lines)
