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

"""MPIR (Multi-Party Intermediate Representation) serialization module.

This module provides functionality for serializing and deserializing
expression-based computation graphs to and from protobuf representations.
It serves as the bridge between in-memory expression trees and their
serialized form for storage or transmission.

Key components:
- Writer: Serializes Expr objects to GraphProto
- Reader: Deserializes GraphProto back to Expr objects
- Conversion functions: Handle mapping between Python types and protobuf types
"""

from __future__ import annotations

from typing import Any

import numpy as np
import spu.libspu as spu_api

from mplang.core.dtype import DATE, JSON, STRING, TIME, TIMESTAMP, DType
from mplang.core.expr import Expr, FuncDefExpr
from mplang.core.expr.ast import (
    AccessExpr,
    CallExpr,
    CondExpr,
    ConvExpr,
    EvalExpr,
    ShflExpr,
    ShflSExpr,
    TupleExpr,
    VariableExpr,
    WhileExpr,
)
from mplang.core.expr.walk import walk
from mplang.core.mask import Mask
from mplang.core.mptype import MPType
from mplang.core.pfunc import PFunction
from mplang.core.table import TableType
from mplang.core.tensor import TensorType
from mplang.protos.v1alpha1 import mpir_pb2

# Single mapping table for dtype conversion
DTYPE_MAPPING = {
    np.float32: mpir_pb2.DataType.F32,
    np.uint8: mpir_pb2.DataType.U8,
    np.int8: mpir_pb2.DataType.I8,
    np.uint16: mpir_pb2.DataType.U16,
    np.int16: mpir_pb2.DataType.I16,
    np.int32: mpir_pb2.DataType.I32,
    np.int64: mpir_pb2.DataType.I64,
    np.str_: mpir_pb2.DataType.STRING,
    np.bool_: mpir_pb2.DataType.BOOL,
    np.float16: mpir_pb2.DataType.F16,
    np.float64: mpir_pb2.DataType.F64,
    np.uint32: mpir_pb2.DataType.U32,
    np.uint64: mpir_pb2.DataType.U64,
    np.complex64: mpir_pb2.DataType.COMPLEX64,
    np.complex128: mpir_pb2.DataType.COMPLEX128,
}

# Additional mapping for table-only DType constants
DTYPE_TO_PROTO_MAPPING = {
    # Map DType constants to protobuf enums
    STRING: mpir_pb2.DataType.STRING,
    DATE: mpir_pb2.DataType.DATE,
    TIME: mpir_pb2.DataType.TIME,
    TIMESTAMP: mpir_pb2.DataType.TIMESTAMP,
    JSON: mpir_pb2.DataType.JSON,
}


def dtype_to_proto(dtype_like: Any) -> Any:
    """Convert dtype (DType, NumPy dtype, or type) to protobuf DataType.

    Args:
        dtype_like: A DType, NumPy dtype, or Python type to convert.

    Returns:
        The corresponding protobuf DataType enum value.

    Raises:
        ValueError: If the dtype is not supported for conversion.
    """
    # If it's already a DType, check for direct mapping first
    if isinstance(dtype_like, DType):
        # Check for table-only types first
        if dtype_like in DTYPE_TO_PROTO_MAPPING:
            return DTYPE_TO_PROTO_MAPPING[dtype_like]

        # For regular types, convert to numpy for protobuf mapping
        try:
            numpy_dtype = dtype_like.to_numpy()
            key_type = numpy_dtype.type
        except ValueError as e:
            # Handle table-only types that can't be converted to numpy
            raise ValueError(
                f"Unsupported dtype for proto conversion: {dtype_like}. This is likely a table-only type that cannot be converted to a numpy dtype. Please ensure the dtype is supported for proto conversion."
            ) from e
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
    """Convert protobuf DataType enum to DType.

    Args:
        dtype_enum: The protobuf DataType enum value to convert.

    Returns:
        The corresponding DType object.

    Raises:
        ValueError: If the enum value is not supported.
    """
    # Check for table-only types first
    for dtype_obj, proto_enum in DTYPE_TO_PROTO_MAPPING.items():
        if proto_enum == dtype_enum:
            return dtype_obj

    # Find the numpy type for the given enum by searching the mapping
    for numpy_type, proto_enum in DTYPE_MAPPING.items():
        if proto_enum == dtype_enum:
            # Convert numpy type to dtype
            try:
                np_dtype = np.dtype(numpy_type)
            except TypeError as e:
                raise ValueError(f"Cannot create numpy dtype from {numpy_type}") from e

            # Special handling for string types since DType.from_numpy doesn't support them
            if np_dtype.kind == "U":  # Unicode string
                # Return the STRING constant for table-only string types
                return STRING
            else:
                try:
                    return DType.from_numpy(np_dtype)
                except ValueError as e:
                    raise ValueError(
                        f"Cannot convert numpy dtype {np_dtype} to DType"
                    ) from e

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
        attr_proto.func.name = py_value.fn_name or ""
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
    elif isinstance(py_value, Mask):
        # Handle Mask objects by storing as int
        attr_proto.type = mpir_pb2.AttrProto.INT
        attr_proto.i = int(py_value)
    else:
        raise TypeError(f"Unsupported attribute type: {type(py_value)}")
    return attr_proto


class Writer:
    """Writer for serializing Expr-based expressions to GraphProto.

    This class traverses an expression tree and converts it into a serialized
    GraphProto representation. It handles various expression types and ensures
    that all dependencies are properly serialized before the expressions that
    depend on them.
    """

    def __init__(self, var_name_mapping: dict[str, str] | None = None):
        """Initialize the Writer.

        Args:
            var_name_mapping: Optional mapping of variable names to replace during serialization.
        """
        self._counter = 0
        self._expr_ids: dict[int, str] = {}  # Use expr id instead of Node
        self._nodes: list[mpir_pb2.NodeProto] = []
        self._var_name_mapping = var_name_mapping or {}

    def expr_name(self, expr: Expr) -> str:
        """Get or create a name for an expression.

        Args:
            expr: The expression to name.

        Returns:
            A unique name for the expression.
        """
        expr_id = id(expr)
        if expr_id not in self._expr_ids:
            self._expr_ids[expr_id] = f"%{self._counter}"
            self._counter += 1
        return self._expr_ids[expr_id]

    def value_name(self, expr: Expr, out_idx: int = 0) -> str:
        """Get value name for expression output.

        Args:
            expr: The expression.
            out_idx: The output index for multi-output expressions.

        Returns:
            A name for the specific output of the expression.
        """
        if len(expr.mptypes) == 1:
            return self.expr_name(expr)
        else:
            return f"{self.expr_name(expr)}:{out_idx}"

    # ------------------------- traversal and deps helpers -------------------------
    @staticmethod
    def _writer_deps(node: Expr) -> list[Expr]:
        """Dependencies for serialization order.

        Similar to dataflow deps, but with two important differences:
        - CallExpr: include the function value (fn) so we emit a func_def node
          in the outer graph before the call node.
        - FuncDefExpr: include body so we emit body producers before func_def.
        """
        if isinstance(node, EvalExpr):
            return list(node.args)
        if isinstance(node, TupleExpr):
            return list(node.args)
        if isinstance(node, CondExpr):
            # pred and actual args only; functions are serialized via attrs (nested graphs)
            return [node.pred, *node.args]
        if isinstance(node, WhileExpr):
            # initial state args only; functions are serialized via attrs (nested graphs)
            return list(node.args)
        if isinstance(node, ConvExpr):
            return list(node.vars)
        if isinstance(node, ShflSExpr):
            return [node.src_val]
        if isinstance(node, ShflExpr):
            return [node.src, node.index]
        if isinstance(node, AccessExpr):
            return [node.src]
        if isinstance(node, VariableExpr):
            return []
        if isinstance(node, FuncDefExpr):
            # ensure body producers are serialized first
            return [node.body]
        if isinstance(node, CallExpr):
            # include fn and args as deps so func_def appears before call
            return [node.fn, *node.args]
        return []

    def reset(self) -> None:
        """Reset writer state.

        Clears all internal state, allowing the writer to be reused for
        serializing a new expression tree.
        """
        self._counter = 0
        self._expr_ids.clear()
        self._nodes.clear()

    def _create_node_proto(self, expr: Expr, op_type: str) -> mpir_pb2.NodeProto:
        """Helper: Create a basic NodeProto with common fields set.

        Args:
            expr: The expression this node represents.
            op_type: The operation type for this node.

        Returns:
            A new NodeProto with basic fields set.
        """
        op = mpir_pb2.NodeProto()
        op.op_type = op_type
        op.name = self.expr_name(expr)
        return op

    def _add_output_info(self, op: mpir_pb2.NodeProto, expr: Expr) -> None:
        """Helper: Add output type information to a NodeProto.

        This method populates the output type information for a node based
        on the expression's mptypes.

        Args:
            op: The NodeProto to populate.
            expr: The expression providing the type information.
        """
        for out_info in expr.mptypes:
            out_proto = op.outs_info.add()

            if out_info.is_tensor:
                # Handle tensor type
                tensor_type = out_proto.tensor_type
                tensor_type.dtype = dtype_to_proto(out_info.dtype)
                tensor_type.shape_dims.extend(list(out_info.shape))
            elif out_info.is_table:
                # Handle table type
                table_type = out_proto.table_type
                for col_name, col_dtype in out_info.schema.columns:
                    column = table_type.columns.add()
                    column.name = col_name
                    column.dtype = dtype_to_proto(col_dtype)

            # Set pmask (now int64, -1 for dynamic mask)
            if out_info.pmask is not None:
                out_proto.pmask = int(out_info.pmask)
            else:
                out_proto.pmask = -1  # Dynamic mask

    def _add_expr_inputs(self, op: mpir_pb2.NodeProto, *exprs: Expr) -> None:
        """Helper: Add expression inputs to NodeProto.

        For multi-output expressions, this adds all outputs as inputs.

        Args:
            op: The NodeProto to add inputs to.
            exprs: The expressions to add as inputs.
        """
        for expr in exprs:
            op.inputs.extend([
                self.value_name(expr, i) for i in range(len(expr.mptypes))
            ])

    def _add_single_expr_inputs(self, op: mpir_pb2.NodeProto, *exprs: Expr) -> None:
        """Helper: Add single-output expression inputs to NodeProto.

        For expressions, this adds only the first (primary) output as input.

        Args:
            op: The NodeProto to add inputs to.
            exprs: The expressions to add as inputs.
        """
        for expr in exprs:
            op.inputs.append(self.value_name(expr, 0))

    def _add_attrs(self, op: mpir_pb2.NodeProto, **attrs: Any) -> None:
        """Helper: Add attributes to NodeProto.

        Args:
            op: The NodeProto to add attributes to.
            **attrs: The attributes to add (key-value pairs).
        """
        for key, value in attrs.items():
            if value is not None:  # Skip None values
                op.attrs[key].CopyFrom(attr_to_proto(value))

    def _finalize_node(self, op: mpir_pb2.NodeProto, expr: Expr) -> str:
        """Helper: Add output info, append to nodes, and return expr name.

        This method completes the node creation process by adding output
        information, appending the node to the list of nodes, and returning
        the expression name.

        Args:
            op: The completed NodeProto.
            expr: The expression the node represents.

        Returns:
            The name of the expression.
        """
        self._add_output_info(op, expr)
        self._nodes.append(op)
        return self.expr_name(expr)

    def dumps(self, expr: Expr) -> mpir_pb2.GraphProto:
        """Dump an expression to GraphProto using iterative walk traversal."""
        self.reset()

        # Walk in post-order so deps are serialized before users
        for node in walk(expr, get_deps=self._writer_deps, traversal="dfs_post_iter"):
            # Avoid double-emit if the same Expr object appears multiple times
            node_id = id(node)
            if node_id in self._expr_ids:
                continue
            # Emit node
            self._serialize_node(node)

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

    # ------------------------------- emitters --------------------------------
    def _serialize_node(self, expr: Expr) -> None:
        """Create and append a NodeProto for the given expr."""
        if isinstance(expr, EvalExpr):
            op = self._create_node_proto(expr, "eval")
            self._add_expr_inputs(op, *expr.args)
            self._add_attrs(op, pfunc=expr.pfunc, rmask=expr.rmask)
            self._finalize_node(op, expr)
        elif isinstance(expr, VariableExpr):
            op = self._create_node_proto(expr, "variable")
            mapped_name = self._var_name_mapping.get(expr.name, expr.name)
            self._add_attrs(op, name=mapped_name)
            self._finalize_node(op, expr)
        elif isinstance(expr, TupleExpr):
            op = self._create_node_proto(expr, "tuple")
            self._add_single_expr_inputs(op, *expr.args)
            self._finalize_node(op, expr)
        elif isinstance(expr, CondExpr):
            op = self._create_node_proto(expr, "cond")
            self._add_single_expr_inputs(op, expr.pred)
            self._add_expr_inputs(op, *expr.args)
            self._add_attrs(op, then_fn=expr.then_fn, else_fn=expr.else_fn)
            self._finalize_node(op, expr)
        elif isinstance(expr, CallExpr):
            op = self._create_node_proto(expr, "call")
            self._add_single_expr_inputs(op, expr.fn)
            self._add_expr_inputs(op, *expr.args)
            self._finalize_node(op, expr)
        elif isinstance(expr, WhileExpr):
            op = self._create_node_proto(expr, "while")
            self._add_expr_inputs(op, *expr.args)
            self._add_attrs(op, cond_fn=expr.cond_fn, body_fn=expr.body_fn)
            self._finalize_node(op, expr)
        elif isinstance(expr, ConvExpr):
            op = self._create_node_proto(expr, "conv")
            self._add_expr_inputs(op, *expr.vars)
            self._finalize_node(op, expr)
        elif isinstance(expr, ShflSExpr):
            op = self._create_node_proto(expr, "shfl_s")
            self._add_single_expr_inputs(op, expr.src_val)
            self._add_attrs(op, pmask=expr.pmask, src_ranks=expr.src_ranks)
            self._finalize_node(op, expr)
        elif isinstance(expr, ShflExpr):
            op = self._create_node_proto(expr, "shfl")
            self._add_single_expr_inputs(op, expr.src, expr.index)
            self._finalize_node(op, expr)
        elif isinstance(expr, AccessExpr):
            op = self._create_node_proto(expr, "access")
            op.inputs.append(self.value_name(expr.src, expr.index))
            self._add_attrs(op, index=expr.index)
            self._finalize_node(op, expr)
        elif isinstance(expr, FuncDefExpr):
            op = self._create_node_proto(expr, "func_def")
            self._add_expr_inputs(op, expr.body)
            self._add_attrs(op, params=expr.params)
            self._finalize_node(op, expr)
        else:
            raise TypeError(f"Unsupported expr type for serialization: {type(expr)}")


class Reader:
    """Reader for deserializing GraphProto back to Expr-based expressions.

    This class is responsible for converting serialized GraphProto representations
    back into executable expression trees. It handles the deserialization of
    various node types and manages dependencies between nodes to ensure proper
    reconstruction of the expression graph.
    """

    def __init__(self) -> None:
        self._value_cache: dict[str, Expr] = {}

    def loads(self, graph_proto: mpir_pb2.GraphProto) -> Expr | None:
        """Load an expression from a GraphProto.

        Args:
            graph_proto: The protobuf graph to deserialize

        Returns:
            The deserialized expression or None if empty
        """
        self._value_cache.clear()

        # Create a mapping for faster node lookup, checking for duplicate node names
        node_map = {}
        for node in graph_proto.nodes:
            if node.name in node_map:
                raise ValueError(
                    f"Duplicate node name detected in graph: '{node.name}'"
                )
            node_map[node.name] = node

        # Process nodes in topological order
        processed_nodes = set()

        def process_node(node_proto: mpir_pb2.NodeProto) -> None:
            """Process a single node and its dependencies."""
            if node_proto.name in processed_nodes:
                return

            # First process all dependencies
            for input_name in node_proto.inputs:
                dep_node_name = input_name.split(":")[0]
                if dep_node_name in node_map and dep_node_name not in processed_nodes:
                    process_node(node_map[dep_node_name])

            # Now process this node
            try:
                expr = self._create_expr_from_proto(node_proto)
                processed_nodes.add(node_proto.name)
                # Cache the expression
                self._value_cache[node_proto.name] = expr
            except Exception as e:
                raise ValueError(
                    f"Error processing node '{node_proto.name}' "
                    f"of type '{node_proto.op_type}': {e!s}"
                ) from e

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
        """Create an Expression from a NodeProto.

        This method delegates to specific creation methods based on the node type.
        """
        # Dispatch to appropriate creation method based on op_type
        creation_methods = {
            "eval": self._create_eval_expr,
            "variable": self._create_variable_expr,
            "tuple": self._create_tuple_expr,
            "cond": self._create_cond_expr,
            "while": self._create_while_expr,
            "access": self._create_access_expr,
            "func_def": self._create_func_def_expr,
            "shfl_s": self._create_shfl_s_expr,
            "shfl": self._create_shfl_expr,
            "conv": self._create_conv_expr,
            "call": self._create_call_expr,
        }

        if node_proto.op_type in creation_methods:
            return creation_methods[node_proto.op_type](node_proto)
        else:
            raise ValueError(f"Unsupported node type: {node_proto.op_type}")

    def _create_eval_expr(self, node_proto: mpir_pb2.NodeProto) -> EvalExpr:
        """Create an EvalExpr from a NodeProto."""
        # Parse inputs
        input_exprs = []
        for input_name in node_proto.inputs:
            dep_name = input_name.split(":")[0]
            if dep_name in self._value_cache:
                input_exprs.append(self._value_cache[dep_name])
            else:
                raise ValueError(f"Input {input_name} not found for eval node")

        # Parse function
        pfunc = self._proto_to_attr(node_proto.attrs["pfunc"])
        rmask = None
        if "rmask" in node_proto.attrs:
            rmask = self._proto_to_attr(node_proto.attrs["rmask"])

        # Fill in ins_info and outs_info for PFunction
        # ins_info from input expressions (use mptype for single type per value)
        ins_info: list[TensorType | TableType] = []
        for input_expr in input_exprs:
            # Use mptype directly for single MPType
            mptype = input_expr.mptype
            if mptype.is_tensor:
                ins_info.append(TensorType(mptype.dtype, mptype.shape))
            elif mptype.is_table:
                ins_info.append(mptype.schema)
            else:
                raise ValueError(f"unsupported type: {mptype}")

        # outs_info from NodeProto.outs_info
        outs_info: list[TensorType | TableType] = []
        for out_proto in node_proto.outs_info:
            if out_proto.HasField("tensor_type"):
                tensor_type_proto = out_proto.tensor_type
                dtype = proto_to_dtype(tensor_type_proto.dtype)
                shape = tuple(tensor_type_proto.shape_dims)
                outs_info.append(TensorType(dtype, shape))
            elif out_proto.HasField("table_type"):
                columns = [
                    (col.name, proto_to_dtype(col.dtype))
                    for col in out_proto.table_type.columns
                ]
                outs_info.append(TableType.from_pairs(columns))
            else:
                raise ValueError("Eval node currently only supports tensor types")

        # Create a complete PFunction with proper type information
        complete_pfunc = PFunction(
            fn_type=pfunc.fn_type,
            ins_info=ins_info,
            outs_info=outs_info,
            fn_name=pfunc.fn_name,
            fn_text=pfunc.fn_text,
            **pfunc.attrs,  # Restore attributes
        )

        return EvalExpr(complete_pfunc, input_exprs, rmask)

    def _create_variable_expr(self, node_proto: mpir_pb2.NodeProto) -> VariableExpr:
        """Create a VariableExpr from a NodeProto."""
        # Parse variable name
        name = self._proto_to_attr(node_proto.attrs["name"])

        # Parse type info from output info (VariableExpr needs a single MPType)
        if not node_proto.outs_info:
            raise ValueError("Variable node missing output info")

        mptype = self._proto_to_mptype(node_proto.outs_info[0])
        return VariableExpr(name, mptype)

    def _create_tuple_expr(self, node_proto: mpir_pb2.NodeProto) -> TupleExpr:
        """Create a TupleExpr from a NodeProto."""
        # Parse inputs
        input_exprs = []
        for input_name in node_proto.inputs:
            dep_name = input_name.split(":")[0]
            if dep_name in self._value_cache:
                input_exprs.append(self._value_cache[dep_name])
            else:
                raise ValueError(f"Input {input_name} not found for tuple node")

        return TupleExpr(input_exprs)

    def _create_cond_expr(self, node_proto: mpir_pb2.NodeProto) -> CondExpr:
        """Create a CondExpr from a NodeProto."""
        # Parse predicate and arguments
        pred_name = node_proto.inputs[0].split(":")[0]
        pred_expr = self._value_cache[pred_name]

        arg_exprs = []
        for input_name in node_proto.inputs[1:]:
            dep_name = input_name.split(":")[0]
            if dep_name in self._value_cache:
                arg_exprs.append(self._value_cache[dep_name])
            else:
                raise ValueError(f"Input {input_name} not found for cond node")

        # Parse functions
        then_fn = self._proto_to_attr(node_proto.attrs["then_fn"])
        else_fn = self._proto_to_attr(node_proto.attrs["else_fn"])

        return CondExpr(pred_expr, then_fn, else_fn, arg_exprs)

    def _create_while_expr(self, node_proto: mpir_pb2.NodeProto) -> WhileExpr:
        """Create a WhileExpr from a NodeProto."""
        # Parse arguments
        arg_exprs = []
        for input_name in node_proto.inputs:
            dep_name = input_name.split(":")[0]
            if dep_name in self._value_cache:
                arg_exprs.append(self._value_cache[dep_name])
            else:
                raise ValueError(f"Input {input_name} not found for while node")

        # Parse functions
        cond_fn = self._proto_to_attr(node_proto.attrs["cond_fn"])
        body_fn = self._proto_to_attr(node_proto.attrs["body_fn"])

        return WhileExpr(cond_fn, body_fn, arg_exprs)

    def _create_access_expr(self, node_proto: mpir_pb2.NodeProto) -> AccessExpr:
        """Create an AccessExpr from a NodeProto."""
        # Parse source expression
        input_name = node_proto.inputs[0]
        dep_name = input_name.split(":")[0]
        src_expr = self._value_cache[dep_name]

        # Parse index
        index = self._proto_to_attr(node_proto.attrs["index"])

        return AccessExpr(src_expr, index)

    def _create_func_def_expr(self, node_proto: mpir_pb2.NodeProto) -> FuncDefExpr:
        """Create a FuncDefExpr from a NodeProto."""
        # Parse body expression
        input_names = node_proto.inputs
        if not input_names:
            raise ValueError("FuncDef node missing body input")

        body_name = input_names[0].split(":")[0]
        body_expr = self._value_cache[body_name]

        # Parse parameters
        params = self._proto_to_attr(node_proto.attrs["params"])

        return FuncDefExpr(params, body_expr)

    def _create_shfl_s_expr(self, node_proto: mpir_pb2.NodeProto) -> ShflSExpr:
        """Create a ShflSExpr from a NodeProto."""
        # Parse source expression
        input_name = node_proto.inputs[0]
        dep_name = input_name.split(":")[0]
        src_val = self._value_cache[dep_name]

        # Parse attributes
        pmask = self._proto_to_attr(node_proto.attrs["pmask"])
        src_ranks = self._proto_to_attr(node_proto.attrs["src_ranks"])

        return ShflSExpr(src_val, pmask, src_ranks)

    def _create_shfl_expr(self, node_proto: mpir_pb2.NodeProto) -> ShflExpr:
        """Create a ShflExpr from a NodeProto."""
        # Parse source and index expressions
        src_name = node_proto.inputs[0].split(":")[0]
        index_name = node_proto.inputs[1].split(":")[0]
        src_expr = self._value_cache[src_name]
        index_expr = self._value_cache[index_name]

        return ShflExpr(src_expr, index_expr)

    def _create_conv_expr(self, node_proto: mpir_pb2.NodeProto) -> ConvExpr:
        """Create a ConvExpr from a NodeProto."""
        # Parse variable expressions
        var_exprs = []
        for input_name in node_proto.inputs:
            dep_name = input_name.split(":")[0]
            if dep_name in self._value_cache:
                var_exprs.append(self._value_cache[dep_name])
            else:
                raise ValueError(f"Input {input_name} not found for conv node")

        return ConvExpr(var_exprs)

    def _create_call_expr(self, node_proto: mpir_pb2.NodeProto) -> CallExpr:
        """Create a CallExpr from a NodeProto."""
        # Parse function and arguments
        fn_name = node_proto.inputs[0].split(":")[0]
        fn_expr = self._value_cache[fn_name]

        # Ensure function is FuncDefExpr
        if not isinstance(fn_expr, FuncDefExpr):
            raise ValueError(f"Call function must be FuncDefExpr, got {type(fn_expr)}")

        arg_exprs = []
        for input_name in node_proto.inputs[1:]:
            dep_name = input_name.split(":")[0]
            if dep_name in self._value_cache:
                arg_exprs.append(self._value_cache[dep_name])
            else:
                raise ValueError(f"Input {input_name} not found for call node")

        return CallExpr(fn_expr, arg_exprs)

    def _proto_to_mptype(self, type_proto: mpir_pb2.MPTypeProto) -> MPType:
        """Convert MPTypeProto to MPType."""
        # Convert pmask (now int64, -1 means dynamic mask (None))
        pmask_int = type_proto.pmask
        pmask = None if pmask_int == -1 else Mask(pmask_int)

        # Convert attributes
        attrs = {}
        for attr_name, attr_proto in type_proto.attrs.items():
            attrs[attr_name] = self._proto_to_attr(attr_proto)

        # Handle tensor type
        if type_proto.HasField("tensor_type"):
            tensor_type_proto = type_proto.tensor_type
            dtype = proto_to_dtype(tensor_type_proto.dtype)
            shape = tuple(tensor_type_proto.shape_dims)
            tensor_type = TensorType(dtype, shape)
            return MPType(tensor_type, pmask, attrs)

        # Handle table type
        elif type_proto.HasField("table_type"):
            table_type_proto = type_proto.table_type
            columns = []
            for column_proto in table_type_proto.columns:
                col_name = column_proto.name
                col_dtype = proto_to_dtype(column_proto.dtype)
                columns.append((col_name, col_dtype))

            table_type = TableType(tuple(columns))
            return MPType(table_type, pmask, attrs)

        else:
            raise ValueError(
                "MPTypeProto must specify either tensor_type or table_type"
            )

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
                ins_info=[],  # Will be inferred from input expressions
                outs_info=[],  # Will be inferred from context
                fn_name=attr_proto.func.name or None,
                fn_text=attr_proto.func.body if attr_proto.func.body else None,
                **attrs,  # Restore serialized attributes
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

    # Version information with compatibility check
    try:
        version = graph_proto.version
        version_str = f"{version.major}.{version.minor}.{version.patch}"
        lines.append(f"- Version: {version_str}")

        # Version compatibility check
        if version.major != 1:
            lines.append(f"  WARNING: Expected major version 1, got {version.major}")
    except AttributeError:
        lines.append("- Version: Unknown (missing version info)")
        version_str = "unknown"

    # Node and output counts
    lines.append(f"- Number of nodes: {len(graph_proto.nodes)}")
    lines.append(f"- Number of outputs: {len(graph_proto.outputs)}")
    lines.append(f"- Graph attributes: {len(graph_proto.attrs)}")
    lines.append("")

    # Node breakdown by operation type
    lines.append("Node breakdown by operation type:")
    op_counts: dict[str, int] = {}
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
