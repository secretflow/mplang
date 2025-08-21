# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from mplang.core.dtype import FLOAT32, INT32
from mplang.core.mask import Mask
from mplang.core.tensor import TensorType
from mplang.expr import ConstExpr, Expr, ExprTransformer, Printer
from mplang.expr.ast import RankExpr


@pytest.fixture
def pmask_2p():
    """Provide a 2-party mask for testing."""
    return Mask(0b11)


class TestPrinter:
    """Test Printer visitor implementation."""

    def test_printer_basic_output(self, pmask_2p):
        """Test that printer generates expected output for expression graph."""
        # Create a simple expression graph with concrete expressions
        import numpy as np

        # Create proper data bytes for a 2x3 float32 array
        data1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).tobytes()
        const1 = ConstExpr(TensorType(FLOAT32, (2, 3)), data1, pmask_2p)
        # Create proper data bytes for a scalar int32
        data2 = np.array(42, dtype=np.int32).tobytes()
        const2 = ConstExpr(TensorType(INT32, ()), data2, pmask_2p)
        from mplang.expr.ast import AccessExpr, TupleExpr

        tuple_expr = TupleExpr([const1, const2])
        access_expr = AccessExpr(tuple_expr, 0)

        # Use the printer
        printer = Printer()
        output = printer.print_expr(access_expr)

        # Verify expected components are in the output
        assert "pconst" in output
        assert "tuple" in output

    def test_printer_simple_expression(self, pmask_2p):
        """Test printer with a simple constant expression."""
        import numpy as np

        # Create proper data bytes for a 2x3 float32 array
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).tobytes()
        const = ConstExpr(TensorType(FLOAT32, (2, 3)), data, pmask_2p)

        printer = Printer()
        output = printer.print_expr(const)

        assert "pconst" in output
        assert isinstance(output, str)
        assert len(output) > 0


class TestExprTransformer:
    """Test ExprTransformer visitor implementation."""

    def test_const_transformer(self, pmask_2p):
        """Test transformer that modifies constant expressions."""
        # Create a simple expression
        import numpy as np

        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).tobytes()
        const = ConstExpr(TensorType(FLOAT32, (2, 3)), data, pmask_2p)
        original_length = len(const.data_bytes)

        # Create a transformer that doubles constants
        def double_const(expr: Expr) -> Expr:
            if isinstance(expr, ConstExpr):
                return ConstExpr(expr.typ, expr.data_bytes * 2, expr.pmask)
            return expr

        transformer = ExprTransformer({"const": double_const})
        result = const.accept(transformer)

        # Verify transformation
        assert isinstance(result, ConstExpr)
        assert len(result.data_bytes) == original_length * 2
        assert result.typ == const.typ
        assert result.pmask == const.pmask

    def test_identity_transformer(self, pmask_2p):
        """Test transformer that doesn't modify expressions."""
        import numpy as np

        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).tobytes()
        const = ConstExpr(TensorType(FLOAT32, (2, 3)), data, pmask_2p)

        # Identity transformer - returns expression unchanged
        transformer = ExprTransformer({})
        result = const.accept(transformer)

        # Should be unchanged (but may be a different instance)
        assert isinstance(result, ConstExpr)
        assert result.data_bytes == const.data_bytes
        assert result.typ == const.typ
        assert result.pmask == const.pmask

    @pytest.mark.parametrize(
        "expr_type,transform_key",
        [
            ("const", "const"),
            ("rank", "rank"),
        ],
    )
    def test_selective_transformation(self, pmask_2p, expr_type, transform_key):
        """Test that transformer only affects specified expression types."""
        # Create expressions
        import numpy as np

        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32).tobytes()
        const = ConstExpr(TensorType(FLOAT32, (2, 3)), data, pmask_2p)
        rank = RankExpr(pmask_2p)

        expressions = {"const": const, "rank": rank}
        target_expr = expressions[expr_type]

        # Create transformer that only transforms the target type
        def modify_expr(expr: Expr) -> Expr:
            if expr_type == "const" and isinstance(expr, ConstExpr):
                modified_data = np.array(
                    [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]], dtype=np.float32
                ).tobytes()
                return ConstExpr(expr.typ, modified_data, expr.pmask)
            return expr

        transformer = ExprTransformer({transform_key: modify_expr})

        # Transform and verify
        if expr_type == "const":
            result = target_expr.accept(transformer)
            expected_data = np.array(
                [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]], dtype=np.float32
            ).tobytes()
            assert result.data_bytes == expected_data
        else:
            # For other types, just verify no error occurs
            result = target_expr.accept(transformer)
            assert result is not None
