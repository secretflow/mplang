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

from mplang.core.dtype import FLOAT32, INT32, UINT64
from mplang.core.expr import (
    AccessExpr,
    CallExpr,
    CondExpr,
    ConvExpr,
    EvalExpr,
    FuncDefExpr,
    ShflExpr,
    ShflSExpr,
    TupleExpr,
    VariableExpr,
    WhileExpr,
)
from mplang.core.mask import Mask
from mplang.core.mptype import MPType, Rank


class TestEvalExpr:
    """Test EvalExpr expression type."""

    def test_basic_evaluation(self, pmask_2p, tensor_info_2d, pfunc_2i1o):
        """Test basic eval expression with valid arguments."""
        mptype1 = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        mptype2 = MPType.tensor(INT32, (), pmask_2p)
        arg1 = VariableExpr("x", mptype1)
        arg2 = VariableExpr("y", mptype2)

        expr = EvalExpr(pfunc_2i1o, [arg1, arg2])

        assert len(expr.mptypes) == 1
        assert expr.mptypes[0].dtype == FLOAT32
        assert expr.mptypes[0].shape == (2, 3)
        assert expr.mptypes[0].pmask == pmask_2p

    def test_wrong_argument_count(self, pfunc_2i1o, pmask_2p, tensor_info_2d):
        """Test that EvalExpr raises ValueError with wrong number of arguments."""
        mptype1 = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        arg1 = VariableExpr("x", mptype1)

        with pytest.raises(ValueError):
            EvalExpr(pfunc_2i1o, [arg1])  # Missing second argument

    def test_mask_logic(self, pfunc_2i1o, tensor_info_2d):
        """Test comprehensive mask logic for EvalExpr."""
        # Create argument expressions with different pmasks
        pmask1 = Mask(7)  # 0b111 (parties 0, 1, 2)
        pmask2 = Mask(3)  # 0b011 (parties 0, 1)

        mptype1 = MPType.tensor(FLOAT32, (2, 3), pmask1)
        mptype2 = MPType.tensor(INT32, (), pmask2)
        arg1 = VariableExpr("x", mptype1)
        arg2 = VariableExpr("y", mptype2)

        # Test case 1: No rmask provided - should use deduced pmask (intersection)
        expr_no_rmask = EvalExpr(pfunc_2i1o, [arg1, arg2])
        expected_deduced = pmask1 & pmask2  # Should be Mask(3)
        assert expr_no_rmask.mptypes[0].pmask == expected_deduced

        # Test case 2: rmask is subset of deduced pmask - should use rmask
        rmask_subset = Mask(1)  # 0b001 (party 0 only)
        expr_rmask_subset = EvalExpr(pfunc_2i1o, [arg1, arg2], rmask_subset)
        assert expr_rmask_subset.mptypes[0].pmask == rmask_subset

        # Test case 3: rmask is not subset of deduced pmask - should raise error
        rmask_not_subset = Mask(4)  # 0b100 (party 2 only)
        expr_invalid = EvalExpr(pfunc_2i1o, [arg1, arg2], rmask_not_subset)
        with pytest.raises(ValueError, match="not a subset of deduced pmask"):
            _ = expr_invalid.mptypes  # Exception happens when accessing mptypes

        # Test case 4: Create expressions with None pmask using ShflExpr
        shuffle_src_type = MPType.tensor(FLOAT32, (2, 3), Mask(3))
        shuffle_index_type = MPType.tensor(INT32, (), Mask(3))
        shuffle_src = VariableExpr("shuffle_src", shuffle_src_type)
        shuffle_index = VariableExpr("shuffle_index", shuffle_index_type)
        shuffle_expr = ShflExpr(shuffle_src, shuffle_index)  # This will have None pmask

        # When one arg has None pmask, rmask should be used if provided
        rmask_force = Mask(5)  # 0b101
        expr_force_rmask = EvalExpr(pfunc_2i1o, [shuffle_expr, arg2], rmask_force)
        assert expr_force_rmask.mptypes[0].pmask == rmask_force

        # Test case 5: No rmask and some args have None pmask - should get None
        expr_mixed_none = EvalExpr(pfunc_2i1o, [shuffle_expr, arg2])
        assert expr_mixed_none.mptypes[0].pmask is None


class TestVariableExpr:
    """Test VariableExpr expression type."""

    def test_type_computation(self, pmask_2p):
        """Test that variable expression returns the provided type."""
        mptype = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        expr = VariableExpr("param1", mptype)

        # VariableExpr should now return the provided type
        assert len(expr.mptypes) == 1
        assert expr.mptypes[0] == mptype
        assert expr.mptype == mptype

    def test_parameter_name(self, pmask_2p):
        """Test that parameter name is stored correctly."""
        mptype = MPType.tensor(UINT64, (), pmask_2p)
        expr = VariableExpr("test_param", mptype)
        assert expr.name == "test_param"


class TestTupleExpr:
    """Test TupleExpr expression type."""

    def test_type_computation(self, pmask_2p, tensor_info_2d):
        """Test tuple expression type computation."""
        mptype1 = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        mptype2 = MPType.tensor(INT32, (), pmask_2p)
        arg1 = VariableExpr("x", mptype1)
        arg2 = VariableExpr("y", mptype2)

        expr = TupleExpr([arg1, arg2])

        assert len(expr.mptypes) == 2
        assert expr.mptypes[0].dtype == FLOAT32
        assert expr.mptypes[0].shape == (2, 3)
        assert expr.mptypes[1].dtype == INT32
        assert expr.mptypes[1].shape == ()

    def test_empty_arguments(self):
        """Test tuple expression with empty arguments."""
        expr = TupleExpr([])
        assert len(expr.mptypes) == 0

    def test_multi_output_argument_rejection(self, pmask_2p, tensor_info_2d):
        """Test that TupleExpr rejects multi-output arguments."""
        # Create a multi-output expression by using another TupleExpr
        mptype1 = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        mptype2 = MPType.tensor(INT32, (), pmask_2p)
        arg1 = VariableExpr("x", mptype1)
        arg2 = VariableExpr("y", mptype2)

        # First create a valid TupleExpr with 2 outputs
        multi_output_expr = TupleExpr([arg1, arg2])
        assert multi_output_expr.num_outputs == 2  # This has 2 outputs

        # TupleExpr should reject multi-output expressions
        with pytest.raises(ValueError, match="single-output expressions"):
            TupleExpr([multi_output_expr])


class TestAccessExpr:
    """Test AccessExpr expression type."""

    def test_valid_access(self, pmask_2p, tensor_info_2d):
        """Test valid element access."""
        mptype1 = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        mptype2 = MPType.tensor(INT32, (), pmask_2p)
        arg1 = VariableExpr("x", mptype1)
        arg2 = VariableExpr("y", mptype2)
        tuple_expr = TupleExpr([arg1, arg2])

        # Test access to first element
        expr1 = AccessExpr(tuple_expr, 0)
        assert len(expr1.mptypes) == 1
        assert expr1.mptypes[0].dtype == FLOAT32
        assert expr1.mptypes[0].shape == (2, 3)

        # Test access to second element
        expr2 = AccessExpr(tuple_expr, 1)
        assert len(expr2.mptypes) == 1
        assert expr2.mptypes[0].dtype == INT32
        assert expr2.mptypes[0].shape == ()

    def test_out_of_bounds_access(self, pmask_2p, tensor_info_2d):
        """Test that out of bounds access raises IndexError."""
        mptype1 = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        mptype2 = MPType.tensor(INT32, (), pmask_2p)
        arg1 = VariableExpr("x", mptype1)
        arg2 = VariableExpr("y", mptype2)
        tuple_expr = TupleExpr([arg1, arg2])

        expr_out_of_bounds = AccessExpr(tuple_expr, 2)
        with pytest.raises(IndexError):
            _ = expr_out_of_bounds.mptypes


class TestFuncDefExpr:
    """Test FuncDefExpr expression type."""

    def test_identity_function(self, pmask_2p):
        """Test function definition expression with identity function."""
        # Create a variable expression for the function body
        mptype = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        root = VariableExpr("x", mptype)

        # Create function definition with string parameter
        identity_expr = FuncDefExpr(["x"], root)

        # The function's type should be the same as its body's type
        assert identity_expr.params == ["x"]
        assert isinstance(identity_expr.body, VariableExpr)
        assert identity_expr.body.name == "x"

        # The mptypes should be the same as the body
        assert len(identity_expr.mptypes) == 1
        assert identity_expr.mptypes[0] == mptype


class TestCallExpr:
    """Test CallExpr expression type."""

    def test_function_call(self, pmask_2p, tensor_info_2d):
        """Test function call expression."""
        # Create a function definition
        mptype = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        x_var = VariableExpr("x", mptype)
        func = FuncDefExpr(["x"], x_var)  # Identity function

        # Create an argument
        arg_mptype = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        arg = VariableExpr("arg", arg_mptype)

        expr = CallExpr(func, [arg])

        # The call should have the correct structure
        assert isinstance(expr.fn, FuncDefExpr)
        assert expr.fn.params == ["x"]
        assert len(expr.args) == 1
        assert isinstance(expr.args[0], VariableExpr)

        # mptypes should now work since VariableExpr has explicit type
        assert len(expr.mptypes) == 1
        assert expr.mptypes[0] == mptype


class TestCondExpr:
    """Test CondExpr expression type."""

    def test_conditional_expression(self, pmask_2p, tensor_info_2d):
        """Test conditional expression."""
        # Create predicate (boolean scalar)
        pred_mptype = MPType.tensor(INT32, (), pmask_2p)
        pred = VariableExpr("pred", pred_mptype)

        # Create then and else functions
        mptype = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        x_var = VariableExpr("x", mptype)
        then_fn = FuncDefExpr(["x"], x_var)  # Identity
        else_fn = FuncDefExpr(["x"], x_var)  # Identity

        # Create arguments
        arg_mptype = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        arg = VariableExpr("arg", arg_mptype)

        expr = CondExpr(pred, then_fn, else_fn, [arg])

        # Check structure
        assert isinstance(expr.pred, VariableExpr)
        assert isinstance(expr.then_fn, FuncDefExpr)
        assert isinstance(expr.else_fn, FuncDefExpr)
        assert len(expr.args) == 1

        # mptypes should now work since VariableExpr has explicit type
        assert len(expr.mptypes) == 1
        assert expr.mptypes[0] == mptype


class TestWhileExpr:
    """Test WhileExpr expression type."""

    def test_while_loop(self, pmask_2p, tensor_info_2d):
        """Test while loop expression."""
        # Create condition and body functions
        mptype = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        x_var = VariableExpr("x", mptype)

        # Create condition function returning boolean
        bool_mptype = MPType.tensor(INT32, (), pmask_2p)
        bool_var = VariableExpr("bool_result", bool_mptype)
        cond_fn = FuncDefExpr(["x"], bool_var)
        body_fn = FuncDefExpr(["x"], x_var)  # Identity

        # Create initial state
        init_mptype = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        init = VariableExpr("init", init_mptype)

        expr = WhileExpr(cond_fn, body_fn, [init])

        # Check structure
        assert isinstance(expr.cond_fn, FuncDefExpr)
        assert isinstance(expr.body_fn, FuncDefExpr)
        assert isinstance(expr.args[0], VariableExpr)  # First arg is the init value
        assert len(expr.args) == 1
        assert expr.args[0] is init

        # The while expression should return the type of init
        assert len(expr.mptypes) == 1
        assert expr.mptypes[0].dtype == FLOAT32
        assert expr.mptypes[0].shape == (2, 3)
        assert expr.mptypes[0].pmask == pmask_2p


class TestConvExpr:
    """Test ConvExpr expression type."""

    def test_convergence_expression(self, pmask_1p, tensor_info_2d):
        """Test convergence expression."""
        # Use disjoint masks for convergence - party 0 and party 1 separately
        pmask_party0 = pmask_1p  # Mask(1) = party 0
        pmask_party1 = Mask(2)  # party 1 only

        mptype1 = MPType.tensor(FLOAT32, (2, 3), pmask_party0)
        mptype2 = MPType.tensor(FLOAT32, (2, 3), pmask_party1)
        var1 = VariableExpr("var1", mptype1)
        var2 = VariableExpr("var2", mptype2)

        expr = ConvExpr([var1, var2])

        assert len(expr.mptypes) == 1
        assert expr.mptypes[0].dtype == FLOAT32
        assert expr.mptypes[0].shape == (2, 3)
        # The output pmask should be the union of input pmasks
        assert expr.mptypes[0].pmask == Mask(3)  # 0b11 = parties 0 and 1


class TestShflSExpr:
    """Test ShflSExpr expression type."""

    def test_static_shuffle(self, tensor_info_2d):
        """Test static shuffle expression."""
        pmask = Mask(3)  # 0b11 (parties 0 and 1)
        mptype = MPType.tensor(FLOAT32, (2, 3), pmask)
        src_val = VariableExpr("src", mptype)
        new_pmask = Mask(7)  # 0b111 (parties 0, 1, and 2)
        # new_pmask has 3 bits set, so we need 3 src_ranks
        # These should correspond to which source parties provide data for each destination party
        src_ranks = [Rank(0), Rank(1), Rank(0)]  # Party 2 gets data from party 0

        expr = ShflSExpr(src_val, new_pmask, src_ranks)

        assert len(expr.mptypes) == 1
        assert expr.mptypes[0].dtype == FLOAT32
        assert expr.mptypes[0].shape == (2, 3)
        assert expr.mptypes[0].pmask == new_pmask


class TestShflExpr:
    """Test ShflExpr expression type."""

    def test_dynamic_shuffle(self, pmask_2p, tensor_info_2d):
        """Test dynamic shuffle expression."""
        src_mptype = MPType.tensor(FLOAT32, (2, 3), pmask_2p)
        index_mptype = MPType.tensor(INT32, (), pmask_2p)
        src = VariableExpr("src", src_mptype)
        index = VariableExpr("index", index_mptype)

        expr = ShflExpr(src, index)

        assert len(expr.mptypes) == 1
        assert expr.mptypes[0].dtype == FLOAT32
        assert expr.mptypes[0].shape == (2, 3)
        assert expr.mptypes[0].pmask is None  # pmask becomes None for dynamic shuffle
