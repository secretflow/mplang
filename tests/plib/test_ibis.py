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

import ibis
import pytest


class TestIbis:
    def test_train_test_split(self):
        pass

    @pytest.mark.parametrize("op", ["+", "-", "*", "/"])
    @pytest.mark.parametrize(
        "features,new_feature_name",
        [
            (["a", "b"], "r"),
            (["a", "b"], "a"),
            (["a", "a"], "r"),
            (["a", "a"], "a"),
            (["a", "c"], "r"),
        ],
    )
    def test_binary_op(self, op: str, features, new_feature_name):
        t = ibis.table(schema={"a": "int", "b": "int", "c": "float"}, name="table")

        match op:
            case "+":
                result_expr = t[features[0]] + t[features[1]]
            case "-":
                result_expr = t[features[0]] - t[features[1]]
            case "*":
                result_expr = t[features[0]] * t[features[1]]
            case "/":
                result_expr = t[features[0]] / t[features[1]]
            case _:
                raise ValueError(f"Unsupported operation: {op}")

        new_table = t.mutate(**{new_feature_name: result_expr})

        assert new_feature_name in new_table.schema()
        res = self._eval(
            new_table, {"a": [1, 2, 3], "b": [4, 5, 6], "c": [4.0, 5.0, 6.0]}
        )
        print(f"{new_feature_name} = {features[0]} {op} {features[1]}:\n {res}")

    # def test_case_when(self):
    #     pass
    @pytest.mark.parametrize("columns", [["a", "b", "c", "d"]])
    @pytest.mark.parametrize("astype", ["int", "float", "string"])
    def test_cast(self, columns: list[str], astype: str):
        t = ibis.table(
            schema={"a": "int", "b": "float", "c": "string", "d": "bool"}, name="table"
        )
        # exprs = {col: t[col].cast(astype) for col in columns}
        exprs = {}
        for name in columns:
            col = t[name]
            # substrait donot support strip/trim and try_cast
            if col.type().is_string():
                col = col.strip()
            exprs[name] = col.cast(astype)
        new_t = t.mutate(**exprs)
        input = {
            "a": [1, 2, 3],
            "b": [4.1, 5.1, 6.1],
            "c": ["7", "8", "9"],  # "8.1" -> 8 will panic
            "d": [True, False, True],
        }
        res = self._eval(new_t, input)
        print(f"columns:{columns} cast to {astype}\n {res}")

    @pytest.mark.parametrize(
        "features,op,operands",
        [
            (["a1", "a2", "b1"], "standardize", []),
            (["a1", "a2", "b1"], "normalize", []),
            (["a1", "b1"], "range_limit", ["1", "2"]),
            (["a1", "b1"], "unary", ["+", "+", "1"]),  # unary_+
            (["a1", "b1"], "unary", ["+", "-", "1"]),  # unary_-
            (["a1", "b1"], "unary", ["-", "-", "1"]),  # unary_reverse_-
            (["a1", "b1"], "unary", ["+", "*", "2"]),  # unary_*
            (["a1", "b1"], "unary", ["+", "/", "3"]),  # unary_/
            (["a1", "b1"], "reciprocal", []),
            (["a1", "b1"], "round", []),
            (["a1", "b1"], "log_round", ["10"]),
            (["a1", "b1"], "log", ["e", "10"]),
            (["a1", "b1"], "log", ["2", "10"]),
            (["b1"], "sqrt", []),
            (["a1", "b1"], "exp", []),
            (["a3"], "length", []),
            (["a3"], "substr", ["0", "2"]),
        ],
    )
    def test_feature_calculate(self, features: list[str], op: str, operands: list[str]):
        t: ibis.Table = ibis.table(
            schema={"a1": "float", "a2": "float", "a3": "string", "b1": "int"},
            name="table",
        )

        def _standardize(col: ibis.Column):
            mean_expr = col.mean()
            std_expr = col.std()
            return (
                ibis.case()
                .when(std_expr == 0, 0)
                .else_((col - mean_expr) / std_expr)
                .end()
            )

        def _normalize(col: ibis.Column):
            # norm = (x-min)/(max-min)
            min_expr = col.min()
            max_expr = col.max()

            denominator = max_expr - min_expr

            return (
                ibis.case()
                .when(denominator == 0, 0)
                .else_((col - min_expr) / denominator)
                .end()
            )

        def _range_limit(col: ibis.Column):
            op_cnt = len(operands)
            if op_cnt != 2:
                raise ValueError(
                    f"range limit operator need 2 operands, but got {op_cnt}"
                )
            op0 = float(operands[0])
            op1 = float(operands[1])
            if op0 > op1:
                raise ValueError(
                    f"range limit operator expect min <= max, but get [{op0}, {op1}]"
                )
            new_col = (
                ibis.case()
                .when(col < op0, op0)  # 条件 1: col < op0 → 返回 op0
                .when(col > op1, op1)  # 条件 2: col > op1 → 返回 op1
                .else_(col)  # 默认情况 → 返回 col
                .end()
            )
            return new_col

        def _unary(col: ibis.Column):
            op_cnt = len(operands)
            if op_cnt != 3:
                raise ValueError(f"unary operator needs 3 operands, but got {op_cnt}")
            op0 = operands[0]
            if op0 not in ["+", "-"]:
                raise ValueError(f"unary op0 should be [+ -], but get {op0}")
            op1 = operands[1]
            if op1 not in ["+", "-", "*", "/"]:
                raise ValueError(f"unary op1 should be [+ - * /], but get {op1}")
            op3 = float(operands[2])
            if op1 == "+":
                new_col = col + op3
            elif op1 == "-":
                new_col = col - op3 if op0 == "+" else op3 - col
            elif op1 == "*":
                new_col = col * op3
            elif op1 == "/":
                if op0 == "+":
                    if op3 == 0:
                        raise ValueError("unary operator divide zero")
                    new_col = col / op3
                else:
                    new_col = op3 / col
            return new_col

        def _reciprocal(col: ibis.Column):
            return 1 / col

        def _round(col: ibis.Column):
            return col.round()

        def _log_round(col: ibis.Column):
            op_cnt = len(operands)
            if op_cnt != 1:
                raise ValueError(f"log operator needs 1 operands, but got {op_cnt}")
            op0 = float(operands[0])
            return (col + op0).log(2).round(digits=2)

        def _log(col: ibis.Column):
            import math

            op_cnt = len(operands)
            if op_cnt != 2:
                raise ValueError(f"log operator needs 2 operands, but got {op_cnt}")

            op0 = operands[0]
            op1 = float(operands[1])
            adjusted_col = col + op1

            if op0 == "e":
                # 自然对数:log_e(x) = ln(x)
                new_col = adjusted_col.log(math.e)
            else:
                # 任意底数对数:log_base(x)
                base = float(op0)
                new_col = adjusted_col.log(base)

            return new_col

        def _sqrt(col: ibis.Column):
            return col.sqrt()

        def _exp(col: ibis.Column):
            return col.exp()

        def _length(col: ibis.Column):
            return col.length()

        def _substr(col: ibis.Column):
            op_cnt = len(operands)
            if op_cnt != 2:
                raise ValueError(f"substr operator needs 2 operands, but got {op_cnt}")
            start = int(operands[0])
            length = int(operands[1])
            return col.substr(start + 1, length)

        fn_map = {
            "standardize": _standardize,
            "normalize": _normalize,
            "range_limit": _range_limit,
            "unary": _unary,
            "reciprocal": _reciprocal,
            "round": _round,
            "log_round": _log_round,
            "log": _log,
            "sqrt": _sqrt,
            "exp": _exp,
            "length": _length,
            "substr": _substr,
        }
        fn = fn_map[op]
        exprs = {name: fn(t[name]) for name in features}
        new_t = t.mutate(**exprs)

        input = {
            "a1": [i * (-0.8) for i in range(3)],
            "a2": [0.1] * 3,
            "a3": ["AAA", "BBB", "CCC"],
            "b1": list(range(3)),
        }
        res = self._eval(new_t, input)
        print(f"calulate {op},{operands}\n {res}")

    def test_fillna(self):
        pass

    def test_onehot(self):
        pass

    def test_binning(self):
        pass

    def test_woe_binning(self):
        pass

    def test_sql(self):
        # ibis don't support parse sql
        pass

    def test_score_card(self):
        pass

    def _eval(self, expr: ibis.Table, input: dict):
        sql = ibis.to_sql(expr)

        import duckdb
        import pandas as pd

        df = pd.DataFrame(input)
        conn = duckdb.connect(":memory:")
        conn.register("table", df)
        result_df = conn.execute(sql).fetchdf()
        return result_df

    # def _eval(self, expr: ibis.Table, input: dict):
    #     import pyarrow as pa
    #     import pyarrow.substrait as substrait
    #     from ibis_substrait.compiler.core import SubstraitCompiler

    #     # encode ir
    #     compiler = SubstraitCompiler()
    #     plan = compiler.compile(expr)
    #     fn_txt = plan.SerializeToString()

    #     # # decode ir
    #     # substrait_plan = substrait.Plan()
    #     # substrait_plan.ParseFromString(fn_txt)

    #     # exec ir use pyarrow
    #     tbl = pa.Table.from_pydict(input)

    #     def table_provider(names, schema):
    #         return tbl

    #     reader = pa.substrait.run_query(fn_txt, table_provider=table_provider)
    #     result = reader.read_all()
    #     return result
