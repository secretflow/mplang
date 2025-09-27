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

import math

import ibis
import numpy.testing as npt
import pandas as pd
import pytest

# you can switch engine in ["duckdb", "substrait"]
# some testing may crash in "substrait"
_ibis_engine = "duckdb"


class TestIbis:
    def test_scalar_udf(self):
        def uppercase_string(s: str) -> str:
            return s.upper()

        ibis_uppercase_string = ibis.udf.scalar.builtin(name="uppercase_string")(
            uppercase_string
        )

        t = ibis.table(schema={"col": "str"}, name="table")
        expr = t.select(upper_str=ibis_uppercase_string(t["col"]))

        data = {"col": ["hello", "world"]}
        res = self._eval_duckdb(expr, [data], udfs=[uppercase_string])
        print(res)

    @pytest.mark.parametrize(
        "train_size,test_size,shuffle,random_state",
        [
            (0.75, None, False, None),
            (10, None, False, None),
            # (None, 0.35, True, None),
            # (None, 30, True, None),
        ],
    )
    def test_train_test_split(self, train_size, test_size, shuffle, random_state):
        if random_state is None:
            random_state = 1024
        if train_size is None and test_size is None:
            raise ValueError("train_size and test_size cannot all be None")

        data = {
            "id": [f"id{i}" for i in range(100)],
            "a1": [f"a{i}" for i in range(100)],
            "a2": list(range(100)),
        }

        schema = {"id": "str", "a1": "str", "a2": "int"}
        t = self._memtable(data, schema=schema)
        total = t.count().execute()

        if train_size is not None and train_size <= 1.0:
            train_size = int(total * train_size)
        if test_size is not None and test_size <= 0:
            test_size = int(total * test_size)
        if train_size is None:
            train_size = total - test_size
        if test_size is None:
            test_size = total - train_size
        if train_size + test_size > total:
            raise ValueError(f"split size overflow: {train_size}, {test_size}, {total}")

        # shuffle will crash.
        # duckdb.duckdb.InvalidInputException: Invalid Input Error:
        # More than one row returned by a subquery used as an expression - scalar subqueries can only return a single row.
        # if shuffle:
        #     # TODO: random_state
        #     t = t.order_by(ibis.random())

        train_data = t.limit(train_size).execute()
        test_data = t.limit(test_size, offset=train_size).execute()
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        print(
            f"totol_size: {total}, train_size:{train_size}, test_size:{test_size}, shuffle: {shuffle}"
        )
        print(f"train_data: {train_data.shape} \n {train_data.head(5)}")
        print(f"test_data:  {test_data.shape}  \n {test_data.head(5)}")

    def test_join(self):
        schema1 = {"id1": "str", "id2": "str", "a1": "int", "a2": "bool"}
        schema2 = {"id3": "str", "id4": "str", "b1": "float"}

        keys1 = ["id1", "id2"]
        keys2 = ["id3", "id4"]

        def _join(t0: ibis.Table, t1: ibis.Table) -> ibis.Table:
            cond = ibis.and_(*[
                getattr(t0, k1) == getattr(t1, k2)
                for k1, k2 in zip(keys1, keys2, strict=True)
            ])
            return t0.join(t1, cond, how="inner")

        t0 = ibis.table(schema=schema1, name="table0")
        t1 = ibis.table(schema=schema2, name="table1")
        aligned = _join(t0, t1)
        data1 = {
            "id1": ["k1", "k2"],
            "id2": ["k3", "k4"],
            "a1": [1, 2],
            "a2": [True, False],
        }
        data2 = {
            "id3": ["k1", "k2", "ka"],
            "id4": ["k3", "k5", "kb"],
            "b1": [1.1, 2.1, 3.1],
        }
        res = self._eval(aligned, data1, data2, table_names=["table0", "table1"])
        print(f"join res: \n {res}")

    def test_union(self):
        schema = {"id": "str", "f0": "str", "f1": "int"}
        t1 = ibis.table(schema=schema, name="table1")
        t2 = ibis.table(schema=schema, name="table2")
        expr = t1.union(t2, distinct=False)
        data1 = {"id": ["ID1", "ID2"], "f0": ["x1", "x2"], "f1": [1, 2]}
        data2 = {"id": ["ID3", "ID4"], "f0": ["x3", "x4"], "f1": [3, 4]}
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

        res = self._eval(expr, data1, data2, table_names=["table1", "table2"])
        print(f"union res: \n{res}")
        combined_df = pd.concat([df1, df2], ignore_index=True)
        npt.assert_equal(res, combined_df.to_numpy())

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
            return ibis.cases((std_expr == 0, 0), else_=(col - mean_expr) / std_expr)

        def _normalize(col: ibis.Column):
            # norm = (x-min)/(ma-min)
            min_expr = col.min()
            max_expr = col.max()

            denominator = max_expr - min_expr

            return ibis.cases(
                (denominator == 0, 0), else_=(col - min_expr) / denominator
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
            new_col = ibis.cases((col < op0, op0), (col > op1, op1), else_=col)
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
            return (col + op0).log(2).round(2)

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

    @pytest.mark.parametrize(
        "drop_type,min_freq,features",
        [
            ("first", 0.1, ["a1", "a2", "a3"]),
            ("mode", 0.1, ["a1", "a2", "a3"]),
        ],
    )
    def test_onehot(self, drop_type: str, min_freq: float, features: list):
        # onehot 需要生成相对负责的rule,无法一次性生成expr
        data = {
            "id": [str(i) for i in range(17)],
            "a1": ["K"] + ["F"] * 13 + ["", "M", "N"],
            "a2": [0.1, 0.2, 0.3] * 5 + [0.4] * 2,
            "a3": [1] * 17,
            "y": [0] * 17,
        }
        schema = {"id": "str", "a1": "str", "a2": "float", "a3": "int", "y": "int"}
        t = self._memtable(data, schema=schema)
        row_count = t.count().execute()
        print(f"row_count: {type(row_count)} {row_count}")
        min_freq_count = round(row_count * min_freq)

        def _fit_col(col_name: str):
            col = t[col_name]
            dist_count = col.nunique().execute()

            if dist_count >= 100:
                raise ValueError(
                    f"the feature has too many categories: {col_name}, {dist_count}"
                )

            if drop_type == "mode":
                expr = (
                    t.group_by(col_name)
                    .aggregate(count=col.count())
                    .order_by(ibis.desc("count"))
                    .limit(1)
                )
                res = expr.execute()
                drop_category = res[col_name].iloc[0]
            elif drop_type == "first":
                res = t.limit(1).execute()
                drop_category = res[col_name].iloc[0]
            elif drop_type == "no_drop":
                drop_category = None
            else:
                raise ValueError(f"unsupported drop type: {drop_type}")

            if drop_category is None:
                filtered = t.filter(col.isnull().not_())
            else:
                filtered = t.filter(col != drop_category)
            # 聚合出每个元素个数
            freq_table: pd.DataFrame = (
                filtered.group_by(col_name).aggregate(freq=col.count()).execute()
            )
            # category
            category = []
            infrequent = []
            for _, row in freq_table.iterrows():
                name = row[col_name]
                freq = row["freq"]
                if freq < min_freq_count:
                    infrequent.append(name)
                else:
                    category.append([name])

            if infrequent:
                category.append(infrequent)

            return category, drop_category

        onehot_rules = {}
        drop_rules = {}
        for col_name in features:
            category, drop_category = _fit_col(col_name)
            onehot_rules[col_name] = category
            if drop_category:
                drop_rules[col_name] = drop_category

        # onehot_rules: {'a1': [['F'], ['', 'N', 'M']], 'a2': [[0.3], [0.4], [0.2]], 'a3': []}, drop_rules: {'a1': 'K', 'a2': 0.1, 'a3': 1}
        print(f"onehot_rules: {onehot_rules}, drop_rules: {drop_rules}")
        # apply rule
        exprs = {}
        for col_name, col_rules in onehot_rules.items():
            col = t[col_name]
            for idx, values in enumerate(col_rules):
                new_col = ibis.ifelse(col.isin(values), 1.0, 0.0)
                exprs[f"{col_name}_{idx}"] = new_col
        old_columns = [col for col in t.columns if col not in onehot_rules]
        new_t = t.mutate(**exprs).select(old_columns + list(exprs.keys()))
        res = new_t.execute()
        print(f"onehot res: \n {res}")

    @pytest.mark.parametrize(
        "bin_method,bin_num,features",
        [
            ("eq_range", 3, ["a1", "a2"]),
            ("quantile", 3, ["a1", "a2"]),
            ("", 0, ["s"]),
        ],
    )
    def test_binning(self, bin_method: str, bin_num: int, features: list):
        # 需要对不同列,不同的类型,不同的分箱方法生成不同的规则,也没有办法使用一个表达式完成
        schema = ibis.schema({"a1": "int", "a2": "float", "s": "str"})
        data = {
            "a1": [1, 2, 3, 4, 5],
            "a2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "s": ["s1", "s2", "s2", "s4", "s5"],
        }

        t = self._memtable(data, schema=schema)

        def _fit_feature_bins(col_name: str) -> dict:
            col = t[col_name]
            col_dtype = t.schema().fields[col_name]
            if col_dtype.is_string():
                categories_df = t.drop_null(col_name).distinct(on=col_name).execute()
                categories = categories_df[col_name]
                split_points = sorted(categories)
                return {"type": "str", "split_points": split_points}
            else:
                if bin_method == "quantile":
                    # bins, split_points = pd.qcut(
                    #    f_data, bin_num, labels=False, duplicates='drop', retbins=True
                    # )
                    quantiles = [i / bin_num for i in range(1, bin_num)]
                    split_points_ibis = col.approx_quantile(quantiles)
                    split_points = split_points_ibis.execute()
                    split_points = sorted(set(split_points))
                elif bin_method == "eq_range":
                    # bins, split_points = pd.cut(
                    #     f_data, bin_num, labels=False, duplicates="drop", retbins=True
                    # )
                    # 计算等宽分箱
                    min_val = col.min().execute()
                    max_val = col.max().execute()
                    split_width = (max_val - min_val) / bin_num
                    split_points = [
                        min_val + i * split_width for i in range(1, bin_num)
                    ]
                else:
                    raise ValueError(
                        f"binning_method {self.binning_method} not supported"
                    )

                return {"type": "numeric", "split_points": split_points}

        def _apply_feature_rule(tbl: ibis.Table, rule: dict):
            col_name = rule["name"]
            col = tbl[col_name]

            split_points = rule["split_points"]
            branches = []
            if rule["type"] == "str":
                branches = [(col == c, c) for c in split_points]
            else:
                branches = [(col < c, c) for c in split_points]
                branches.append((col > split_points[-1], split_points[-1]))
            bin_expr = ibis.cases(*branches)

            nt = t.mutate(row_number=ibis.row_number().over())
            nt = nt.select([col_name, "row_number", bin_expr.name("bin_index")])

            na_index_df = nt.filter(col.isnull()).execute()
            na_index = na_index_df["row_number"].to_numpy()

            non_null_nt = nt.filter(col.notnull())
            non_null_df = non_null_nt.execute()

            grouped = non_null_df.groupby("bin_index")["row_number"].apply(list)
            bin_indices = grouped.tolist()
            return bin_indices, na_index

        rules = {}
        for col_name in features:
            rule = _fit_feature_bins(col_name)
            rule["name"] = col_name
            rules[col_name] = rule

        print(f"binning rules: {rules}")
        for col_name, rule in rules.items():
            bin_indices, na_index = _apply_feature_rule(t, rule)
            print(f"use binning rule: {col_name} {bin_indices}, {na_index}")

    def test_woe_binning(self):
        pass

    def test_sql(self):
        schema = {
            "date_id": "date",
            "store_id": "int32",
            "sold_amount": "double",
            "quantity": "double",
        }
        catalog = {"tx": schema}
        sql_str = "SELECT store_id, SUM(sold_amount) AS total FROM tx GROUP BY store_id"
        expr = ibis.parse_sql(sql_str, catalog)

        data = {
            "date_id": pd.to_datetime(["2024-08-01", "2024-08-01", "2024-08-02"]),
            "store_id": [1, 2, 1],
            "sold_amount": [100.0, 200.0, 300.0],
            "quantity": [1, 2, 3],
        }
        res = self._eval(expr, data, table_names=["tx"])
        print(f"sql res:\n {res}")

    def test_score_card(self):
        predict_name: str = "pred"
        predict_score_name: str = "predict_score"
        scaled_value: float = 600
        odd_base: float = 20.0
        pdo: float = 20.0
        positive: int = 1
        min_score: float = 0.0
        max_score: float = 1000.0

        t = ibis.table(schema={"id": "str", "pred": "float"}, name="table")

        factor = pdo / math.log(2)
        offset = scaled_value - factor * math.log(odd_base)
        pred = t[predict_name]
        log_odds = factor * ((pred / (1 - pred)).log())
        if positive == 1:
            score = offset - log_odds
        else:
            score = offset + log_odds

        new_col = ibis.ifelse(
            score <= min_score,
            min_score,
            ibis.ifelse(score >= max_score, max_score, score),
        )
        exprs = {predict_score_name: new_col}
        new_t = t.mutate(**exprs)
        #
        input = {
            "id": ["id1", "id2"],
            "pred": [0.1, 0.2],
        }
        res = self._eval(new_t, input)
        print(f"score_card:\n {res}")
        except_res = [576.959938, 553.561438]
        npt.assert_allclose(res[predict_score_name], except_res)

    @pytest.mark.parametrize(
        "feature,op,bound_value",
        [
            ("a1", "==", "K1"),
        ],
    )
    def test_condition_filter(self, feature, op, bound_value):
        # step 1: compute row mask
        schema = {"id": "str", "a1": "str", "a2": "str", "a3": "int", "a4": "float"}
        t = ibis.table(schema, name="table")
        col_dt = t.schema().fields[feature]
        if col_dt.is_string():
            value_type = str
        elif col_dt.is_floating():
            value_type = float
        elif col_dt.is_integer():
            value_type = int
        else:
            raise ValueError("only support INT, FLOAT, STRING for now")

        value = None
        if op != "NOTNULL":
            if bound_value == "":
                raise ValueError("bound_value is empty")
            bound_value_list = bound_value.split(",")
            values = [value_type(val) for val in bound_value_list]
            value = values if op == "IN" else values[0]

        col = t[feature]
        if op == "==":
            mask_expr = ibis.and_(col.notnull(), col == value)
        elif op == "<":
            mask_expr = ibis.and_(col.notnull(), col < value)
        elif op == "<=":
            mask_expr = ibis.and_(col.notnull(), col <= value)
        elif op == ">":
            mask_expr = ibis.and_(col.notnull(), col > value)
        elif op == ">=":
            mask_expr = ibis.and_(col.notnull(), col >= value)
        elif op == "IN":
            mask_expr = ibis.and_(col.notnull(), col.isin(value))
        elif op == "NOTNULL":
            mask_expr = col.notnull()
        else:
            raise ValueError(f"not support op: {op}")

        new_t = t.mutate(mask=mask_expr).select("mask")

        data = {
            "id": ["1", "2", "3", "4"],
            "a1": ["K5", "K1", None, "K6"],
            "a2": ["A5", "A1", "A2", "A6"],
            "a3": [5, 1, 2, 6],
            "a4": [0, 1, 1, 0],
        }
        res = self._eval(new_t, data)

        # step2: filter by row mask
        # TODO: use ibis?
        df = pd.DataFrame(data)
        filterd_df = df[res["mask"]]
        print(filterd_df)

    @pytest.mark.parametrize("by", [["a"], ["a", "b"]])
    @pytest.mark.parametrize(
        "agg_pairs",
        [
            [("c", "sum")],
            [("c", "count"), ("d", "sum")],
        ],
    )
    def test_stat_groupby(self, by: list[str], agg_pairs: list[tuple[str, str]]):
        data = {
            "a": ["9", "6", "5", "5"],
            "b": [5, 5, 6, 7],
            "c": [1, 1, 2, 4],
            "d": [11, 55, 1, 99],
        }
        schema = {"a": "str", "b": "int", "c": "float", "d": "int"}
        t = ibis.table(schema=schema, name="table")

        agg_map = {
            "count": lambda x: x.count(),
            "sum": lambda x: x.sum(),
            "mean": lambda x: x.mean(),
            "min": lambda x: x.min(),
            "max": lambda x: x.max(),
            "var": lambda x: x.var(),  # Variance
        }

        grouped = t.group_by(by)
        aggs = []
        for p in agg_pairs:
            agg_col = p[0]
            agg_fn = p[1]
            if agg_fn not in agg_map:
                raise ValueError(f"Unsupported aggregation function: {agg_fn}")
            expr = agg_map[agg_fn](t[agg_col]).name(f"{agg_col}_{agg_fn}")
            aggs.append(expr)

        new_t = grouped.aggregate(aggs)
        res = self._eval(new_t, data)
        print(f"groupby: {by}\n {res}")

    def test_stat_pearsonr(self):
        pass

    @pytest.mark.parametrize("features", [["a1", "a2"]])
    def test_stat_vif(self, features: list[str]):
        pass

    def test_stat_psi(self):
        pass

    @pytest.mark.parametrize("features", [["a", "b", "c", "d", "f"]])
    def test_stat_table_stats(self, features: list[str]):
        # duckdb has impl skewness, register for ibis.
        @ibis.udf.agg.builtin
        def skewness(a: float) -> float:
            """Compute the skewness."""

        schema_pairs = {"a": "int", "b": "int", "c": "int", "d": "int", "f": "str"}
        table = ibis.table(schema=schema_pairs, name="table")

        def _compute_if(col_expr: ibis.Column, fn):
            if not col_expr.type().is_string():
                return fn(col_expr).cast("float64")
            return ibis.null("float64")

        def _central_moment(col: ibis.Column, k):
            #  "t0"."c" - AVG("t0"."c") OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND...
            mean_expr = col.mean()
            return ((col - mean_expr) ** k).mean()

        def get_stat_expr(col_expr: ibis.Column, order: int):
            return ibis.struct({
                "order": ibis.literal(order),
                "column": ibis.literal(col_expr.get_name()),
                "datatype": ibis.literal(str(col_expr.type())),
                "nunique": col_expr.nunique(),
                "count": col_expr.count(),
                "count_na": col_expr.isnull().sum(),
                "na_ratio": col_expr.isnull().sum() / col_expr.count(),
                "min": _compute_if(col_expr, lambda c: c.min()),
                "max": _compute_if(col_expr, lambda c: c.max()),
                "mean": _compute_if(col_expr, lambda c: c.mean()),
                "var": _compute_if(col_expr, lambda c: c.var()),
                "std": _compute_if(col_expr, lambda c: c.std()),
                "sem": _compute_if(col_expr, lambda c: c.std() / c.count().sqrt()),
                "skew": _compute_if(col_expr, lambda c: skewness(c)),
                "kurtosis": _compute_if(col_expr, lambda c: c.kurtosis()),
                "q1": _compute_if(col_expr, lambda c: c.quantile(0.25)),
                "q2": _compute_if(col_expr, lambda c: c.quantile(0.5)),
                "q3": _compute_if(col_expr, lambda c: c.quantile(0.75)),
                "moment_2": _compute_if(col_expr, lambda c: (c**2).mean()),
                "moment_3": _compute_if(col_expr, lambda c: (c**3).mean()),
                "moment_4": _compute_if(col_expr, lambda c: (c**4).mean()),
                # "central_moment_2": _compute_if(
                #     col_expr, lambda c: _central_moment(c, 2)
                # ),
                # "central_moment_3": _compute_if(
                #     col_expr, lambda c: _central_moment(c, 3)
                # ),
                # "central_moment_4": _compute_if(
                #     col_expr, lambda c: _central_moment(c, 4)
                # ),
                "sum": _compute_if(col_expr, lambda c: c.sum()),
                "sum_2": _compute_if(col_expr, lambda c: (c**2).sum()),
                "sum_3": _compute_if(col_expr, lambda c: (c**3).sum()),
                "sum_4": _compute_if(col_expr, lambda c: (c**4).sum()),
            }).name("stats")

        stats_tables = []
        for idx, col_name in enumerate(features):
            col = table[col_name]
            stat_expr = get_stat_expr(col, idx)
            stats_table = table.aggregate([stat_expr])
            stats_tables.append(stats_table)

        result_expr = ibis.union(*stats_tables)
        struct_fields = result_expr["stats"].type().names
        mutations = {field: result_expr["stats"][field] for field in struct_fields}

        result_expr = (
            result_expr.mutate(**mutations)
            .drop("stats")
            .order_by("order")
            .drop("order")
        )

        data = {
            "a": [9, 6, 5, 5],
            "b": [5, 5, 6, 7],
            "c": [1, 1, 2, 4],
            "d": [11, 55, 1, 99],
            "f": ["hello", "world", "are you", "ok?"],
        }
        res = self._eval(result_expr, data)
        print(f"table_stats: {features}\n {res}")

    def _memtable(self, data: dict, schema, name: str = "table") -> ibis.Table:
        return ibis.memtable(data, schema=schema, name=name)

    def _eval(
        self, expr: ibis.Table, *inputs: dict, table_names: list[str] | None = None
    ) -> pd.DataFrame:
        if table_names is None:
            table_names = ["table"]

        if _ibis_engine == "duckdb":
            return self._eval_duckdb(expr, inputs, table_names)
        else:
            return self._eval_substrait(expr, inputs, table_names)

    def _eval_duckdb(
        self,
        expr: ibis.Table,
        inputs: list[dict],
        table_names: list[str] | None = None,
        udfs: list | None = None,
    ):
        if not table_names:
            table_names = ["table"]
        assert len(inputs) == len(table_names)

        sql = ibis.to_sql(expr, dialect="duckdb")

        print(f"eval sql: \n{sql}\n")

        import duckdb
        import pandas as pd

        conn = duckdb.connect(":memory:")
        for data, name in zip(inputs, table_names, strict=True):
            df = pd.DataFrame(data)
            conn.register(name, df)

        if udfs:
            for fn in udfs:
                name = fn.__name__
                conn.create_function(name, fn)

        result_df = conn.execute(sql).fetchdf()
        return result_df

    def _eval_substrait(
        self, expr: ibis.Table, inputs: list[dict], table_names: list[str] | None = None
    ):
        # You need to run cmd 'pip install ibis-substrait' manually.
        # It will cause the ibis-framework version to roll back
        import pyarrow as pa
        import pyarrow.substrait as substrait  # noqa: F401
        from ibis_substrait.compiler.core import SubstraitCompiler

        assert len(inputs) == len(table_names)

        # encode ir
        compiler = SubstraitCompiler()
        plan = compiler.compile(expr)
        fn_txt = plan.SerializeToString()

        # # decode ir
        # substrait_plan = substrait.Plan()
        # substrait_plan.ParseFromString(fn_txt)

        # exec ir use pyarrow

        tables = {
            name: pa.Table.from_pydict(data)
            for name, data in zip(table_names, inputs, strict=True)
        }

        def table_provider(names, schema):
            if not names:
                raise Exception("No names provided")
            for name in names:
                if name in tables:
                    return tables[name]
            raise Exception(f"Unrecognized table name, {names}, {table_names}")

        reader = pa.substrait.run_query(fn_txt, table_provider=table_provider)
        result = reader.read_all()
        return result
