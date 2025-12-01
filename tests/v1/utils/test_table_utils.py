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

import pyarrow as pa
import pytest

from mplang.v1.utils.table_utils import decode_table, encode_table


@pytest.mark.parametrize("format", ["csv", "orc", "parquet"])
def test_readwrite_table(format: str):
    data = {
        "a": [1, 2, 3],
        "b": ["a", None, "c"],
        "c": [1.1, 2.1, 3.1],
        "d": [True, False, None],
    }
    columns = ["a", "c"]
    csv_options = {"null_values": [""], "strings_can_be_null": True}
    options = csv_options if format == "csv" else {}
    buffer = encode_table(data, format=format)
    result_all = decode_table(buffer, format=format, **options)
    result_sub = decode_table(buffer, format=format, columns=columns, **options)

    assert result_all.equals(pa.table(data))
    assert result_sub.equals(pa.table({col: data[col] for col in columns}))


def test_write_dataframe():
    import pandas as pd

    data = {
        "a": [1, 2, 3],
        "b": ["a", None, "c"],
        "c": [1.1, 2.1, 3.1],
        "d": [True, False, None],
    }

    df = pd.DataFrame(data)
    buffer = encode_table(df)
    result = decode_table(buffer)
    assert result.equals(pa.table(data))


def test_readwrite_table_fail():
    data = pa.table({})
    with pytest.raises(ValueError, match="Cannot convert Table with no columns"):
        encode_table(data)

    with pytest.raises(ValueError, match="unsupported data format"):
        encode_table({"a": [1, 2, 3]}, format="")
    with pytest.raises(ValueError, match="unsupported data format"):
        decode_table(b"1,2,3", format="")
