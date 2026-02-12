# Copyright 2026 Ant Group Co., Ltd.
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


import pyarrow as pa

_STR_TO_TYPE_MAP = {
    "float16": pa.float16(),
    "float32": pa.float32(),
    "float64": pa.float64(),
    "int8": pa.int8(),
    "int16": pa.int16(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "uint8": pa.uint8(),
    "uint16": pa.uint16(),
    "uint32": pa.uint32(),
    "uint64": pa.uint64(),
}

_TYPE_TO_STR_MAP = {v: k for k, v in _STR_TO_TYPE_MAP.items()}


def _str_to_type(v: str) -> pa.DataType:
    return _STR_TO_TYPE_MAP[v]


def _type_to_str(t: pa.DataType) -> str:
    return _TYPE_TO_STR_MAP[t]
