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

from io import StringIO
from typing import Any

import pandas as pd

__all__ = ["csv_to_dataframe", "dataframe_to_csv"]


def dataframe_to_csv(df: Any) -> bytes:
    """Convert DataFrame to CSV format as bytes.

    Args:
        df: DataFrame to convert (pandas DataFrame)

    Returns:
        CSV formatted data as bytes

    Raises:
        TypeError: If df is not a pandas DataFrame
        ValueError: If DataFrame is empty or has no columns
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

    if len(df.columns) == 0:
        raise ValueError("Cannot convert DataFrame with no columns to CSV")

    # Convert DataFrame to CSV string, then to bytes
    csv_str: str = df.to_csv(index=False)
    return csv_str.encode("utf-8")


def csv_to_dataframe(content: bytes) -> Any:
    """Convert CSV bytes to DataFrame.

    Args:
        content: CSV formatted data as bytes

    Returns:
        pandas DataFrame

    Raises:
        TypeError: If content is not bytes
        ValueError: If content cannot be parsed as CSV
    """
    if not isinstance(content, bytes):
        raise TypeError(f"Expected bytes, got {type(content)}")

    try:
        # Decode bytes to string, then parse as CSV
        csv_str = content.decode("utf-8")
        df = pd.read_csv(StringIO(csv_str))
        return df
    except UnicodeDecodeError as e:
        raise ValueError(f"Invalid UTF-8 encoding in CSV content: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to parse CSV content: {e}") from e
