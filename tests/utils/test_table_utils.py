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

import pytest


class TestTableUtilsCSVHelpers:
    """Test CSV helper functions."""

    def test_dataframe_to_csv_basic(self):
        """Test basic DataFrame to CSV conversion."""
        pytest.importorskip("pandas")
        import pandas as pd

        from mplang.utils.table_utils import dataframe_to_csv

        # Create a test DataFrame
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.2, 92.0],
            "active": [True, False, True],
        })

        # Convert to CSV
        csv_bytes = dataframe_to_csv(df)

        # Verify the result
        assert isinstance(csv_bytes, bytes)
        csv_str = csv_bytes.decode("utf-8")

        # Check that CSV contains expected headers and data
        assert "id,name,score,active" in csv_str
        assert "1,Alice,95.5,True" in csv_str
        assert "2,Bob,87.2,False" in csv_str
        assert "3,Charlie,92.0,True" in csv_str

    def test_dataframe_to_csv_empty(self):
        """Test DataFrame to CSV conversion with empty DataFrame."""
        pytest.importorskip("pandas")
        import pandas as pd

        from mplang.utils.table_utils import dataframe_to_csv

        # Create an empty DataFrame with schema
        df = pd.DataFrame(columns=["user_id", "username"])
        df = df.astype({"user_id": "int64", "username": "string"})

        csv_bytes = dataframe_to_csv(df)

        assert isinstance(csv_bytes, bytes)
        csv_str = csv_bytes.decode("utf-8")
        # Should contain headers but no data rows
        assert "user_id,username" in csv_str

    def test_dataframe_to_csv_no_columns(self):
        """Test that DataFrame with no columns raises ValueError."""
        pytest.importorskip("pandas")
        import pandas as pd

        from mplang.utils.table_utils import dataframe_to_csv

        # Create DataFrame with no columns
        df = pd.DataFrame()

        with pytest.raises(
            ValueError, match="Cannot convert DataFrame with no columns to CSV"
        ):
            dataframe_to_csv(df)

    def test_dataframe_to_csv_wrong_type(self):
        """Test that non-DataFrame input raises TypeError."""
        pytest.importorskip("pandas")

        from mplang.utils.table_utils import dataframe_to_csv

        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            dataframe_to_csv([1, 2, 3])

        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            dataframe_to_csv("not a dataframe")

        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            dataframe_to_csv(None)

    def test_csv_to_dataframe_basic(self):
        """Test basic CSV to DataFrame conversion."""
        pytest.importorskip("pandas")
        import pandas as pd

        from mplang.utils.table_utils import csv_to_dataframe

        # Create test CSV content
        csv_content = b"id,name,score,active\n1,Alice,95.5,True\n2,Bob,87.2,False\n3,Charlie,92.0,True\n"

        # Convert to DataFrame
        df = csv_to_dataframe(csv_content)

        # Verify the result
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["id", "name", "score", "active"]

        # Check data values
        assert df.loc[0, "id"] == 1
        assert df.loc[0, "name"] == "Alice"
        assert df.loc[0, "score"] == 95.5
        # Check boolean value - pandas reads "True" as boolean
        assert df.loc[0, "active"]

    def test_csv_to_dataframe_empty(self):
        """Test CSV to DataFrame conversion with empty data."""
        pytest.importorskip("pandas")
        import pandas as pd

        from mplang.utils.table_utils import csv_to_dataframe

        # CSV with headers but no data
        csv_content = b"user_id,username\n"

        df = csv_to_dataframe(csv_content)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["user_id", "username"]

    def test_csv_to_dataframe_wrong_type(self):
        """Test that non-bytes input raises TypeError."""
        pytest.importorskip("pandas")

        from mplang.utils.table_utils import csv_to_dataframe

        with pytest.raises(TypeError, match="Expected bytes"):
            csv_to_dataframe("not bytes")

        with pytest.raises(TypeError, match="Expected bytes"):
            csv_to_dataframe([1, 2, 3])

        with pytest.raises(TypeError, match="Expected bytes"):
            csv_to_dataframe(None)

    def test_csv_to_dataframe_invalid_utf8(self):
        """Test that invalid UTF-8 content raises ValueError."""
        pytest.importorskip("pandas")

        from mplang.utils.table_utils import csv_to_dataframe

        # Invalid UTF-8 bytes
        invalid_content = b"\xff\xfe\x00\x00"

        with pytest.raises(ValueError, match="Invalid UTF-8 encoding in CSV content"):
            csv_to_dataframe(invalid_content)

    def test_csv_to_dataframe_invalid_csv(self):
        """Test error handling with various edge cases."""
        pytest.importorskip("pandas")

        from mplang.utils.table_utils import csv_to_dataframe

        # Test case 1: Invalid UTF-8 encoding should raise ValueError
        invalid_utf8 = b"invalid\xff\xfe\x00\x00"
        with pytest.raises(ValueError, match="Invalid UTF-8 encoding in CSV content"):
            csv_to_dataframe(invalid_utf8)

        # Test case 2: Edge case content that pandas handles gracefully
        # (pandas is very tolerant, so we mainly test that our function doesn't crash)
        edge_case_csv = b"this is not really csv content"
        result = csv_to_dataframe(edge_case_csv)
        # pandas will treat this as a single column with one row
        assert len(result.columns) == 1

    def test_roundtrip_conversion(self):
        """Test that DataFrame -> CSV -> DataFrame roundtrip works correctly."""
        pytest.importorskip("pandas")
        import pandas as pd

        from mplang.utils.table_utils import csv_to_dataframe, dataframe_to_csv

        # Create original DataFrame
        original_df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [95.5, 87.2, 92.0],
        })

        # Convert to CSV and back
        csv_bytes = dataframe_to_csv(original_df)
        restored_df = csv_to_dataframe(csv_bytes)

        # Compare DataFrames (accounting for potential type differences)
        assert len(original_df) == len(restored_df)
        assert list(original_df.columns) == list(restored_df.columns)

        # Check data values (convert to same types for comparison)
        pd.testing.assert_frame_equal(
            original_df.astype(str), restored_df.astype(str), check_dtype=False
        )
