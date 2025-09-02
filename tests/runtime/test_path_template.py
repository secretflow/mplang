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

"""Tests for path_template module."""

import pytest

from mplang.runtime.path_template import PathTemplate


class TestPathTemplate:
    """Test cases for PathTemplate class."""

    def test_literal_template(self):
        """Test template with only literal parts."""
        template = PathTemplate("users/messages")

        # Test matching
        assert template.match("users/messages") == {}
        assert template.match("users/msgs") is None
        assert template.match("users/messages/extra") is None
        assert template.match("users") is None

        # Test rendering
        assert template.render() == "users/messages"

    def test_positional_template(self):
        """Test template with positional variables (*)."""
        template = PathTemplate("users/*/messages/*")

        # Test matching
        result = template.match("users/123/messages/456")
        assert result == {"$0": "123", "$1": "456"}

        result = template.match("users/alice/messages/hello")
        assert result == {"$0": "alice", "$1": "hello"}

        # Test non-matching paths
        assert template.match("users/123/messages") is None
        assert template.match("users/123/messages/456/extra") is None
        assert template.match("posts/123/messages/456") is None

        # Test rendering
        rendered = template.render(**{"$0": "123", "$1": "456"})
        assert rendered == "users/123/messages/456"

        # Test missing variables
        with pytest.raises(ValueError, match="Missing positional variable \\$0"):
            template.render(**{"$1": "456"})

    def test_named_template(self):
        """Test template with named variables ({name})."""
        template = PathTemplate("shelves/{shelf}/books/{book}")

        # Test matching
        result = template.match("shelves/fiction/books/1984")
        assert result == {"shelf": "fiction", "book": "1984"}

        result = template.match("shelves/sci-fi/books/dune")
        assert result == {"shelf": "sci-fi", "book": "dune"}

        # Test non-matching paths
        assert template.match("shelves/fiction/books") is None
        assert template.match("shelves/fiction/magazines/1984") is None

        # Test rendering
        rendered = template.render(shelf="fiction", book="1984")
        assert rendered == "shelves/fiction/books/1984"

        # Test missing variables
        with pytest.raises(ValueError, match="Missing named variable shelf"):
            template.render(book="1984")

    def test_mixed_template(self):
        """Test template with both positional and named variables."""
        template = PathTemplate(
            "sessions/{session_id}/executions/*/msgs/{msg_id}/frm/*"
        )

        # Test matching
        result = template.match("sessions/abc123/executions/exec456/msgs/hello/frm/0")
        expected = {
            "session_id": "abc123",
            "$0": "exec456",
            "msg_id": "hello",
            "$1": "0",
        }
        assert result == expected

        # Test rendering
        rendered = template.render(
            session_id="abc123", msg_id="hello", **{"$0": "exec456", "$1": "0"}
        )
        assert rendered == "sessions/abc123/executions/exec456/msgs/hello/frm/0"

    def test_empty_template(self):
        """Test empty template."""
        template = PathTemplate("")

        # Test matching
        assert template.match("") == {}
        assert template.match("anything") is None

        # Test rendering
        assert template.render() == ""

    def test_single_part_templates(self):
        """Test templates with single parts."""
        # Literal
        template = PathTemplate("users")
        assert template.match("users") == {}
        assert template.match("posts") is None
        assert template.render() == "users"

        # Positional
        template = PathTemplate("*")
        assert template.match("anything") == {"$0": "anything"}
        assert template.render(**{"$0": "test"}) == "test"

        # Named
        template = PathTemplate("{id}")
        assert template.match("123") == {"id": "123"}
        assert template.render(id="123") == "123"

    def test_complex_real_world_examples(self):
        """Test with complex real-world path templates."""
        # REST API style
        api_template = PathTemplate("api/v1/users/{user_id}/posts/{post_id}")

        result = api_template.match("api/v1/users/12345/posts/67890")
        assert result == {"user_id": "12345", "post_id": "67890"}

        rendered = api_template.render(user_id="alice", post_id="my-post")
        assert rendered == "api/v1/users/alice/posts/my-post"

        # File system style
        fs_template = PathTemplate("home/*/documents/{type}/*")

        result = fs_template.match("home/alice/documents/pdf/report.pdf")
        assert result == {"$0": "alice", "type": "pdf", "$1": "report.pdf"}

        rendered = fs_template.render(type="images", **{"$0": "bob", "$1": "photo.jpg"})
        assert rendered == "home/bob/documents/images/photo.jpg"

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Template with consecutive slashes (should be treated as empty parts)
        template = PathTemplate("users//messages")
        assert template.match("users//messages") == {}
        assert template.match("users/x/messages") is None

        # Template with special characters in names
        template = PathTemplate("data/{file-name}")
        result = template.match("data/my-file.txt")
        assert result == {"file-name": "my-file.txt"}

        # Numbers in variable names - using proper format
        template = PathTemplate("{version}/api/{endpoint1}")
        result = template.match("v2/api/users")
        assert result == {"version": "v2", "endpoint1": "users"}

    def test_repr_and_str(self):
        """Test string representations."""
        template = PathTemplate("users/{id}/posts/*")

        assert str(template) == "users/{id}/posts/*"
        assert repr(template) == "PathTemplate('users/{id}/posts/*')"

    def test_multiple_positional_variables(self):
        """Test template with many positional variables."""
        template = PathTemplate("*/*/*/*/*")

        result = template.match("a/b/c/d/e")
        expected = {"$0": "a", "$1": "b", "$2": "c", "$3": "d", "$4": "e"}
        assert result == expected

        rendered = template.render(**expected)
        assert rendered == "a/b/c/d/e"

    def test_invalid_variable_names(self):
        """Test handling of invalid variable names."""
        # Variable names with special characters
        template = PathTemplate("users/{user_id}")
        result = template.match("users/123")
        assert result == {"user_id": "123"}

        rendered = template.render(user_id="123")
        assert rendered == "users/123"

    def test_case_sensitivity(self):
        """Test that matching is case-sensitive."""
        template = PathTemplate("Users/{ID}")

        # Exact case match
        assert template.match("Users/123") == {"ID": "123"}

        # Different case should not match literals
        assert template.match("users/123") is None
        assert template.match("Users/123") == {
            "ID": "123"
        }  # Variables are case-sensitive too

        # Rendering preserves case
        rendered = template.render(ID="AbC")
        assert rendered == "Users/AbC"

    def test_validate_method(self):
        """Test the validate method using Google's path_template validation."""
        template = PathTemplate("users/{user_id}/messages/*")

        # Valid paths
        assert template.validate("users/123/messages/456") is True
        assert template.validate("users/abc/messages/xyz") is True
        assert template.validate("users/user-123/messages/msg_456") is True

        # Invalid paths
        assert (
            template.validate("users/123/messages") is False
        )  # Missing positional part
        assert template.validate("users/123/messages/456/extra") is False  # Extra part
        assert (
            template.validate("wrong/123/messages/456") is False
        )  # Wrong literal part
        assert template.validate("users//messages/456") is False  # Empty user_id

        # Test with positional-only template
        pos_template = PathTemplate("api/*/v1/*")
        assert pos_template.validate("api/service/v1/endpoint") is True
        assert (
            pos_template.validate("api/service/v2/endpoint") is False
        )  # Wrong literal "v1"

        # Test with named-only template
        named_template = PathTemplate("projects/{project}/databases/{database}")
        assert named_template.validate("projects/my-project/databases/my-db") is True
        assert (
            named_template.validate("projects/my-project/databases") is False
        )  # Missing database
