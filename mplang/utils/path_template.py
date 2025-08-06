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

"""Path template utilities for resource path parsing and rendering.

This module provides PathTemplate class for handling path templates with
named and positional variables, commonly used for resource path management
in REST APIs and similar systems.
"""

from __future__ import annotations

from google.api_core import path_template


class PathTemplate:
    """A utility class for parsing and rendering path templates.

    PathTemplate supports both positional (*) and named ({variable}) placeholders
    in path strings. It can match paths against templates to extract variables
    and render templates with provided variable values.

    Examples:
        >>> template = PathTemplate("users/{user_id}/messages/*")
        >>> variables = template.match("users/123/messages/456")
        >>> print(variables)  # {"user_id": "123", "$0": "456"}

        >>> rendered = template.render(user_id="123", **{"$0": "456"})
        >>> print(rendered)  # "users/123/messages/456"
    """

    def __init__(self, tmpl: str):
        """Initialize the PathTemplate with a given template string.

        Args:
            tmpl: The template string to parse, e.g.,
                 "users/*/messages/*" or "shelves/{shelf}/books/{book}".
        """
        self.tmpl = tmpl
        self.parsed_parts: list[tuple[str, str | None]] = []
        self.position_keys: list[str] = []

        # Parse the template and populate parsed_parts and position_keys
        position_index = 0
        for part in tmpl.split("/"):
            if part == "*":
                self.parsed_parts.append(("position", None))
                self.position_keys.append(f"${position_index}")
                position_index += 1
            elif part.startswith("{") and part.endswith("}"):
                var_name = part[1:-1]
                self.parsed_parts.append(("named", var_name))
            else:
                self.parsed_parts.append(("literal", part))

    def validate(self, path: str) -> bool:
        """Validate that a path matches this template.

        Args:
            path: The path to validate.

        Returns:
            True if the path matches the template, False otherwise.
        """
        return path_template.validate(self.tmpl, path)

    def match(self, path: str) -> dict[str, str] | None:
        """Match a path against the template and extract variables.

        Args:
            path: The path to match, e.g., "users/me/messages/123".

        Returns:
            A dictionary of extracted variables (e.g., {'$0': 'me', '$1': '123'})
            if the path matches the template, otherwise None.
        """
        if not self.validate(path):
            return None

        # If valid, extract variables using our custom parsing
        path_parts = path.split("/")
        if len(path_parts) != len(self.parsed_parts):
            return None

        variables: dict[str, str] = {}
        position_index = 0

        # Compare each template part with the corresponding path part
        for tmpl_part, path_part in zip(self.parsed_parts, path_parts):
            if tmpl_part[0] == "literal":
                if tmpl_part[1] != path_part:
                    return None
            elif tmpl_part[0] == "position":
                variables[self.position_keys[position_index]] = path_part
                position_index += 1
            elif tmpl_part[0] == "named":
                var_name = tmpl_part[1]
                if var_name is not None:
                    variables[var_name] = path_part

        return variables

    def render(self, **vars_dict) -> str:
        """Render the template into a path using the provided variables.

        Args:
            **vars_dict: A dictionary of variables, e.g., {'$0': 'me', '$1': '123'} or
                        {'shelf': '1', 'book': '2'}.

        Returns:
            The rendered path, e.g., "users/me/messages/123".

        Raises:
            ValueError: If a required positional or named variable is missing in vars_dict.
        """
        positional_args = []
        named_kwargs = {}

        # Collect positional arguments in order
        for key in self.position_keys:
            if key not in vars_dict:
                raise ValueError(f"Missing positional variable {key}")
            positional_args.append(vars_dict[key])

        # Collect named arguments
        for part in self.parsed_parts:
            if part[0] == "named":
                var_name = part[1]
                if var_name is not None:
                    if var_name not in vars_dict:
                        raise ValueError(f"Missing named variable {var_name}")
                    named_kwargs[var_name] = vars_dict[var_name]

        return path_template.expand(self.tmpl, *positional_args, **named_kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the PathTemplate."""
        return f"PathTemplate('{self.tmpl}')"

    def __str__(self) -> str:
        """Return the template string."""
        return self.tmpl
