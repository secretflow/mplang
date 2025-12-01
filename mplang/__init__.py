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

"""Multi-Party Programming Language for Secure Computation.

This module provides access to both v1 and v2 APIs:

    # Use v1 (current default, will be deprecated)
    import mplang as mp
    # or explicitly:
    import mplang.v1 as mp

    # Use v2 (recommended for new code)
    import mplang.v2 as mp

v1 API will be deprecated in a future release. Please migrate to v2.
"""

# Version is managed by hatch-vcs and available after package installation
try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("mplang")
except PackageNotFoundError:
    # Fallback for development/editable installs when package is not installed
    __version__ = "0.0.0-dev"

# =============================================================================
# Default: Re-export v1 API for backward compatibility
# =============================================================================
# This will be changed to v2 in a future release

# Make v1 and v2 subpackages directly accessible
from mplang import v1, v2  # noqa: F401
from mplang.v1 import *  # noqa: F403
