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

"""
Serde registration for MPLang dialects.

This module ensures all dialect types are registered with the serde system.
Each dialect module (bfv, crypto, tee, spu) now registers its own types
inline with the class definitions.

This module imports all dialects to trigger their registrations, ensuring
types are available for serialization/deserialization.

Import this module after edsl.graph to ensure base types are registered first.
"""

from __future__ import annotations

# Import dialects to trigger their type registrations
# Each dialect module registers its types at import time via _register_*_types()
from mplang.v2.dialects import bfv as _bfv  # noqa: F401
from mplang.v2.dialects import crypto as _crypto  # noqa: F401
from mplang.v2.dialects import spu as _spu  # noqa: F401
from mplang.v2.dialects import store as _store  # noqa: F401
from mplang.v2.dialects import tee as _tee  # noqa: F401
