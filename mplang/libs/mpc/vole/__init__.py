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

"""Vector Oblivious Linear Evaluation (VOLE) protocols.

Submodules:
- gilboa: Gilboa VOLE protocol
- silver: Silver VOLE (LDPC-based)
- ldpc: LDPC matrix operations
"""

from .gilboa import vole
from .silver import estimate_silver_communication, silver_vole, silver_vole_ldpc

__all__ = [
    "estimate_silver_communication",
    "silver_vole",
    "silver_vole_ldpc",
    "vole",
]
