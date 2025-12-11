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

"""Private Set Intersection (PSI) protocols.

Submodules:
- rr22: VOLE-masked PSI protocol (formerly okvs.py)
- unbalanced: Unbalanced PSI (O(n) communication)
- oprf: KKRT OPRF protocol
- cuckoo: Cuckoo hashing
- okvs_gct: Sparse OKVS data structure (Garbled Cuckoo Table)
- okvs: OKVS Abstract Base Class
"""

from .oprf import eval_oprf, sender_eval_prf, sender_eval_prf_batch
from .rr22 import psi_intersect
from .unbalanced import psi_unbalanced

# Alias for backward compatibility
eval = psi_intersect

__all__ = [
    "eval",
    "eval_oprf",
    "psi_intersect",
    "psi_unbalanced",
    "sender_eval_prf",
    "sender_eval_prf_batch",
]
