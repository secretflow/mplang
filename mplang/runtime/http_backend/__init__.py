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

"""
HTTP Backend for MPLang Runtime.

This package provides HTTP-based alternatives to gRPC components for
distributed multi-party computation.
"""

from mplang.runtime.http_backend.communicator import HttpCommunicator
from mplang.runtime.http_backend.driver import HttpDriver, HttpDriverVar
from mplang.runtime.http_backend.server import app

__all__ = [
    "HttpCommunicator",
    "HttpDriver",
    "HttpDriverVar",
    "app",
]
