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

"""Device library for MPLang2.

This module provides the high-level device-centric programming interface.
"""

from mplang.v2.dialects.tensor import jax_fn

from .api import (
    DeviceContext,
    DeviceError,
    DeviceInferenceError,
    DeviceNotFoundError,
    device,
    fetch,
    get_dev_attr,
    is_device_obj,
    put,
    set_dev_attr,
)
from .cluster import ClusterSpec, Device, Node

__all__ = [
    "ClusterSpec",
    "Device",
    "DeviceContext",
    "DeviceError",
    "DeviceInferenceError",
    "DeviceNotFoundError",
    "Node",
    "device",
    "fetch",
    "get_dev_attr",
    "is_device_obj",
    "jax_fn",
    "put",
    "set_dev_attr",
]
