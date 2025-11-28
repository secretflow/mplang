"""Device library for MPLang2.

This module provides the high-level device-centric programming interface.
"""

from .api import (
    device,
    get_dev_attr,
    is_device_obj,
    put,
    set_dev_attr,
)
from .cluster import (
    ClusterSpec,
    Device,
    Node,
    get_global_cluster,
    set_global_cluster,
)

__all__ = [
    "ClusterSpec",
    "Device",
    "Node",
    "device",
    "get_dev_attr",
    "get_global_cluster",
    "is_device_obj",
    "put",
    "set_dev_attr",
    "set_global_cluster",
]
