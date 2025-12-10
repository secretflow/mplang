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

"""Runtime implementations for MPLang2 dialects."""

import importlib
import logging

logger = logging.getLogger(__name__)


def load_backend(module_name: str) -> None:
    """Load a backend implementation module by name.

    This is a helper to avoid 'unused import' warnings when importing backend
    implementations solely for their side effects (registration).

    Args:
        module_name: The dotted name of the module to load.
    """
    try:
        importlib.import_module(module_name)
        logger.debug(f"Loaded backend: {module_name}")
    except ImportError as e:
        # We re-raise the error so the user knows their backend failed to load
        raise ImportError(f"Failed to load backend '{module_name}': {e}") from e


def load_builtins() -> None:
    """Load all built-in backend implementations."""
    # Core backends that are expected to be present
    builtin_backends = [
        "mplang.v2.backends.spu_impl",
        "mplang.v2.backends.tensor_impl",
        "mplang.v2.backends.table_impl",
        "mplang.v2.backends.crypto_impl",
        "mplang.v2.backends.tee_impl",
        "mplang.v2.backends.bfv_impl",
        "mplang.v2.backends.store_impl",
    ]

    for module_name in builtin_backends:
        try:
            load_backend(module_name)
        except ImportError as e:
            logger.warning(f"Could not load built-in backend '{module_name}': {e}")
