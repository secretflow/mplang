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

"""Store dialect: save/load primitives for internal state."""

from __future__ import annotations

import mplang.v2.edsl as el
import mplang.v2.edsl.typing as elt

save_p: el.Primitive[el.Object] = el.Primitive("store.save")
load_p: el.Primitive[el.Object] = el.Primitive("store.load")


@save_p.def_abstract_eval
def _save_abstract(obj: elt.BaseType, *, uri_base: str) -> elt.BaseType:
    # Save is an identity operation: returns the input object type
    return obj


@load_p.def_abstract_eval
def _load_abstract(*, uri_base: str, expected_type: elt.BaseType) -> elt.BaseType:
    # Load returns an object of the expected type
    return expected_type


def save(obj: el.Object, uri_base: str) -> el.Object:
    """Save an object to persistent storage.

    This is an SPMD operation. Each party holding the object will save its
    local portion to the location specified by `uri_base`.

    Returns:
        The input object (identity), allowing for dependency chaining.
    """
    return save_p.bind(obj, uri_base=uri_base)


def load(uri_base: str, expected_type: elt.BaseType) -> el.Object:
    """Load an object from persistent storage.

    This is an SPMD operation. Each party will load its local portion from
    a path derived from `uri_base`.

    Args:
        uri_base: Base URI for the checkpoint package.
        expected_type: The type of the object to load (reconstructed from manifest).

    Returns:
        The loaded object.
    """
    return load_p.bind(uri_base=uri_base, expected_type=expected_type)
