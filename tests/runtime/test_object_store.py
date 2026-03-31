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

import pytest

from mplang.runtime.object_store import MemoryBackend, ObjectStore


def test_memory_backend():
    backend = MemoryBackend()
    backend.put("k1", "v1")
    assert backend.get("k1") == "v1"
    assert backend.exists("k1")
    backend.delete("k1")
    assert not backend.exists("k1")
    with pytest.raises(KeyError):
        backend.get("k1")


def test_object_store_transient():
    store = ObjectStore()

    # put returns a mem:// URI
    uri = store.put("value")
    assert isinstance(uri, str)
    assert uri.startswith("mem://")
    assert store.get(uri) == "value"
    assert store.exists(uri)

    store.delete(uri)
    assert not store.exists(uri)
    with pytest.raises(KeyError):
        store.get(uri)


def test_object_store_persistent(tmp_path):
    from mplang.runtime.object_store import FileSystemBackend

    store = ObjectStore(persistent=FileSystemBackend(root_path=str(tmp_path)))

    # Bare key -> auto-prefix with persistent scheme
    uri = store.put("my_value", uri="my_key")
    assert uri == "fs://my_key"
    assert store.get("fs://my_key") == "my_value"

    # Explicit scheme
    store.put("v2", uri="fs://explicit")
    assert store.get("fs://explicit") == "v2"


def test_object_store_no_persistent():
    store = ObjectStore()
    with pytest.raises(RuntimeError, match="No persistent backend configured"):
        store.put("v", uri="bare_key")


def test_object_store_invalid_scheme():
    store = ObjectStore()
    with pytest.raises(ValueError, match="No backend registered for scheme"):
        store.put("v", uri="unknown://key")
    with pytest.raises(ValueError, match="Invalid URI format"):
        store.get("no-scheme")
