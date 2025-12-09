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

from mplang.v2.runtime.object_store import MemoryBackend, ObjectStore


def test_memory_backend():
    backend = MemoryBackend()
    backend.put("k1", "v1")
    assert backend.get("k1") == "v1"
    assert backend.exists("k1")
    backend.delete("k1")
    assert not backend.exists("k1")
    with pytest.raises(KeyError):
        backend.get("k1")


def test_object_store(tmp_path):
    store = ObjectStore(fs_root=str(tmp_path))

    # Test default put (mem://)
    uri = store.put("value")
    assert uri.startswith("mem://")
    assert store.get(uri) == "value"
    assert store.exists(uri)

    store.delete(uri)
    assert not store.exists(uri)
    with pytest.raises(KeyError):
        store.get(uri)

    # Test explicit URI
    uri2 = "mem://custom-key"
    store.put("value2", uri2)
    assert store.get(uri2) == "value2"

    # Test invalid scheme
    with pytest.raises(ValueError, match="No backend registered"):
        store.put("val", "invalid://key")

    with pytest.raises(ValueError, match="Invalid URI"):
        store.put("val", "invalid-uri")
