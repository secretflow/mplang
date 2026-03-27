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

import os
import shutil
import tempfile
import unittest

from mplang.runtime.object_store import FileSystemBackend, ObjectStore


class TestFileSystemBackend(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.backend = FileSystemBackend(root_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_put_get(self):
        key = "test_key"
        value = {"a": 1, "b": [2, 3]}
        self.backend.put(key, value)

        retrieved = self.backend.get(key)
        self.assertEqual(retrieved, value)
        self.assertTrue(self.backend.exists(key))

    def test_nested_key(self):
        key = "folder/subfolder/key"
        value = "nested_value"
        self.backend.put(key, value)

        retrieved = self.backend.get(key)
        self.assertEqual(retrieved, value)
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "folder/subfolder/key"))
        )

    def test_delete(self):
        key = "to_delete"
        self.backend.put(key, "val")
        self.assertTrue(self.backend.exists(key))

        self.backend.delete(key)
        self.assertFalse(self.backend.exists(key))
        with self.assertRaises(KeyError):
            self.backend.get(key)

    def test_list_keys(self):
        keys = ["a", "b/c", "d/e/f"]
        for k in keys:
            self.backend.put(k, k)

        listed = self.backend.list_keys()
        self.assertEqual(sorted(listed), sorted(keys))

    def test_traversal_prevention(self):
        with self.assertRaises(ValueError):
            self.backend.get("../outside")

        with self.assertRaises(ValueError):
            self.backend.put("../outside", "val")


class TestObjectStoreWithFS(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.store = ObjectStore(persistent=FileSystemBackend(self.test_dir))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_load(self):
        value = [1, 2, 3]
        uri = self.store.put(value, uri="mydata")
        self.assertEqual(uri, "fs://mydata")

        retrieved = self.store.get("fs://mydata")
        self.assertEqual(retrieved, value)

        # Check file existence
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "mydata")))

    def test_save_load_nested(self):
        value = "hello"
        self.store.put(value, uri="path/to/data")
        self.assertEqual(self.store.get("fs://path/to/data"), value)

    def test_transient_independent(self):
        """Transient put/get works independently from persistent."""
        uri = self.store.put("transient_val")
        self.assertTrue(uri.startswith("mem://"))
        self.assertEqual(self.store.get(uri), "transient_val")

        self.store.put("persistent_val", uri="persistent_key")
        self.assertEqual(self.store.get("fs://persistent_key"), "persistent_val")


class TestOpenData(unittest.TestCase):
    """Tests for StoreBackend.open_data / ObjectStore.open_data."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.backend = FileSystemBackend(root_dir=self.test_dir)
        self.store = ObjectStore(persistent=self.backend)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    # -- FileSystemBackend.open_data --

    def test_backend_open_data_read(self):
        """open_data('r') yields the resolved local path."""
        with self.backend.open_data("data/input.csv", "r") as path:
            expected = os.path.join(self.test_dir, "data/input.csv")
            self.assertEqual(path, expected)

    def test_backend_open_data_write_creates_dirs(self):
        """open_data('w') creates parent directories automatically."""
        with self.backend.open_data("output/nested/result.parquet", "w") as path:
            expected = os.path.join(self.test_dir, "output/nested/result.parquet")
            self.assertEqual(path, expected)
            self.assertTrue(os.path.isdir(os.path.dirname(path)))

    def test_backend_open_data_traversal_prevention(self):
        """open_data rejects directory traversal attempts."""
        with self.assertRaises(ValueError):
            with self.backend.open_data("../outside", "r"):
                pass

    def test_backend_open_data_write_roundtrip(self):
        """Write a file via open_data('w'), then read it back via open_data('r')."""
        with self.backend.open_data("test.txt", "w") as path:
            with open(path, "w") as f:
                f.write("hello")

        with self.backend.open_data("test.txt", "r") as path:
            with open(path) as f:
                self.assertEqual(f.read(), "hello")

    # -- ObjectStore.open_data --

    def test_store_open_data_relative_path(self):
        """Relative paths are resolved via persistent backend."""
        with self.store.open_data("data/file.csv", "r") as path:
            self.assertTrue(path.startswith(self.test_dir))
            self.assertTrue(path.endswith("data/file.csv"))

    def test_store_open_data_absolute_path(self):
        """Absolute paths are yielded as-is (backward compat)."""
        abs_path = "/tmp/some/absolute/path.csv"
        with self.store.open_data(abs_path, "r") as path:
            self.assertEqual(path, abs_path)

    def test_store_open_data_no_persistent_backend(self):
        """Without persistent backend, relative paths resolve against cwd."""
        store_no_persist = ObjectStore()
        with store_no_persist.open_data("relative/path.csv", "r") as path:
            self.assertEqual(path, os.path.abspath("relative/path.csv"))

    # -- MemoryBackend.open_data --

    def test_memory_backend_open_data_raises(self):
        """MemoryBackend does not support open_data."""
        from mplang.runtime.object_store import MemoryBackend

        mem = MemoryBackend()
        with self.assertRaises(NotImplementedError):
            with mem.open_data("key", "r"):
                pass
