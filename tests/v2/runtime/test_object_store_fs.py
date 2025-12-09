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

from mplang.v2.runtime.object_store import FileSystemBackend, ObjectStore


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
        self.store = ObjectStore(fs_root=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_fs_uri(self):
        uri = "fs://mydata"
        value = [1, 2, 3]

        # Put with explicit URI
        returned_uri = self.store.put(value, uri=uri)
        self.assertEqual(returned_uri, uri)

        # Get
        retrieved = self.store.get(uri)
        self.assertEqual(retrieved, value)

        # Check file existence
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "mydata")))

    def test_fs_uri_nested(self):
        uri = "fs://path/to/data"
        value = "hello"
        self.store.put(value, uri=uri)
        self.assertEqual(self.store.get(uri), value)
