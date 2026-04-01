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
        self.backend = FileSystemBackend(root_path=self.test_dir)

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
        self.store = ObjectStore(persistent=FileSystemBackend(root_path=self.test_dir))

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


class TestDownloadUpload(unittest.TestCase):
    """Tests for StoreBackend.download/upload / ObjectStore.download/upload."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.local_dir = tempfile.mkdtemp()
        self.backend = FileSystemBackend(root_path=self.test_dir)
        self.store = ObjectStore(persistent=self.backend)

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.local_dir)

    # -- FileSystemBackend.download --

    def test_backend_download(self):
        """download creates a symlink from root to local dest."""
        # Create a source file in root_path
        src = os.path.join(self.test_dir, "data", "input.csv")
        os.makedirs(os.path.dirname(src), exist_ok=True)
        with open(src, "w") as f:
            f.write("a,b\n1,2\n")

        dest = os.path.join(self.local_dir, "input.csv")
        self.backend.download("data/input.csv", dest)
        self.assertTrue(os.path.islink(dest))
        with open(dest) as f:
            self.assertEqual(f.read(), "a,b\n1,2\n")

    def test_backend_download_noop_same_path(self):
        """download is a no-op when src and dest resolve to the same path."""
        src = os.path.join(self.test_dir, "file.csv")
        with open(src, "w") as f:
            f.write("data")

        # dest == src (absolute)
        self.backend.download("file.csv", src)
        with open(src) as f:
            self.assertEqual(f.read(), "data")

    # -- FileSystemBackend.upload --

    def test_backend_upload(self):
        """upload copies data from local source to root."""
        source = os.path.join(self.local_dir, "result.parquet")
        with open(source, "w") as f:
            f.write("parquet_data")

        self.backend.upload(source, "output/nested/result.parquet")
        dest = os.path.join(self.test_dir, "output/nested/result.parquet")
        self.assertTrue(os.path.exists(dest))
        with open(dest) as f:
            self.assertEqual(f.read(), "parquet_data")

    def test_backend_download_traversal_prevention(self):
        """download rejects directory traversal attempts."""
        dest = os.path.join(self.local_dir, "out.csv")
        with self.assertRaises(ValueError):
            self.backend.download("../outside", dest)

    def test_backend_upload_traversal_prevention(self):
        """upload rejects directory traversal attempts."""
        source = os.path.join(self.local_dir, "in.csv")
        with open(source, "w") as f:
            f.write("data")
        with self.assertRaises(ValueError):
            self.backend.upload(source, "../outside")

    def test_backend_upload_download_roundtrip(self):
        """Upload a file, then download it back."""
        source = os.path.join(self.local_dir, "test.txt")
        with open(source, "w") as f:
            f.write("hello")

        self.backend.upload(source, "test.txt")

        dest = os.path.join(self.local_dir, "test_downloaded.txt")
        self.backend.download("test.txt", dest)
        with open(dest) as f:
            self.assertEqual(f.read(), "hello")

    def test_backend_download_rejects_existing_dest(self):
        """download raises FileExistsError when dest already exists."""
        src = os.path.join(self.test_dir, "src.csv")
        with open(src, "w") as f:
            f.write("data")

        dest = os.path.join(self.local_dir, "existing.csv")
        with open(dest, "w") as f:
            f.write("old")

        with self.assertRaises(FileExistsError):
            self.backend.download("src.csv", dest)

    def test_backend_upload_rejects_existing_dest(self):
        """upload raises FileExistsError when dest already exists."""
        source = os.path.join(self.local_dir, "new.csv")
        with open(source, "w") as f:
            f.write("new_data")

        # Pre-create the destination in root_path
        dst = os.path.join(self.test_dir, "existing_key")
        with open(dst, "w") as f:
            f.write("old_data")

        with self.assertRaises(FileExistsError):
            self.backend.upload(source, "existing_key")

    # -- FileSystemBackend.download with absolute key --

    def test_backend_download_absolute_key(self):
        """download with absolute key treats it as direct source path."""
        abs_src = os.path.join(self.local_dir, "abs_src.csv")
        with open(abs_src, "w") as f:
            f.write("abs_data")

        dest = os.path.join(self.local_dir, "abs_dest.csv")
        self.backend.download(abs_src, dest)
        with open(dest) as f:
            self.assertEqual(f.read(), "abs_data")

    def test_backend_upload_absolute_key(self):
        """upload with absolute key treats it as direct dest path."""
        source = os.path.join(self.local_dir, "src.csv")
        with open(source, "w") as f:
            f.write("upload_abs")

        abs_dest = os.path.join(self.local_dir, "abs_uploaded.csv")
        self.backend.upload(source, abs_dest)
        with open(abs_dest) as f:
            self.assertEqual(f.read(), "upload_abs")

    # -- ObjectStore.download --

    def test_store_download_relative_path(self):
        """Relative paths are resolved via persistent backend."""
        src = os.path.join(self.test_dir, "data", "file.csv")
        os.makedirs(os.path.dirname(src), exist_ok=True)
        with open(src, "w") as f:
            f.write("content")

        dest = os.path.join(self.local_dir, "file.csv")
        self.store.download("data/file.csv", dest)
        with open(dest) as f:
            self.assertEqual(f.read(), "content")

    def test_store_download_absolute_path(self):
        """Absolute paths are handled by the backend directly."""
        abs_src = os.path.join(self.test_dir, "abs_file.csv")
        with open(abs_src, "w") as f:
            f.write("abs_content")

        dest = os.path.join(self.local_dir, "abs_out.csv")
        self.store.download(abs_src, dest)
        with open(dest) as f:
            self.assertEqual(f.read(), "abs_content")

    def test_store_download_no_persistent_backend_raises(self):
        """Without persistent backend, download raises RuntimeError."""
        store_no_persist = ObjectStore()
        with self.assertRaises(RuntimeError):
            store_no_persist.download("relative/path.csv", "/tmp/dest.csv")

    def test_store_upload_no_persistent_backend_raises(self):
        """Without persistent backend, upload raises RuntimeError."""
        store_no_persist = ObjectStore()
        with self.assertRaises(RuntimeError):
            store_no_persist.upload("/tmp/source.csv", "relative/path.csv")

    # -- ObjectStore.upload --

    def test_store_upload_relative_path(self):
        """Relative paths upload via persistent backend."""
        source = os.path.join(self.local_dir, "upload.csv")
        with open(source, "w") as f:
            f.write("upload_data")

        self.store.upload(source, "uploaded/file.csv")
        stored = os.path.join(self.test_dir, "uploaded/file.csv")
        self.assertTrue(os.path.exists(stored))
        with open(stored) as f:
            self.assertEqual(f.read(), "upload_data")

    # -- MemoryBackend.download/upload --

    def test_memory_backend_download_raises(self):
        """MemoryBackend raises RuntimeError on download."""
        from mplang.runtime.object_store import MemoryBackend

        mem = MemoryBackend()
        with self.assertRaises(RuntimeError):
            mem.download("key", "/tmp/dest")

    def test_memory_backend_upload_raises(self):
        """MemoryBackend raises RuntimeError on upload."""
        from mplang.runtime.object_store import MemoryBackend

        mem = MemoryBackend()
        with self.assertRaises(RuntimeError):
            mem.upload("/tmp/source", "key")
