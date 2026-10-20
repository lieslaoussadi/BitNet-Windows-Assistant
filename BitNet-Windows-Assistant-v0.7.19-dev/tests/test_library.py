import tempfile
import unittest
from pathlib import Path

from bitnet_windows.library import (
    LIBRARY_DIRS,
    UnsafeLibraryRootError,
    initialize_library,
)


class LibraryTests(unittest.TestCase):
    def test_initializes_expected_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "LocalAI"
            created = initialize_library(root)
            self.assertEqual(len(created), len(LIBRARY_DIRS))
            self.assertTrue(all(path.is_dir() for path in created))

    def test_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "LocalAI"
            initialize_library(root)
            initialize_library(root)
            self.assertTrue((root / "Library" / "models").is_dir())

    def test_rejects_filesystem_root(self):
        filesystem_root = Path(Path(tempfile.gettempdir()).resolve().anchor)
        with self.assertRaises(UnsafeLibraryRootError):
            initialize_library(filesystem_root)

    def test_rejects_git_checkout_without_creating_target(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            (repo / ".git").mkdir(parents=True)
            target = repo / "LocalAI"
            with self.assertRaises(UnsafeLibraryRootError):
                initialize_library(target)
            self.assertFalse(target.exists())

    def test_rejects_file_blocking_layout_before_creating_other_folders(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "LocalAI"
            (root / "Library").mkdir(parents=True)
            (root / "Library" / "models").write_text(
                "blocking file", encoding="utf-8"
            )
            with self.assertRaises(UnsafeLibraryRootError):
                initialize_library(root)
            self.assertFalse((root / "Library" / "downloads").exists())

    def test_rejects_symbolic_link_in_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "LocalAI"
            outside = Path(temp) / "outside"
            root.mkdir()
            outside.mkdir()
            try:
                (root / "Library").symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Symbolic links are unavailable on this Windows setup")
            with self.assertRaises(UnsafeLibraryRootError):
                initialize_library(root)
            self.assertFalse((outside / "downloads").exists())

    def test_rejects_symbolic_link_in_root_ancestors(self):
        with tempfile.TemporaryDirectory() as temp:
            outside = Path(temp) / "outside"
            alias = Path(temp) / "alias"
            outside.mkdir()
            try:
                alias.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Directory symbolic links are unavailable on this Windows setup")
            target = alias / "LocalAI"
            with self.assertRaises(UnsafeLibraryRootError):
                initialize_library(target)
            self.assertFalse((outside / "LocalAI").exists())


if __name__ == "__main__":
    unittest.main()