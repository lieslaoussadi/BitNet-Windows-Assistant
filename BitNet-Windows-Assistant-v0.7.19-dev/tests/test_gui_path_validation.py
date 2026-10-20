import unittest
from pathlib import Path

from bitnet_windows.gui import parse_library_root


class LibraryPathValidationTests(unittest.TestCase):
    def test_accepts_normal_windows_path_with_spaces(self):
        self.assertEqual(
            parse_library_root(r'"C:\AI\Folder With Spaces"'),
            Path(r'C:\AI\Folder With Spaces'),
        )

    def test_rejects_embedded_newline(self):
        with self.assertRaisesRegex(ValueError, 'retour à la ligne'):
            parse_library_root('C:\\AI\\Library\n2')

    def test_rejects_tab_character(self):
        with self.assertRaisesRegex(ValueError, 'caractère de contrôle'):
            parse_library_root('C:\\AI\\Library\t2')

    def test_rejects_empty_path(self):
        with self.assertRaisesRegex(ValueError, 'racine'):
            parse_library_root('   ')


if __name__ == '__main__':
    unittest.main()
