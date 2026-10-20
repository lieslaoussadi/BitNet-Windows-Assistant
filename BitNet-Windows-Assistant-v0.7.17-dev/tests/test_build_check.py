import unittest
from types import SimpleNamespace
from unittest.mock import patch

from bitnet_windows.diagnostics import run_cpp_smoke_test


class BuildCheckTests(unittest.TestCase):
    @patch("bitnet_windows.diagnostics.shutil.which", return_value=None)
    def test_reports_missing_compiler(self, _which):
        result = run_cpp_smoke_test()
        self.assertEqual(result["status"], "missing")
        self.assertIsNone(result["compiler"])

    @patch("bitnet_windows.diagnostics.subprocess.run")
    @patch("bitnet_windows.diagnostics.shutil.which", return_value=r"C:\Tools\clang++.exe")
    def test_reports_successful_compile_and_run(self, _which, run):
        run.side_effect = [
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=0, stdout="BitNet compiler smoke test passed\n", stderr=""),
        ]
        result = run_cpp_smoke_test()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["compile_exit_code"], 0)
        self.assertEqual(result["program_exit_code"], 0)
        self.assertIn("BitNet compiler smoke test passed", result["program_output"])
        self.assertEqual(run.call_count, 2)

    @patch("bitnet_windows.diagnostics.subprocess.run")
    @patch("bitnet_windows.diagnostics.shutil.which", return_value=r"C:\Tools\clang++.exe")
    def test_reports_compile_failure(self, _which, run):
        run.return_value = SimpleNamespace(returncode=1, stdout="", stderr="compile error")
        result = run_cpp_smoke_test()
        self.assertEqual(result["status"], "compile_failed")
        self.assertEqual(result["compile_exit_code"], 1)
        self.assertEqual(run.call_count, 1)

    @patch("bitnet_windows.diagnostics.subprocess.run")
    @patch("bitnet_windows.diagnostics.shutil.which", return_value=r"C:\Tools\clang++.exe")
    def test_reports_program_failure(self, _which, run):
        run.side_effect = [
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=3, stdout="", stderr="runtime error"),
        ]
        result = run_cpp_smoke_test()
        self.assertEqual(result["status"], "run_failed")
        self.assertEqual(result["program_exit_code"], 3)


if __name__ == "__main__":
    unittest.main()
