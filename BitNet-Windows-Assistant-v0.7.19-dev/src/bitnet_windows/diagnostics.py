"""Read-only diagnostics and an explicitly requested temporary C++ smoke test."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from . import __version__


@dataclass
class Check:
    name: str
    status: str
    detail: str
    path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _version_output(executable: str, args: list[str]) -> tuple[Optional[str], Optional[str]]:
    path = shutil.which(executable)
    if not path:
        return None, None
    try:
        result = subprocess.run(
            [path, *args], capture_output=True, text=True, timeout=8, check=False
        )
        lines = (result.stdout or result.stderr).strip().splitlines()
        first_line = lines[0].strip() if lines else f"process exited with code {result.returncode}"
        return path, first_line
    except (OSError, subprocess.TimeoutExpired) as exc:
        return path, f"Found at {path}, but could not query version: {exc}"


def _tool_check(name: str, exe: str, args: list[str]) -> Check:
    path, version = _version_output(exe, args)
    if path is None:
        return Check(name, "missing", f"{exe} was not found on PATH.")
    return Check(name, "found", version or f"Found {exe}.", path)


def _find_vswhere() -> Optional[Path]:
    candidates = []
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    program_files = os.environ.get("ProgramFiles")
    if program_files_x86:
        candidates.append(Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    if program_files:
        candidates.append(Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = shutil.which("vswhere")
    return Path(found) if found else None


def _visual_studio_check() -> Check:
    if platform.system() != "Windows":
        return Check("Visual Studio 2022", "not_applicable", "Visual Studio detection is Windows-specific.")
    vswhere = _find_vswhere()
    if vswhere:
        try:
            result = subprocess.run(
                [str(vswhere), "-latest", "-products", "*", "-requires",
                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                capture_output=True, text=True, timeout=8, check=False,
            )
            lines = result.stdout.strip().splitlines()
            if result.returncode == 0 and lines:
                return Check("Visual Studio C++ build tools", "found", "vswhere found an installation with the C++ toolchain component. Further validation is required.", lines[0])
            return Check("Visual Studio C++ build tools", "uncertain", "vswhere exists, but no matching C++ toolchain installation was returned.", str(vswhere))
        except (OSError, subprocess.TimeoutExpired) as exc:
            return Check("Visual Studio C++ build tools", "uncertain", f"Could not query vswhere: {exc}", str(vswhere))
    if os.environ.get("VCINSTALLDIR") or os.environ.get("VCToolsInstallDir"):
        return Check("Visual Studio C++ build tools", "likely", "Developer-shell environment variables are present; exact toolchain version was not verified.")
    return Check("Visual Studio C++ build tools", "uncertain", "Could not find vswhere or recognizable Developer-shell variables. This is not proof that Visual Studio is absent.")


def run_diagnostics() -> dict:
    """Collect environment facts only; do not install, write, or change settings."""
    checks = [
        Check("Operating system", "info", f"{platform.system()} {platform.release()} ({platform.machine()})"),
        Check("Python", "found" if sys.version_info >= (3, 10) else "warning", sys.version.split()[0], sys.executable),
        _tool_check("Git", "git", ["--version"]),
        _tool_check("CMake", "cmake", ["--version"]),
        _tool_check("Clang C compiler", "clang", ["--version"]),
        _tool_check("Clang C++ compiler", "clang++", ["--version"]),
        _visual_studio_check(),
    ]
    return {
        "app": "BitNet Windows Assistant",
        "version": __version__,
        "mode": "read-only",
        "checks": [item.to_dict() for item in checks],
        "notice": "This is a preliminary environment report, not a guarantee that BitNet will build. No system changes were made.",
    }


def run_cpp_smoke_test(timeout: int = 30) -> dict:
    """Compile, link and run a tiny C++ program in a temporary directory."""
    compiler = shutil.which("clang++")
    result = {
        "status": "missing" if compiler is None else "not_run",
        "compiler": compiler,
        "compile_exit_code": None,
        "compile_output": [],
        "program_exit_code": None,
        "program_output": [],
        "detail": "clang++ was not found on PATH." if compiler is None else "C++ smoke test was not completed.",
    }
    if compiler is None:
        return result

    try:
        with tempfile.TemporaryDirectory(prefix="bitnet-assistant-cpp-") as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "bitnet_smoke.cpp"
            executable = temp_path / ("bitnet_smoke.exe" if os.name == "nt" else "bitnet_smoke")
            source.write_text(
                '#include <iostream>\nint main() { std::cout << "BitNet compiler smoke test passed" << std::endl; return 0; }\n',
                encoding="utf-8",
            )
            try:
                compiled = subprocess.run(
                    [compiler, str(source), "-o", str(executable)],
                    capture_output=True, text=True, timeout=timeout, check=False,
                )
            except subprocess.TimeoutExpired:
                result.update(status="timeout", detail=f"Compilation exceeded {timeout} seconds.")
                return result
            except OSError as exc:
                result.update(status="error", detail=f"Could not start clang++: {exc}")
                return result

            result["compile_exit_code"] = compiled.returncode
            result["compile_output"] = (compiled.stdout + compiled.stderr).splitlines()
            if compiled.returncode != 0:
                result.update(status="compile_failed", detail="clang++ failed to compile or link the smoke-test program.")
                return result

            try:
                executed = subprocess.run(
                    [str(executable)], capture_output=True, text=True,
                    timeout=timeout, check=False,
                )
            except subprocess.TimeoutExpired:
                result.update(status="timeout", detail=f"The compiled program exceeded {timeout} seconds.")
                return result
            except OSError as exc:
                result.update(status="error", detail=f"Could not launch the compiled program: {exc}")
                return result

            result["program_exit_code"] = executed.returncode
            result["program_output"] = (executed.stdout + executed.stderr).splitlines()
            if executed.returncode == 0:
                result.update(status="passed", detail="Compilation, linking and execution succeeded in a temporary directory.")
            else:
                result.update(status="run_failed", detail="The program compiled, but execution returned a non-zero exit code.")
            return result
    except OSError as exc:
        result.update(status="error", detail=f"Temporary smoke-test setup failed: {exc}")
        return result
