"""Command-line interface for the safe BitNet Windows prototype."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .diagnostics import run_cpp_smoke_test, run_diagnostics
from .library import initialize_library


def _print_report(report: dict) -> None:
    print(f"{report['app']} {report['version']} - {report['mode']}")
    print("=" * 64)
    for check in report["checks"]:
        status = check["status"].upper()
        print(f"[{status:14}] {check['name']}")
        print(f"                {check['detail']}")
        if check.get("path"):
            print(f"                Path: {check['path']}")
    if "build_check" in report:
        check = report["build_check"]
        print("\n=== TEMPORARY C++ BUILD CHECK ===")
        print(f"Status: {check['status']}")
        print(f"Detail: {check['detail']}")
        if check.get("compiler"):
            print(f"Compiler: {check['compiler']}")
        if check.get("program_output"):
            print("Output: " + " | ".join(check["program_output"]))
    print("\n" + report["notice"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bitnet-windows", description="Safe preparation and diagnostics for the official BitNet runtime.")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Read-only environment diagnostics.")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    doctor.add_argument(
        "--build-check", action="store_true",
        help="Explicitly compile, link and run a tiny C++ program in a temporary directory.",
    )
    init = sub.add_parser("init-library", help="Create the local library folders at a path you choose.")
    init.add_argument("--root", required=True, type=Path, help="Explicit root directory to create/use.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        report = run_diagnostics()
        if args.build_check:
            report["build_check"] = run_cpp_smoke_test()
            report["mode"] = "diagnostic+temporary-build-test"
            report["notice"] = (
                "A temporary C++ source and executable were used for the explicit build check. "
                "Temporary files are cleaned up; no BitNet repository, model, PATH, registry, "
                "or installed software is changed. This is not a guarantee that BitNet itself builds."
            )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_report(report)
        if args.build_check and report["build_check"]["status"] != "passed":
            return 1
        return 0
    if args.command == "init-library":
        try:
            paths = initialize_library(args.root)
        except OSError as exc:
            print(f"Could not initialize library: {exc}", file=sys.stderr)
            return 2
        print(f"Library initialized under: {args.root.expanduser().resolve()}")
        for path in paths:
            print(f"  {path}")
        print("No downloads, installations, or system settings were changed.")
        return 0
    return 2
