#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Run the full legacy unittest discovery without a nested container."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import unittest


def test_ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from test_ids(test)
        else:
            yield test.id()


def run_suite(suite, report_dir, stream=None, environment=None):
    """Keep real failures fatal, and reject empty or entirely skipped suites."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    inventory = list(test_ids(suite))
    (report_dir / "test-inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )
    result = unittest.TextTestRunner(
        stream=stream or sys.stdout, verbosity=2
    ).run(suite)
    executed = result.testsRun - len(result.skipped)
    success = result.wasSuccessful() and executed > 0
    summary = {
        "collected": len(inventory),
        "tests_run": result.testsRun,
        "executed_not_skipped": executed,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "success": success,
        "environment": environment or {},
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if not inventory or executed == 0:
        print("ERROR: no non-skipped tests executed", file=stream or sys.stdout)
    return 0 if success else 1


def main():
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    report_dir = root / "ci_artifacts"
    report_dir.mkdir(exist_ok=True)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "image": os.environ.get("CI_TEST_IMAGE", "local"),
        "requested_revision": os.environ.get("CI_TESTED_REVISION", "local"),
        "k8s": os.environ.get("HCU_CI_IN_K8S", "0"),
        "node": os.environ.get("NODE_NAME", ""),
        "packages": {},
    }
    for package in ("pyyaml", "tabulate"):
        try:
            environment["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment["packages"][package] = "not installed"
    (report_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(environment, indent=2), flush=True)
    # Legacy `exec python -m unittest` actually discovers all tests, not only ST.
    suite = unittest.TestLoader().discover(
        start_dir=str(root / "tests"), pattern="test*.py", top_level_dir=str(root)
    )
    return run_suite(suite, report_dir, environment=environment)


if __name__ == "__main__":
    sys.exit(main())
