# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import io
import json
from pathlib import Path
import tempfile
import unittest

from run_simulation_tests import run_suite


class ReportRunnerTests(unittest.TestCase):
    def run_fixture(self, case):
        with tempfile.TemporaryDirectory() as directory:
            suite = unittest.TestSuite([] if case is None else [case])
            code = run_suite(suite, directory, stream=io.StringIO())
            summary = json.loads((Path(directory) / "summary.json").read_text())
            inventory = json.loads(
                (Path(directory) / "test-inventory.json").read_text()
            )
        return code, summary, inventory

    def test_success(self):
        code, summary, inventory = self.run_fixture(unittest.FunctionTestCase(lambda: None))
        self.assertEqual(code, 0)
        self.assertEqual(summary["executed_not_skipped"], 1)
        self.assertEqual(len(inventory), 1)

    def test_assertion_failure_is_fatal(self):
        def fail():
            raise AssertionError("fixture")

        code, summary, _ = self.run_fixture(unittest.FunctionTestCase(fail))
        self.assertEqual(code, 1)
        self.assertEqual(summary["failures"], 1)

    def test_exception_is_fatal(self):
        def error():
            raise RuntimeError("fixture")

        code, summary, _ = self.run_fixture(unittest.FunctionTestCase(error))
        self.assertEqual(code, 1)
        self.assertEqual(summary["errors"], 1)

    def test_empty_suite_is_fatal(self):
        code, summary, inventory = self.run_fixture(None)
        self.assertEqual(code, 1)
        self.assertEqual(summary["tests_run"], 0)
        self.assertEqual(inventory, [])

    def test_all_skipped_is_fatal(self):
        def skip():
            raise unittest.SkipTest("fixture")

        code, summary, _ = self.run_fixture(unittest.FunctionTestCase(skip))
        self.assertEqual(code, 1)
        self.assertEqual(summary["skipped"], 1)


if __name__ == "__main__":
    unittest.main()
