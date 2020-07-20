import unittest
import subprocess
import os
import pytest
import sys


class MainTest(unittest.TestCase):
    def test_cpp(self):
        print("\n\nRunning C++ tests...")
        all_files = [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(
                os.path.join(os.path.dirname(os.path.relpath(__file__)), "bin")
            )
            for f in filenames
        ]
        for filename in all_files:
            print(filename)
            subprocess.check_call(filename)

    def test_python(self):
        print("\n\nRunning Python tests...")
        sys.path.append("src")

        args = [os.path.join(os.path.basename(__file__), "..", "compy")]

        # Verbose
        args += ["-v"]

        # Show outputs
        args += ["-s"]

        # Coverage check
        args += ["--cov=compy"]

        # # HTML report for coverage check
        # args = ['--cov-report', 'html'] + args

        # XML report for codecov
        args = ["--cov-report=xml"] + args

        pytest.main(args)


if __name__ == "__main__":
    unittest.main()
