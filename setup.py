#! /usr/bin/env python3

import os
import re
import sys
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from shutil import copyfile, copymode

install_requires = [
    "appdirs",
    "gitpython",
    "numpy",
    "torch",
    "torch-geometric",
    "tensorflow",
    "pygraphviz",
    "appdirs",
    "pandas",
    "tqdm",
]
tests_require = ["pytest", "pytest-cov"]


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        cmake_args = ["-DPYTHON_EXECUTABLE=" + sys.executable]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
        build_args += ["--", "-j" + os.environ.get("COMPY_BUILD_JOBS", '8')]

        print(" ".join(cmake_args))

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

        # Deploy shared library files to python folder structure.
        all_files = [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(self.build_temp)
            for f in filenames
        ]
        for filename in all_files:
            if filename.endswith(".so"):
                self.copy_pybind11_file(filename)

        # Copy *_test file to tests directory
        all_files = [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(self.build_temp)
            for f in filenames
        ]
        for filename in all_files:
            if filename.endswith("_tests"):
                self.copy_test_file(filename)
        print()

    def copy_pybind11_file(self, src_file):
        dest_file = os.path.relpath(src_file, self.build_temp)
        # dest_file = os.path.join(os.path.dirname(dest_file), '..', os.path.basename(dest_file))
        print("copying {} -> {}".format(src_file, dest_file))
        copyfile(src_file, dest_file)
        copymode(src_file, dest_file)

    def copy_test_file(self, src_file):
        """
        Copy ``src_file`` to ``dest_file`` ensuring parent directory exists.
        """
        # Create directory if needed
        dest_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tests", "bin"
        )
        if dest_dir != "" and not os.path.exists(dest_dir):
            print("creating directory {}".format(dest_dir))
            os.makedirs(dest_dir)

        # Copy file
        dest_file = os.path.join(dest_dir, os.path.basename(src_file))
        print("copying {} -> {}".format(src_file, dest_file))
        copyfile(src_file, dest_file)
        copymode(src_file, dest_file)


setup(
    name="ComPy",
    version="0.1b0",
    description="",
    author="ComPy-Learn authors",
    author_email="",
    long_description="",
    install_requires=install_requires,
    tests_require=tests_require,
    packages=find_packages("."),
    ext_modules=[CMakeExtension("")],
    cmdclass=dict(build_ext=CMakeBuild),
    test_suite="tests",
    zip_safe=False,
)
