"""
Build the `fasttracker` Python extension (C++ FastTracker via pybind11).

Usage:
    pip install pybind11            # build-time dependency
    # then either:
    pip install ./FastTracker       # from the project root
    # or, in-place for development:
    cd FastTracker && python setup.py build_ext --inplace

Dependencies: a C++14 compiler, Eigen3 (header-only), pybind11.
Only Eigen + pybind11 are required (no OpenCV).
"""
import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

HERE = os.path.dirname(os.path.abspath(__file__))


def find_eigen():
    """Return an Eigen include dir, trying common locations and env override."""
    candidates = [
        os.environ.get("EIGEN_INCLUDE_DIR"),
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "/opt/homebrew/include/eigen3",
        "/usr/include",  # some distros install Eigen headers directly here
    ]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "Eigen", "Core")):
            return c
    raise RuntimeError(
        "Could not find Eigen3 headers. Install it (e.g. `sudo apt install libeigen3-dev`) "
        "or set EIGEN_INCLUDE_DIR to the directory containing the 'Eigen' folder."
    )


eigen_inc = find_eigen()

sources = [
    "bindings.cpp",
    os.path.join("src", "FastTracker.cpp"),
    os.path.join("src", "STrack.cpp"),
    os.path.join("src", "kalmanFilter.cpp"),
    os.path.join("src", "lapjv.cpp"),
    os.path.join("src", "utils.cpp"),
]

extra_compile_args = ["-O3"]
if sys.platform != "win32":
    extra_compile_args += ["-fvisibility=hidden"]

ext_modules = [
    Pybind11Extension(
        "fasttracker",
        sources=sources,
        include_dirs=[os.path.join(HERE, "include"), eigen_inc],
        cxx_std=14,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="fasttracker",
    version="0.1.0",
    description="FastTracker C++ multi-object tracker with pybind11 bindings",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
