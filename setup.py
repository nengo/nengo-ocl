#!/usr/bin/env python
import imp
import io
import os
import sys

try:
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py"
    )

from setuptools import find_packages, setup  # noqa: F811


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    "version", os.path.join(root, "nengo_ocl", "version.py")
)
testing = "test" in sys.argv or "pytest" in sys.argv

# Don't mess with added options if any are passed
sysargs_overridden = False

if testing and "--addopts" not in sys.argv:
    # Enable nengo tests by default
    old_sysargs = sys.argv[:]
    sys.argv[:] = old_sysargs + ["--addopts", "--pyargs nengo"]
    sysargs_overridden = True

setup(
    name="nengo-ocl",
    version=version_module.version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    packages=find_packages(),
    scripts=[],
    data_files=[],
    url="https://github.com/nengo/nengo-ocl",
    license="Free for non-commercial use",
    description=(
        "OpenCL-backed neural simulations using the " "Neural Engineering Framework"
    ),
    long_description=read("README.rst", "CHANGES.rst"),
    zip_safe=False,
    setup_requires=["pytest-runner"] if testing else [],
    install_requires=["nengo>=3.0.0,<3.1.0", "mako", "pyopencl"],
    tests_require=[
        "matplotlib>=1.4",
        "pytest>=2.9",
        "pytest-allclose>=1.0.0",
        "pytest-plt>=1.0.0",
        "pytest-rng>=1.0.0",
    ],
    entry_points={
        "nengo.backends": [
            "ocl = nengo_ocl:Simulator",
        ]
    },
    python_requires=">=3.5",
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Framework :: Nengo",
        "Intended Audience :: Science/Research",
        "License :: Free for non-commercial use",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

if sysargs_overridden:
    sys.argv[:] = old_sysargs
