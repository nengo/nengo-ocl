#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    try:
        from ez_setup import use_setuptools
        use_setuptools()
        from setuptools import setup
    except Exception, e:
        print "Forget setuptools, trying distutils..."
        from distutils.core import setup


description = ("OpenCL neural simulations using the methods "
               + "of the Neural Engineering Framework")
setup(
    name="nengo_ocl",
    version="0.0.1.dev",
    author="CNRGlab at UWaterloo",
    author_email="https://github.com/jaberg/nengo_ocl/issues",
    packages=['nengo_ocl'],
    scripts=[],
    url="https://github.com/ctn-waterloo/nengo_theano",
    license="BSD",
    description=description,
    long_description="",
    requires=[
        "numpy (>=1.5.0)",
        "networkx",
        "nengo",
        "pyopencl",
    ],
    test_suite='nengo_ocl.test',
)
