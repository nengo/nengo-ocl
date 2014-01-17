"""
Black-box testing of the sim_ocl Simulator.

TestCase classes are added automatically from
nengo.tests, but you can still run individual
test files like this:

$ py.test test/test_sim_ocl.py -k test_ensemble.test_scalar

See http://pytest.org/latest/usage.html for more invocations.

"""

import inspect
import os.path

import nengo.tests
import pyopencl as cl
import pytest

from nengo_ocl import sim_ocl


ctx = cl.create_some_context()


def Ocl2Simulator(*args, **kwargs):
    kwargs['context'] = ctx
    return sim_ocl.Simulator(*args, **kwargs)


def pytest_funcarg__Simulator(request):
    """the Simulator class being tested.

    For this file, it's sim_npy.Simulator.
    """
    return Ocl2Simulator


def pytest_funcarg__RefSimulator(request):
    """the Simulator class being tested.

    For this file, it's sim_npy.Simulator.
    """
    return Ocl2Simulator


nengotestdir = os.path.dirname(nengo.tests.__file__)

for testfile in os.listdir(nengotestdir):
    if not testfile.startswith('test_') or not testfile.endswith('.py'):
        continue
    m = __import__("nengo.tests." + testfile[:-3], globals(), locals(), ['*'])
    for k in dir(m):
        if k.startswith('test_'):
            tst = getattr(m, k)
            args = inspect.getargspec(tst).args
            if 'Simulator' in args or 'RefSimulator' in args:
                locals()[testfile[:-3] + '.' + k] = tst
        if k.startswith('pytest'):
            locals()[k] = getattr(m, k)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
