from nengo.tests.conftest import *


def pytest_generate_tests(metafunc):
    if "nl" in metafunc.funcargnames:
        metafunc.parametrize(
            "nl", [Direct, LIF, LIFRate])
    if "nl_nodirect" in metafunc.funcargnames:
        metafunc.parametrize(
            "nl_nodirect", [LIF, LIFRate])
