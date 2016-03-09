import nengo
from nengo.conftest import *  # noqa: F403


def pytest_generate_tests(metafunc):
    if "nl" in metafunc.funcargnames:
        metafunc.parametrize(
            "nl", [nengo.Direct, nengo.LIF, nengo.LIFRate])
    if "nl_nodirect" in metafunc.funcargnames:
        metafunc.parametrize(
            "nl_nodirect", [nengo.LIF, nengo.LIFRate])
