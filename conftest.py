import pyopencl as cl
import pytest


@pytest.fixture(scope="session")
def ctx(request):
    return cl.create_some_context()
