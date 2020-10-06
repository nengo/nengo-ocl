# pylint: disable=missing-module-docstring,missing-function-docstring

import pkg_resources

import nengo_ocl


def test_entry_point():
    sims = []
    for ep in pkg_resources.iter_entry_points(group="nengo.backends"):
        try:
            sims.append(ep.load(require=False))
        except Exception:  # pylint: disable=broad-except
            pass

    assert nengo_ocl.Simulator in sims
