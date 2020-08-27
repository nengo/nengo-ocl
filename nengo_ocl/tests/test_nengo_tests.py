import pkg_resources

import nengo_ocl


def test_entry_point():
    sims = [ep.load() for ep in pkg_resources.iter_entry_points(group="nengo.backends")]
    assert nengo_ocl.Simulator in sims
