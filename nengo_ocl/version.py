"""Nengo OCL version information.

We use semantic versioning (see http://semver.org/).
and conform to PEP440 (see https://www.python.org/dev/peps/pep-0440/).
'.devN' will be added to the version unless the code base represents
a release version. Release versions are git tagged with the version.
"""

# --- version of this release
name = "nengo_ocl"
version_info = (1, 1, 1)  # (major, minor, patch)
dev = 0
version = "{v}{dev}".format(v='.'.join(str(v) for v in version_info),
                            dev=('.dev%d' % dev) if dev is not None else '')

bad_nengo_versions = [
    (2, 0, 0),
    (2, 0, 1),
    (2, 0, 2),
    (2, 0, 3),
    (2, 0, 4),
    (2, 1, 1),
]

# --- latest Nengo version at time of release
latest_nengo_version_info = (2, 3, 0)  # (major, minor, patch)
latest_nengo_version = '.'.join(str(v) for v in latest_nengo_version_info)
