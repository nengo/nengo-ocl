import logging
import sys

from nengo.utils.logging import log

from .version import version as __version__
from .sim_ocl import Simulator

# logging (default to no handler; use imported `log` fn to change this)
try:
    logging.root.addHandler(logging.NullHandler())
except AttributeError:
    # No NullHandler in Python 2.6
    pass
