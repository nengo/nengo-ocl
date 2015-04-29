import logging
import sys

from .version import version as __version__
from nengo.utils.logging import log
from sim_ocl import Simulator

# logging (default to no handler; use imported `log` fn to change this)
logging.root.addHandler(logging.NullHandler())
