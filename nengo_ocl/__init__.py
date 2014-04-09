import logging
import sys

from nengo.utils.logging import log
from sim_ocl import Simulator

# logging (default to no handler; use imported `log` fn to change this)
logging.root.addHandler(logging.NullHandler())
