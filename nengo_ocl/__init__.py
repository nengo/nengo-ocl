import logging
import sys

from nengo.utils.logging import log
from sim_ocl import Simulator

# console logging (default to no handler; use imported `log` fn to change this)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
