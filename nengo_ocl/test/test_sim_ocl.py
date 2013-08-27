try:
    # For Python <=2.6
    import unittest2 as unittest
except ImportError:
    import unittest

import pyopencl as cl

if __name__ == "__main__":
    unittest.main()
