
try:
    from collections import OrderedDict
except ImportError:
    # For Python <=2.6
    from ordereddict import OrderedDict

try:
    # For Python <=2.6
    import unittest2 as unittest
except ImportError:
    import unittest
