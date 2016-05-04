===============
Release History
===============

.. Changelog entries should follow this format:

   version (release date)
   ======================

   **section**

   - One-line description of change (link to Github issue/PR)

.. Changes should be organized in one of several sections:

   - Features
   - Improvements
   - Bugfixes
   - Documentation

1.0.0 (unreleased)
==================

Release in support of Nengo 2.1.0.

**Features**

- Added support for ``Process`` class and subclasses, new in Nengo in 2.1.0.
  We specifically support the ``WhiteNoise``, ``WhiteSignal``, and
  ``PresentInput`` processes. We also support the ``Conv2d`` and ``Pool2d``
  processes in ``nengo_extras``.
- ``LinearFilter`` is now fully supported, allowing for general synapses.

**Improvements**

- The Numpy simulator in this project (``sim_npy``) has been phased out and
  combined with the OCL simulator (``sim_ocl``). It is now called ``Simulator``
  and resides in ``simulator.py``.
- Operator scheduling (i.e. the planner) is much faster. We still have only
  one planner (``greedy_planner``), which now resides in ``planners.py``.
- Many small speed improvements, including a number of cases where data was
  needlessly copied off the device to check sizes, dtypes, etc.

**Documentation**

- Updated examples to use up-to-date Nengo syntax.

0.1.0 (June 8, 2015)
====================

Initial release of Nengo OpenCL!
Supports Nengo 2.0.x on Python 2.6+ and 3.3+.
