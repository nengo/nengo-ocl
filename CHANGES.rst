***************
Release History
***************

.. Changelog entries should follow this format:

   version (release date)
   ======================

   **section**

   - One-line description of change (link to Github issue/PR)

.. Changes should be organized in one of several sections:

   - Added
   - Changed
   - Deprecated
   - Removed
   - Fixed

2.0.1 (unreleased)
==================

2.0.0 (Sept 4, 2020)
====================

**Added**

- The documentation is now online at https://labs.nengo.ai/nengo-ocl/ (`#179`_)
- ``Sparse`` transforms are now supported. (`#176`_)
- Added ``Simulator.clear_probes`` method to clear probe data stored in memory.
  (`#179`_)

**Changed**

- Now requires Python >= 3.5. (`#172`_)
- Now supports Nengo 3.0.0. Note that support for previous Nengo
  versions has been dropped. (`#172`_)
- ``Convolution`` transforms are now supported. The previous code supporting ``Conv2d``
  and ``Pool2d`` processes (from NengoExtras) has been removed. (`#172`_)

.. _#172: https://github.com/nengo-labs/nengo-ocl/pull/172
.. _#176: https://github.com/nengo-labs/nengo-ocl/pull/176
.. _#179: https://github.com/nengo-labs/nengo-ocl/pull/179

1.4.0 (July 4, 2018)
====================

**Improvements**

- Supports recent Nengo versions, up to 2.8.0.
- Supports the new ``SpikingRectifiedLinear`` neuron type.


1.3.0 (October 6, 2017)
=======================

**Improvements**

- Supports recent Nengo versions, up to 2.6.0.

**Bugfixes**

- Fixed an issue in which stochastic processes would not be
  fully reset on simulator reset.
- Fixed an issue in which building a model multiple times
  could result in old probe data persisting.

1.2.0 (February 23, 2017)
=========================

**Improvements**

- Supports all Nengo versions from 2.1.2 to 2.3.1.
- ``nengo_ocl.Simulator`` is no longer a subclass of ``nengo.Simulator``,
  reducing the chances that Nengo OCL will be affected by changes in Nengo.

1.1.0 (November 30, 2016)
=========================

**Features**

- Added support for ``RectifiedLinear`` and ``Sigmoid`` neuron types.
- Added support for arbitrary ``Process`` subclasses. Unlike the processes
  that are explicitly supported like ``WhiteSignal``, these processes
  may not fully utilize the OpenCL device.
- Added support for applying synaptic filters to matrices,
  which is commonly done when probing connection weights.

**Improvements**

- Supports all Nengo versions from 2.1.2 to 2.3.0.
- The ``LIF`` model is now more accurate, and matches the implementation
  in Nengo (see `Nengo#975 <https://github.com/nengo/nengo/pull/975>`_).
- Several operators have been optimized and should now run faster.

**Bugfixes**

- Fixed compatibility issues with Python 3,
  and certain versions of NumPy and Nengo.

1.0.0 (June 6, 2016)
====================

Release in support of Nengo 2.1.0. Since Nengo no longer supports Python 2.6,
we now support Python 2.7+ and 3.3+.

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
