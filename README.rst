.. image:: https://img.shields.io/pypi/v/nengo-ocl.svg
  :target: https://pypi.org/project/nengo-ocl
  :alt: Latest PyPI version

.. image:: https://img.shields.io/pypi/pyversions/nengo-ocl.svg
  :target: https://pypi.org/project/nengo-ocl
  :alt: Python versions

|

.. image:: https://www.nengo.ai/design/_images/nengo-ocl-full-light.svg
  :target: https://labs.nengo.ai/nengo-ocl
  :alt: NengoOCL
  :width: 400px

****************************
OpenCL-based Nengo Simulator
****************************

NengoOCL is an OpenCL-based simulator for
brain models built using `Nengo <https://github.com/nengo/nengo>`_.
It can be orders of magnitude faster than the reference simulator
in ``nengo`` for large models.

Usage
=====

To use the ``nengo_ocl`` project's OpenCL simulator,
build a Nengo model as usual,
but use ``nengo_ocl.Simulator`` when creating a simulator for your model::

    import numpy as np
    import matplotlib.pyplot as plt
    import nengo
    import nengo_ocl

    # define the model
    with nengo.Network() as model:
        stim = nengo.Node(np.sin)
        a = nengo.Ensemble(100, 1)
        b = nengo.Ensemble(100, 1)
        nengo.Connection(stim, a)
        nengo.Connection(a, b, function=lambda x: x**2)

        probe_a = nengo.Probe(a, synapse=0.01)
        probe_b = nengo.Probe(b, synapse=0.01)

    # build and run the model
    with nengo_ocl.Simulator(model) as sim:
        sim.run(10)

    # plot the results
    plt.plot(sim.trange(), sim.data[probe_a])
    plt.plot(sim.trange(), sim.data[probe_b])
    plt.show()

If you are running within ``nengo_gui`` make sure the ``PYOPENCL_CTX``
environment variable has been set. If this variable is not set it will open
an interactive prompt which will cause ``nengo_gui`` to get stuck during build.

Dependencies and Installation
=============================

The requirements are the same as Nengo, with the additional Python packages
``mako`` and ``pyopencl`` (where the latter requires installing OpenCL).

General:

* Python 2.7+ or Python 3.3+ (same as Nengo)
* One or more OpenCL implementations (test with e.g. PyOpenCl)

A working installation of OpenCL is the most difficult
part of installing NengoOCL. See below for more details
on how to install OpenCL.

Python packages:

* NumPy
* nengo
* mako
* PyOpenCL

In the ideal case, all of the Python dependencies
will be automatically installed when installing ``nengo_ocl`` with

.. code-block:: bash

   pip install nengo-ocl

If that doesn't work, then do a developer install
to figure out what's going wrong.

Developer Installation
----------------------

First, ``pip install nengo``.
For best performance, first make sure a fast version of Numpy is installed
by following the instructions in the
`Nengo README <http://github.com/nengo/nengo/blob/master/README.rst>`_.

This repository can then be installed with:

.. code-block:: bash

   git clone https://github.com/nengo/nengo-ocl.git
   cd nengo-ocl
   python setup.py develop --user

If you’re using a ``virtualenv`` (recommended!)
then you can omit the ``--user`` flag.
Check the output to make sure everything installed correctly.
Some dependencies (e.g. ``pyopencl``) may require manual installation.

Installing OpenCL
=================

How you install OpenCL is dependent on your hardware and operating system.
A good resource for various cases is found in the PyOpenCL documentation:

* `Installing PyOpenCL on Windows <http://wiki.tiker.net/PyOpenCL/Installation/Windows>`_
* `Installing PyOpenCL on Mac OS X <http://wiki.tiker.net/PyOpenCL/Installation/Mac>`_
* `Installing PyOpenCL on Linux <http://wiki.tiker.net/PyOpenCL/Installation/Linux>`_,
  and a `more detailed guide <http://wiki.tiker.net/OpenCLHowTo>`_

Below are instructions that have worked for the
NengoOCL developers at one point in time.

AMD OpenCL on Debian Unstable
-----------------------------

On Debian unstable (sid) there are packages in non-free and contrib
to install AMD's OpenCL implementation easily.
Actually, the easiest thing would be to apt-get install
`python-pyopencl <http://packages.debian.org/sid/python-pyopencl>`_.
But if you're using a virtual environment, you can
``sudo apt-get install opencl-headers libboost-python-dev amd-opencl-icd amd-libopencl1``
and then ``pip install pyopencl``.

Nvidia OpenCL on Debian/Ubuntu Linux
------------------------------------

On Debian unstable (sid) there are packages
for installing the Nvidia OpenCL implementation as well.

.. code-block:: bash

   sudo apt-get install nvidia-opencl-common nvidia-libopencl1

Ensure that the Nvidia driver version matches the OpenCL library version.
You can check the Nvidia driver version by running ``nvidia-smi`` in the
command line. You can find the OpenCL library version by looking at the
libnvidia-opencl.so.XXX.XX file in the ``/usr/lib/x86_64-linux-gnu/`` folder.

Intel OpenCL on Debian/Ubuntu Linux
-----------------------------------

The Intel SDK for OpenCL is no longer available. Intel OpenCL drivers
can be found `on Intel's website <https://software.intel.com/en-us/articles/opencl-drivers>`_.
See `the PyOpenCL wiki <http://wiki.tiker.net/OpenCLHowTo#Installing_the_Intel_CPU_ICD>`_
for instructions.

Running Tests
=============

From the ``nengo-ocl`` source directory, run:

.. code-block:: bash

    py.test nengo_ocl/tests --pyargs nengo -v

This will run the tests using the default context. If you wish to use another
context, configure it with the ``PYOPENCL_CTX`` environment variable
(run the Python command ``pyopencl.create_some_context()`` for more info).
