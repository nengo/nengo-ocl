****************************
OpenCL-based Nengo Simulator
****************************

This project is an OpenCL-based simulator for
brain models built using
`Nengo <https://github.com/nengo/nengo>`_.
It can be orders of magnitude
faster than the reference simulator
in ``nengo`` for large models.

Usage
=====

To use the ``nengo_ocl`` project's OpenCL simulator,
build a nengo model as usual,
but pass ``sim_ocl.Simulator``
when creating a simulator for your model::

   import numpy as np
   import nengo

   # define the model
   model = nengo.Network()
   with model:
       stim = nengo.Node(np.sin)
       a = nengo.Ensemble(n_neurons=100, dimensions=1)
       b = nengo.Ensemble(n_neurons=100, dimensions=1)
       nengo.Connection(stim, a)
       nengo.Connection(a, b, function=lambda x: x**2, synapse=0.01)

       probe_a = nengo.Probe(a, synapse=0.01)
       probe_b = nengo.Probe(b, synapse=0.01)

   import nengo_ocl
   import pyopencl as cl
   ctx = cl.create_some_context()

   # build the model
   sim = nengo_ocl.Simulator(model, context=ctx)
   # run the model
   sim.run(10)

   # plot the results
   import matplotlib.pyplot as plt
   plt.plot(sim.trange(), sim.data[probe_a])
   plt.plot(sim.trange(), sim.data[probe_b])
   plt.show()

Dependencies and Installation
=============================

General:
* Python 2.6 or better
* One or more OpenCL implementations (test with e.g. PyOpenCl)

A working installation of OpenCL is the most difficult
part of installing Nengo OCL. See below for more details
on how to install OpenCL.

Python packages:
* NumPy
* nengo
* networkx
* mako
* PyOpenCL

In the ideal case, all of the Python dependencies
will be automatically installed
when installing ``nengo_ocl`` with

.. code-block:: bash

   pip install nengo_ocl

If that doesn't work, then do a developer install
to figure out what's going wrong.

Developer Installation
----------------------

First, ``pip install nengo``.
For best performance, make sure a fast version of Numpy is installed
by following the instructions in the
`Nengo README <http://github.com/nengo/nengo/blob/master/README.rst>`_.
Currently, ``nengo_ocl`` is compatible with Nengo 2.0.x,
supporting most features.

Once Nengo is installed, install the remaining dependencies:

.. code-block:: bash

   pip install networkx mako pyopencl

This repository can then be installed with:

.. code-block:: bash

   git clone https://github.com/nengo/nengo_ocl.git
   cd nengo_ocl
   python setup.py develop --user

If you’re using a ``virtualenv`` (recommended!)
then you can omit the ``--user`` flag.

Installing OpenCL
=================

How you install OpenCL is dependent on your
hardware and operating system.
A good resource for various cases is found
in the PyOpenCL documentation:

* `Installing PyOpenCL on Windows <http://wiki.tiker.net/PyOpenCL/Installation/Windows>`_
* `Installing PyOpenCL on Mac OS X <http://wiki.tiker.net/PyOpenCL/Installation/Mac>`_
* `Installing PyOpenCL on Linux <http://wiki.tiker.net/PyOpenCL/Installation/Linux>`_,
  and a `more detailed guide <http://wiki.tiker.net/OpenCLHowTo>`_

Below are instructions that have worked for the
Nengo OCL developers at one point in time.

Intel OCL on Debian/Ubuntu Linux
--------------------------------

NOTE: the Intel SDK for OpenCL is no longer available. Intel OpenCL drivers
can be found `on Intel's website <https://software.intel.com/en-us/articles/opencl-drivers>`_,
but the installation procedure may differ from below
(see `the PyOpenCL wiki <http://wiki.tiker.net/OpenCLHowTo#Installing_the_Intel_CPU_ICD>`_
for more up-to-date instructions).

Intel provides an OpenCL driver for at least some of their multicore processors.
Core-i7 and Xeon chips can be quite good for running Nengo simulations.

Details: http://software.intel.com/en-us/forums/topic/390630

1. Download Intel SDK for OpenCL for applications from `Intel's OpenCL website <http://software.intel.com/en-us/articles/vcsource-tools-opencl-sdk/>`_

2. Extract

   .. code-block:: bash

      tar zxvf intel_sdk_for_ocl_applications_2012_x64.tgz

3. Convert RPM files to ``.deb``

   .. code-block:: bash

      sudo apt-get install -y rpm alien libnuma1  # Get conversion packages
      fakeroot alien --to-deb opencl-1.2-*.rpm  # Convert all RPMs

4. Install ``.deb`` packages. They will be put in ``/opt/intel``

   .. code-block:: bash

      sudo dpkg -i opencl-1.2-*.deb # Install all .debs

5. Add library to search path

   .. code-block:: bash

      sudo touch /etc/ld.so.conf.d/intelOpenCL.conf

    Put in the line: ``/opt/intel/opencl-1.2-3.0.67279/lib64``

6. Link the Intel ICD file

   .. code-block:: bash

      sudo ln /opt/intel/opencl-1.2-3.0.67279/etc/intel64.icd /etc/OpenCL/vendors/intel64.icd

7. Run ``ldconfig``

   .. code-block:: bash

      sudo ldconfig

AMD OCL on Debian Unstable
--------------------------

On Debian unstable (sid) there are packages in non-free and contrib
to install AMD's OCL implementation easily.
Actually, the easiest thing would be to apt-get install
`python-pyopencl <http://packages.debian.org/sid/python-pyopencl>`_.
But if you're using a virtual environment, you can
``sudo apt-get install opencl-headers libboost-python-dev amd-opencl-icd amd-libopencl1``
and then ``pip install pyopencl``.


Nvidia OCL on Debian/Ubuntu Linux
---------------------------------

On Debian unstable (sid) there are packages
for installing the Nvidia OpenCL implementation as well.

.. code-block:: bash

   sudo apt-get install nvidia-opencl-common nvidia-libopencl1

Ensure that the Nvidia driver version matches the OpenCL library version.
You can check the Nvidia driver version by running ``nvidia-smi`` in the
command line. You can find the OpenCL library version by looking at the
libnvidia-opencl.so.XXX.XX file in the ``/usr/lib/x86_64-linux-gnu/`` folder.

Note! At the time of writing (Sept 2013) these drivers provide only
OpenCL-1.1 rather than the more current OpenCL-1.2.
Consequently, you may find that pyopencl's default build
creates a binary Python module (_cl.so) that cannot be loaded (i.e.
``import pyopencl`` fails in the Python interpreter).
You can fix this one of two ways:

1. Use the generic libOpenCL.so driver-loading library
   from another provider (by e.g. following the Intel
   instructions above), and simply don't try to use new 1.2 features on
   NVidia devices.
2. Follow PyOpenCL's build instructions to compile an OpenCL-1.1 version of
   PyOpenCL.

It's nice to have a CPU OpenCL driver, so we recommend option (1).
