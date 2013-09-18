OpenCL-based Nengo simulator
============================

This project is an OpenCL-based simulator for
brain models built from NEural ENGineering Objects in
[Nengo](https://github.com/ctn-waterloo/nengo). It can be orders of magnitude
faster than the default simulator in `nengo` for large models.

Usage
-----

To use the `nengo_ocl` projects OpenCL simulator, build a nengo model as
usual, but pass `sim_ocl.Simulator` when creating a simulator for your model.

```python
import nengo
from nengo_ocl.sim_ocl import Simulator
import pyopencl as cl
from functools import partial

ctx = cl.create_some_context()

# -- build a model
m = nengo.Model('foo')
m.make_node('in', output=1)
m.make_ensemble('A', nengo.LIF(40), 1)
m.connect('in', 'A')
m.probe('A', filter=0.01)

# -- create an OpenCL-backed simulator using a
#    particular device context:
sim = m.simulator(sim_class=partial(Simulator, context=ctx))

sim.run(1.0)
print sim.data('A')
```


Dependencies
------------

General:
* Python 2.6 or better (Python 3 untested)
* One or more OpenCL implementations (test with e.g. PyOpenCl)

Python packages:

* mako
* nengo
* networkx
* NumPy
* PyOpenCL

```bash
( set -e ; for PCK in networkx numpy mako pyopencl ; do pip install $PCK ; done )
```


Install Intel OCL on Debian/Ubuntu Linux
----------------------------------------

Intel provides an OpenCL driver for at least some of their multicore processors.
Core-i7 and Xeon chips can be quite good for running nengo simulations.

Details: http://software.intel.com/en-us/forums/topic/390630

1. Download Intel SDK for OpenCL for applications from Intel website
    http://software.intel.com/en-us/articles/vcsource-tools-opencl-sdk/
2. Extract
    ```
    $ tar zxvf intel_sdk_for_ocl_applications_2012_x64.tgz
    ```
3. Convert RPM files to .deb
    ```
    $ sudo apt-get install -y rpm alien libnuma1   # Get conversion packages
    $ fakeroot alien --to-deb opencl-1.2-*.rpm     # Convert all RPMs
    ```
4. Install .deb packages. They will be put in /opt/intel
    ```
    $ sudo dpkg -i opencl-1.2-*.deb # Install all .debs
    ```
5. Add library to search path
    ```
    $ sudo touch /etc/ld.so.conf.d/intelOpenCL.conf
    ```
    Put in the line: `/opt/intel/opencl-1.2-3.0.67279/lib64`
6. Link the Intel icd file
    ```
    $ sudo ln /opt/intel/opencl-1.2-3.0.67279/etc/intel64.icd /etc/OpenCL/vendors/intel64.icd
    ```
7. Run ldconfig
    ```
    $ sudo ldconfig
    ```

Install AMD OCL on Debian/Ubuntu Linux
--------------------------------------
Can be easy: AMD provides binary drivers and wants people to use OCL.
TODO / link to PyOpenCL wiki.


Install Nvidia OCL on Debian/Ubuntu Linux
--------------------------------------
Can be tricky: Nvidia provides binary drivers but does not seem to want people
to use them.
TODO / link to PyOpenCL wiki.



