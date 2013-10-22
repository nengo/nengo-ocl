OpenCL-based Nengo Simulator
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

ctx = cl.create_some_context()

# -- build a model
m = nengo.Model('foo')
m.make_node('in', output=1)
m.make_ensemble('A', nengo.LIF(40), 1)
m.connect('in', 'A')
m.probe('A', filter=0.01)

# -- create an OpenCL-backed simulator using a
#    particular device context:
sim = m.simulator(sim_class=Simulator, context=ctx)

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

1. Download Intel SDK for OpenCL for applications from [Intel OpenCL website](http://software.intel.com/en-us/articles/vcsource-tools-opencl-sdk/)
2. Extract

    ```bash
    tar zxvf intel_sdk_for_ocl_applications_2012_x64.tgz
    ```

3. Convert RPM files to .deb

    ```bash
    sudo apt-get install -y rpm alien libnuma1   # Get conversion packages
    fakeroot alien --to-deb opencl-1.2-*.rpm     # Convert all RPMs
    ```

4. Install .deb packages. They will be put in /opt/intel

    ```bash
    sudo dpkg -i opencl-1.2-*.deb # Install all .debs
    ```

5. Add library to search path

    ```bash
    sudo touch /etc/ld.so.conf.d/intelOpenCL.conf
    ```

    Put in the line: `/opt/intel/opencl-1.2-3.0.67279/lib64`

6. Link the Intel icd file

    ```bash
    sudo ln /opt/intel/opencl-1.2-3.0.67279/etc/intel64.icd /etc/OpenCL/vendors/intel64.icd
    ```

7. Run ldconfig

    ```bash
    sudo ldconfig
    ```

Install AMD OCL on Debian/Ubuntu Linux
--------------------------------------
Can be easy: AMD provides binary drivers and wants people to use OCL.
[Instructions on PyOpenCL
wiki](http://wiki.tiker.net/PyOpenCL/Installation/Linux/Ubuntu)

Install AMD OCL on Debian Unstable
----------------------------------

On Debian unstable (sid) there are packages in non-free and contrib
to install AMD's OCL implementation easily.
Actually, the easiest thing would be to apt-get install
[python-pyopencl](http://packages.debian.org/sid/python-pyopencl).
But if you're using a virtual environment, you can
`apt-get install opencl-headers libboost-python-dev
amd-opencl-icd amd-libopencl1`
and then `pip install pyopencl`.

Install Nvidia OCL on Debian/Ubuntu Linux
--------------------------------------
Debian (at least the rolling "sid" distribution)
provides easily-installable .deb files for OpenCL:

```
sudo apt-get install nvidia-opencl-common nvidia-libopencl1

```

Ensure that the nvidia driver version matches the opencl library version.
You can check the nvidia driver version by running `nvidia-smi` in the 
command line. You can find the opencl library version by looking at the 
libnvidia-opencl.so.XXX.XX file in the `/usr/lib/x86_64-linux-gnu/` folder.

N.B. that at the time of writing (Sept 2013) these drivers provide only
OpenCL-1.1 rather than the more current OpenCL-1.2.
Consequently, you may find that pyopencl's default build
creates a binary Python module (_cl.so) that cannot be loaded (i.e.
`import pyopencl` fails in the Python interpreter).
You can fix this one of two ways:

1. Use the generic libOpenCL.so driver-loading library
   from another provider (by e.g. following the Intel
   instructions above), and simply don't try to use new 1.2 features on
   NVidia devices,
2. Follow PyOpenCL's build instructions to compile an OpenCL-1.1 version of
   PyOpenCL.

It's nice to have a CPU OpenCL driver, so we recommend option (1).
