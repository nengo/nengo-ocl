`nengo_ocl`: OpenCL-backed Nengo simulator
==========================================



Installation of Intel OCL on Debian/Ubuntu-based linux
------------------------------------------------------

Details: http://software.intel.com/en-us/forums/topic/390630

1. Download Intel SDK for OpenCL for applications from Intel website
    http://software.intel.com/en-us/articles/vcsource-tools-opencl-sdk/
2. Extract
    ```
    $ tar zxvf intel_sdk_for_ocl_applications_2012_x64.tgz
    ``
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
