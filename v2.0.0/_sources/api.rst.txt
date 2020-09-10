*************
API reference
*************

.. default-role:: obj

Simulator
=========

This is the NengoOCL simulator.
It uses the Nengo builder to take a model
and turn it into signals and operators.
Then, we copy all the signals into OpenCL,
and create OpenCL versions of all the operators.
This is what the ``Simulator.plan_*`` functions do;
each one of them is responsible for
creating an OpenCL kernel (or kernels)
to execute the corresponding operator.

.. autoclass:: nengo_ocl.Simulator

clra_nonlinearities
===================

This is where the kernels for most operators are
(i.e. most of the ``Simulator.plan_*`` functions
call a ``plan_*`` function in here to generate the kernel).
Each plan function follows roughly the same template:

1. Do some checks on the arguments.
2. Generate the C code for the kernel,
   using Mako to fill in variable things like datatypes.
3. Compile the C code as a PyOpenCL Program, and set the arguments.
4. Create and return a ``Plan`` object responsible for executing
   that `pyopencl.Program`.

.. automodule:: nengo_ocl.clra_nonlinearities
    :members:

clra_gemv
=========

This contains kernels specific to the GEMV
(matrix-vector multiply) operation,
since it's such an important and specialized operation.
Most people will not need to know the details
of this module.

.. automodule:: nengo_ocl.clra_gemv
    :members:

RaggedArrays
============

You can think of a `.RaggedArray` a list of arrays.
Whereas NumPy allows lists of arrays
of the same size to be made into one big array
(e.g. if you've got five 3x2 arrays,
you can make a 5x3x2 array),
there's no way to make arrays
of different sizes into one big one.
RaggedArray does this
by making one big memory buffer where all the data is stored,
and managing the reading and writing of that.

.. autoclass:: nengo_ocl.raggedarray.RaggedArray

.. autoclass:: nengo_ocl.clraggedarray.CLRaggedArray

.. autofunction:: nengo_ocl.clraggedarray.data_ptr

.. autofunction:: nengo_ocl.clraggedarray.to_host

Operators
=========

.. automodule:: nengo_ocl.operators
    :members:

Python AST conversion
=====================

.. automodule:: nengo_ocl.ast_conversion
    :members:

Utils
=====

.. automodule:: nengo_ocl.utils
    :members:
