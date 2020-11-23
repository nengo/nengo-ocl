**************************
Frequently asked questions
**************************

How do I add a new kernel?
==========================

When you make a new Nengo neuron type or learning rule,
it's unlikely that NengoOCL
will know how to simulate it.
To teach NengoOCL how to simulate it,
you have to write an OpenCL kernel.

A good starting point for this
is to look at the existing kernels
in ``nengo_ocl/clra_nonlinearities.py``
and study one that is similar
to the kernel you wish to add.
For each kernel, you'll see
that there are a lot of arguments that go in.
Essentially, they're all lists
of different aspects of the input ragged arrays.
When we're doing computations with ragged arrays,
essentially we want to write kernels
that can take in a list of arrays of different sizes,
and perform the operation on each one of those arrays.

Let's take the BCM kernel as an example.
It gets arguments ``shape0s`` and ``shape1s``;
these are lists of the ``.shape[0]`` and ``.shape[1]`` for each output.
Then we have ``pre_stride0s`` and ``pre_starts``;
these give the strides along axis 0
and a pointer to where the data starts
for each array in the ``pre`` ragged array.
``pre_data`` is the buffer itself.
Then we have similar things for the ``post`` ragged array,
followed by ``theta`` and ``delta``.
Finally, there's a list of the alphas (learning rates)
for all the different operators.
So each of these lists will be of length N,
where N is the number of ``SimBCM`` operators
that we've combined into this single kernel.

Then there's the kernel itself.
It starts by getting ids, as is typical with OpenCL kernels.
Here, ``k`` is the index of which array
we're treating right now (0 <= k < N),
so we use it to index into all of the input lists.
``ij`` tells the kernel which individual element to treat.
We split it into ``i`` and ``j`` (row and column indices),
and then this kernel computes element (i, j) of the output (``delta``).
