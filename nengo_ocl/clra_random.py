"""
This file implements the MRG31k3p random number generator, as well
as functions for using the generator to create various types of
random numbers.

The generator code is modified from Theano, which in turn was based off code
from the SSJ package (L'Ecuyer & Simard)
http://www.iro.umontreal.ca/~simardr/ssj/indexe.html
The Theano code is used under the following licence.

-------------------------------------------------------------------------------
Copyright (c) 2008--2013, Theano Development Team
All rights reserved.

Contains code from NumPy, Copyright (c) 2005-2011, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Theano nor the names of its contributors may be
      used to endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import pyopencl as cl
from mako.template import Template
from .plan import Plan


def get_sample_constants():
    return """
    const int i0 = 0;
    const int i7 = 7;
    const int i9 = 9;
    const int i15 = 15;
    const int i16 = 16;
    const int i22 = 22;
    const int i24 = 24;

    const int M1 = 2147483647;    //2^31 - 1
    const int M2 = 2147462579;    //2^31 - 21069
    const int MASK12 = 511;       //2^9 - 1
    const int MASK13 = 16777215;  //2^24 - 1
    const int MASK2 = 65535;      //2^16 - 1
    const int MULT2 = 21069;
    """


def get_sample_code(ocldtype, oname='sample'):
    """Implements the MRG31k3p random number generator.

    Adapted from Theano under their license (reprinted above).
    """
    if ocldtype == 'float':
        norm = '4.6566126e-10f'  # numpy.float32(1.0/(2**31+65))
        # this was determined by finding the biggest number such that
        # numpy.float32(number * M1) < 1.0
    else:
        norm = '4.656612873077392578125e-10'

    return """
    y1 = ((x12 & MASK12) << i22) + (x12 >> i9)
       + ((x13 & MASK13) << i7) + (x13 >> i24);
    y1 -= (y1 < 0 || y1 >= M1) ? M1 : 0;
    y1 += x13;
    y1 -= (y1 < 0 || y1 >= M1) ? M1 : 0;
    x13 = x12;
    x12 = x11;
    x11 = y1;

    y1 = ((x21 & MASK2) << i15) + (MULT2 * (x21 >> i16));
    y1 -= (y1 < 0 || y1 >= M2) ? M2 : 0;
    y2 = ((x23 & MASK2) << i15) + (MULT2 * (x23 >> i16));
    y2 -= (y2 < 0 || y2 >= M2) ? M2 : 0;
    y2 += x23;
    y2 -= (y2 < 0 || y2 >= M2) ? M2 : 0;
    y2 += y1;
    y2 -= (y2 < 0 || y2 >= M2) ? M2 : 0;

    x23 = x22;
    x22 = x21;
    x21 = y2;

    if (x11 <= x21) {
        %(oname)s = (x11 - x21 + M1) * %(norm)s;
    }
    else
    {
        %(oname)s = (x11 - x21) * %(norm)s;
    }
    """ % dict(norm=norm, oname=oname)


def plan_rand(queue, state, samples, tag=None):
    assert len(state) == 1
    n_streams = state.shape0s[0]
    assert all(s == 6 for s in state.shape1s)
    assert all(s == 6 for s in state.stride0s)
    assert all(s == 1 for s in state.stride1s)

    nY = len(samples)
    assert all(s1 == st0 for s1, st0 in zip(samples.shape1s, samples.stride0s))
    assert all(s == 1 for s in samples.stride1s)

    ocldtype = samples.cl_buf.ocldtype
    sample_constants = get_sample_constants()
    sample_code = get_sample_code(ocldtype)

    text = """
    __kernel void fn(
        __global int *state_data,
        __global const int *Ystarts,
        __global const int *Yshape0s,
        __global const int *Yshape1s,
        __global ${Ytype} *Ydata
    )
    {
        const int gid0 = get_global_id(0);
        const int i = get_global_id(1);
        int j = gid0;
        int n = Yshape0s[i] * Yshape1s[i];
        if (j >= n)
            return;

        ${sample_constants}

        int y1, y2, x11, x12, x13, x21, x22, x23;
        x11 = state_data[gid0*6+0];
        x12 = state_data[gid0*6+1];
        x13 = state_data[gid0*6+2];
        x21 = state_data[gid0*6+3];
        x22 = state_data[gid0*6+4];
        x23 = state_data[gid0*6+5];

        __global ${Ytype} *y = Ydata + Ystarts[i];
        for (; j < n; j += ${n_streams})
        {
            ${Ytype} sample;

            ${sample_code}

            y[j] = sample;
        }

        state_data[gid0*6+0] = x11;
        state_data[gid0*6+1] = x12;
        state_data[gid0*6+2] = x13;
        state_data[gid0*6+3] = x21;
        state_data[gid0*6+4] = x22;
        state_data[gid0*6+5] = x23;
    }
    """

    textconf = dict(
        Ytype=ocldtype, n_streams=n_streams, nY=nY,
        sample_constants=sample_constants, sample_code=sample_code)
    text = Template(text, output_encoding='ascii').render(
        **textconf).decode('ascii')

    full_args = (
        state.cl_buf,
        samples.cl_starts,
        samples.cl_shape0s,
        samples.cl_shape1s,
        samples.cl_buf,
    )

    _fn = cl.Program(queue.context, text).build().fn
    _fn.set_args(*[arr.data for arr in full_args])

    assert n_streams <= queue.device.max_work_group_size
    gsize = (n_streams, nY)
    lsize = (n_streams, 1)
    rval = Plan(
        queue, _fn, gsize, lsize=lsize, name="cl_filter_synapses", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    return rval


def plan_randn(queue, state, samples, tag=None):
    assert len(state) == 1
    n_streams = state.shape0s[0]
    assert all(s == 6 for s in state.shape1s)
    assert all(s == 6 for s in state.stride0s)
    assert all(s == 1 for s in state.stride1s)

    nY = len(samples)
    assert all(s1 == st0 for s1, st0 in zip(samples.shape1s, samples.stride0s))
    assert all(s == 1 for s in samples.stride1s)

    ocldtype = samples.cl_buf.ocldtype
    sample_constants = get_sample_constants()
    sample_code0 = get_sample_code(ocldtype, oname='sample0')
    sample_code1 = get_sample_code(ocldtype, oname='sample1')

    text = """
    __kernel void fn(
        __global int *state_data,
        __global const int *Ystarts,
        __global const int *Yshape0s,
        __global const int *Yshape1s,
        __global ${Ytype} *Ydata
    )
    {
        const int gid0 = get_global_id(0);
        const int i = get_global_id(1);
        int j = gid0;
        int n = Yshape0s[i] * Yshape1s[i];
        if (j >= n)
            return;

        ${sample_constants}

        int y1, y2, x11, x12, x13, x21, x22, x23;
        x11 = state_data[gid0*6+0];
        x12 = state_data[gid0*6+1];
        x13 = state_data[gid0*6+2];
        x21 = state_data[gid0*6+3];
        x22 = state_data[gid0*6+4];
        x23 = state_data[gid0*6+5];

        __global ${Ytype} *y = Ydata + Ystarts[i];
        for (; j < n; j += ${n_streams})
        {
            ${Ytype} sample0, sample1, r2, f;

            do {
                ${sample_code0}
                ${sample_code1}
                sample0 = 2*sample0 - 1;
                sample1 = 2*sample1 - 1;
                r2 = sample0*sample0 + sample1*sample1;
            } while (r2 >= 1.0 || r2 == 0.0);

            // Box-Muller transform
            f = sqrt(-2.0*log(r2)/r2);

            y[j] = f*sample0;
            j += ${n_streams};
            if (j < n)
                y[j] = f*sample1;
            else
                break;
        }

        state_data[gid0*6+0] = x11;
        state_data[gid0*6+1] = x12;
        state_data[gid0*6+2] = x13;
        state_data[gid0*6+3] = x21;
        state_data[gid0*6+4] = x22;
        state_data[gid0*6+5] = x23;
    }
    """

    textconf = dict(
        Ytype=ocldtype, n_streams=n_streams, nY=nY,
        sample_constants=sample_constants,
        sample_code0=sample_code0,
        sample_code1=sample_code1)
    text = Template(text, output_encoding='ascii').render(
        **textconf).decode('ascii')

    full_args = (
        state.cl_buf,
        samples.cl_starts,
        samples.cl_shape0s,
        samples.cl_shape1s,
        samples.cl_buf,
    )

    _fn = cl.Program(queue.context, text).build().fn
    _fn.set_args(*[arr.data for arr in full_args])

    assert n_streams <= queue.device.max_work_group_size
    gsize = (n_streams, nY)
    lsize = (n_streams, 1)
    rval = Plan(
        queue, _fn, gsize, lsize=lsize, name="cl_filter_synapses", tag=tag)
    rval.full_args = full_args     # prevent garbage-collection
    return rval
