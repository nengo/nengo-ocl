import re
import pyopencl as cl
from plan import Plan



def plan_elemwise(queue, body, inputs, outputs):
    # THIS IS A REFERENCE IMPLEMENTATION

    if len(outputs) > 1:
        raise NotImplementedError()

    if outputs[0].ndim > 3:
        raise NotImplementedError()

    # this will be modified many times
    full_body = body
    for anum, arr in enumerate(inputs):
        varname = '\$IN_%i' % anum
        ptrname = 'IPTR_%i' % anum
        indexes = []
        for jj in range(arr.ndim):
            indexes.append('gid%i * %i' % (jj, arr.itemstrides[jj]))
        repl = '%s[%s]' % (ptrname, ' + '.join(indexes))
        full_body = re.sub(varname, repl, full_body)

    for anum, arr in enumerate(outputs):
        varname = '\$OUT_%i' % anum
        ptrname = 'OPTR_%i' % anum
        indexes = []
        for jj in range(arr.ndim):
            indexes.append('gid%i * %i' % (jj, arr.itemstrides[jj]))
        repl = '%s[%s]' % (ptrname, ' + '.join(indexes))
        full_body = re.sub(varname, repl, full_body)

    #print full_body

    params = []
    params.extend(
        ['__global const %s * IPTR_%s' % (arr.ocldtype, inum)
         for inum, arr in enumerate(inputs)])
    params.extend(
        ['__global %s * OPTR_%s' % (arr.ocldtype, inum)
         for inum, arr in enumerate(outputs)])
    joined_params = ', '.join(params)

    text = """
        __kernel void fn(
            %(joined_params)s
                     )
        {
            const int gid0 = get_global_id(0);
            const int gid1 = get_global_id(1);
            const int gid2 = get_global_id(2);

            %(full_body)s
        }
        """ % locals()

    # TODO: support for larger arrays than workgroup size
    _fn = cl.Program(queue.context, text).build().fn
    _fn_args = (queue, outputs[0].shape, None,)
    _fn_args = _fn_args + tuple([arr.data
                                 for arr in inputs + outputs])
    return [Plan(locals())]


def plan_copy(queue, src, dst, tag=None):
    if not (src.shape == dst.shape):
        raise ValueError()
    if (src.data.size != dst.data.size):
        raise NotImplementedError('size', (src, dst))
    if (src.strides != dst.strides):
        raise NotImplementedError('strides', (src, dst))
    if (src.offset != dst.offset):
        raise NotImplementedError('offset', (src, dst))
    if src.offset != 0:
        raise NotImplementedError('offset', (src, dst))
    # XXX: only copy the parts of the buffer that are part of the logical Array
    # XXX: use the elemwise kernel generator above
    config = {'src_type': src.ocldtype,
              'dst_type': dst.ocldtype}
    _fn = cl.Program(queue.context, """
        __kernel void fn(__global const %(src_type)s *src,
                         __global %(dst_type)s *dst
                         )
        {
            dst[get_global_id(0)] = src[get_global_id(0)];
        }
        """ % config).build().fn
    _fn.set_args(src.data, dst.data)
    L, = src.shape
    return Plan(queue, _fn, (L,), None, name='copy', tag=tag)

def plan_inc(queue, buf, amt, tag=None):
    # XXX: only copy the parts of the buffer that are part of the logical Array
    # XXX: use the elemwise kernel generator above
    config = {'buf_type': buf.ocldtype, 'amt': amt}
    _fn = cl.Program(queue.context, """
        __kernel void fn(__global %(buf_type)s *dst)
        {
            dst[get_global_id(0)] = dst[get_global_id(0)] + %(amt)s;
        }
        """ % config).build().fn
    assert buf.offset == 0
    L, = buf.shape
    _fn.set_args(buf.data,)
    return Plan(queue, _fn, (L,), None, name='inc', amt=amt, tag=tag)

