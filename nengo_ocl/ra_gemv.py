
import numpy as np


def raw_ragged_gather_gemv(
    BB,
    Ns,
    alphas,
    A_starts,
    A_ldas,
    A_data,
    A_js_starts,
    A_js_lens,
    A_js_data,
    X_starts,
    X_data,
    X_js_starts,
    X_js_data,
    betas,
    Y_in_starts,
    Y_in_data,
    Y_starts,
    Y_lens,
    Y_data):
    for bb in xrange(BB):
        alpha = alphas[bb]
        beta = betas[bb]
        n_dot_products = A_js_lens[bb]
        y_offset = Y_starts[bb]
        y_in_offset = Y_in_starts[bb]
        M = Y_lens[bb]
        for mm in xrange(M):
            Y_data[y_offset + mm] = beta * Y_in_data[y_in_offset + mm]

        for ii in xrange(n_dot_products):
            x_i = X_js_data[X_js_starts[bb] + ii]
            a_i = A_js_data[A_js_starts[bb] + ii]
            N_i = Ns[a_i]
            a_lda = A_ldas[a_i]
            x_offset = X_starts[x_i]
            a_offset = A_starts[a_i]
            for mm in xrange(M):
                y_sum = 0.0
                for nn in xrange(N_i):
                    a_i_m_n = A_data[a_offset + nn * a_lda + mm]
                    y_sum += X_data[x_offset + nn] * a_i_m_n
                Y_data[y_offset + mm] += alpha * y_sum


def ragged_gather_gemv(alpha, A, A_js, X, X_js,
                       beta, Y, Y_in=None,
                       use_raw_fn=False,
                       tag=None, seq=None,
                      ):
    """
    """
    try:
        float(alpha)
        alpha = [alpha] * len(Y)
    except TypeError:
        pass

    try:
        float(beta)
        beta = [beta] * len(Y)
    except TypeError:
        pass

    if Y_in is None:
        Y_in = Y

    if use_raw_fn:
        # This is close to the OpenCL reference impl
        return raw_ragged_gather_gemv(
            len(Y),
            A.shape1s,
            alpha,
            A.starts,
            A.ldas,
            A.buf,
            A_js.starts,
            A_js.shape0s,
            A_js.buf,
            X.starts,
            X.buf,
            X_js.starts,
            X_js.buf,
            beta,
            Y_in.starts,
            Y_in.buf,
            Y.starts,
            Y.shape0s,
            Y.buf)
    else:
        # -- less-close to the OpenCL impl
        #print alpha
        #print A.buf, 'A'
        #print X.buf, 'X'
        #print Y_in.buf, 'in'

        for i in xrange(len(Y)):
            try:
                y_i = beta[i] * Y_in[i]  # -- ragged getitem
            except:
                print i, beta, Y_in
                raise
            alpha_i = alpha[i]

            x_js_i = X_js[i] # -- ragged getitem
            A_js_i = A_js[i] # -- ragged getitem
            assert len(x_js_i) == len(A_js_i)
            for xi, ai in zip(x_js_i, A_js_i):
                x_ij = X[xi]  # -- ragged getitem
                A_ij = A[ai]  # -- ragged getitem
                try:
                    y_i += alpha_i * np.dot(A_ij, x_ij)
                except:
                    print tag, seq[i]
                    print i, xi, ai, A_ij, x_ij
                    print y_i.shape, A_ij.shape, x_ij.shape
                    raise
            if 0:
                if 'ai' in locals():
                    print 'Gemv writing', tag, A_ij, x_ij, y_i
                else:
                    print 'Gemv writing', tag, y_i
            Y[i] = y_i

