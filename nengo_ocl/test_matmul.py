import time
import numpy as np
from base import Model
from base import net_get_bias
from base import net_get_encoders
from base import net_get_decoders
from sim_npy import Simulator
from matplotlib import pyplot as plt

import nengo.nef_theano as nef

def net_matrixmul(D1=1, D2=2, D3=3, seed=123, N=50):
    # Adjust these values to change the matrix dimensions
    #  Matrix A is D1xD2
    #  Matrix B is D2xD3
    #  result is D1xD3

    net=nef.Network('Matrix Multiplication', seed=seed)

    # values should stay within the range (-radius,radius)
    radius=1

    # make 2 matrices to store the input
    print "make_array: input matrices A and B"
    net.make_array('A',N,D1*D2,radius=radius, neuron_type='lif')
    net.make_array('B',N,D2*D3,radius=radius, neuron_type='lif')

    # connect inputs to them so we can set their value
    net.make_input('input A',[0]*D1*D2)
    net.make_input('input B',[0]*D2*D3)
    print "connect: input matrices A and B"
    net.connect('input A','A')
    net.connect('input B','B')

    # the C matrix holds the intermediate product calculations
    #  need to compute D1*D2*D3 products to multiply 2 matrices together
    print "make_array: intermediate C"
    net.make_array('C',4 * N,D1*D2*D3,dimensions=2,radius=1.5*radius,
        encoders=[[1,1],[1,-1],[-1,1],[-1,-1]], neuron_type='lif')

    #  determine the transformation matrices to get the correct pairwise
    #  products computed.  This looks a bit like black magic but if
    #  you manually try multiplying two matrices together, you can see
    #  the underlying pattern.  Basically, we need to build up D1*D2*D3
    #  pairs of numbers in C to compute the product of.  If i,j,k are the
    #  indexes into the D1*D2*D3 products, we want to compute the product
    #  of element (i,j) in A with the element (j,k) in B.  The index in
    #  A of (i,j) is j+i*D2 and the index in B of (j,k) is k+j*D3.
    #  The index in C is j+k*D2+i*D2*D3, multiplied by 2 since there are
    #  two values per ensemble.  We add 1 to the B index so it goes into
    #  the second value in the ensemble.  
    transformA=[[0]*(D1*D2) for i in range(D1*D2*D3*2)]
    transformB=[[0]*(D2*D3) for i in range(D1*D2*D3*2)]
    for i in range(D1):
        for j in range(D2):
            for k in range(D3):
                transformA[(j+k*D2+i*D2*D3)*2][j+i*D2]=1
                transformB[(j+k*D2+i*D2*D3)*2+1][k+j*D3]=1
                
    net.connect('A','C',transform=transformA)            
    net.connect('B','C',transform=transformB)            
                
                
    # now compute the products and do the appropriate summing
    print "make_array: output D"
    net.make_array('D',N,D1*D3,radius=radius, neuron_type='lif')

    def product(x):
        return x[0]*x[1]
    # the mapping for this transformation is much easier, since we want to
    # combine D2 pairs of elements (we sum D2 products together)    
    net.connect('C','D',index_post=[i/D2 for i in range(D1*D2*D3)],func=product)

    return net

def test_ref():
    net = net_matrixmul(1, 2, 2, seed=123, N=50)

    net.get_object('input A').origin['X'].decoded_output.set_value(
        np.asarray([.5, -.5]).astype('float32'))
    net.get_object('input B').origin['X'].decoded_output.set_value(
        np.asarray([0, 1, -1, 0]).astype('float32'))

    Dprobe = net.make_probe('D', dt_sample=0.01, pstc=0.1)

    net.run(1) # run for 1 second

    net_data = Dprobe.get_data()
    print net_data.shape
    plt.plot(net_data[:, 0])
    plt.plot(net_data[:, 1])
    plt.show()


def test_matrix_mult_example(D1=1, D2=2, D3=2, Simulator=Simulator, show=True):
    # construct good way to do the 
    # examples/matrix_multiplication.py model
    # from nengo_theano

    # Adjust these values to change the matrix dimensions
    #  Input matrix A is D1xD2
    #  Input matrix B is D2xD3
    #  intermediate tensor of products D1 x D2 x D3
    #  result D is D1xD3

    m = Model(dt=0.001)

    decay_0p01 = np.exp(-m.dt / 0.01)
    decay_0p1 = np.exp(-m.dt / 0.1)

    # XXX present code matches-ish, but 
    # are the network arrays storing
    # linearized matrices in row-major or col-major order?

    if (1, 2, 2) == (D1, D2, D3):
        net = net_matrixmul(1, 2, 2)
        A_vals = [[.5, -.5]]
        B_vals = [[0, -1], [1, 0]]
    else:
        net = net_matrixmul(D1, D2, D3)
        A_vals = np.random.RandomState(123).rand(D1, D2)
        B_vals = np.random.RandomState(124).rand(D2, D3)

    A = {}
    A_in = {}
    A_dec = {}
    for i in range(D1):
        for j in range(D2):
            idx = i * D2 + j
            A_in[(i, j)] = m.signal(value=A_vals[i][j])
            # XXX ensure all constants are held on line like this
            m.filter(1.0, A_in[(i, j)], A_in[(i, j)])
            A[(i, j)] = m.population(50,
                                     bias=net_get_bias(net, 'A', idx))
            A_dec[(i, j)] = m.signal()

            m.encoder(
                A_in[(i, j)],
                A[(i, j)],
                weights=net_get_encoders(net, 'A', idx))
            m.decoder(
                A[(i, j)],
                A_dec[(i, j)],
                weights=net_get_decoders(net, 'A', 'X', idx))
            m.filter(decay_0p01, A_dec[(i, j)], A_dec[(i, j)])
            m.transform((1.0 - decay_0p01) / 1.5, A_dec[(i, j)], A_dec[(i, j)])
            m.signal_probe(A_dec[(i, j)], dt=0.01)

    B = {}
    B_in = {}
    B_dec = {}
    for i in range(D2):
        for j in range(D3):
            idx = i * D3 + j
            B_in[(i, j)] = m.signal(value=B_vals[i][j])
            # XXX ensure all constants are held on line like this
            m.filter(1.0, B_in[(i, j)], B_in[(i, j)])
            B[(i, j)] = m.population(50,
                                     bias=net_get_bias(net, 'B', idx))
            B_dec[(i, j)] = m.signal()

            m.encoder(
                B_in[(i, j)],
                B[(i, j)],
                weights=net_get_encoders(net, 'B', idx))
            m.decoder(
                B[(i, j)],
                B_dec[(i, j)],
                weights=net_get_decoders(net, 'B', 'X', idx))
            m.filter(decay_0p01, B_dec[(i, j)], B_dec[(i, j)])
            m.transform((1.0 - decay_0p01) / 1.5, B_dec[(i, j)], B_dec[(i, j)])

    D = {}
    D_in = {}
    D_dec = {}
    for i in range(D1):
        for j in range(D3):
            idx = i * D3 + j
            D[(i, j)] = m.population(50,
                                     bias=net_get_bias(net, 'D', idx))
            D_in[(i, j)] = m.signal()
            D_dec[(i, j)] = m.signal()
            m.encoder(
                D_in[(i, j)],
                D[(i, j)],
                weights=net_get_encoders(net, 'D', i * D3 + j))
            m.decoder(
                D[(i, j)],
                D_dec[(i, j)],
                weights=net_get_decoders(net, 'D', 'X', i * D3 + j))
            m.filter(decay_0p01, D_in[(i, j)], D_in[(i, j)])

            m.filter(decay_0p1, D_dec[(i, j)], D_dec[(i, j)])
            m.transform(1.0 - decay_0p1, D_dec[(i, j)], D_dec[(i, j)])

            m.signal_probe(D_in[(i, j)], dt=0.01)
            m.signal_probe(D_dec[(i, j)], dt=0.01)

    C = {}
    C_dec = {}
    for i in range(D1):
        for k in range(D3):
            for j in range(D2):
                idx = i * D2 * D3 + k + j * D3
                C[(i, j, k)] = m.population(200,
                                           bias=net_get_bias(net, 'C', idx))
                C_dec[(i, j, k)] = m.signal()
                m.encoder(
                    A_dec[(i, j)],
                    C[(i, j, k)],
                    weights=net_get_encoders(net, 'C', idx)[:, 0:1])
                m.encoder(
                    B_dec[(i, j)],
                    C[(i, j, k)],
                    weights=net_get_encoders(net, 'C', idx)[:, 1:2])
                m.decoder(
                    C[(i, j, k)],
                    C_dec[(i, j, k)],
                    weights=net_get_decoders(net, 'C', 'product', idx))

                m.signal_probe(C_dec[(i, j, k)], dt=0.01)
                m.filter(decay_0p1, C_dec[(i, j, k)], C_dec[(i, j, k)])
                m.transform(1.0 - decay_0p1,
                            C_dec[(i, j, k)], C_dec[(i, j, k)])

                m.transform(1.0 - decay_0p01,
                            C_dec[(i, j, k)],
                            D_in[(i, k)])


    sim = Simulator(m)

    sim.alloc_all()

    t0 = time.time()
    sim.run_steps(1000)
    t1 = time.time()
    print 'sim_npy takes', (t1 - t0)

    if show:
        plt.subplot(4, 1, 1)
        plt.title('A')
        for i in range(D1):
            for j in range(D2):
                plt.plot(sim.signal(A_dec[(i, j)]))

        plt.subplot(4, 1, 2)
        plt.title("C")
        for i in range(D1):
            for j in range(D2):
                for k in range(D3):
                    plt.plot(sim.signal(C_dec[(i, j, k)]))

        plt.subplot(4, 1, 3)
        plt.title("D")
        for i in range(D1):
            for k in range(D3):
                plt.plot(sim.signal(D_dec[(i, k)]))

    net.get_object('input A').origin['X'].decoded_output.set_value(
        np.asarray(A_vals.flatten()).astype('float32'))
    net.get_object('input B').origin['X'].decoded_output.set_value(
        np.asarray(B_vals.flatten()).astype('float32'))

    #Cprobe = net.make_probe('C', dt_sample=0.01, pstc=0.01)
    Dprobe = net.make_probe('D', dt_sample=0.01, pstc=0.1)
    t0 = time.time()
    net.run(1) # run for 1 second
    t1 = time.time()
    print 'nengo_theano takes', (t1 - t0)

    if show:
        plt.subplot(4, 1, 4)
        plt.plot(Dprobe.get_data()[:, 0])
        plt.plot(Dprobe.get_data()[:, 1])
        #plt.plot(Cprobe.get_data()[:, 2])
        #plt.plot(Cprobe.get_data()[:, 3])

        plt.show()


