
from base import Model
from sim_npy import Simulator


def test_matrix_mult_example(D1=1, D2=2, D3=3):
    # construct good way to do the 
    # examples/matrix_multiplication.py model
    # from nengo_theano

    # Adjust these values to change the matrix dimensions
    #  Input matrix A is D1xD2
    #  Input matrix B is D2xD3
    #  intermediate tensor of products D1 x D2 x D3
    #  result D is D1xD3

    m = Model(dt=0.001)

    A = {}
    A_in = {}
    A_dec = {}
    for i in range(D1):
        for j in range(D2):
            A_in[(i, j)] = m.signal()
            A[(i, j)] = m.population(50)
            A_dec[(i, j)] = m.signal()

            m.encoder(A_in[(i, j)], A[(i, j)])
            m.decoder(A[(i, j)], A_dec[(i, j)])

    B = {}
    B_in = {}
    B_dec = {}
    for i in range(D2):
        for j in range(D3):
            B_in[(i, j)] = m.signal()
            B[(i, j)] = m.population(50)
            B_dec[(i, j)] = m.signal()

            m.encoder(B_in[(i, j)], B[(i, j)])
            m.decoder(B[(i, j)], B_dec[(i, j)])

    C = {}
    C_in = {}
    C_dec = {}
    for i in range(D1):
        for j in range(D2):
            for k in range(D3):
                C[(i, j, k)] = m.population(200)
                C_in[(i, j, k)] = m.signal()
                C_dec[(i, j, k)] = m.signal()
                m.transform(1.0, A_dec[(i, j)], C_in[(i, j, k)])
                m.transform(1.0, B_dec[(i, j)], C_in[(i, j, k)])
                m.encoder(C_in[(i, j, k)], C[(i, j, k)])
                m.decoder(C[(i, j, k)], C_dec[(i, j, k)])

    D = {}
    D_in = {}
    D_dec = {}
    for i in range(D1):
        for j in range(D3):
            D[(i, j)] = m.population(50)
            D_in[(i, j)] = m.signal()
            D_dec[(i, j)] = m.signal()
            for k in range(D2):
                m.transform(1.0, C_dec[(i, k, j)], D_in[(i, j)])
            m.encoder(D_in[(i, j)], D[(i, j)])
            m.decoder(D[(i, j)], D_dec[(i, j)])

    sim = Simulator(m)

    sim.alloc_all()
    sim.step()
