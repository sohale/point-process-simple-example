import numpy as np

def simulate_input():
    """ """

    # Simulation Time Length (Intervals)
    # T =
    K = 100
    #Delta = T / float(K)

    Delta = 100 * MSEC
    T = K * Delta

    last_every_second = -1000.0

    for k in range(K):
        t = k * Delta
        every_second = ( t ) % 1.0
        if every_second < last_every_second:
            I_k = 1.0
        else:
            I_k = 0.0
        last_every_second = every_second
        yield t,k,I_k

range(2)
n0 = {'rho': 0.99, 'alpha': 3.0, 'sigma_eps2':0.001, 'mu': -4.9, 'beta': 0.0}
BETA_RANGE = [0.9, 1.1]
NEURONS = 20
MSEC = 0.001

na = []
for i in range(NEURONS):
    n = n0.copy()
    d = BETA_RANGE[1] - BETA_RANGE[0]
    n['beta'] = (np.random.rand() * d) + BETA_RANGE[0]
    na.append(n)

#for n in na:
#    print n['beta']




for t,k,I_k in simulate_input():
    print t,k,I_k
