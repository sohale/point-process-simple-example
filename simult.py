import math
import numpy as np
import matplotlib.pyplot as plt

global Delta
global K
global T
def simulate_input(_K):
    """ """

    global Delta
    global K
    global T
    # Simulation Time Length (Intervals)
    # T =
    K = _K
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
        yield k,t,I_k

range(2)
n0 = {'rho': 0.99, 'alpha': 3.0, 'sigma_eps':math.sqrt(0.001), 'mu': -4.9, 'beta': 0.0}
BETA_RANGE = [0.9, 1.1]
NEURONS = 20
MSEC = 0.001

na = []
for i in range(NEURONS):
    n = n0.copy()
    d = BETA_RANGE[1] - BETA_RANGE[0]
    n['beta'] = (np.random.rand() * d) + BETA_RANGE[0]
    na.append(n)

#for n in na:eps_k
#    print n['beta']



K = 100
x_arr = np.zeros((K,))
Nc_arr = np.zeros((K,))
last_x_k = 0.0
Nc = 0
for k,t,I_k in simulate_input(K):
    print t,k,I_k

    n = na[0]

    eps_k = n['sigma_eps'] * np.random.randn()
    x_k = n['rho'] * last_x_k  + n['alpha'] * I_k + eps_k
    last_x_k = x_k


    output = x_k * Delta > np.random.rand()
    Nc += output

    x_arr[k] = x_k
    Nc_arr[k] = Nc

print x_arr

plt.plot(x_arr, Nc_arr, 'o-')
plt.ylabel('some numbers')
plt.show()
