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

    Delta = 1 * MSEC
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



K = 3000
x_arr = np.zeros((K,))
Nc_arr = np.zeros((K,))
t_arr = np.zeros((K,))
fire_probability_arr  = np.zeros((K,))
lambda_arr  = np.zeros((K,))

last_x_k = 0.0
Nc = 0
for k,t,I_k in simulate_input(K):
    #print t,k,I_k

    n = na[0]

    # x_k is the State
    eps_k = n['sigma_eps'] * np.random.randn()
    x_k = n['rho'] * last_x_k  + n['alpha'] * I_k + eps_k
    last_x_k = x_k

    lambda_k = math.exp(n['mu'] + n['beta'] * x_k)
    fire_probability = lambda_k * Delta
    fire = fire_probability > np.random.rand()

    #print "n['mu'] + n['beta'] * x_k, lambda_k * Delta", n['mu'],n['beta'],x_k, lambda_k, Delta

    #output = (x_k * Delta) > np.random.rand()
    #Nc += output

    Nc += fire

    x_arr[k] = x_k
    Nc_arr[k] = Nc
    t_arr[k] = t

    fire_probability_arr[k] = fire_probability
    lambda_arr[k] = lambda_k


plt.subplot(2, 1, 1)
# fig, ax = plt.subplots() # http://matplotlib.org/1.3.0/examples/pylab_examples/legend_demo.html

plt.subplot(2, 1, 2)
#plt.plot(x_arr, Nc_arr, 'o-')
plt.plot(t_arr, Nc_arr, 'o-', label='N_c')
#plt.plot(t_arr, lambda_arr, 'r.', label='\lambda')
#plt.plot(t_arr, np.log(fire_probability_arr), 'r.', label='Log(Pr)')
plt.plot(t_arr, fire_probability_arr, 'r', label='Log(Pr)')
plt.plot(t_arr, x_arr, 'k-', label='x_k')
plt.xlabel('Time (Sec)')
#legend = plt.legend(loc='upper center', shadow=True)
plt.legend()
plt.show()
