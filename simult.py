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

    last_every_second = -float('inf')

    for k in range(K):
        t = k * Delta
        every_second = ( t ) % 1.0
        if every_second < last_every_second:
            I_k = 1.0
        else:
            I_k = 0.0
        last_every_second = every_second
        yield k,t,I_k

n0 = {
    # Latent process model
    'rho': 0.99,
    'alpha': 3.0,
    'sigma_eps':math.sqrt(0.001),

    # Latent-to-observable (gain), or, state-to-Lprobability
    'mu': -4.9,
    'beta': 0.0
}
descriptions = {
    'rho': ["", ""],
    'alpha': ["input gain", "volts per amps"],
    'sigma_eps': ["noise amplitude", ""],

    'mu': ["", ""],
    'beta': ["", ""]
}
BETA_RANGE = [0.9, 1.1]
NEURONS = 20
MSEC = 1./ 1000.

na = []
for i in range(NEURONS):
    n = n0.copy()
    d = BETA_RANGE[1] - BETA_RANGE[0]
    n['beta'] = 1.1 # (np.random.rand() * d) + BETA_RANGE[0]
    na.append(n)

#for n in na:eps_k
#    print n['beta']



K = 30000
x_arr = np.zeros((K,))
xlogpr_arr = np.zeros((K,))
Nc_arr = np.zeros((K,))
t_arr = np.zeros((K,))
fire_probability_arr  = np.zeros((K,))
lambda_arr = np.zeros((K,))
I_arr = np.zeros((K,))

last_x_k = 0.0
Nc = 0
for k,t,I_k in simulate_input(K):
    #print t,k,I_k

    n = na[0]

    # x_k is the State
    eps_k = n['sigma_eps'] * np.random.randn()
    x_k = n['rho'] * last_x_k  + n['alpha'] * I_k + eps_k
    last_x_k = x_k

    xlp = n['mu'] + n['beta'] * x_k
    lambda_k = math.exp(xlp)
    # What will guarantee that xlp < 0 ? i.e. probability < 1
    # Where is x reset?
    fire_probability = lambda_k * Delta  # * 100
    fire = fire_probability > np.random.rand()

    #output = (x_k * Delta) > np.random.rand()
    #Nc += output

    Nc += fire

    x_arr[k] = x_k
    Nc_arr[k] = Nc
    t_arr[k] = t
    I_arr[k] = I_k

    xlogpr_arr[k] = xlp

    fire_probability_arr[k] = fire_probability
    lambda_arr[k] = lambda_k

print "T=", T, " spikes/sec=", float(Nc)/T
# fig, ax = plt.subplots() # http://matplotlib.org/1.3.0/examples/pylab_examples/legend_demo.html

#http://matplotlib.org/1.3.0/examples/subplots_axes_and_figures/subplot_demo.html
axes = plt.subplot(3, 1, 1)
tcolor = 'b'
plt.plot(t_arr, x_arr, tcolor+'-', label='$x_k$');
#plt.ylabel('$x_k$ State')
axes.set_ylabel('$x_k$ State', color=tcolor)
#plt.plot(t_arr, xlogpr_arr, 'k-', label='$\\mu + \\beta x_k$'); plt.ylabel('$L(x_k)$ State ($\log \Pr$)')
axes.tick_params('y', colors=tcolor)
plt.legend()

#  q,qq = plt.subplot(4, 1, 2)  # TypeError: 'AxesSubplot' object is not iterable

ax2 = axes.twinx()  # http://matplotlib.org/examples/api/two_scales.html
#plt.subplot(4, 1, 2)
tcolor = 'k'
#plt.plot(t_arr, x_arr, 'k-', label='$x_k$'); plt.ylabel('$x_k$ State')
ax2.plot(t_arr, xlogpr_arr, tcolor + '-', alpha=1.0, label='$\\mu + \\beta x_k$');
ylabel = '$L(x_k)$ State ($\log \Pr$)'
plt.ylabel(ylabel)
ax2.set_ylabel(ylabel, color=tcolor)
ax2.tick_params('y', colors=tcolor)
plt.legend()


plt.subplot(3, 1, 2)
#plt.plot(t_arr, lambda_arr, 'r.', label='\lambda')
#plt.plot(t_arr, np.log(fire_probability_arr), 'r.', label='Log(Pr)')
plt.plot(t_arr, fire_probability_arr, 'r', label='Probability')
plt.legend()


plt.subplot(3, 1, 3)
#plt.plot(x_arr, Nc_arr, 'o-')
plt.plot(t_arr, I_arr, 'k', label='$I_k$ (input)', alpha=0.1)
plt.plot(t_arr, Nc_arr, 'b-', label='$N_c$')

plt.xlabel('Time (Sec)')

#legend = plt.legend(loc='upper center', shadow=True)
plt.legend()

plt.tight_layout()

plt.show()
