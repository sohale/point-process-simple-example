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
        fire = every_second < last_every_second
        #duration = 0.01
        #fire = every_second < duration
        if fire:
            I_k = 1.0
        else:
            I_k = 0.0

        # What is this? Is it Dirac? Then why not multiplied by 1/Delta?

        last_every_second = every_second
        yield k,t,I_k

def report_neuron(n, Delta):
    tau = get_tau(n, Delta)
    print 'Tau=', tau * 1000.0, ' (msec)'
    print 'Noisiness:  sigma_eps = ', n['sigma_eps'] * 1000.0, ' (milli Volts per sample)'

def get_tau(n, Delta):
    tau = - Delta / math.log(n['rho'])
    return tau

n0 = {
    # Latent process model
    'rho': 0.99,
    'alpha': 3.0,
    'sigma_eps':math.sqrt(0.001), # noisiness

    # Latent-to-observable (gain), or, state-to-Lprobability
    'mu': -4.9,
    'beta': 0.0
}
descriptions = {
    'rho': ["", ""],
    'alpha': ["input gain", "volts per amps"],
    'sigma_eps': ["noisiness: noise amplitude", ""],

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

K = 3000
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

    #dirac_factor = 3.0
    #dirac_factor = 7.0  # terrible. Why no refactory period?
    dirac_factor = 1.0

    #dirac_factor = 1.0 / Delta
    #print "dirac_factor,",dirac_factor
    x_k = n['rho'] * last_x_k  + n['alpha'] * I_k * dirac_factor + eps_k
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

    if k == 0:
        report_neuron(n, Delta)

def fix_ylim(ax, arr):
    mn, mx = np.min(arr), np.max(arr)
    m = (mx-mn) * 0.1
    ax.set_ylim([mn - m, mx + m])

print "Simulation time = T =", T, ". Mean rate = ", float(Nc)/T, "(spikes/sec)"


# **********************************************************************************
# *                                  plotting
# **********************************************************************************

PANELS = 3
panel_id = 0

# fig, ax = plt.subplots() # http://matplotlib.org/1.3.0/examples/pylab_examples/legend_demo.html

#http://matplotlib.org/1.3.0/examples/subplots_axes_and_figures/subplot_demo.html
panel_id += 1
axes = plt.subplot(PANELS, 1, panel_id)
tcolor = 'b'
pl1 = plt.plot(t_arr, x_arr, tcolor+'-', label='$x_k$');
#plt.ylabel('$x_k$ State')
fix_ylim(axes, x_arr)
axes.set_ylabel('$x_k$ State', color=tcolor)
axes.tick_params('y', colors=tcolor)

def LIM(narr, a, b=float('inf')):
    rnarr = narr.copy()
    rnarr[rnarr < a] = a
    rnarr[rnarr > b] = b
    return rnarr

def v(n):
    #ht = LIM(t_arr, -float('inf'), 0)
    ht = LIM(t_arr - 1.0, 0, float('inf'))
    tau = get_tau(n, Delta)
    expa = np.exp(- ht / tau) * n['alpha']
    pl = plt.plot(t_arr, expa, 'r', label='$exp(-t/\\tau)$', alpha=0.2, linewidth=5);
    return pl

pl2 = v(na[0])
#plt.legend()

#  q,qq = plt.subplot(4, 1, 2)  # TypeError: 'AxesSubplot' object is not iterable

ax2 = axes.twinx()  # http://matplotlib.org/examples/api/two_scales.html
#plt.subplot(4, 1, 2)
tcolor = 'k'
#plt.plot(t_arr, x_arr, 'k-', label='$x_k$'); plt.ylabel('$x_k$ State')
pl3 = ax2.plot(t_arr, xlogpr_arr, tcolor + '--', alpha=1.0, label='$\\mu + \\beta x_k$')
ylabel = '$L(x_k)$ State ($\log \Pr$)'
#ax2.ylabel(ylabel)
fix_ylim(ax2, xlogpr_arr)
ax2.set_ylabel(ylabel, color=tcolor)
ax2.tick_params('y', colors=tcolor)

lns = pl1+pl2+pl3
labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc=0)
#plt.legend()

panel_id += 1
plt.subplot(PANELS, 1, panel_id)
#plt.plot(t_arr, lambda_arr, 'r.', label='\lambda')
#plt.plot(t_arr, np.log(fire_probability_arr), 'r.', label='Log(Pr)')
plt.plot(t_arr, fire_probability_arr, 'r', label='Probability')
plt.legend()

panel_id += 1
plt.subplot(PANELS, 1, panel_id)
#plt.plot(x_arr, Nc_arr, 'o-')
plt.plot(t_arr, I_arr, 'k', label='$I_k$ (input)', alpha=0.1)
plt.plot(t_arr, Nc_arr, 'b-', label='$N_c$')
plt.xlabel('Time (Sec)')
plt.legend()  # legend = plt.legend(loc='upper center', shadow=True)

plt.tight_layout()

plt.show()

assert panel_id == PANELS
