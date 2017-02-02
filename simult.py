import math
import numpy as np
import matplotlib.pyplot as plt

"""
    Simulates the example 1 of [1].
    [1]. Smith and Brown 2003. "Estimating a State-Space Model from Point Process Observations".
"""

MSEC = 1./ 1000.

# some utility functions
def visualise_analytical_relaxation(n, Delta0, t_arr, plt):
    """ For a given neuron, based on its alpha, rho  """
    ht = clamp_numpyarr(t_arr - 1.0, 0, float('inf'))
    tau = get_neuron_tau(n, Delta0)
    expa = np.exp(- ht / tau) * n['alpha']
    pl = plt.plot(t_arr, expa, 'r', label='$exp(-t/\\tau)$', alpha=0.2, linewidth=5)
    return pl

def clamp_numpyarr(narr, a, b=float('inf')):
    """ clamps (limits) a numpy array between a,b """
    rnarr = narr.copy()
    rnarr[rnarr < a] = a
    rnarr[rnarr > b] = b
    return rnarr

# **********************************************************************************
# *                                  neuron model
# **********************************************************************************

"""
A neurons is characterised by equations 2.2 and 2.6, as in example 1 of [1].
Ref. equations #Eq.1 and #Eq.2
"""

def report_neuron(n, Delta):
    tau = get_neuron_tau(n, Delta)
    print 'Tau=', tau * 1000.0, ' (msec)'
    print 'Noisiness:  sigma_eps = ', n['sigma_eps'] * 1000.0, ' (milli Volts per sample)'

def get_neuron_tau(n, Delta):
    # todo: def get_ER_tau(n, Delta, rho)  # ER: Exponential Relaxation
    tau = - Delta / math.log(n['rho'])
    return tau

def get_neuron_rho(tau, Delta):
    rho = math.exp( - Delta / tau )
    return rho

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

DELTA0 = 1.0 * MSEC

# **********************************************************************************
# *                                  simulation
# **********************************************************************************

class simulator_args(object):

    def __init__(self, _K=None, duration=None):
        #self.Delta = 0.000
        #self.K = -1
        #self.T = 0.0000

        if _K is not None:
            self.K = _K
            #self.Delta = self.T / float(self.K)
            self.Delta = 1 * MSEC
            self.T = self.K * self.Delta

        elif duration is not None:
            # self.K = 3000
            # self.T = 3.0; self.Delta =  # sec
            self.T = duration
            self.Delta = 1 * MSEC  * 0.01
            self.K = int(self.T / self.Delta + 1 - 0.00001)
            print "K=", self.K

        else:
            raise "Error"


global simargs  # simulation args
simargs = None
# simulator.K

def simulate_input(_K=None, duration=None):
    """ """

    global simargs
    # Simulation Time Length (Intervals)
    # simargs.T =
    simargs = simulator_args(_K, duration)

    last_every_second = -float('inf')

    for k in range(simargs.K):
        t = k * simargs.Delta
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


BETA_RANGE = [0.9, 1.1]
NEURONS = 20

na = []
for i in range(NEURONS):
    n = n0.copy()
    d = BETA_RANGE[1] - BETA_RANGE[0]
    n['beta'] = 1.1 # (np.random.rand() * d) + BETA_RANGE[0]
    na.append(n)

#print "Beta: ",
#for n in na:eps_k
#    print n['beta'],
# print

# simargs.K = 3000
# simargs.T = 3.0; simargs.Delta =  # sec
# simargs.K = int(simargs.T / simargs.Delta + 1)

#assert simargs.K
#assert simargs.Delta
#assert simargs.T


last_x_k = 0.0
Nc = 0
for k,t,I_k in simulate_input(duration=3.0):
    if k == 0:
        x_arr = np.zeros((simargs.K,))
        xlogpr_arr = np.zeros((simargs.K,))
        Nc_arr = np.zeros((simargs.K,))
        t_arr = np.zeros((simargs.K,))
        fire_probability_arr  = np.zeros((simargs.K,))
        lambda_arr = np.zeros((simargs.K,))
        I_arr = np.zeros((simargs.K,))

        _tau = get_neuron_tau(na[0], 1.0 * MSEC)
        _rho_corrected = get_neuron_rho(_tau, simargs.Delta)
        _sigma_eps_corrected = na[0]['sigma_eps'] * math.sqrt(simargs.Delta/DELTA0)
        print "_rho_corrected = ", _rho_corrected, "rho=",na[0]['rho']
        print "_sigma_eps_corrected = ", _sigma_eps_corrected, "sigma_eps=",na[0]['sigma_eps']


    #print t,k,I_k

    n = na[0]

    # *************************
    # *  Neuron model
    # *************************
    if False:
        # x_k is the State
        eps_k = n['sigma_eps'] * np.random.randn()

        #dirac_factor = 3.0
        #dirac_factor = 7.0  # terrible. Why no refactory period?
        dirac_factor = 1.0

        #dirac_factor = 1.0 / simargs.Delta
        #print "dirac_factor,",dirac_factor
        x_k = n['rho'] * last_x_k  + n['alpha'] * I_k * dirac_factor + eps_k  #Eq.1

    if True:
        eps_k = _sigma_eps_corrected * np.random.randn()
        x_k = _rho_corrected * last_x_k  + n['alpha'] * I_k + eps_k  #Eq.1


    last_x_k = x_k

    xlp = n['mu'] + n['beta'] * x_k
    lambda_k = math.exp(xlp)   #Eq.2
    # What will guarantee that xlp < 0 ? i.e. probability < 1
    # Where is x reset?


    # *****************************
    # * Point process simulation
    # *****************************

    fire_probability = lambda_k * simargs.Delta  # * 100
    fire = fire_probability > np.random.rand()

    #output = (x_k * simargs.Delta) > np.random.rand()
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
        report_neuron(n, simargs.Delta)

print "Simulation time = T =", simargs.T, ". Mean rate = ", float(Nc)/simargs.T, "(spikes/sec)"
print "Integral of lambda = ", np.sum(lambda_arr) * simargs.Delta

# **********************************************************************************
# *                                  plotting
# **********************************************************************************

class Panels(object):
    def __init__(self, panels):
        self.PANELS = panels
        self.panel_id = 0
        self.cax = None  # current ax
        self.ax1 = None
        self.ax2 = None

    def next_panel(self):
        self.panel_id += 1
        self.ax1 = plt.subplot(self.PANELS, 1, self.panel_id)
        self.cax = self.ax1  # also equal to plt
        self.ax2 = None
        # fig, ax = plt.subplots() # http://matplotlib.org/1.3.0/examples/pylab_examples/legend_demo.html
        #http://matplotlib.org/1.3.0/examples/subplots_axes_and_figures/subplot_demo.html

    def second_y_axis(self):
        self.ax2 = panels.ax1.twinx()  # http://matplotlib.org/examples/api/two_scales.html
        self.cax = self.ax2

    def fix_ylim(self, arr):
        mn, mx = np.min(arr), np.max(arr)
        m = (mx-mn) * 0.1
        self.cax.set_ylim([mn - m, mx + m])

    def ylabels_double(self, ylabel, tcolor):
        # for double y
        self.cax.set_ylabel(ylabel, color=tcolor)
        self.cax.tick_params('y', colors=tcolor)

    def multi_legend(self, added_plts):
        """
            Legend for multiple plots, withe superimposed or double-y-axis.
            Usage: panels.multi_legend(pl1 + pl2 + pl3)
        """
        #added_plts = pl1+pl2+pl3
        labs = [l.get_label() for l in added_plts]
        plt.legend(added_plts, labs, loc=0)


panels = Panels( 4 )

panels.next_panel()

tcolor = 'b'
pl1 = plt.plot(t_arr, x_arr, tcolor+'-', label='$x_k$')
panels.fix_ylim(x_arr)
panels.ylabels_double('$x_k$ State', tcolor)

pl2 = visualise_analytical_relaxation(na[0], DELTA0, t_arr, plt)

panels.second_y_axis()
tcolor = 'k'
pl3 = panels.cax.plot(t_arr, xlogpr_arr, tcolor + '--', alpha=1.0, label='$\\mu + \\beta x_k$')

panels.fix_ylim(xlogpr_arr)
panels.ylabels_double('$L(x_k)$ State ($\\log \\Pr$)', tcolor)

panels.multi_legend(pl1 + pl2 + pl3)

panels.next_panel()
panels.cax.plot(t_arr, lambda_arr, 'r.', label='$\\lambda$')
panels.cax.legend()


panels.next_panel()
#plt.plot(t_arr, lambda_arr, 'r.', label='\lambda')
#plt.plot(t_arr, np.log(fire_probability_arr), 'r.', label='Log(Pr)')
panels.cax.plot(t_arr, fire_probability_arr, 'r', label='$\\Pr$ / bin')
panels.cax.legend()

panels.next_panel()
panels.cax.plot(t_arr, I_arr, 'k', label='$I_k$ (input)', alpha=0.1)
panels.cax.plot(t_arr, Nc_arr, 'b-', label='$N_c$')
panels.cax.legend()  # legend = plt.legend(loc='upper center', shadow=True)
plt.xlabel('Time (Sec)')

plt.tight_layout()
plt.title("Delta = %1.4f (msec)"%(simargs.Delta/MSEC))
plt.show()

assert panels.panel_id == panels.PANELS



# Misc notes:
#  q,qq = plt.subplot(4, 1, 2)  # TypeError: 'AxesSubplot' object is not iterable

#legend/plot label versus panel label:       plt.ylabel(..) versus  ..plot(..,label=...)

# 'xlabel' versus 'set_xlabel': for plt (current panel/plot) and axis (subpanel) respectively. # plt.xlabel('Time (Sec)')
