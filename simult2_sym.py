import math
import numpy as np

# import matplotlib
#matplotlib.use('qtagg')
#matplotlib.use('macosx')
#matplotlib.use("Qt5Agg")
# matplotlib.use("MACOSX")
# matplotlib.use("TkAgg") # black
#
# Failed attempt to change font
# from matplotlib import rcParams
# rcParams['font.family'] = 'Tahoma'
# rcParams['font.family'] = 'Arial'

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# https://matplotlib.org/3.4.0/thirdpartypackages/index.html

"""
    Simulates the example 1 of [1].
    [1]. Smith and Brown 2003. "Estimating a State-Space Model from Point Process Observations".
"""

MSEC = 1. / 1000.

# some utility functions


def visualise_analytical_relaxation(n, Delta0, t_arr, plt):
    """ For a given neuron, based on its alpha, rho  """
    ht = clamp_numpyarr(t_arr - 1.0, 0, float('inf'))
    tau = get_neuron_tau(n, Delta0)
    expa = np.exp(- ht / tau) * n['alpha'] * \
       np.heaviside(t_arr - 1.0, 1.0)
    pl = plt.plot(t_arr, expa, 'r', label='$\exp(-t/\\tau)$',
                  alpha=0.2, linewidth=5)
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
    print('Tau=', tau * 1000.0, ' (msec)')
    print('Noisiness:  sigma_eps = ',
          n['sigma_eps'] * 1000.0, ' (milli Volts per sample)')


def get_neuron_tau(n, Delta):
    # todo: def get_ER_tau(n, Delta, rho)  # ER: Exponential Relaxation
    tau = - Delta / math.log(n['rho'])
    return tau


def get_neuron_rho(tau, Delta):
    rho = math.exp(- Delta / tau)
    return rho


CORRECT_IT = 1.0

n0 = {
    # Latent process model
    'rho': 0.99,
    'alpha': 3.0,
    'sigma_eps': math.sqrt(0.001),  # noisiness

    # Latent-to-observable (gain), or, state-to-Lprobability
    'mu': -4.9 + CORRECT_IT*math.log(1000),
    'beta': 0.0
}
print(repr(n0))

def describe_model_latex(neuron_array):
    def cut_sigbits(x):
        """ shortest representation using 2 signifiacnt digits """
        import math
        s1 = "%g" % x
        s2 = "%.2f" % x
        s3 = "%g" % round(x, 2)
        print('sss>', s1, s2, s3)
        s23 = s2 if len(s2) < len(s3) else s3
        s123 = s1 if len(s1) < len(s23) else s23
        return s123
    def latex_sem_number(param_name):
        values = [n[param_name] for n in neuron_array]
        std = np.std(values)
        ε = 0.000000001
        pm_suffix = "\\pm%s" % (cut_sigbits(std),) if std > ε else ""
        return "%s%s" % ( \
            cut_sigbits(np.mean(values)), \
            pm_suffix, \
        )
    return " $(\\rho=%s, \\alpha=%s, \\sigma_\\epsilon=%s)$ " %(
        latex_sem_number('rho'),
        latex_sem_number('alpha'),
        latex_sem_number('sigma_eps'),
    )

descriptions = {
    'rho': ["", ""],
    'alpha': ["input gain", "volts per amps"],
    'sigma_eps': ["noisiness: noise amplitude", ""],

    'mu': ["", ""],
    'beta': ["", ""]
}

# Part of the problem spects, not the Delta used in the simulation.
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

            """
            elif duration is not None:
                # self.K = 3000
                # self.T = 3.0; self.Delta =  # sec
                self.T = duration
                self.Delta = 1 * MSEC  * 0.01
                self.K = int(self.T / self.Delta + 1 - 0.00001)
                print( "K=", self.K )
            """
        elif duration is not None:
            self.T = duration
            self.Delta = 1 * MSEC * 0.2
            self.K = int(self.T / self.Delta + 1 - 0.00001)
            print("K=", self.K)

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
        every_second = (t) % 1.0
        fire = every_second < last_every_second
        #duration = 0.01
        #fire = every_second < duration
        if fire:
            I_k = 1.0
        else:
            I_k = 0.0

        # What is this? Is it Dirac? Then why not multiplied by 1/Delta?

        last_every_second = every_second
        yield k, t, I_k


BETA_RANGE = [0.9, 1.1]
NEURONS = 20

na = []
for i in range(NEURONS):
    n = n0.copy()
    d = BETA_RANGE[1] - BETA_RANGE[0]
    n['beta'] = 1.1  # (np.random.rand() * d) + BETA_RANGE[0]
    na.append(n)

# print( "Beta: ", end = '')
#for n in na:eps_k
#    print( n['beta'], end = '')
# print()

# simargs.K = 3000
# simargs.T = 3.0; simargs.Delta =  # sec
# simargs.K = int(simargs.T / simargs.Delta + 1)

#assert simargs.K
#assert simargs.Delta
#assert simargs.T


last_x_k = 0.0
Nc = 0
for k, t, I_k in simulate_input(duration=3.0):
    if k == 0:
        x_arr = np.zeros((simargs.K,))
        xlogpr_arr = np.zeros((simargs.K,))
        Nc_arr = np.zeros((simargs.K,))
        t_arr = np.zeros((simargs.K,))
        fire_probability_arr = np.zeros((simargs.K,))
        lambda_arr = np.zeros((simargs.K,))
        I_arr = np.zeros((simargs.K,))

        _tau = get_neuron_tau(na[0], DELTA0)
        _rho_corrected = get_neuron_rho(_tau, simargs.Delta)
        _sigma_eps_corrected = na[0]['sigma_eps'] * \
            math.sqrt(simargs.Delta/DELTA0)
        print("_rho_corrected = ", _rho_corrected, "rho=", na[0]['rho'])
        print("_sigma_eps_corrected = ", _sigma_eps_corrected,
              "sigma_eps=", na[0]['sigma_eps'])

    # print( t, k, I_k )

    n = na[0]

    # *************************
    # *  Neuron model
    # *************************
    if False:
        # x_k is the State
        eps_k = n['sigma_eps'] * np.random.randn()

        # dirac_factor = 7.0  # terrible. Why no refactory period?
        dirac_factor = 1.0

        #dirac_factor = 1.0 / simargs.Delta
        # print( "dirac_factor,",dirac_factor )
        x_k = n['rho'] * last_x_k + n['alpha'] * \
            I_k * dirac_factor + eps_k  # Eq.1

    if True:
        dirac_factor = 1.0
        eps_k = _sigma_eps_corrected * np.random.randn()
        x_k = _rho_corrected * last_x_k + \
            n['alpha'] * I_k * dirac_factor + eps_k  # Eq.1

    last_x_k = x_k

    xlp = n['mu'] + n['beta'] * x_k
    lambda_k = math.exp(xlp)  # Eq.2
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

print("Simulation time = T =", simargs.T, ". Mean rate = ",
      float(Nc)/simargs.T, "(spikes/sec)")
print("Integral of lambda = ", np.sum(lambda_arr) * simargs.Delta)
print("Mean lambda = ", np.sum(lambda_arr) * simargs.Delta / simargs.T)


def cumsum0(x, cutlast=True):
    """ generates a cumsum starting with 0.0, of the same size as x, i.e. removes the last element, and returning it separately. """
    c = np.cumsum(x)
    maxval = c[-1]
    c = np.concatenate((np.array([0.0]), c))
    if cutlast:
        c = c[:-1]

    # return c, maxval
    return c


def generate_unit_isi(total_rate):
    # print( "ISI(%g):"%(total_rate), end='')
    st = 0.0
    l = []
    while st < total_rate:
        if st > 0.0:
            l.append(st)
        isi = -math.log(np.random.rand())
        # print( isi,l, end='')
        # l.append(isi)
        st += isi
        # if st > total_rate:
        #    break
    # print()
    return np.array(l)


# t_arr
#cumintegr_arr, maxv = cumsum0(lambda_arr, cutlast=False)*simargs.Delta
#t_arr_aug = np.concatenate(t_arr, np.array([t_arr[-1]+simargs.Delta]))
cumintegr_arr = cumsum0(lambda_arr, cutlast=True)*simargs.Delta
maxv = np.max(cumintegr_arr)

interp_func = interp1d(cumintegr_arr, t_arr, kind='linear')
# Time-rescaled quantiles:
#quantiles01 = np.arange(0,maxv,maxv/10.0 * 10000)
quantiles01 = generate_unit_isi(maxv)
# print( quantiles01 )


assert quantiles01.shape[0] == 0 or np.max(
    cumintegr_arr) >= np.max(quantiles01)
assert quantiles01.shape[0] == 0 or np.min(
    cumintegr_arr) <= np.min(quantiles01)
if quantiles01.shape == (0,):
    print("Warning: empty spike train. *****")

spike_times = interp_func(quantiles01)

# based on stackoverflow.com/questions/19956388/scipy-interp1d-and-matlab-interp1
# spikes = (spike_times, quantiles01)  # spikes and their accumulated lambda

# **********************************************************************************
# *                                  plotting
# **********************************************************************************


class Panels(object):
    def __init__(self, panels):
        self.PANELS = panels
        self.panel_id = 0
        self.cax = None  # current ax
        #self.axi = []
        self.ax1 = None
        self.ax2 = None

        self.xlim = None

    def next_panel(self):
        self.panel_id += 1
        self.ax1 = plt.subplot(self.PANELS, 1, self.panel_id)
        self.cax = self.ax1  # also equal to plt
        self.ax2 = None
        # fig, ax = plt.subplots() # http://matplotlib.org/1.3.0/examples/pylab_examples/legend_demo.html
        # http://matplotlib.org/1.3.0/examples/subplots_axes_and_figures/subplot_demo.html

    def add_second_y_axis(self):
        self.ax2 = panels.ax1.twinx()  # http://matplotlib.org/examples/api/two_scales.html
        self.cax = self.ax2

    def fix_currenty_ylim(self, arr, margin_ratio=0.1):
        mn, mx = np.min(arr), np.max(arr)
        m = (mx-mn) * margin_ratio
        self.cax.set_ylim([mn - m, mx + m])

    def set_currenty_ylabel(self, ylabel, tcolor):
        # for double y
        self.cax.set_ylabel(ylabel, color=tcolor)
        self.cax.tick_params('y', colors=tcolor)

    def multi_legend(self, added_plts, loc='upper left'):
        """
            Legend for multiple plots, withe superimposed or double-y-axis.
            Usage: panels.multi_legend(pl1 + pl2 + pl3)
        """
        #added_plts = pl1+pl2+pl3
        labs = [l.get_label() for l in added_plts]
        plt.legend(added_plts, labs, loc=loc, shadow=True, fontsize=7)

    # set common xlim
    def set_common_xlims(self, x0, x1):
        self.xlim = [x0, x1]

    def apply_common_xlims(self):
        if self.xlim:
            plt.xlim(self.xlim[0], self.xlim[1])

    def no_xticks(self):
        self.cax.set_xticklabels([])

# ##########################

panels = Panels(4)
#panels.set_common_xlims(1.00 - 0.01, 1.00 + 0.01)
# ##########################
panels.next_panel() # 1


plt.title("Delta = %1.4f (msec), %s" % (simargs.Delta/MSEC, describe_model_latex(na)))

tcolor = 'b'
pl1 = plt.plot(t_arr, x_arr, tcolor+'-', label='$x_k$')
panels.fix_currenty_ylim(x_arr, 0.1)
panels.set_currenty_ylabel('$x_k$ State', tcolor)

pl2 = visualise_analytical_relaxation(na[0], DELTA0, t_arr, plt)


panels.add_second_y_axis()
tcolor = 'k'
pl3 = panels.cax.plot(t_arr, xlogpr_arr, tcolor + '--',
                      alpha=1.0, label='$\\mu + \\beta x_k$')

panels.fix_currenty_ylim(xlogpr_arr, 0.1)
panels.set_currenty_ylabel('$L(x_k)$ State ($\\log \\Pr$)', tcolor)

panels.multi_legend(pl1 + pl2 + pl3)
panels.apply_common_xlims()
panels.no_xticks()

# ##########################
panels.next_panel() # 2
tcolor = 'r'
plt1 = panels.cax.plot(t_arr, lambda_arr, tcolor+'-', alpha=0.5, label='$\\lambda$')
# panels.cax.legend()
panels.fix_currenty_ylim(lambda_arr, 0.1)
panels.set_currenty_ylabel('$\\lambda$ (sec.$^-1$)', tcolor)

panels.add_second_y_axis()
tcolor = 'b'
ISI_arr = 1.0 / lambda_arr
plt2 = panels.cax.plot(t_arr, ISI_arr, tcolor+'-', alpha=0.6, label='ISI')
panels.fix_currenty_ylim(ISI_arr, 0.1)
panels.set_currenty_ylabel('ISI (sec.)', tcolor)
#panels.multi_legend(plt1 + plt2)
panels.apply_common_xlims()
#panels.cax.set_ylim(-0.1, 15.0)

# third axis
panels.add_second_y_axis()
tcolor = 'k'
cumintegr_arr = np.cumsum(lambda_arr)*simargs.Delta
plt3 = panels.cax.plot(t_arr, cumintegr_arr, tcolor+'-',
                       alpha=0.6, label='$\\int\\lambda dt$') # a\n $\\int...
panels.cax.spines['right'].set_position(('data', np.max(t_arr)))
plt4 = panels.cax.plot(spike_times, quantiles01, 'k.',
                       alpha=1.0, label='spikes')


panels.fix_currenty_ylim(cumintegr_arr, 0.1)
panels.set_currenty_ylabel('Integral $\\lambda$', tcolor)
panels.apply_common_xlims()
panels.no_xticks()

panels.multi_legend(plt1 + plt2 + plt3)

# ##########################
panels.next_panel() # 3
#plt.plot(t_arr, lambda_arr, 'r.', label='\lambda')
#plt.plot(t_arr, np.log(fire_probability_arr), 'r.', label='Log(Pr)')
panels.cax.plot(t_arr, fire_probability_arr, 'r', label='$\\Pr$ / bin')
panels.cax.legend()
panels.apply_common_xlims()
panels.no_xticks()

# ##########################


def nc_to_spk(t_arr, nc_arr, shift=+1):
    """
    shift=+1 (default) => post-spike Nc
    shift=0  => pre-spikes Nc
    """
    tarr = np.nonzero(np.diff(nc_arr) > 0)[0] + shift
    return t_arr[tarr], nc_arr[tarr]


spkt, nc = nc_to_spk(t_arr, Nc_arr)
# ##########################
panels.next_panel() # 4
plt1_N =\
    panels.cax.plot(t_arr, Nc_arr, 'b-', label='$N_c$')
random_shift_sz = Nc_arr[-1]
randy = 0  # np.random.rand(spike_times.shape[0]) * random_shift_sz

panels.cax.plot(spike_times, spike_times*0+0.1+randy*0.9, 'k.')
#panels.cax.plot(t_arr, Nc_arr, 'b-', label='$N_c$')
plt3_s2 =\
    panels.cax.plot(spkt, nc, 'k.', label='Spikes', alpha=0.9)
plt.xlabel('Time (Sec)')

panels.add_second_y_axis()
plt4_I =\
    panels.cax.plot(t_arr, I_arr, 'darkgreen',
                    label='$I_k$ (input)', alpha=0.4)
panels.set_currenty_ylabel('$I_k$', tcolor)
panels.multi_legend(plt1_N + plt3_s2 + plt4_I, 'upper left')
panels.apply_common_xlims()

plt.tight_layout()
plt.subplots_adjust(hspace=0) # 0.04

plt.show()

assert panels.panel_id == panels.PANELS, str(
    panels.panel_id) + "==" + str(panels.PANELS)


# Misc notes:
#  q,qq = plt.subplot(4, 1, 2)  # TypeError: 'AxesSubplot' object is not iterable

# legend/plot label versus panel label:       plt.ylabel(..) versus  ..plot(..,label=...)

# 'xlabel' versus 'set_xlabel': for plt (current panel/plot) and axis (subpanel) respectively. # plt.xlabel('Time (Sec)')


#     Variance-stabilizing transformation !
