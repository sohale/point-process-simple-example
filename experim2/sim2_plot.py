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

# https://matplotlib.org/3.4.0/thirdpartypackages/index.html

# **********************************************************************************
# * some utility functions:
# **********************************************************************************

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

def clamp_numpyarr(narr, a, b=float('inf')):
    """ clamps (limits) a numpy array between a,b """
    rnarr = narr.copy()
    rnarr[rnarr < a] = a
    rnarr[rnarr > b] = b
    return rnarr

def visualise_analytical_relaxation(n, Delta0, t_arr, plt, get_neuron_tau):
    """ For a given neuron, based on its alpha, rho  """
    ht = clamp_numpyarr(t_arr - 1.0, 0, float('inf'))
    tau = get_neuron_tau(n, Delta0)
    expa = np.exp(- ht / tau) * n['alpha'] * \
       np.heaviside(t_arr - 1.0, 1.0)
    pl = plt.plot(t_arr, expa, 'r', label='$\exp(-t/\\tau)$',
                  alpha=0.2, linewidth=5)
    return pl

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
        self.ax2 = self.ax1.twinx()  # http://matplotlib.org/examples/api/two_scales.html
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

MSEC = 1. / 1000.

def plot_all(simargs, na, get_neuron_tau, simulation_result, DELTA0):

    (t_arr, x_arr_A, xlogpr_arr_A, lambda_arr_A, spike_times_A, quantiles01_A, fire_probability_arr_A, Nc_arr_A, I_arr_A2) = \
        simulation_result

    neuron_id = 0
    input_id = 0

    x_arr = x_arr_A[neuron_id]
    xlogpr_arr = xlogpr_arr_A[neuron_id]
    lambda_arr = lambda_arr_A[neuron_id]
    spike_times = spike_times_A[neuron_id]
    quantiles01 = quantiles01_A[neuron_id]
    fire_probability_arr = fire_probability_arr_A[neuron_id]
    Nc_arr = Nc_arr_A[neuron_id]
    I_arr = I_arr_A2[input_id]

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

    pl2 = visualise_analytical_relaxation(na[0], DELTA0, t_arr, plt, get_neuron_tau)


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

if __name__ == "__main__":
    pass
