import dataclasses
import math
import numpy as np
from scipy.interpolate import interp1d

# from typing import ...
from operator import xor


"""
    Simulates the example 1 of [1].
    [1]. Smith and Brown 2003. "Estimating a State-Space Model from Point Process Observations".
"""

# -------------------------------
# * epxeriment
#    * simulation
#       * simulation inputs
#       * simulation results
# -------------------------------



# simargs1: SimulatorArgs1
#         ( .Delta )
# simulation_result

# na
# get_neuron_tau

# DELTA0  ??



MSEC = 1. / 1000.


# **********************************************************************************
# *                                  neuron model
# **********************************************************************************


"""
A neurons is characterised by equations 2.2 and 2.6, as in example 1 of [1].
Ref. equations #Eq.1 and #Eq.2
"""


class Neuron_static:
    def report_neuron(n, Delta):
        tau = Neuron_static.get_neuron_tau(n, Delta)
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

    def n0_prot():
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
        return n0

        # see describe_model_latex()


        descriptions = {
            'rho': ["", ""],
            'alpha': ["input gain", "volts per amps"],
            'sigma_eps': ["noisiness: noise amplitude", ""],

            'mu': ["", ""],
            'beta': ["", ""]
        }

# Part of the problem specs, but not the Delta used in the simulation.
DELTA0 = 1.0 * MSEC

# **********************************************************************************
# *                                  simulation
# **********************************************************************************

#simargs1  # simulation args
#simargs1.K
# simulator.K

class SimulatorArgs1(object):
    """
       `.T` Simulation duration (Time length) (all time units in seconds)
       `.K` length (bins/timesteps/intervals): int
       `.Delta`: (seconds)

       invariants:
            duration ~= K * Delta

    """
    # old incorrect comment: Simulation Time Length (Intervals)


    Delta: float
    K: int
    T: float

    def __init__(self, _K=None, duration=None, _deltaT=None):
        """
            Either based on `_K` or `duration`
            They specify the duration of simulation.
        """
        #self.Delta = 0.000
        #self.K = -1
        #self.T = 0.0000
        assert _deltaT is not None, '_deltaT: time-step (bin) size in seconds'

        if _K is not None:
            assert duration is None
            self.K = _K
            #self.Delta = self.T / float(self.K)
            #self.Delta = 1 * MSEC
            self.Delta = _deltaT
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
            assert _K is None
            self.T = duration
            #self.Delta = 1 * MSEC * 0.2
            self.Delta = _deltaT
            self.K = int(self.T / self.Delta + 1 - 0.00001)

        else:
            assert False, "Either `_K` or `duration` needs be specified"

        print("Simulation time-steps: K=%g" % self.K)

    def invar(self):
        # simargs1.K = 3000
        # simargs1.T = 3.0; simargs1.Delta =  # sec
        # simargs1.K = int(simargs1.T / simargs1.Delta + 1)

        assert simargs1.K
        assert simargs1.Delta
        assert simargs1.T

    # produces each timestep? no longer.
    # simulate_input()
    # provides: 1. basic simulatin args (part 1), instantiates simargs1
    # also user-interface for that. Conventient providing of three items: K/dt/T
    # Could be a factory method as part of SimulatorArgs1!
    def simargs_factory(_K=None, duration=None, deltaT=None):

        assert xor(_K is None, duration is None), \
            """ Simulation duration is either based on `_K` or `duration`.
                duration ~= K * Delta
            """
        # todo: remove global
        global simargs1
        # simargs1.T = Simulation Time Length (Sec)
        assert deltaT is not None, 'deltaT: time-step (bin) size in seconds'
        simargs1 = SimulatorArgs1(_K=_K, duration=duration, _deltaT=deltaT)
        return simargs1


# old idea, occluded by the idea of `simulate_step()`:
# ... = simulate_input()

class InputDriver_static:
    # produces each timestep
    # the idea was it actually provided the INPUT signal! (I_k)
    # input also drives the program flow !
    def simulate_input_and_drive_next_step(simargs1):
        last_every_second = -float('inf')

        for k in range(simargs1.K):
            t = k * simargs1.Delta
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
            # yield k, t, I_k, simargs1
            yield k, t, [I_k]


# comprised of multiple neurons
class FullModel:

    def __init__(self) -> None:
        """
        Generates an instance
        """

        BETA_RANGE = [0.9, 1.1]
        NEURONS = 20

        na = []
        for i in range(NEURONS):
            n = Neuron_static.n0_prot().copy()
            d = BETA_RANGE[1] - BETA_RANGE[0]
            n['beta'] = 1.1  # (np.random.rand() * d) + BETA_RANGE[0]
            na.append(n)

        # print( "Beta: ", end = '')
        #for n in na:eps_k
        #    print( n['beta'], end = '')
        # print()

        self.na = na


# local loop-updating variables
# last_x_k = 0.0
# Nc = 0


# `deltaT` was:
#     1 * MSEC * 0.2 (when duration is specified)
#     1 * MSEC (when K is specified)
simargs1 = SimulatorArgs1.simargs_factory(duration=3.0, deltaT=1 * MSEC * 0.2)

simargs1.invar()

full_model = FullModel()

# Slow, hence, cache!
# neuron instance
class Neuron:
    def init_slow_cache(self, n_obj):
        _tau = Neuron_static.get_neuron_tau(n_obj, DELTA0)
        _rho_corrected = Neuron_static.get_neuron_rho(_tau, simargs1.Delta)
        _sigma_eps_corrected = full_model.na[0]['sigma_eps'] * \
            math.sqrt(simargs1.Delta/DELTA0)
        print("_rho_corrected = ", _rho_corrected, "rho=", full_model.na[0]['rho'])
        print("_sigma_eps_corrected = ", _sigma_eps_corrected,
                "sigma_eps=", full_model.na[0]['sigma_eps'])

        self._sigma_eps_corrected = _sigma_eps_corrected
        self._rho_corrected = _rho_corrected


# [neuron_id]
NEURONS_NUM = 1
if True:
    t_arr = np.full((simargs1.K,), np.nan)

    x_arr_A = np.full((NEURONS_NUM, simargs1.K,), np.nan)
    xlogpr_arr_A = np.full((NEURONS_NUM, simargs1.K,), np.nan)
    Nc_arr_A = np.full((NEURONS_NUM, simargs1.K,), 99999, dtype=int)
    fire_probability_arr_A = np.full((NEURONS_NUM, simargs1.K,), np.nan)
    lambda_arr_A = np.full((NEURONS_NUM, simargs1.K,), np.nan)
    I_arr_A2 = np.full((NEURONS_NUM, simargs1.K,), np.nan)

    # output.
    # Non-square. Hence, list of nparrays
    spike_times_Al = [None] * NEURONS_NUM
    # todo: rename
    quantiles01_Al = [None] * NEURONS_NUM

    # local loop-updating variable(s)
    Nc_A = np.zeros((NEURONS_NUM,))
    last_x_k_A = np.zeros((NEURONS_NUM,))
    # last_x_k = 0.0
    # Nc = 0

    # for neuron_id in range(1):
    neuron_id = 0
    neur_instance = [Neuron()]
    neur_instance[neuron_id].init_slow_cache(full_model.na[neuron_id])

for k, t, I_k_A2 in InputDriver_static.simulate_input_and_drive_next_step(simargs1):

    # print( t, k, I_k_A2 )

    neuron_id = 0
    n = full_model.na[neuron_id]

    is_first_step = (k == 0)

    # def STEP(..., is_first_step)


    # *************************
    # *  Neuron model
    # *************************
    if False:
        # x_k is the State
        eps_k = n['sigma_eps'] * np.random.randn()

        # dirac_factor = 7.0  # terrible. Why no refactory period?
        dirac_factor = 1.0

        #dirac_factor = 1.0 / simargs1.Delta
        # print( "dirac_factor,",dirac_factor )
        x_k = n['rho'] * last_x_k_A[neuron_id] + n['alpha'] * \
            I_k_A2[inp_id] * dirac_factor + eps_k  # Eq.1

    inp_id = 0
    if True:
        dirac_factor = 1.0
        eps_k = neur_instance[neuron_id]._sigma_eps_corrected * np.random.randn()
        x_k = neur_instance[neuron_id]._rho_corrected * last_x_k_A[neuron_id] + \
            n['alpha'] * I_k_A2[inp_id] * dirac_factor + eps_k  # Eq.1

    # last_x_k_A should be outside a loop function
    # last_x_k = x_k
    last_x_k_A[neuron_id] = x_k

    xlp = n['mu'] + n['beta'] * x_k
    lambda_k = math.exp(xlp)  # Eq.2
    # What will guarantee that xlp < 0 ? i.e. probability < 1
    # Where is x reset?

    # *****************************
    # * Point process simulation
    # *****************************

    fire_probability = lambda_k * simargs1.Delta  # * 100
    fire = fire_probability > np.random.rand()

    #output = (x_k * simargs1.Delta) > np.random.rand()
    #Nc += output

    # total count
    Nc_A[neuron_id] += fire

    # set all (x_k, Nc, xlp),
    # not: ?( t, I_k)
    t_arr[k] = t

    x_arr_A[neuron_id][k] = x_k
    Nc_arr_A[neuron_id][k] = Nc_A[neuron_id]
    I_arr_A2[inp_id][k] = I_k_A2[inp_id]

    xlogpr_arr_A[neuron_id][k] = xlp
    del xlp

    fire_probability_arr_A[neuron_id][k] = fire_probability
    lambda_arr_A[neuron_id][k] = lambda_k

    if is_first_step:
        Neuron_static.report_neuron(n, simargs1.Delta)

    # del x_arr,     xlogpr_arr,    Nc_arr,    fire_probability_arr,    lambda_arr,    I_arr,
    del lambda_k, fire_probability, t, x_k, fire

print("Simulation time = T =", simargs1.T, ". Mean rate = ",
      Nc_arr_A[:][-1].astype(float)/simargs1.T, "(spikes/sec)")

for neuron_id in range(1):
    print("Integral of lambda = ", np.sum(lambda_arr_A[neuron_id]) * simargs1.Delta)
    print("Mean lambda = ", np.sum(lambda_arr_A[neuron_id]) * simargs1.Delta / simargs1.T)


def cumsum0(x, cutlast=True):
    """ generates a cumsum starting with 0.0, of the same size as x, i.e. removes the last element, and returning it separately. """
    c = np.cumsum(x)
    maxval = c[-1]
    c = np.concatenate((np.array([0.0]), c))
    if cutlast:
        c = c[:-1]

    # return c, maxval
    return c


def generate_isi_samples_unit_exp1(total_rate):
    """
    Generates samples from
    Exponential distribution
    λ = 1.0

    PDF(x) = λ exp(-λx)

    where x = ISI in temrs of "virtual-time" (Λ)
    (Not really ISI: but ISI-Λ )

    Enough number of samples to fit the whole `total_rate` (Λ)

    `total_rate` units are in "virtual-time" (rate, λ, intensity-integrated, capital Lambda: Λ )
    ISIΛ: Inter-spike inter-val -> inter-Λ-val
    interval implies physical "time". But this is virtual-time (Λ)
    """
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
#cumintegr_arr, maxΛ = cumsum0(lambda_arr_A[neuron_id], cutlast=False)*simargs1.Delta
#t_arr_aug = np.concatenate(t_arr, np.array([t_arr[-1]+simargs1.Delta]))
cumintegr_arr = cumsum0(lambda_arr_A[neuron_id], cutlast=True)*simargs1.Delta
maxΛ = cumintegr_arr[-1]
assert cumintegr_arr[-1] == np.max(cumintegr_arr), "monotonically increaseing"

# Time-Rescaling: Quantile ~ (physical) time (of spikes)
# todo: rename time_quantiles
# time_quantiles is ...
interp_func = interp1d(cumintegr_arr, t_arr, kind='linear')
# Time-rescaled quantiles:
#time_quantiles = np.arange(0,maxΛ,maxΛ/10.0 * 10000)
time_quantiles = generate_isi_samples_unit_exp1(maxΛ)
# print( time_quantiles )

assert time_quantiles.shape[0] == 0 or np.max(
    cumintegr_arr) >= np.max(time_quantiles)
assert time_quantiles.shape[0] == 0 or np.min(
    cumintegr_arr) <= np.min(time_quantiles)
if time_quantiles.shape == (0,):
    print("Warning: empty spike train. *****")

spike_times = interp_func(time_quantiles)

# based on stackoverflow.com/questions/19956388/scipy-interp1d-and-matlab-interp1
# spikes = (spike_times, time_quantiles)  # spikes and their accumulated lambda

spike_times_Al[neuron_id] = spike_times
quantiles01_Al[neuron_id] = time_quantiles
del spike_times, time_quantiles, maxΛ, cumintegr_arr

simulation_result = \
    (t_arr, x_arr_A, xlogpr_arr_A, lambda_arr_A, spike_times_Al, quantiles01_Al, fire_probability_arr_A, Nc_arr_A, I_arr_A2)

# import sys
# sys.path.append('/ufs/guido/lib/python')
from sim2_plot import *

plot_all(simargs1, full_model.na, Neuron_static.get_neuron_tau, simulation_result, DELTA0)
