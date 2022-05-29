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
    # the idea was it actually provided the INPUT signal! (Iₖ)
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
                Iₖ = 1.0
            else:
                Iₖ = 0.0

            # What is this? Is it Dirac? Then why not multiplied by 1/Delta?

            last_every_second = every_second
            # yield k, t, Iₖ, simargs1
            yield k, t, [Iₖ]


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
        #for n in na:epsₖ
        #    print( n['beta'], end = '')
        # print()

        self.na = na


# local loop-updating variables
# last_xₖ = 0.0
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
    tΞ = np.full((simargs1.K,), np.nan)

    x_ΞΞ = np.full((NEURONS_NUM, simargs1.K,), np.nan)
    xlogpr_ΞΞ = np.full((NEURONS_NUM, simargs1.K,), np.nan)
    Nᶜ_ΞΞ = np.full((NEURONS_NUM, simargs1.K,), 99999, dtype=int)
    fire_probabilityΞΞ = np.full((NEURONS_NUM, simargs1.K,), np.nan)
    λ_ΞΞ = np.full((NEURONS_NUM, simargs1.K,), np.nan, dtype=float)
    Iₖ_ΞΞ = np.full((NEURONS_NUM, simargs1.K,), np.nan)

    # output.
    # Non-square. Hence, list of nparrays
    ϟ_timesϟ_Ξ = [None] * NEURONS_NUM
    # todo: rename
    Λ_at_spikes_Ξ = [None] * NEURONS_NUM

    # local loop-updating variable(s)
    Nᶜ_ξ = np.zeros((NEURONS_NUM,))
    last_xₖ_ξ = np.zeros((NEURONS_NUM,))
    # last_xₖ = 0.0
    # Nc = 0

    # for neuron_id in range(1):
    neuron_id = 0
    neur_instance = [Neuron()]
    neur_instance[neuron_id].init_slow_cache(full_model.na[neuron_id])

for k, t, Iₖ_Ξ in InputDriver_static.simulate_input_and_drive_next_step(simargs1):

    # print( t, k, Iₖ_Ξ )

    neuron_id = 0
    n = full_model.na[neuron_id]

    is_first_step = (k == 0)

    # def STEP(..., is_first_step)


    # *************************
    # *  Neuron model
    # *************************
    if False:
        # xₖ is the State
        epsₖ = n['sigma_eps'] * np.random.randn()

        # dirac_factor = 7.0  # terrible. Why no refactory period?
        dirac_factor = 1.0

        #dirac_factor = 1.0 / simargs1.Delta
        # print( "dirac_factor,",dirac_factor )
        xₖ = n['rho'] * last_xₖ_ξ[neuron_id] + n['alpha'] * \
            Iₖ_Ξ[inp_id] * dirac_factor + epsₖ  # Eq.1

    inp_id = 0
    if True:
        dirac_factor = 1.0
        epsₖ = neur_instance[neuron_id]._sigma_eps_corrected * np.random.randn()
        xₖ = neur_instance[neuron_id]._rho_corrected * last_xₖ_ξ[neuron_id] + \
            n['alpha'] * Iₖ_Ξ[inp_id] * dirac_factor + epsₖ  # Eq.1

    # last_xₖ_ξ should be outside a loop function
    # last_xₖ = xₖ
    last_xₖ_ξ[neuron_id] = xₖ

    # xlp is x at ?
    xlp = n['mu'] + n['beta'] * xₖ
    λₖ = math.exp(xlp)  # Eq.2
    # What will guarantee that xlp < 0 ? i.e. probability < 1
    # Where is x reset?

    # *****************************
    # * Point process simulation
    # *****************************

    fire_probability = λₖ * simargs1.Delta  # * 100
    fire = fire_probability > np.random.rand()

    #output = (xₖ * simargs1.Delta) > np.random.rand()
    #Nc += output

    # total count
    Nᶜ_ξ[neuron_id] += fire

    # set all (xₖ, Nc, xlp),
    # not: ?( t, Iₖ)
    # t_arr:
    tΞ[k] = t

    x_ΞΞ[neuron_id][k] = xₖ
    Nᶜ_ΞΞ[neuron_id][k] = Nᶜ_ξ[neuron_id]
    Iₖ_ΞΞ[inp_id][k] = Iₖ_Ξ[inp_id]
    del xₖ, Iₖ_Ξ

    xlogpr_ΞΞ[neuron_id][k] = xlp
    del xlp

    fire_probabilityΞΞ[neuron_id][k] = fire_probability
    λ_ΞΞ[neuron_id][k] = λₖ

    if is_first_step:
        Neuron_static.report_neuron(n, simargs1.Delta)

    # del x_arr,     xlogpr_arr,    Nc_arr,    fire_probability_arr,    λ_arr,    I_arr,
    del λₖ, fire_probability, t, fire

print("Simulation time = T =", simargs1.T, ". Mean rate = ",
      Nᶜ_ΞΞ[:][-1].astype(float)/simargs1.T, "(spikes/sec)")

for neuron_id in range(1):
    print("Integral of λ = ", np.sum(λ_ΞΞ[neuron_id]) * simargs1.Delta)
    print("Mean λ = ", np.sum(λ_ΞΞ[neuron_id]) * simargs1.Delta / simargs1.T)


def cumsum0(x, cutlast=True):
    """
        Numerical integration.
        generates a cumsum that starts with 0.0,
        of the same size as x
        To maintain the size, it removes the last element,
        and returns it separately.
        """
    c = np.cumsum(x)
    maxval = c[-1]
    c = np.concatenate((np.array([0.0]), c))
    if cutlast:
        c = c[:-1]

    return c, maxval


def generate_Λ_samples_unit_exp1(total_rate):
    """
    Generates samples from
    Exponential distribution
    λ = 1.0

    PDF(x) = λ exp(-λx)

    Name evolution: Old names:
        generate_unit_isi
        generate_isi_samples_unit_exp1
        generate_Λ_samples_unit_exp1

    where x = ISI in temrs of "virtual-time" (Λ)
    (Not really ISI: but ISI-Λ )

    Enough number of samples to fit the whole `total_rate` (Λ)

    `total_rate` units are in "virtual-time" (rate, λ, intensity-integrated, capital Lambda: Λ )
    ISIΛ: Inter-spike inter-val -> inter-Λ-val
    interval implies physical "time". But this is virtual-time (Λ)

    True-time (internal, canonical, natural!)

    ISI = TR(R)

    time_quantiles01 -> Λ_quantiles
    By q01 I meant the ISI in terms of this.
    Max of sum(q01) is Sum(λ) = (disrete)Sum(Λ) = Λ
    Evolution:
        Sum(λ) = Λ
        (disrete)Sum(λ) = Λ
        Sum(λ) = (disrete)Sum(Λ) = Λ
        ∫λ = (disrete)Sum(ΔΛ) = Λ
        ∫λ = Σ(ΔΛ) = Λ
        ∫λ dt = Σ(ΔΛ) = Λ

    But here:
        Λ = Σ (isi?)
        Λ = Σ(ΔΛ)

        isi? = ΔΛ
    On the other hands, the units of λ are 1/time?

    No, in fact, the time is added to it by the integration:
        ∫λ dt  is unit-less
        Hence, λ is 1/t or better to say: 1/Δt?
    I am adding back:
        1. physical Dim
        2. Δ or whole (Σ). ΣΔ = whole. ΣΔ = id
            Δ = Σ^{-1} * id ?!
            Δ = 1/Σ * id
            Δ = id/Σ (but in opposite order)

        3. Adding back the natures other than time. e.g. Λ-ness.

        4. Maybe: DIM of `rand()`. (?)

        5. log-ness as DIM. log-ness is important.
            log-transforms changre DIM. (BEcause they are in sa differnt space!)

            Σ,Δ are kind of transformations:    the results are relative/absolute (to reference points).
            jens (material) is different.
            This is a very fundamental quality.

            Σ ( seq1 ) = cusum
            Δ (seq2) = diff
            Σ(Δ) is obviously are transfomrations.

        6. Nature of space (of transformation)
            eg rand is in its own "space".


    The `rand()` itself has some DIM (?)

    The idea is to keep track of the dimentionality, so that you dont need to remember what transformations are done inside.
    So the output "type" is about interface: to avoid the inside.
    But to give sufficient info about inside.

    The language of input/output needs "type".

    So what is the type of output here?

    Σ
    (
        desribe type in terms of the transfomrations. a "composition" of them.
        A composition is an architecture. I the shape of the circuit.
        DIMension is about the general shape of ciruit - in that sense.
        A homeomorphism of the circuit itself?
        The black box has in its outputs a homemorphism of inside?
        (But hiding certain aspects of inside. Not all)
    )
    Σ(x)
    Σ(Δ)
    Σ(x = Δ(.)) =  // not to be confuse with a function x=>
    Σ(x = (Δ=(∫λ dt))) =
    Σ(Δ=(∫λ dt))
    // ^ This also confirms that what is inside the Σ is a Δ. Hence, we end of having a Σ(Δ) or ΣΔ.
    Σ(Δ: ∫λ dt)
    Σ(Δ: ∫λ dt st. input = Δ)
    By input I mean rand !
    But "Σ(Δ: ∫λ dt)" is the charactger ofd the type.
        It is a whole picture. And you mark the inpt and output there.
    "output" Σ("input" Δ: ∫λ dt)
    Then attach: (Occam: IO language: a? a!)
    "input"! = `rand()`   # attach
    Occam language:
    "output!" Σ("input?" Δ: ∫λ dt)
    In fact !? already sa it is output or input. ut we still use "" (verbal) labels.
    "output!" Σ("input?" Δ: ∫λ dt)

    "output"! Σ("input"? Δ: ∫λ dt)  "input"! `rand`
    "output"! Σ("input"? Δ: ∫λ dt)  "input"!: `rand`
    *  "input"!: `rand` <---- simlar pattern to: (Δ: ...)
    *  `rand` "input"!  <--- input outputs the previous expression
    label, after or before?
    "After-" notation:
        Σ("input"? Δ: ∫λ dt) "output"! ;  `rand` "input"!
        Σ("input"? Δ: ∫λ dt) -> "output"! ;  `rand` -> "input"!

    "After-" notation: for bother direcitons of input/output?
        Σ("input"? <- Δ: ∫λ dt) -> "output"! ;  `rand` -> "input"!
        Σ("input"? -> Δ: ∫λ dt) -> "output"! ;  `rand` -> "input"!

        (Σ( Δ: ∫λ dt) -> "output"! , "input"? ->Δ) ;  `rand` -> "input"!

    Full picture:
        Σ( Δ: ∫λ dt)
        Σ( Δ: (∫λ dt) = "input": `rand`)
        Σ( Δ? (∫λ dt) = input? `rand`)
        Σ( Δ?! (∫λ dt) = input?! = `rand`)
        Σ( (∫λ dt)  =  Δ?! = input?! = `rand`)
        Σ( Δ?!  = (∫λ dt) = input?! = `rand`)

    How time is eliminated?
            Σ( Δ?!  = (∫ (λ?) dt) = input?! = `rand`)
    λ? ---> λ is not geivern here
    :  -----> as
    =  -----> as
    ?!  -----> as, attached immediately.

    Σ( Δ  = (∫ (λ= 1/ISI) dt) = input = `rand`)
    not sure about ISI here
    Where is ISI? In fact:
    ISI is closely related to Δ.
    ISI = time-quantile (Δ)?
    NO, we need the map of almbda.
    Yes we do have it.

    time = inv(Λ(·))

    I didn't utilise ∘ !

    Σ ∘ Δ = (∫dt) ∘ (λ(·)?),  ... = `rand` := input
    aha:
    Σ ∘ (Δ = (∫dt) ∘ (λ(·)?)),  ... = `rand` = input
    ∘ means apply.
    Σ ∘ (Δ = (∫dt) ∘ (λ?∘(·)))

    Σ ∘ (Δ := (∫dt) ∘ (λ?∘(·)))

    Now the meaning of / correct name of this function is clear
    instead of ISI, we should say, Λ
    Λ_quantiles = generate_Λ_samples_unit_exp1(maxΛ)
    was: Λ_quantiles = generate_isi_samples_unit_exp1(maxΛ)
    was: time_quantiles = generate_isi_samples_unit_exp1(maxΛ)
    was: quantiles01 = generate_unit_isi(maxv)

    simply from a mythtake:! st += isi
    Important:
        isi <-> ΔΛ

    """
    """
    st -> sΛ
    sΛ = sumΛ = Λ

    quantiles01 <-> Λ
    aks: time_quantiles01 (which is totally wrong!)

    Correspondance: Λquantiles <-> spike_timesϟ

    spike_times_Al -> ϟ_times_ξ -> ϟ_times_Ξ -> ϟ_timesξ_Ξ? -> ϟ_timesϟ_Ξ (good: ϟ indicates some kind of array suffix (redunndancy in the name that specifies the type)). Also it is not soft (list)

    Λ_at_spikes_Al -> Λϟ_ξ -> Λ_atϟ_ξ  -> Λ_at_spikes_ξ -> Λ_at_spikes_Ξ

    Λ_quantiles -> Λ_atϟ ->? Λ_at_spikes

    t_arr -> tΞ  (should I use t_Ξ instead?)
    """
    # print( "ISI(%g):"%(total_rate), end='')
    Λ = 0.0
    Λa = []
    while Λ < total_rate:
        if Λ > 0.0:
            Λa.append(Λ)
        ΔΛ = -math.log(np.random.rand())
        # print( ΔΛ, Λa, end='')
        # Λa.append(ΔΛ)
        Λ += ΔΛ
        # if Λ > total_rate:
        #    break
    # print()
    return np.array(Λa)

# λ_arr is already small. -> λ_Ξ

# Generates a Point Process

def generates_time_points(λ_Ξ, ΔT, tΞ):
    # tΞ
    #Λcumintegrλ_Ξ, maxΛ = cumsum0(λ_ΞΞ[neuron_id], cutlast=False)*simargs1.Delta
    #t_arr_aug = np.concatenate(tΞ, np.array([tΞ[-1]+simargs1.Delta]))
    #Λcumintegrλ_Ξ, _ = cumsum0(λ_ΞΞ[neuron_id], cutlast=True)*simargs1.Delta
    cumintegrλ_Ξξ2, _ignore_max = cumsum0(λ_Ξ, cutlast=True)
    Λcumintegrλ_Ξ = cumintegrλ_Ξξ2 * ΔT
    # Λcumintegrλ_Ξ = Λ(t) = Λt   Λt_arr
    # todo: find a unicode substitute for `_arr` suffix.

    maxΛ = Λcumintegrλ_Ξ[-1]
    assert Λcumintegrλ_Ξ[-1] == np.max(Λcumintegrλ_Ξ), "monotonically increaseing"

    # time_reversal_transform

    # Time-Rescaling: Quantile ~ (physical) time (of spikes)
    # todo: rename Λ_quantiles
    # Λ_quantiles is ...
    # Converts Λ -> time. time(Λ)
    time_rescaling_interp_func = interp1d(Λcumintegrλ_Ξ, tΞ, kind='linear')
    Λⁱⁿᵛ = time_rescaling_interp_func
    # (x,y, ...)  y = F(x).  tΞ = F(Λcumintegrλ_Ξ)
    # time_rescaling_interp_func: Λ -> t
    # Hence, the opposiute of Λ(t)
    # t(Λ)  tΛ
    # => `time_rescaling_interp_func` IS the Time-Rescaling transformation function
    # It is a continuous function
    # It is in fact better called Λ_rescaling
    #     But its standard Mathematical name is Λ_rescaling
    #     It is about Time-rescaling "Theorem"
    # Converts Λ -> time. time(Λ)
    # In a sense, rescaling means calculating "quantile"s. Hence the name: Λquantiles (spike_timesϟ), spike_Λs
    #    Λquantiles, spike_Λs, Λ_at_spikes, Λ_at_points, Λ_points (time_points)
    #   time_of_spikes, Λ_of_spikes
    # Λquantiles -> Λ_at_spikes
    # However, note that it is about uotput spikes

    # Time-rescaled quantiles:
    #Λ_quantiles = np.arange(0,maxΛ,maxΛ/10.0 * 10000)
    Λ_atϟ = generate_Λ_samples_unit_exp1(maxΛ)
    # print( Λ_atϟ )

    # empty_spikes, empty_spiketrain, no_spikes
    no_spikes = Λ_atϟ.shape[0] == 0
    assert no_spikes or \
        np.max(Λcumintegrλ_Ξ) >= np.max(Λ_atϟ)
    assert no_spikes or \
        np.min(Λcumintegrλ_Ξ) <= np.min(Λ_atϟ)
    if no_spikes:
        print("Warning: empty spike train. *****")

    spike_timesϟ = time_rescaling_interp_func(Λ_atϟ)
    spike_timesϟ = Λⁱⁿᵛ(Λ_atϟ)
    # why changed to this?
    #spike_timesϟ = time_rescaling_interp_func(Λcumintegrλ_Ξ)

    del maxΛ, Λcumintegrλ_Ξ
    # del spike_timesϟ, Λ_atϟ
    print( spike_timesϟ.shape , Λ_atϟ.shape )
    assert spike_timesϟ.shape == Λ_atϟ.shape
    return Λ_atϟ, spike_timesϟ


# Λ_quantiles
Λ_atϟ, spike_timesϟ = \
    generates_time_points(λ_ΞΞ[neuron_id], simargs1.Delta, tΞ)

# based on stackoverflow.com/questions/19956388/scipy-interp1d-and-matlab-interp1
# spikes = (spike_timesϟ, Λ_atϟ)  # spikes and their accumulated Λ

# ϟ_times_Ξ <- ϟ_times_ξ = spike_times_Al
ϟ_timesϟ_Ξ[neuron_id] = spike_timesϟ
# Λ_at_spikes_ξ = Λ_atϟ_ξ = Λϟ_ξ = Λ_at_spikes_Al
Λ_at_spikes_Ξ[neuron_id] = Λ_atϟ
del spike_timesϟ, Λ_atϟ

# todo: (Λ_at_spikes_ξ) Λ_at_spikes_Ξ -> Λ_atϟξ ? or Λ_atϟ_ξ ?  or Λϟ_ξ ?
assert len(ϟ_timesϟ_Ξ) == len(Λ_at_spikes_Ξ)
print( ϟ_timesϟ_Ξ[0].shape , Λ_at_spikes_Ξ[0].shape )
assert ϟ_timesϟ_Ξ[0].shape == Λ_at_spikes_Ξ[0].shape

simulation_result = \
    (tΞ, x_ΞΞ, xlogpr_ΞΞ, λ_ΞΞ, ϟ_timesϟ_Ξ, Λ_at_spikes_Ξ, fire_probabilityΞΞ, Nᶜ_ΞΞ, Iₖ_ΞΞ)

# import sys
# sys.path.append('/ufs/guido/lib/python')
from sim2_plot import *

plot_all(simargs1, full_model.na, Neuron_static.get_neuron_tau, simulation_result, DELTA0)
