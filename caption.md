

### Screenshot
Statistical modelling of Spike Trains as Point Processes
![Spike Trains as Point Processes](https://repository-images.githubusercontent.com/80567584/04691a80-5aa6-11eb-863e-9f2bab06be1b "Statisticall modelling of Spike Trains as Point Processes")
Note Λ(t) = ∫ λ(t) dt is the times-rescaling function. x[k] is the input (i.e. the "state"). In physiological terms, the input signal is the EPSP/IPSP.

As spike rate estimator: The derived statistical model (with parameters) can also be seen as a rigorous method of calculating the "spiking rates" based on observed spike trains. The estimated moodel generates λ(t), which is an estimation of the probability of spikes (also can be seen as spiking rate). It's a quantity that is difficult to estimate, or would be incorrect if estimated in the naïve way simply by averaging (over bins) or smoothing of spike trains.

The estimations are based on EM (Expectation Maximisation) (the procedure devised by Ghahramani & Hinton 1996) apllied to Point Processes.

* `x(t+δt) = A x(t) + αI + ε`
* `λ = exp(βx + μ)`

### Model Diagram (in progress)
It is comprised of a State Space model ( _x_ = state) and a Point Process ( _λ_ = CIF )
```txt
┌───┐   ┌───┐    ┌──────────────────────────────..─┐
│ x │ → │ λ │ →  │ │ │║ │║│  │ │    │║║     │   .. │
└───┘   └───┘    └──────────────────────────────..─┘ 
```


A State-State model is a Dynamical System, i.e. a system with a feelback loop.
```txt
┌───┐      ┌───┐     ┌──────────────────────────────··─┐
│ x │ ───→ │ λ │───→ │ │ │║ │║│  │ │    │║║     │   ·· │
└╥──┘══>╗  └───┘     └──────────────────────────────··─┘ 
 ║      ║*A
 ║      ╟─<─ α * I ←─ I
 ╚═+══<═╝
   +ε
```

_I_ is the input and ε is additive noise, both are injected into the loop.
```txt
         ┌───┐          ┌───┐     ┌──────────────────────────────···
         │ x │ ───────→ │ λ │───→ │ │ │║ │║│  │ │    │║║     │   ···
         └╥──┘══>╗      └───┘     └──────────────────────────────···
          ║      ║
 I ─→*α──→╢      ╟*A
          ╚═+══<═╝
            +ε
```


LaTex:
* $x_{t+δt} = A^{δt} \ x_t + αI + ε$
* $λ = \exp(βx + μ)$

Coloured: $\color{red}{\lambda}$ Characterises a Point Process.
```
`
```

│         ┌───┐          ┌───┐     ┌──────────────────────────────..
│         │ x │ ───────→ │ λ │───→ │ <span style="color: red; opacity: 0.80;">│ │║ │║│  │ │    │║║     │   </span>.. 
│         └╥──┘══>╗      └───┘     └──────────────────────────────..
│          ║      ║
│ I ─→*α──→╢      ╟*A
│          ╚═+══<═╝
│            +ε
│

Misc:
* $\log λ = βx + μ$
* λ ∝ $\Pr$(spike) / dt
.

In fact,

*  Pr(?) ∝ ∫ λ dt
*  $\Pr(?) ∝ \int λ dt $

.
