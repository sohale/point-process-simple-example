# Point Process example

A simple straightforward example of a Point Process + State Space with manually set model parameters.

Old name: `smithbrown2003-example1`

This implements the forward model (not the estimatiom part) of the following.
For the reverse (parameter estimation), a separate repo is used (contact author).

## Smith-Brown's example 1
A quick implementation of: **Smith & Brown, 2003** (See [1] below)

This implements the Example 1, that is the case of "Multiple Neurons Driven by a Common Latent Process", from [1].

My code implements an EM Algorithm for decoding based on Point-process observations (Spike train observation, Inter-spike interval observations) using Python 3.

Parameter Estimation of the Kalman filter is implemented using Em Algorithm.

Assessment of the results includes ML and use of Time-Rescaling Theorem.

## Reference:
[1]
Anne C **Smith**, Emery N **Brown**.
"*Estimating a State-Space Model from Point Process Observations*".
Neural Computation.
Volume 15, Issue 5, May **2003**.
pp 965-991.

Details: Link: [doi: 10.1162/089976603765202622](https://doi.org/10.1162/089976603765202622),
[pubmed](https://pubmed.ncbi.nlm.nih.gov/12803953/) Download links:
[dl1](https://www.cmu.edu/dickson-prize/images/ENBrown_Dickson_Prize_Publications_12_06_18.pdf)
[dl2](http://annecsmith.net/images/State_Space_2003.pdf)


### Keywords:
My code has implementations the following concepts:

EM Algorithm, Kalman filter, Estimation, Maximum Likelihood, Point Processes, Time-Rescaling Theorem.

### Screenshot 1
Statistical modelling of Spike Trains as Point Processes
![Spike Trains as Point Processes](https://repository-images.githubusercontent.com/80567584/04691a80-5aa6-11eb-863e-9f2bab06be1b "Statisticall modelling of Spike Trains as Point Processes")
Note Λ(t) = ∫ λ(t) dt is the times-rescaling function. x[k] is the input (i.e. the "state"). In physiological terms, the input signal is the EPSP/IPSP.

As spike rate estimator: The derived statistical model (with parameters) can also be seen as a rigorous method of calculating the "spiking rates" based on observed spike trains. The estimated moodel generates λ(t), which is an estimation of the probability of spikes (also can be seen as spiking rate). It's a quantity that is difficult to estimate, or would be incorrect if estimated in the naïve way simply by averaging (over bins) or smoothing of spike trains.

The estimations are based on EM (Expectation Maximisation) (the procedure devised by Ghahramani & Hinton 1996) apllied to Point Processes.

* `x(t+δt) = A x(t) + αI + ε`
* `λ = exp(βx + μ)`
 
### Screenshot 2
Multiple trials, better formulas
![Spike Trains as Point Processes](https://github.com/sohale/point-process-simple-example/releases/download/v0.4.0/my-eps3b-1-resz.png "Statisticall modelling of Spike Trains as Point Processes")
