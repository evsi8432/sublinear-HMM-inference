# sublinear-HMM-inference

This repository contains all of the code and data from "Variance-Reduced Stochastic Optimization for Efficient Inference of Hidden Markov Models" by Sidrow et al. (2024). All code was run using Python 3.9.12.

## simulation study

In order to run the simualtion study, navigate to the `src` directory and run the command `python sim_study.py x y`, where `x` is the number hours for the code to run and `y` is an integer between 0 and 399. `y` corresponds to an experiment index that varies the following:

- the length of the time series ($T \in \{10^3,10^5\}$)
- the number of hidden states of the HMM ($N \in \{3,6\}$)
- the dimension of the observations ($d \in \{3,6\}$)
- the parameter initializations ($\phi_0$, 5 total)
- the observation dataset (5 total)
- the optimization algorithm (All baselines, `SVRG` vs `SAGA` in the M step, whether to do a partial E step, and either $T$ or $10T$ M step iterations)

In the paper, `x` was set to 12 and the code was run on all values of `y` from 0 to 399.

## case study

In order to run the case study, navigate to the `src` directory and run the command `python case_study.py x y`, where `x` is the number hours for the code to run and `y` is an integer between 0 and 499. In the paper, `x` was set to 12 and the code was run on all values of `y` from 0 to 499. `y` corresponds to an experiment index that varies the following:

- the parameter initializations ($\phi_0$, 50 total)
- the optimization algorithm (All baselines, `SVRG` vs `SAGA` in the M step, whether to do a partial E step, and either $T$ or $10T$ M step iterations)

In the paper, `x` was set to 12 and the code was run on all values of `y` from 0 to 499.

## plotting results

Running the simulation and case studies will create a two new subdirectories: `params/case_study` and `params/sim_study`, both of which contain pkl files from their respective experiments.

In order to plot the results, run any of the .ipynb files after running the simualtion or case studies:

- `sim-study-optimization-results.ipynb` will plot the optimizaiton results from the simulation study
- `case-study-optimization-results.ipynb` will plot the optimizaiton results from the case study
- `case-study-model-results.ipynb` will plot the parameter estimates from the case study
