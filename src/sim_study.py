import sys
#sys.path.remove('/Users/evsi8432/Documents/Research/CarHHMM-DFT/Repository/Code')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import vonmises
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
from scipy.stats import circstd
from scipy.special import iv
from scipy.special import expit
from scipy.special import logit
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import LinearConstraint
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.linalg import block_diag

from math import isclose

from copy import deepcopy

from datetime import datetime

import importlib

import time
import pickle
import random

import optimizor

from optimizor import delta_2_eta0
from optimizor import Gamma_2_eta
from optimizor import eta0_2_delta
from optimizor import eta_2_Gamma
from optimizor import logdotexp

# parse command-line args
experiment = int(sys.argv[1])
T = int(sys.argv[2])
max_time = 3600*float(sys.argv[3])
id = int(sys.argv[4])

# set method
if id % 5 == 4:
    method = "control"
    partial_E = 0
elif id % 5 == 3:
    method = "BFGS"
    partial_E = 0
else:
    method = "SAGA"
    partial_E = float(id % 5) / 2.0

rand_seed = int(id / 5)
random.seed(rand_seed)
np.random.seed(rand_seed)

print("experiment: %d" % experiment)
print("method: %s" % method)
print("partial E_step: %.1f" % partial_E)
print("random seed: %d" % rand_seed)
print("max time : %.3f hours" % (max_time/3600))

# select parameters for optimization
num_epochs = 100
tol = 1e-8
grad_tol = 1e-16

step_sizes = {"EM"  : [None,None],
              "CG"  : [None,None],
              "BFGS": [None,None],
              "GD"  : [0.01,0.01],
              "SGD" : [0.01,0.01],
              "SAG" : [0.01,0.01],
              "SVRG": [0.01,0.01],
              "SAGA": [0.01,0.01]}

jump_every = 1

### checks on optimization parameters ###
if partial_E > 0 and method in ["EM","BFGS","Nelder-Mead","CG"]:
    raise("partial_E not consistent with method")

### features of data ###
if experiment == 1:

    features = [{},
                {'Y'      : {'f'           : 'normal',
                             'lower_bound' : None,
                             'upper_bound' : None},
                 'Y_star' : {'f'           : 'normal',
                             'lower_bound' : None,
                             'upper_bound' : None}}]

    K = [2,2]

    jump_every = 1

    Gamma = [np.array([[0.99,0.01],
                       [0.01,0.99]]),
             [np.array([[0.75,0.25],
                        [0.25,0.75]]),
              np.array([[0.25,0.75],
                        [0.75,0.25]])]]

    delta = [np.array([0.5,0.5]),
             [np.array([0.5,0.5]),
              np.array([0.5,0.5])]]


    Gamma_coarse_full = np.kron(Gamma[0],np.ones((2,2)))
    Gamma_fine_full = np.block([[Gamma[1][0],np.tile(delta[1][1],(2,1))],
                                [np.tile(delta[1][0],(2,1)),Gamma[1][1]]])

    Gamma_full = Gamma_coarse_full * Gamma_fine_full
    delta_full = np.repeat(delta[0],2) * np.concatenate(delta[1])

    mus = {'Y'      : np.array([1.0,1.0,2.0,2.0]),
           'Y_star' : np.array([1.0,2.0,1.0,2.0])}

    sigs = {'Y'      : np.exp(np.array([-1.0,-1.0,-1.0,-1.0])), # coarse-scale sigs
            'Y_star' : np.exp(np.array([-1.0,-1.0,-1.0,-1.0]))} # fine-scale sigs

elif experiment == 2:

    features = [{},
                {'Y'      : {'f'           : 'normal',
                             'lower_bound' : None,
                             'upper_bound' : None}}]

    K = [1,3]

    jump_every = 1

    Gamma = [np.array([[1.0]]),
             [np.array([[0.99,0.005,0.005],
                        [0.005,0.99,0.005],
                        [0.005,0.005,0.99]])]]

    delta = [np.array([1.0]),
             [np.array([0.3,0.3,0.4])]]

    Gamma_full = Gamma[1][0]
    delta_full = delta[1][0]

    mus =  {'Y' : np.array([1.0,2.0,3.0])}
    sigs = {'Y' : np.exp(np.array([-1.0,-1.0,-1.0]))}

### load in data ###
data_fname = "../dat/data_Y_exp-%d_T-%d" % (experiment,T)
with open(data_fname,"rb") as f:
    data = pickle.load(f)

# pick intial parameters
optim = optimizor.optimizor(data,features,K)
optim.param_bounds[1]["Y"]["mu"] = [-100,100]
optim.param_bounds[1]["Y"]["log_sig"] = [-5,5]
if experiment == 1:
    optim.param_bounds[1]["Y_star"]["mu"] = [-100,100]
    optim.param_bounds[1]["Y_star"]["log_sig"] = [-5,5]

if method == "control":
    optim.step_size = step_sizes["SAGA"]
    if not (step_sizes["SAGA"][0] is None):
        optim.L_theta = 1.0 / step_sizes["SAGA"][0]
        optim.L_eta = 1.0 / step_sizes["SAGA"][1]
else:
    optim.step_size = step_sizes[method]
    if not (step_sizes[method][0] is None):
        optim.L_theta = 1.0 / step_sizes[method][0]
        optim.L_eta = 1.0 / step_sizes[method][1]

# print initial parameters
print("initial theta:")
print(optim.theta)
print("")
print("initial eta0:")
print(optim.eta0)
print("")
print("initial eta:")
print(optim.eta)
print("")

# get optimal value via SAGA:
if method == "control":
    if experiment == 1:
        optim.theta = [{},
                        [{'Y'      : {'mu': np.array([1.0,1.0]),
                                     'log_sig': np.array([-1.0,-1.0])},
                          'Y_star' : {'mu': np.array([1.0,2.0]),
                                     'log_sig': np.array([-1.0,-1.0])}},
                         {'Y'      : {'mu': np.array([2.0,2.0]),
                                     'log_sig': np.array([-1.0,-1.0])},
                          'Y_star' : {'mu': np.array([1.0,2.0]),
                                     'log_sig': np.array([-1.0,-1.0])}}]]

        optim.Gamma = [np.array([[0.99,0.01],
                                  [0.01,0.99]]),
                        [np.array([[0.75,0.25],
                                   [0.25,0.75]]),
                         np.array([[0.25,0.75],
                                   [0.75,0.25]])]]

        optim.delta = [np.array([0.5,0.5]),
                         [np.array([0.5,0.5]),
                          np.array([0.5,0.5])]]

        optim.eta0 = delta_2_eta0(optim.delta)
        optim.eta = Gamma_2_eta(optim.Gamma)

    elif experiment == 2:
        optim.theta = [{},
                        [{'Y': {'mu': np.array([1.0,2.0,3.0]),
                                'log_sig': np.array([-1.0,-1.0,-1.0])}}]]

        optim.Gamma = [np.array([[1.0]]),
                                     [np.array([[0.99,0.005,0.005],
                                                [0.005,0.99,0.005],
                                                [0.005,0.005,0.99]])]]

        optim.delta = [np.array([1.0]),
                                     [np.array([0.3,0.3,0.4])]]

        optim.eta0 = delta_2_eta0(optim.delta)
        optim.eta = Gamma_2_eta(optim.Gamma)

    optim.train_HHMM(num_epochs=200,
                     max_time=max_time,
                     method="SAGA",
                     max_iters=T,
                     partial_E=True,
                     alpha_theta=step_sizes["SAGA"][0],
                     alpha_eta=step_sizes["SAGA"][1],
                     tol=1e-4*tol,
                     grad_tol=1e-4*grad_tol,
                     record_like=True)

elif partial_E == 0:
    optim.train_HHMM(num_epochs=num_epochs,
                      max_time=max_time,
                      method=method,
                      max_iters=T,
                      partial_E=False,
                      alpha_theta=step_sizes[method][0],
                      alpha_eta=step_sizes[method][1],
                      tol=tol,
                      grad_tol=grad_tol,
                      record_like=True)

elif partial_E == 0.5:
    if method in ["SGD","SAG","SVRG","SAGA"]:
        optim.train_HHMM(num_epochs=num_epochs,
                          max_time=max_time,
                          method=method,
                          max_iters=T,
                          partial_E=True,
                          alpha_theta=step_sizes[method][0],
                          alpha_eta=step_sizes[method][1],
                          tol=tol,
                          grad_tol=grad_tol,
                          record_like=True)

elif partial_E == 1:
    if method in ["SGD","SAG","SVRG","SAGA"]:
        optim.train_HHMM(num_epochs=num_epochs,
                          max_time=max_time,
                          method=method,
                          max_iters=10*T,
                          partial_E=True,
                          alpha_theta=step_sizes[method][0],
                          alpha_eta=step_sizes[method][1],
                          tol=tol,
                          grad_tol=grad_tol,
                          record_like=True)


# save file
optim.data = data_fname
fname = "../params/experiment_%d_%d_%s_%.1f_%d" % (experiment,T,method,partial_E,rand_seed)
with open(fname, 'wb') as f:
    pickle.dump(optim, f)
