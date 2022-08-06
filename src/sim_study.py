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

experiment = int(sys.argv[1])
rand_seed = int(sys.argv[2])

random.seed(rand_seed)
np.random.seed(rand_seed)

# select parameters for optimization
num_epochs = 100
tol = 1e-8
grad_tol = 1e-16

# optimization parameters
methods = ["SAGA","BFGS"]
# methods = ["EM","SGD","SAG","SVRG","SAGA","GD","BFGS","Nelder-Mead"]

# pick optimization settings
step_sizes = {"EM"  : [None,None],
              "CG"  : [None,None],
              "BFGS": [None,None],
              "GD"  : [0.01,0.01],
              "SGD" : [0.01,0.01],
              "SAG" : [0.01,0.01],
              "SVRG": [0.01,0.01],
              "SAGA": [0.01,0.01]}

# Select parameters for data generating process

if experiment == 1:

    features = [{},
            {'Y'      : {'f'           : 'normal',
                         'lower_bound' : None,
                         'upper_bound' : None},
             'Y_star' : {'f'           : 'normal',
                         'lower_bound' : None,
                         'upper_bound' : None}}]

    K = [2,2]
    T = 1000

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
    T = 1000

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

### GENERATE DATA ###
X = np.zeros(T,dtype=int)
data = []

for t in range(T):

    if t == 0:
        X[t] = np.random.choice(K[0]*K[1],p=delta_full)
    else:
        X[t] = np.random.choice(K[0]*K[1],p=Gamma_full[X[t-1]])

    data.append({'Y'      : mus['Y'][X[t]] + sigs['Y'][X[t]]*np.random.normal(),
                 'Y_star' : mus['Y_star'][X[t]] + sigs['Y_star'][X[t]]*np.random.normal()})


### TRAIN HMM ###
optims = {}
times = {}

# pick initial theta, eta, and eta01
init_optim = optimizor.optimizor(data,features,K)
theta = deepcopy(init_optim.theta)
eta0 = deepcopy(init_optim.eta0)
eta = deepcopy(init_optim.eta)
del init_optim

for method in methods:

    for partial_E in [0,0.5,1]:

        if partial_E > 0 and method in ["EM","BFGS","Nelder-Mead","CG"]:
            continue

        print(method,partial_E)
        print("")

        optims[(method,partial_E)] = optimizor.optimizor(data,features,K)
        optims[(method,partial_E)].step_size = step_sizes[method]
        optims[(method,partial_E)].param_bounds[1]["Y"]["mu"] = [-100,100]
        optims[(method,partial_E)].param_bounds[1]["Y"]["log_sig"] = [-5,5]
        optims[(method,partial_E)].param_bounds[1]["Y_star"]["mu"] = [-100,100]
        optims[(method,partial_E)].param_bounds[1]["Y_star"]["log_sig"] = [-5,5]
        optims[(method,partial_E)].jump_every = jump_every

        # set parameters
        optims[(method,partial_E)].theta = deepcopy(theta)
        optims[(method,partial_E)].eta0 = deepcopy(eta0)
        optims[(method,partial_E)].eta = deepcopy(eta)


        if not (step_sizes[method][0] is None):
            optims[(method,partial_E)].L_theta = 1.0 / step_sizes[method][0]
            optims[(method,partial_E)].L_eta = 1.0 / step_sizes[method][1]

        if partial_E == 0:
            optims[(method,partial_E)].train_HHMM(num_epochs=num_epochs,
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
                optims[(method,partial_E)].train_HHMM(num_epochs=num_epochs,
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
                optims[(method,partial_E)].train_HHMM(num_epochs=num_epochs,
                                                      method=method,
                                                      max_iters=10*T,
                                                      partial_E=True,
                                                      alpha_theta=step_sizes[method][0],
                                                      alpha_eta=step_sizes[method][1],
                                                      tol=tol,
                                                      grad_tol=grad_tol,
                                                      record_like=True)



# get optimal value via SAGA:
optims["control"] = optimizor.optimizor(data,features,K)
optims["control"].step_size = step_sizes["SAGA"]
optims["control"].param_bounds[1]["Y"]["mu"] = [-100,100]
optims["control"].param_bounds[1]["Y"]["log_sig"] = [-5,5]
optims["control"].param_bounds[1]["Y_star"]["mu"] = [-100,100]
optims["control"].param_bounds[1]["Y_star"]["log_sig"] = [-5,5]

if experiment == 1:
    optims["control"].theta = [{},
                                [{'Y'      : {'mu': np.array([1.0,1.0]),
                                             'log_sig': np.array([-1.0,-1.0])},
                                  'Y_star' : {'mu': np.array([1.0,2.0]),
                                             'log_sig': np.array([-1.0,-1.0])}},
                                 {'Y'      : {'mu': np.array([2.0,2.0]),
                                             'log_sig': np.array([-1.0,-1.0])},
                                  'Y_star' : {'mu': np.array([1.0,2.0]),
                                             'log_sig': np.array([-1.0,-1.0])}}]]

    optims["control"].Gamma = [np.array([[0.99,0.01],
                                          [0.01,0.99]]),
                                [np.array([[0.75,0.25],
                                           [0.25,0.75]]),
                                 np.array([[0.25,0.75],
                                           [0.75,0.25]])]]

    optims["control"].delta = [np.array([0.5,0.5]),
                                 [np.array([0.5,0.5]),
                                  np.array([0.5,0.5])]]
    optims["control"].eta0 = delta_2_eta0(optims["control"].delta)
    optims["control"].eta = Gamma_2_eta(optims["control"].Gamma)

elif experiment == 2:
    optims["control"].theta = [{},
                                [{'Y': {'mu': np.array([1.0,2.0,3.0]),
                                        'log_sig': np.array([-1.0,-1.0,-1.0])}}]]

    optims["control"].Gamma = [np.array([[1.0]]),
                                 [np.array([[0.99,0.005,0.005],
                                            [0.005,0.99,0.005],
                                            [0.005,0.005,0.99]])]]

    optims["control"].delta = [np.array([1.0]),
                                 [np.array([0.3,0.3,0.4])]]

    optims["control"].eta0 = delta_2_eta0(optims["control"].delta)
    optims["control"].eta = Gamma_2_eta(optims["control"].Gamma)

optims["control"].train_HHMM(num_epochs=200,
                             method="SAGA",
                             max_iters=T,
                             partial_E=True,
                             alpha_theta=step_sizes[method][0],
                             alpha_eta=step_sizes[method][1],
                             tol=1e-4*tol,
                             grad_tol=1e-4*grad_tol,
                             record_like=True)

# save file
fname = "../params/experiment_%d_%d" % (experiment,rand_seed)
with open(fname, 'wb') as f:
    pickle.dump(optims, f)
