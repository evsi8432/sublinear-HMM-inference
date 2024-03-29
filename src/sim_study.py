import sys
import os

import pandas as pd
import numpy as np
from numpy import array
import numdifftools as ndt
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.dates import DateFormatter

from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import truncnorm
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

from datetime import datetime
from time import gmtime, strftime

from math import isclose
from copy import deepcopy
from itertools import product
import importlib
import time
import pickle
import random

import HHMM
import optimizor
import stoch_optimizor

from helper_funcs import eta_2_log_Gamma
from helper_funcs import log_Gamma_2_eta
from helper_funcs import eta0_2_log_delta
from helper_funcs import log_delta_2_eta0
from helper_funcs import logdotexp
from helper_funcs import generate_data

'''
This script runs the simulation study of VARIANCE-REDUCED STOCHASTIC
OPTIMIZATION FOR EFFICIENT INFERENCE OF HIDDEN MARKOV MODELS
by Sidrow et al. (2023). This script was run in using the command
`python sim_study.py 12 x` for x in 0:399.
'''

# parse command-line args
max_time = 3600*float(sys.argv[1])
id = int(sys.argv[2])

method_partialEs = [("control",0.0),
                    ("BFGS",0.0),
                    ("CG",0.0),
                    ("GD",0.0),
                    ("SAGA",0.0),
                    ("SAGA",0.5),
                    ("SAGA",1.0),
                    ("SVRG",0.0),
                    ("SVRG",0.5),
                    ("SVRG",1.0)]

Ts = [1e3,1e5]
Ks = [3,6]
ds = [3,6]
rand_seed = range(5)
data_set = range(5)

# set T, K, d, random seed, method, P (partial E step), and M (steps before refresh)
for i,settings0 in enumerate(product(Ts,Ks,ds,rand_seed,data_set,method_partialEs)):
    if i == id:
        settings = settings0
        break

T = int(settings[0])
K = [settings[1],1]
d = settings[2]
rand_seed = settings[3]
data_set = settings[4]
method = settings[5][0]
partial_E = settings[5][1]

random.seed(rand_seed)
np.random.seed(rand_seed)

# print settings for this experiment
print("method: %s" % method)
print("partial E_step: %.1f" % partial_E)
print("T: %d" % T)
print("K: %s" % str(K))
print("d: %d" % d)
print("random seed: %d" % rand_seed)
print("data set: %d" % data_set)
print("max time : %.3f hours" % (max_time/3600))

# select parameters for optimization
num_epochs = 10000
tol = 1e-16
grad_tol = 1e-16
grad_buffer = "none"
weight_buffer = "none"

step_sizes = {"EM"  : [None,None],
              "CG"  : [None,None],
              "BFGS": [None,None],
              "GD"  : [0.01,0.01],
              "SGD" : [0.01,0.01],
              "SAG" : [0.01,0.01],
              "SVRG": [0.01,0.01],
              "SAGA": [0.01,0.01],
              "Adam": [0.001,0.001]}

# checks on optimization parameters
if partial_E > 0 and method in ["EM","BFGS","Nelder-Mead","CG"]:
    raise("partial_E not consistent with method")

# define distributions for all features
features = {'Y%d'%d0  : {'f'           : 'normal',
                          'lower_bound' : None,
                          'upper_bound' : None,
                          'share_coarse': False,
                          'share_fine'  : False} for d0 in range(d)}

# define shared parameters for all features (no parameters are shared)
share_params = []
for feature in features:
    for param in ['mu','log_sig']:
        for k0 in range(K[0]):
            for k1 in range(K[1]):
                share_params.append({"features":[feature],
                                     "params"  :[param],
                                     "K_coarse":[k0],
                                     "K_fine"  :[k1]})

# load in (or generate) data
data_fname = "../dat/data_Y_T-%d_K-%d-%d_d-%d_%03d" % (T,K[0],K[1],d,data_set)

if not os.path.isfile(data_fname):
    generate_data(T,K,d,data_set)

with open(data_fname,"rb") as f:
    data = pickle.load(f)

# pick intial parameters
optim = stoch_optimizor.StochOptimizor(data,features,share_params,K)
for d0 in range(d):
    optim.param_bounds["Y%d"%d0] = {}
    optim.param_bounds["Y%d"%d0]["mu"] = [-100,100]
    optim.param_bounds["Y%d"%d0]["log_sig"] = [-5,5]

if method == "control":
    if not (step_sizes["SAGA"][0] is None):
        optim.L_theta = 1.0 / (3.0*step_sizes["SAGA"][0])
        optim.L_eta = 1.0 / (3.0*step_sizes["SAGA"][1])
else:
    if not (step_sizes[method][0] is None):
        optim.L_theta = 1.0 / (3.0*step_sizes[method][0])
        optim.L_eta = 1.0 / (3.0*step_sizes[method][1])

optim.divider = 3.0

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

fname_p = "../dat/data_P_T-%d_K-%d-%d_d-%d_%03d" % (T,K[0],K[1],d,data_set)
with open(fname_p, 'rb') as f:
    true_params = pickle.load(f)

print("true parameters:")
print(true_params)
print("")

# get optimal value via A = SVRG, P = True, and M = T with true initial parameters
if method == "control":

    optim.theta = []

    for k0 in range(K[0]):
        optim.theta.append({'Y%d'%d0 : {'mu': true_params["mus"]['Y%d'%d0][(k0*K[1]):((k0+1)*K[1])],
                                        'log_sig': np.log(true_params["sigs"]['Y%d'%d0][(k0*K[1]):((k0+1)*K[1])])} \
                               for d0 in range(d)})

    log_Gamma = [np.log(true_params["Gamma"][0]),
                 [np.log(Gamma) for Gamma in true_params["Gamma"][1]]]
    log_delta = [np.log(true_params["delta"][0]),
                 [np.log(Gamma) for Gamma in true_params["delta"][1]]]

    optim.eta0 = log_delta_2_eta0(log_delta)
    optim.eta = log_Gamma_2_eta(log_Gamma)

    optim.train_HHMM_stoch(num_epochs=num_epochs,
                           max_time=max_time,
                           method="SAGA",
                           max_epochs=1,
                           partial_E=True,
                           tol=1e-4*tol,
                           grad_tol=1e-4*grad_tol,
                           record_like=True,
                           weight_buffer=weight_buffer,
                           grad_buffer=grad_buffer,
                           buffer_eps=1e-3)

# run model with P = False and M = T:
elif partial_E == 0:
    optim.train_HHMM_stoch(num_epochs=num_epochs,
                           max_time=max_time,
                           method=method,
                           max_epochs=1,
                           partial_E=False,
                           tol=tol,
                           grad_tol=grad_tol,
                           record_like=True,
                           weight_buffer=weight_buffer,
                           grad_buffer=grad_buffer,
                           buffer_eps=1e-3)

# run model with P = True and M = T:
elif partial_E == 0.5:
    if method in ["SGD","SAG","SVRG","SAGA","Adam"]:
        optim.train_HHMM_stoch(num_epochs=num_epochs,
                              max_time=max_time,
                              method=method,
                              max_epochs=1,
                              partial_E=True,
                              tol=tol,
                              grad_tol=grad_tol,
                              record_like=True,
                              weight_buffer=weight_buffer,
                              grad_buffer=grad_buffer,
                              buffer_eps=1e-3)

# run model with P = False and M = 10T:
elif partial_E == 1:
    if method in ["SGD","SAG","SVRG","SAGA","Adam"]:
        optim.train_HHMM_stoch(num_epochs=num_epochs,
                              max_time=max_time,
                              method=method,
                              max_epochs=10,
                              partial_E=True,
                              tol=tol,
                              grad_tol=grad_tol,
                              record_like=True,
                              weight_buffer=weight_buffer,
                              grad_buffer=grad_buffer,
                              buffer_eps=1e-3)


# store file name of data rather than the data itself to reduce storage
optim.data = data_fname

# delete variables with size complexity O(T) to reduce storage
# users can recalculate these by running `optim.E_step()`
optim.grad_eta_t = None
optim.grad_eta0_t = None
optim.grad_theta_t = None

optim.log_alphas = None
optim.log_betas = None

optim.p_Xt = None
optim.p_Xtm1_Xt = None

# save file
if not os.path.isdir('../params'):
    os.mkdir('../params')
if not os.path.isdir('../params/sim_study'):
    os.mkdir('../params/sim_study')

fname = "../params/sim_study/T-%d_K-%d-%d_d-%d_%s_%.1f_%03d_%03d" % (T,K[0],K[1],d,method,partial_E,rand_seed,data_set)
with open(fname, 'wb') as f:
    pickle.dump(optim, f)
