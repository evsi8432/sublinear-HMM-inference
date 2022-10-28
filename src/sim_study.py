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

# set methods
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
              "SAGA": [0.01,0.01]}

jump_every = 1

### checks on optimization parameters ###
if partial_E > 0 and method in ["EM","BFGS","Nelder-Mead","CG"]:
    raise("partial_E not consistent with method")

### features of data ###
features = {'Y%d'%d0  : {'f'           : 'normal',
                          'lower_bound' : None,
                          'upper_bound' : None,
                          'share_coarse': False,
                          'share_fine'  : False} for d0 in range(d)}

### load in data ###
data_fname = "../dat/data_Y_T-%d_K-%d-%d_d-%d_%03d" % (T,K[0],K[1],d,data_set)

if not os.path.isfile(data_fname):
    generate_data(T,K,d,data_set)

with open(data_fname,"rb") as f:
    data = pickle.load(f)

# pick intial parameters
optim = stoch_optimizor.StochOptimizor(data,features,K)
for d0 in range(d):
    optim.param_bounds["Y%d"%d0] = {}
    optim.param_bounds["Y%d"%d0]["mu"] = [-100,100]
    optim.param_bounds["Y%d"%d0]["log_sig"] = [-5,5]

if method == "control":
    optim.step_size = step_sizes["SAGA"]
    if not (step_sizes["SAGA"][0] is None):
        optim.L_theta = 1.0 / (3.0*step_sizes["SAGA"][0]) * np.ones(optim.K_total)
        optim.L_eta = 1.0 / (3.0*step_sizes["SAGA"][1])
else:
    optim.step_size = step_sizes[method]
    if not (step_sizes[method][0] is None):
        optim.L_theta = 1.0 / (3.0*step_sizes[method][0]) * np.ones(optim.K_total)
        optim.L_eta = 1.0 / (3.0*step_sizes[method][1])

### TEMPORARY INITIAL PARAMETERS TO CHECK
'''
optim.theta = [{'Y0': {'mu': array([-0.51222136]), 'log_sig': array([-0.23499933])},
              'Y1': {'mu': array([-1.98611241]), 'log_sig': array([0.53954341])},
              'Y2': {'mu': array([0.87687884]), 'log_sig': array([-0.33830869])}},
             {'Y0': {'mu': array([-0.23291642]), 'log_sig': array([-0.23499933])},
              'Y1': {'mu': array([0.85151215]), 'log_sig': array([0.53954341])},
              'Y2': {'mu': array([0.07060724]), 'log_sig': array([-0.33830869])}},
             {'Y0': {'mu': array([-0.75236709]), 'log_sig': array([-0.23499933])},
              'Y1': {'mu': array([-1.71663788]), 'log_sig': array([0.53954341])},
              'Y2': {'mu': array([0.99496944]), 'log_sig': array([-0.33830869])}},
             {'Y0': {'mu': array([-1.18605903]), 'log_sig': array([-0.23499933])},
              'Y1': {'mu': array([2.79871248]), 'log_sig': array([0.53954341])},
              'Y2': {'mu': array([0.74045027]), 'log_sig': array([-0.33830869])}},
             {'Y0': {'mu': array([-1.7465371]), 'log_sig': array([-0.23499933])},
              'Y1': {'mu': array([-2.14622667]), 'log_sig': array([0.53954341])},
              'Y2': {'mu': array([0.85510889]), 'log_sig': array([-0.33830869])}},
             {'Y0': {'mu': array([1.01952506]), 'log_sig': array([-0.23499933])},
              'Y1': {'mu': array([0.06911714]), 'log_sig': array([0.53954341])},
              'Y2': {'mu': array([0.40894029]), 'log_sig': array([-0.33830869])}}]

optim.eta = [array([[ 0.        , -1.33840574,  0.28388504, -4.25919032, -3.70010475,
         -0.07835999],
        [-2.43483635,  0.        , -0.25316353, -2.22276674, -4.07607753,
         -4.01895965],
        [-4.11651312, -0.68743184,  0.        , -5.47730859, -1.79367401,
         -3.2433337 ],
        [-1.44856391, -4.18134977, -3.21997051,  0.        ,  1.38365226,
         -3.49590748],
        [-3.16159445, -2.22150794,  2.08405749, -1.10495862,  0.        ,
         -1.95422805],
        [-0.28553145, -1.63213884, -2.83222316,  0.5001001 ,  0.49659957,
          0.        ]]),
[array([[0.]]),
array([[0.]]),
array([[0.]]),
array([[0.]]),
array([[0.]]),
array([[0.]])]]

optim.eta0 = [array([ 0.        , -0.16659957,  0.91719602,  0.08025059,  0.22823877,
        -0.8804768 ]),
 [array([0.]),
  array([0.]),
  array([0.]),
  array([0.]),
  array([0.]),
  array([0.])]]

optim.get_log_Gamma(jump=False)
optim.get_log_Gamma(jump=True)
optim.get_log_delta()
'''
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

# print mus and sigs
#print("theta priors:")
#print(optim.theta_mus)
#print(optim.theta_sigs)
#print("")
#print("eta0 priors:")
#print(optim.eta0_mus)
#print(optim.eta0_sigs)
#print("")
#print("eta priors:")
#print(optim.eta_mus)
#print(optim.eta_sigs)
#print("")

fname_p = "../dat/data_P_T-%d_K-%d-%d_d-%d_%03d" % (T,K[0],K[1],d,data_set)
with open(fname_p, 'rb') as f:
    true_params = pickle.load(f)

print("true parameters:")
print(true_params)
print("")

# get optimal value via SAGA:
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

elif partial_E == 0.5:
    if method in ["SGD","SAG","SVRG","SAGA"]:
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

elif partial_E == 1:
    if method in ["SGD","SAG","SVRG","SAGA"]:
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


# reduce storage
optim.data = data_fname

# cut variables with size complexity O(T) (can recalculate with E-step)
optim.grad_eta_t = None
optim.grad_eta0_t = None
optim.grad_theta_t = None

optim.log_alphas = None
optim.log_betas = None

optim.p_Xt = None
optim.p_Xtm1_Xt = None

fname = "../params/sim_study/T-%d_K-%d-%d_d-%d_%s_%.1f_%03d_%03d" % (T,K[0],K[1],d,method,partial_E,rand_seed,data_set)
with open(fname, 'wb') as f:
    pickle.dump(optim, f)
