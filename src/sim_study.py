import sys
#sys.path.remove('/Users/evsi8432/Documents/Research/CarHHMM-DFT/Repository/Code')

import pandas as pd
import numpy as np
from numpy import array
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

from itertools import product

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

Ts = [1e3,1e4,1e5]
Ks = [3,5,10]
ds = [1,4,10]
rand_seed = [0,1,2,3,4,5,6,7,8,9]

# set methods
for i,settings in enumerate(product(method_partialEs,Ts,Ks,ds,rand_seed)):
    if i == id:
        break


method = settings[0][0]
partial_E = settings[0][1]
T = int(settings[1])
K = [settings[2],1]
d = settings[3]
rand_seed = settings[4]
random.seed(rand_seed)
np.random.seed(rand_seed)

print("method: %s" % method)
print("partial E_step: %.1f" % partial_E)
print("T: %d" % T)
print("K: %s" % str(K))
print("d: %d" % d)
print("random seed: %d" % rand_seed)
print("max time : %.3f hours" % (max_time/3600))

# select parameters for optimization
num_epochs = 100
tol = 1e-16
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
features = [{},
            {'Y%d'%d0  : {'f'           : 'normal',
                          'lower_bound' : None,
                          'upper_bound' : None} for d0 in range(d)}]

### load in data ###
data_fname = "../dat/data_Y_T-%d_K-%d-%d_d-%d" % (T,K[0],K[1],d)
with open(data_fname,"rb") as f:
    data = pickle.load(f)

# pick intial parameters
optim = optimizor.optimizor(data,features,K)
for d0 in range(d):
    optim.param_bounds[1]["Y%d"%d0] = {}
    optim.param_bounds[1]["Y%d"%d0]["mu"] = [-100,100]
    optim.param_bounds[1]["Y%d"%d0]["log_sig"] = [-5,5]

if method == "control":
    optim.step_size = step_sizes["SAGA"]
    if not (step_sizes["SAGA"][0] is None):
        optim.L_theta = 1.0 / (3.0*step_sizes["SAGA"][0])
        optim.L_eta = 1.0 / (3.0*step_sizes["SAGA"][1])
else:
    optim.step_size = step_sizes[method]
    if not (step_sizes[method][0] is None):
        optim.L_theta = 1.0 / (3.0*step_sizes[method][0])
        optim.L_eta = 1.0 / (3.0*step_sizes[method][1])

# TEMPORARAY THING
optim.theta = [{},[{'Y0': {'mu': array([1.81734315]), 'log_sig': array([0.17850115])}},
               {'Y0': {'mu': array([1.18815644]), 'log_sig': array([2.01923715])}},
               {'Y0': {'mu': array([1.90027093]), 'log_sig': array([-1.19893393])}},
               {'Y0': {'mu': array([1.16520267]), 'log_sig': array([-0.37301326])}},
               {'Y0': {'mu': array([0.32130249]), 'log_sig': array([0.18894245])}}]]

optim.eta = [array([[ 0.        ,  0.45427351, -0.23896227, -0.87832498, -0.55613677],
                    [-0.66632567,  0.        , -1.20515826, -0.6869323 , -1.85409574],
                    [-3.55298982, -0.3463814 ,  0.        , -1.74216502,  1.26975462],
                    [-2.45436567, -0.95424148, -1.18718385,  0.        ,  0.46935877],
                    [-0.84505257, -0.62183748, -1.88778575, -2.98079647,  0.        ]]),
            [array([[0.]]), array([[0.]]), array([[0.]]), array([[0.]]), array([[0.]])]]

optim.eta0 = [array([ 0.        , -1.04855297, -1.42001794, -1.70627019,  1.9507754 ]),
              [array([0.]), array([0.]), array([0.]), array([0.]), array([0.])]]

optim.log_Gamma = optim.get_log_Gamma(jump=False)[0]
optim.log_Gamma_jump = optim.get_log_Gamma(jump=True)[0]
optim.log_delta = optim.get_log_delta()[0]

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

    fname_p = "../dat/data_P_T-%d_K-%d-%d_d-%d" % (T,K[0],K[1],d)
    with open(fname_p, 'rb') as f:
        true_params = pickle.load(f)

    optim.theta = [{},[]]

    for k0 in range(K[0]):
        optim.theta[1].append({'Y%d'%d0 : {'mu': true_params["mus"]['Y%d'%d0][(k0*K[1]):((k0+1)*K[1])],
                                           'log_sig': np.log(true_params["sigs"]['Y%d'%d0][(k0*K[1]):((k0+1)*K[1])])} \
                               for d0 in range(d)})

    optim.Gamma = true_params["Gamma"]
    optim.delta = true_params["delta"]

    optim.eta0 = delta_2_eta0(optim.delta)
    optim.eta = Gamma_2_eta(optim.Gamma)


    optim.train_HHMM(num_epochs=num_epochs,
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
fname = "../params/T-%d_K-%d-%d_d-%d_%s_%.1f_%03d" % (T,K[0],K[1],d,method,partial_E,rand_seed)
with open(fname, 'wb') as f:
    pickle.dump(optim, f)
