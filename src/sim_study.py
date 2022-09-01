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
for i,settings0 in enumerate(product(Ts,Ks,ds,rand_seed,method_partialEs)):
    if i == id:
        settings = settings0
        break

T = int(settings[0])
K = [settings[1],1]
d = settings[2]
rand_seed = settings[3]
method = settings[4][0]
partial_E = settings[4][1]

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
              "GD"  : [0.005,0.005],
              "SGD" : [0.005,0.005],
              "SAG" : [0.005,0.005],
              "SVRG": [0.005,0.005],
              "SAGA": [0.005,0.005]}

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

# save file
fname = "../params/T-%d_K-%d-%d_d-%d_%s_%.1f_%d" % (T,K[0],K[1],d,method,partial_E,rand_seed)
with open(fname, 'wb') as f:
    pickle.dump(optim, f)
