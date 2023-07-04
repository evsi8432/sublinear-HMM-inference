import sys

import pandas as pd
import numpy as np
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

from numpy import array

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

rand_seed = range(50)

# set methods
for i,settings0 in enumerate(product(rand_seed,method_partialEs)):
    if i == id:
        settings = settings0
        break

rand_seed = settings[0]
method = settings[1][0]
partial_E = settings[1][1]

random.seed(rand_seed)
np.random.seed(rand_seed)

print("method: %s" % method)
print("partial E_step: %.1f" % partial_E)
print("random seed: %d" % rand_seed)
print("max time : %.3f hours" % (max_time/3600))

# Load in Data
df = pd.read_csv("../dat/Killer_Whale_Data.csv")

# pick jump indices
jump_inds = df.index[np.isnan(df["delt_d"])].to_list()
jump_inds = [x for x in jump_inds]

# populate a data object
data = []
initial_ts = [0]
final_ts = []
features = ['delt_d','e_dive']

for t,row in enumerate(df.iterrows()):

    if t != 0 and df.iloc[t]["ID"] != df.iloc[t-1]["ID"]:
        final_ts.append(t-1)
        initial_ts.append(t)

    data.append({feature : row[1][feature] for feature in features})

final_ts.append(t)

initial_ts = np.array(initial_ts)
final_ts = np.array(final_ts)

# Set HMM Parameters
K = [3,3]
T = len(data)

share_params = []
for feature in ['delt_d']:
    for param in ['mu','log_sig']:
        for k0 in range(K[0]):
            for k1 in range(K[1]):
                share_params.append({"features":[feature],
                                     "params"  :[param],
                                     "K_coarse":[k0],
                                     "K_fine"  :[k1]})

for feature in ['e_dive']:
    for param in ['logit_p']:
        for k0 in range(K[0]):
            for k1 in range(K[1]):
                share_params.append({"features":[feature],
                                     "params"  :[param],
                                     "K_coarse":[k0],
                                     "K_fine"  :[k1]})


features = {'delt_d'     : {'f'           : 'normal',
                            'lower_bound' : None,
                            'upper_bound' : None,
                            'share_coarse': False,
                            'share_fine'  : False},
            'e_dive'     : {'f'           : 'bern',
                            'lower_bound' : None,
                            'upper_bound' : None,
                            'share_coarse': False,
                            'share_fine'  : False}}

fix_theta = [{'delt_d': {'mu': array([None, None, None], dtype=object),
                         'log_sig': array([None, None, None], dtype=object)},
              'e_dive': {'logit_p': array([-100, -100,  None], dtype=object)}},
             {'delt_d': {'mu': array([None, None, None], dtype=object),
                         'log_sig': array([None, None, None], dtype=object)},
              'e_dive': {'logit_p': array([-100, -100,  None], dtype=object)}},
             {'delt_d': {'mu': array([None, None, None], dtype=object),
                         'log_sig': array([None, None, None], dtype=object)},
              'e_dive': {'logit_p': array([-100, -100,  None], dtype=object)}}]

fix_eta = [array([[ 0.0, None, None],
                  [None,  0.0, None],
                  [None, None,  0.0]], dtype=object),
           [array([[ 0.0, None, None],
                   [-100,  0.0, None],
                   [-100, -100,  0.0]], dtype=object),
            array([[ 0.0, None, None],
                   [-100,  0.0, None],
                   [-100, -100,  0.0]], dtype=object),
            array([[ 0.0, None, None],
                   [-100,  0.0, None],
                   [-100, -100,  0.0]], dtype=object)]]

fix_eta0 = [array([ 0.0, None, None], dtype=object),
            [array([ 0.0, -100, -100], dtype=object),
             array([ 0.0, -100, -100], dtype=object),
             array([ 0.0, -100, -100], dtype=object)]]


# Initialize HMM
optim = stoch_optimizor.StochOptimizor(data,features,share_params,K,
                                       fix_theta=fix_theta,
                                       fix_eta=fix_eta,
                                       fix_eta0=fix_eta0)

optim.initial_ts = initial_ts
optim.final_ts = final_ts
optim.jump_inds = jump_inds

optim.get_log_Gamma(jump=False)
optim.get_log_Gamma(jump=True)
optim.get_log_delta()

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
print("parameter bounds:")
print(optim.param_bounds)
print("")
print("length of data:")
print(T)
print("")

# Set Optimization Parameters
num_epochs = 1000
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

### checks on optimization parameters ###
if partial_E > 0 and method in ["EM","BFGS","Nelder-Mead","CG"]:
    raise("partial_E not consistent with method")

if method == "control":
    optim.step_size = step_sizes["SAGA"]
    if not (step_sizes["SAGA"][0] is None):
        optim.L_theta = 1.0 / step_sizes["SAGA"][0]
        optim.L_eta = 1.0 / step_sizes["SAGA"][1]
else:
    optim.step_size = step_sizes[method]
    if not (step_sizes[method][0] is None):
        optim.L_theta = 1.0 / (3.0 * step_sizes[method][0])
        optim.L_eta = 1.0 / (3.0 * step_sizes[method][1])

# get optimal value via SVRG:
if method == "control":
    optim.train_HHMM_stoch(num_epochs=2*num_epochs,
                         max_time=max_time,
                         method="SVRG",
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
optim.data = "../dat/Killer_Whale_Data.csv"

# cut variables with size complexity O(T) (can recalculate with E-step)
optim.grad_eta_t = None
optim.grad_eta0_t = None
optim.grad_theta_t = None

optim.log_alphas = None
optim.log_betas = None

optim.p_Xt = None
optim.p_Xtm1_Xt = None

# save file
fname = "../params/case_study/case_study_updown_K-%d-%d_%s_%.1f_%03d" % (K[0],K[1],method,partial_E,rand_seed)
with open(fname, 'wb') as f:
    pickle.dump(optim, f)
