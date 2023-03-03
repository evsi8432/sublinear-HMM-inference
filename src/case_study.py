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

# select parameters for optimization
num_epochs = 10000
tol = 1e-8
grad_tol = 1e-8
grad_buffer = "none"
weight_buffer = "none"

print("num-epochs: %d" % num_epochs)
print("grad_buffer: %s" % grad_buffer)
print("weight_buffer: %s" % weight_buffer)

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
features = {'diveDuration'     : {'f'           : 'normal',
                                   'lower_bound' : None,
                                   'upper_bound' : None,
                                   'share_coarse': True,
                                   'share_fine'  : False},
             'maxDepth'         : {'f'           : 'normal',
                                   'lower_bound' : np.array([-np.infty,np.log(5),np.log(5)]),
                                   'upper_bound' : np.array([np.log(20),np.infty,np.infty]),
                                   'share_coarse': True,
                                   'share_fine'  : False},
             'avg_bot_htv'      : {'f'           : 'normal',
                                   'lower_bound' : None,
                                   'upper_bound' : None,
                                   'share_coarse': True,
                                   'share_fine'  : False}}



### load in data ###
df = pd.read_csv("../../dat/Final_Data_Beth.csv")

# only take whale I107
#whales = ["I145"]
#df = df[df["ID"].isin(whales)]

# convert times
df["stime"] = pd.to_datetime(df["stime"])
df["etime"] = pd.to_datetime(df["etime"])

# force dives to be at least 2 seconds long
df = df[df["diveDuration"] > np.log(2.0)]

# replace -inf
df["max_bot_jp"][df["max_bot_jp"] == -np.infty] = np.NAN

df["broadDiveType"] = np.NAN #3  # unknown
df.loc[df["maxDepth"] > np.log(20),"broadDiveType"] = 1  # deep
df.loc[df["maxDepth"] < np.log(5),"broadDiveType"] = 0  # shallow

# populate a data object

data = []

initial_ts = [0]
final_ts = []

for t,row in enumerate(df.iterrows()):

    if t != 0 and df.iloc[t]["ID"] != df.iloc[t-1]["ID"]:
        final_ts.append(t-1)
        initial_ts.append(t)

    data.append({"diveDuration"     : row[1]["diveDuration"],
                 "maxDepth"         : row[1]["maxDepth"],
                 "avg_bot_htv"      : row[1]["avg_bot_htv"]})

final_ts.append(t)

initial_ts = np.array(initial_ts)
final_ts = np.array(final_ts)

T = len(data)
K = [2,3]

# pick intial parameters
optim = stoch_optimizor.StochOptimizor(data,features,K)

optim.initial_ts = initial_ts
optim.final_ts = final_ts

#for feature in features:
#    optim.param_bounds[feature] = {}
#    optim.param_bounds[feature]["mu"] = [-100,100]
#    optim.param_bounds[feature]["log_sig"] = [-5,5]

if method == "control":
    optim.step_size = step_sizes["SAGA"]
    if not (step_sizes["SAGA"][0] is None):
        optim.L_theta = 1.0 / step_sizes["SAGA"][0] * np.ones(optim.K_total)
        optim.L_eta = 1.0 / step_sizes["SAGA"][1]
else:
    optim.step_size = step_sizes[method]
    if not (step_sizes[method][0] is None):
        optim.L_theta = 1.0 / (3.0 * step_sizes[method][0]) * np.ones(optim.K_total)
        optim.L_eta = 1.0 / (3.0 * step_sizes[method][1])


# set initial parameters
'''
from numpy import array

optim.theta = [{'diveDuration': {'mu': array([2.93782794, 4.88516625, 4.99495526]),
                                'log_sig': array([-0.54510259, -0.2382056 , -0.88471453])},
                'maxDepth': {'mu': array([0.69089463, 1.15212687, 2.49187761]),
                            'log_sig': array([-0.61032425,  0.87554853, -0.45978847])},
                'avg_bot_htv': {'mu': array([-1.93947452, -1.02800198, -2.21785862]),
                                'log_sig': array([-0.5753301 , -0.59439428, -0.97079874])}},
                {'diveDuration': {'mu': array([2.93782794, 4.88516625, 4.99495526]),
                                'log_sig': array([-0.54510259, -0.2382056 , -0.88471453])},
                'maxDepth': {'mu': array([0.69089463, 1.15212687, 2.49187761]),
                            'log_sig': array([-0.61032425,  0.87554853, -0.45978847])},
                'avg_bot_htv': {'mu': array([-1.93947452, -1.02800198, -2.21785862]),
                                'log_sig': array([-0.5753301 , -0.59439428, -0.97079874])}}]

optim.eta = [array([[ 0.        , -5.36783081],
                    [-5.96191054,  0.        ]]),
            [array([[ 0.        , -3.0982348 , -7.29077604],
                    [ 3.11176201,  0.        , -3.55053225],
                    [ 3.46523309, -1.26690601,  0.        ]]),
             array([[ 0.        , -5.86956604, -1.71754442],
                    [ 1.99618808,  0.        , -2.53244688],
                    [ 5.64160625, -3.03739264,  0.        ]])]]

optim.eta0 = [array([ 0.        , -0.68861739]),
             [array([ 0.        ,  0.9141165 , -0.81107036]),
              array([ 0.        , -0.27046925,  0.41425818])]]

optim.theta = [{'diveDuration': {'mu': array([2.93675689, 4.56558096, 4.99335053]),
                                'log_sig': array([-0.54571896, -0.01499696, -0.8641769 ])},
                'maxDepth': {'mu': array([0.68897808, 1.12012189, 2.01582831]),
                            'log_sig': array([-0.61349408,  0.89911473, -0.03025042])},
                'avg_bot_htv': {'mu': array([-1.94116168, -0.96998491, -2.18468672]),
                                'log_sig': array([-0.57793561, -0.57307262, -0.89863459])}},
               {'diveDuration': {'mu': array([2.93675689, 4.56558096, 4.99335053]),
                                'log_sig': array([-0.54571896, -0.01499696, -0.8641769 ])},
                'maxDepth': {'mu': array([0.68897808, 1.12012189, 2.01582831]),
                            'log_sig': array([-0.61349408,  0.89911473, -0.03025042])},
                'avg_bot_htv': {'mu': array([-1.94116168, -0.96998491, -2.18468672]),
                                'log_sig': array([-0.57793561, -0.57307262, -0.89863459])}}]


optim.eta = [array([[ 0.        , -4.50183528],
                   [-6.50307218,  0.        ]]),
            [array([[ 0.        , -3.16085052, -7.26383487],
                   [ 3.01076427,  0.        , -3.53290107],
                   [ 3.45238053, -1.26436704,  0.        ]]),
             array([[ 0.        , -6.02752494, -1.67368254],
                   [ 1.95335965,  0.        , -2.52425799],
                   [ 5.62807992, -3.03345158,  0.        ]])]]

optim.eta0 =  [array([ 0.        , -0.62029701]),
                [array([ 0.        ,  0.89605201, -0.83471787]),
                 array([ 0.        , -0.36191171,  0.40401095])]]


optim.get_log_Gamma(jump=False)
optim.get_log_Gamma(jump=True)
optim.get_log_delta()
'''

#optim.L_theta = 133.4947781051537
#optim.L_eta = 0.2601961298328865

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

# get optimal value via SAGA:
if method == "control":
    optim.train_HHMM_stoch(num_epochs=2*num_epochs,
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
optim.data = "../../dat/Final_Data_Beth.csv"

# cut variables with size complexity O(T) (can recalculate with E-step)
optim.grad_eta_t = None
optim.grad_eta0_t = None
optim.grad_theta_t = None

optim.log_alphas = None
optim.log_betas = None

optim.p_Xt = None
optim.p_Xtm1_Xt = None

# save file
fname = "../params/case_study/case_study_K-%d-%d_%s_%.1f_%03d" % (K[0],K[1],method,partial_E,rand_seed)
with open(fname, 'wb') as f:
    pickle.dump(optim, f)
