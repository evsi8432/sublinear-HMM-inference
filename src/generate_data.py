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

# set random seed
random.seed(0)
np.random.seed(0)

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

### GENERATE DATA ###
X = np.zeros(T,dtype=int)
data_y = []

for t in range(T):

    if t == 0:
        X[t] = np.random.choice(K[0]*K[1],p=delta_full)
    else:
        X[t] = np.random.choice(K[0]*K[1],p=Gamma_full[X[t-1]])

    if experiment == 1:
        data_y.append({'Y'      : mus['Y'][X[t]] + sigs['Y'][X[t]]*np.random.normal(),
                       'Y_star' : mus['Y_star'][X[t]] + sigs['Y_star'][X[t]]*np.random.normal()})
    elif experiment == 2:
        data_y.append({'Y'      : mus['Y'][X[t]] + sigs['Y'][X[t]]*np.random.normal()})

# save data
fname_y = "../dat/data_Y_exp-%d_T-%d" % (experiment,T)
with open(fname_y, 'wb') as f:
    pickle.dump(data_y, f)

fname_x = "../dat/data_X_exp-%d_T-%d" % (experiment,T)
with open(fname_x, 'wb') as f:
    pickle.dump(X, f)
