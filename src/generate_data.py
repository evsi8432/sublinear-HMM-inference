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
#T = int(sys.argv[1])
#K = [int(sys.argv[2]),int(sys.argv[3])]
#d = int(sys.argv[4])

def generate_data(T,K,d):

    # set random seed
    random.seed(0)
    np.random.seed(0)

    # Select parameters for data generating process
    features = [{},
                {'Y%d'%d0  : {'f'           : 'normal',
                              'lower_bound' : None,
                              'upper_bound' : None} for d0 in range(d)}]

    jump_every = 1

    # generate Gamma_coarse
    Gamma_coarse = np.zeros((K[0],K[0]))
    for i in range(K[0]):
        Gamma_coarse[i] = 100.0/(T*(K[0]-1)) * np.ones(K[0])
        Gamma_coarse[i,i] = 1.0-100.0/T

    # generate Gamma_fines
    Gamma_fines = []
    for k0 in range(K[0]):
        Gamma_fine = np.zeros((K[1],K[1]))
        for k1 in range(K[1]):
            Gamma_fine[k1] = np.random.dirichlet(np.ones(K[1]))
        Gamma_fines.append(Gamma_fine)

    delta_coarse = np.random.dirichlet(np.ones(K[0]))
    delta_fines = [np.random.dirichlet(np.ones(K[1])) for _ in range(K[0])]

    Gamma_coarse_full = np.kron(Gamma_coarse,np.ones((K[1],K[1])))
    Gamma_fine_full = np.zeros((0,K[0]*K[1]))

    for k0 in range(K[0]):
        coarse_row_elements = [np.tile(delta_fines[i],(K[1],1)) for i in range(K[0])]
        coarse_row_elements[k0] = Gamma_fines[k0]
        coarse_row = np.hstack(coarse_row_elements)
        Gamma_fine_full = np.vstack([Gamma_fine_full,coarse_row])

    Gamma_full = Gamma_coarse_full * Gamma_fine_full
    delta_full = np.repeat(delta_coarse,K[1]) * np.concatenate(delta_fines)

    mus =  {'Y%d'%d0 : np.random.normal(size=K[0]*K[1]) for d0 in range(d)}
    sigs = {'Y%d'%d0 : np.exp(-1.0)*np.ones(K[0]*K[1]) for d0 in range(d)}

    ### GENERATE DATA ###
    X = np.zeros(T,dtype=int)
    data_y = []

    for t in range(T):

        if t == 0:
            X[t] = np.random.choice(K[0]*K[1],p=delta_full)
        else:
            X[t] = np.random.choice(K[0]*K[1],p=Gamma_full[X[t-1]])

        datum = {}
        for feature in features[1]:
            datum[feature] =  mus[feature][X[t]] + sigs[feature][X[t]]*np.random.normal()
        data_y.append(datum)

    # save data
    fname_y = "../dat/data_Y_T-%d_K-%d-%d_d-%d" % (T,K[0],K[1],d)
    with open(fname_y, 'wb') as f:
        pickle.dump(data_y, f)

    fname_x = "../dat/data_X_T-%d_K-%d-%d_d-%d" % (T,K[0],K[1],d)
    with open(fname_x, 'wb') as f:
        pickle.dump(X, f)

    fname_p = "../dat/data_P_T-%d_K-%d-%d_d-%d" % (T,K[0],K[1],d)
    with open(fname_p, 'wb') as f:
        pickle.dump({"mus"  : mus,
                     "sigs" : sigs,
                     "Gamma": [Gamma_coarse,Gamma_fines],
                     "delta": [delta_coarse,delta_fines]}, f)

    return
