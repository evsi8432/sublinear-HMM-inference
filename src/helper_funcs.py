import sys
import os

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

def eta_2_log_Gamma(eta):

    # get coarse-scale Gamma
    log_Gamma_coarse = (eta[0].T - logsumexp(eta[0],axis=1)).T

    # get fine-scale Gammas
    log_Gammas_fine = []
    for eta_fine in eta[1]:
        log_Gamma_fine = (eta_fine.T - logsumexp(eta_fine,axis=1)).T
        log_Gammas_fine.append(log_Gamma_fine)

    return [log_Gamma_coarse,log_Gammas_fine]

def log_Gamma_2_eta(log_Gamma):

    # get Coarse-scale eta
    eta_coarse = np.zeros_like(log_Gamma[0])
    N = len(log_Gamma[0])
    for i in range(N):
        eta_coarse[i] = log_Gamma[0][i] - log_Gamma[0][i,i]

    # get fine-scale eta
    etas_fine = []
    N = len(log_Gamma[1][0])
    for log_Gamma_fine in log_Gamma[1]:
        eta_fine = np.zeros_like(log_Gamma_fine)
        for i in range(N):
            eta_fine[i] = log_Gamma_fine[i] - log_Gamma_fine[i,i]
        etas_fine.append(eta_fine)

    return [eta_coarse,etas_fine]

def eta0_2_log_delta(eta0):

    # get coarse-scale delta
    log_delta_coarse = eta0[0] - logsumexp(eta0[0])

    # get fine-scale Gammas
    log_deltas_fine = []
    for eta0_fine in eta0[1]:
        log_delta_fine = eta0_fine - logsumexp(eta0_fine)
        log_deltas_fine.append(log_delta_fine)

    return [log_delta_coarse,log_deltas_fine]

def log_delta_2_eta0(log_delta):

    # get coarse-scale eta0
    eta0_coarse = log_delta[0] - log_delta[0][0]

    # get fine-scale eta
    eta0s_fine = []
    for log_delta_fine in log_delta[1]:
        eta0_fine = log_delta_fine - log_delta_fine[0]
        eta0s_fine.append(eta0_fine)

    return [eta0_coarse,eta0s_fine]

def logdotexp(log_delta, log_Gamma):
    max_log_delta, max_log_Gamma = np.max(log_delta), np.max(log_Gamma)
    delta, Gamma = log_delta - max_log_delta, log_Gamma - max_log_Gamma
    np.exp(delta, out=delta)
    np.exp(Gamma, out=Gamma)
    C = np.dot(delta, Gamma)
    np.log(C, out=C)
    C += max_log_delta + max_log_Gamma
    return C

def generate_data(T,K,d,rand_seed):

    # set random seed
    random.seed(rand_seed)
    np.random.seed(rand_seed)

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
    fname_y = "../dat/data_Y_T-%d_K-%d-%d_d-%d_%03d" % (T,K[0],K[1],d,rand_seed)
    with open(fname_y, 'wb') as f:
        pickle.dump(data_y, f)

    fname_x = "../dat/data_X_T-%d_K-%d-%d_d-%d_%03d" % (T,K[0],K[1],d,rand_seed)
    with open(fname_x, 'wb') as f:
        pickle.dump(X, f)

    fname_p = "../dat/data_P_T-%d_K-%d-%d_d-%d_%03d" % (T,K[0],K[1],d,rand_seed)
    with open(fname_p, 'wb') as f:
        pickle.dump({"mus"  : mus, 
                     "sigs" : sigs,
                     "Gamma": [Gamma_coarse,Gamma_fines],
                     "delta": [delta_coarse,delta_fines]}, f)

    return
