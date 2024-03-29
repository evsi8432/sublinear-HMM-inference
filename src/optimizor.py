import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
from scipy.special import digamma
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import LinearConstraint
from scipy.signal import convolve
from scipy.interpolate import interp1d

from itertools import product

from copy import deepcopy
import warnings
import time
import pickle
import sys

from HHMM import HHMM

from helper_funcs import eta_2_log_Gamma
from helper_funcs import log_Gamma_2_eta
from helper_funcs import eta0_2_log_delta
from helper_funcs import log_delta_2_eta0

'''
This file defines the Optimizor class in VARIANCE-REDUCED STOCHASTIC
OPTIMIZATION FOR EFFICIENT INFERENCE OF HIDDEN MARKOV MODELS
by Sidrow et al. (2023). This class inherets from HHMM. It contains all
of the gradients with respect to theta, eta, and eta0 at every observation
index. It also contains the full gradients with respect to the parameters both
as an object as well as a vector (xprime). It keeps track of the time of
optimization as well as the epoch number of the optimization algorithm. Finally,
it stores traces of the epoch, time, parameters, log-likelihood, and gradient
norm at each step of the algorithm.

It can initialize the gradients at each observation. It can get the gradient
of the emission distribution with respect to theta, the gradient of log-Gamma
with respect to eta, and the gradient of log-delta with respect to eta0. It can
get the gradient of the surrogate function at a point F_t with respect to theta,
eta, and eta0. It can convert the full gradient from an object to a vector.
It can also perform an E-step, which updates all conditional probabilites and
gradients for a given set of parameters. Finally, it can run the baseline
optimization algorithms using the scipy package. For our optimization algorithms,
see `stoch_optimizor.py`.
'''

class Optimizor(HHMM):

    def __init__(self,data,features,share_params,K,fix_theta=None,fix_eta=None,fix_eta0=None):

        '''
        constructor for optimizor class
        '''

        # inherent from HHMM
        super(Optimizor, self).__init__(data,features,share_params,K,
                                        fix_theta=fix_theta,
                                        fix_eta=fix_eta,
                                        fix_eta0=fix_eta0)

        # gradients wrt theta
        self.grad_theta_t = [deepcopy(self.theta) for _ in range(self.T)]
        self.grad_theta = deepcopy(self.theta)

        # gradients wrt eta
        self.grad_eta_t = [deepcopy(self.eta) for _ in range(self.T)]
        self.grad_eta = deepcopy(self.eta)

        # gradients wrt eta0
        self.grad_eta0_t = [deepcopy(self.eta0) for _ in range(self.T)]
        self.grad_eta0 = deepcopy(self.eta0)

        # initialize gradients
        self.initialize_grads()
        self.xprime = self.grad_params_2_xprime()

        # keep track of time
        self.start_time = None
        self.train_time = None

        # keep track of epochs
        self.epoch_num = 0.0

        # traces
        self.theta_trace = []
        self.eta_trace = []
        self.eta0_trace = []

        self.log_like_trace = []
        self.grad_norm_trace = []

        self.time_trace = []
        self.epoch_trace = []

        return

    def initialize_grads(self):

        # initialize overall gradient wrt theta
        for feature,dist in self.features.items():
            if dist['f'] == 'normal':
                for k0 in range(self.K[0]):

                    self.grad_theta[k0][feature]['mu'] = np.zeros(self.K[1])
                    self.grad_theta[k0][feature]['log_sig'] = np.zeros(self.K[1])

            elif dist['f'] == 'normal_AR':
                for k0 in range(self.K[0]):

                    self.grad_theta[k0][feature]['mu'] = np.zeros(self.K[1])
                    self.grad_theta[k0][feature]['log_sig'] = np.zeros(self.K[1])
                    self.grad_theta[k0][feature]['logit_rho'] = np.zeros(self.K[1])

            elif dist['f'] == 'gamma':
                for k0 in range(self.K[0]):

                    self.grad_theta[k0][feature]['log_mu'] = np.zeros(self.K[1])
                    self.grad_theta[k0][feature]['log_sig'] = np.zeros(self.K[1])

            elif dist['f'] == 'bern':
                for k0 in range(self.K[0]):

                    self.grad_theta[k0][feature]['logit_p'] = np.zeros(self.K[1])

            elif dist['f'].startswith('cat'):
                for k0 in range(self.K[0]):

                    for cat_num in range(1,int(dist[3:])):
                        self.grad_theta[k0][feature]['psi%d'%cat_num] = np.zeros(self.K[1])

            else:
                raise('only independent normal distributions supported at this time')

        # initialize overall gradient wrt eta
        self.grad_eta = [np.zeros((self.K[0],self.K[0])),
                         [np.zeros((self.K[1],self.K[1])) for _ in range(self.K[0])]]

        # initialize overall gradient wrt eta0
        self.grad_eta0 = [np.zeros(self.K[0]),
                          [np.zeros(self.K[1]) for _ in range(self.K[0])]]

        # initialize gradients at each observation
        for t in range(self.T):

            # initialize gradients wrt eta
            self.grad_eta_t[t] = [np.zeros((self.K[0],self.K[0])),
                                  [np.zeros((self.K[1],self.K[1])) for _ in range(self.K[0])]]

            # initialize gradients wrt eta0
            self.grad_eta0_t[t] = [np.zeros(self.K[0]),
                                   [np.zeros(self.K[1]) for _ in range(self.K[0])]]

            # initialize gradients wrt theta
            for feature,dist in self.features.items():
                if dist['f'] == 'normal':
                    for k0 in range(self.K[0]):

                        self.grad_theta_t[t][k0][feature]['mu'] = np.zeros(self.K[1])
                        self.grad_theta_t[t][k0][feature]['log_sig'] = np.zeros(self.K[1])

                elif dist['f'] == 'normal_AR':
                    for k0 in range(self.K[0]):

                        self.grad_theta_t[t][k0][feature]['mu'] = np.zeros(self.K[1])
                        self.grad_theta_t[t][k0][feature]['log_sig'] = np.zeros(self.K[1])
                        self.grad_theta_t[t][k0][feature]['logit_rho'] = np.zeros(self.K[1])

                elif dist['f'] == 'gamma':
                    for k0 in range(self.K[0]):

                        self.grad_theta_t[t][k0][feature]['log_mu'] = np.zeros(self.K[1])
                        self.grad_theta_t[t][k0][feature]['log_sig'] = np.zeros(self.K[1])

                elif dist['f'] == 'bern':
                    for k0 in range(self.K[0]):

                        self.grad_theta_t[t][k0][feature]['logit_p'] = np.zeros(self.K[1])

                elif dist['f'].startswith('cat'):
                    for k0 in range(self.K[0]):

                        for cat_num in range(1,int(dist[3:])):
                            self.grad_theta_t[t][k0][feature]['psi%d'%cat_num] = np.zeros(self.K[1])

                else:
                    raise('only independent normal distributions supported at this time')

        return

    def get_grad_log_f(self,t,theta=None):

        # set theta if not provided
        if theta is None:
            theta = self.theta

        # set the observation
        y = self.data[t]

        # initialize the gradient
        grad_log_f = [{} for _ in range(self.K[0])]

        # go through each feature and add to the gradient
        for feature,value in y.items():

            # error checking
            if feature not in self.features:
                print("unidentified feature in y: %s" % feature)
                return

            dist = self.features[feature]['f']

            if dist == 'normal':

                for k0 in range(self.K[0]):

                    # extract parameters
                    mu = theta[k0][feature]['mu']
                    log_sig = theta[k0][feature]['log_sig']
                    sig = np.exp(log_sig)

                    # get gradients
                    if np.isnan(y[feature]):
                        grad_log_f[k0][feature] = {'mu': 0, 'log_sig': 0}
                    else:
                        grad_log_f[k0][feature] = {'mu': (y[feature]-mu)/(sig**2),
                                                   'log_sig': ((y[feature]-mu)/sig)**2 - 1}

                        # change gradient if the distribution is a truncated normal
                        a = self.features[feature]['lower_bound']
                        b = self.features[feature]['upper_bound']

                        if (a is not None) and (b is not None):

                            ratio_a = np.exp(norm.logpdf(a,loc=mu,scale=sig) - \
                                             norm.logsf(a,loc=mu,scale=sig))
                            ratio_b = np.exp(norm.logpdf(b,loc=mu,scale=sig) - \
                                             norm.logcdf(b,loc=mu,scale=sig))

                            for k1 in range(self.K[1]):

                                if a[k1] > -np.infty and b[k1] < np.infty:
                                    print("cannot handle two-sided truncated normals at this time.")

                                elif a[k1] > -np.infty:

                                    if y[feature] >= np.array(a[k1]):
                                        grad_log_f[k0][feature]['mu'][k1] += -ratio_a[k1]
                                        grad_log_f[k0][feature]['log_sig'][k1] += -(a[k1]-mu[k1])*ratio_a[k1]
                                    else:
                                        grad_log_f[k0][feature]['mu'][k1] = 0.0
                                        grad_log_f[k0][feature]['log_sig'][k1] = 0.0

                                elif b[k1] < np.infty:

                                    if y[feature] <= np.array(b[k1]):
                                        grad_log_f[k0][feature]['mu'][k1] += ratio_b[k1]
                                        grad_log_f[k0][feature]['log_sig'][k1] += (b[k1]-mu[k1])*ratio_b[k1]
                                    else:
                                        grad_log_f[k0][feature]['mu'][k1] = 0.0
                                        grad_log_f[k0][feature]['log_sig'][k1] = 0.0



            elif dist == 'normal_AR':

                for k0 in range(self.K[0]):

                    # extract parameters
                    mu = theta[k0][feature]['mu']
                    log_sig = theta[k0][feature]['log_sig']
                    sig = np.exp(log_sig)
                    logit_rho = theta[k0][feature]['logit_rho']
                    rho = expit(logit_rho)

                    # get previous data point
                    if t == 0 or np.isnan(self.data[t-1][feature]):
                        y_tm1 = mu
                    else:
                        y_tm1 = self.data[t-1][feature]

                    # get new mean
                    mu_t = rho*y_tm1 + (1.0-rho)*mu

                    # extract gradient
                    if np.isnan(y[feature]):
                        grad_log_f[k0][feature] = {'mu': 0, 'log_sig': 0, 'logit_rho' : 0}
                    else:
                        grad_log_f[k0][feature] = {'mu': (1.0-rho)*(y[feature]-mu_t)/(sig**2),
                                                   'log_sig': ((y[feature]-mu_t)/sig)**2 - 1,
                                                   'logit_rho': rho*(1.0-rho)*(y_tm1-mu)*(y[feature]-mu_t)/(sig**2)}

            elif dist == 'gamma':

                for k0 in range(self.K[0]):

                    # extract parameters
                    log_mu = theta[k0][feature]['log_mu']
                    mu = np.exp(log_mu)
                    log_sig = theta[k0][feature]['log_sig']
                    sig = np.exp(log_sig)
                    alpha = mu**2 / sig**2
                    beta = mu / sig**2

                    # extract gradient
                    if np.isnan(y[feature]):

                        grad_log_f[k0][feature] = {'log_mu': 0, 'log_sig': 0}

                    else:

                        d_logp_d_alpha = np.log(beta) - digamma(alpha) + np.log(y[feature])
                        d_logp_d_beta = alpha / beta - y[feature]

                        d_alpha_d_logmu = 2.0 * alpha
                        d_alpha_d_logsig = -2.0 * alpha

                        d_beta_d_logmu = beta
                        d_beta_d_logsig = -2.0 * beta

                        d_logp_d_logmu = d_logp_d_alpha*d_alpha_d_logmu
                        d_logp_d_logmu += d_logp_d_beta*d_beta_d_logmu

                        d_logp_d_logsig = d_logp_d_alpha*d_alpha_d_logsig
                        d_logp_d_logsig += d_logp_d_beta*d_beta_d_logsig

                        grad_log_f[k0][feature] = {'log_mu': d_logp_d_logmu,
                                                   'log_sig': d_logp_d_logsig}

            elif dist == 'bern':

                for k0 in range(self.K[0]):

                    # extract parameters
                    logit_p = theta[k0][feature]['logit_p']

                    # extract gradient
                    if np.isnan(y[feature]):
                        grad_log_f[k0][feature] = {'logit_p': np.zeros(self.K[1])}
                    elif y[feature] == 0:
                        grad_log_f[k0][feature] = {'logit_p': -expit(logit_p)}
                    elif y[feature] == 1:
                        grad_log_f[k0][feature] = {'logit_p': expit(-logit_p)}
                    else:
                        print("invalid data point %s for %s, which is bernoulli." % (y[feature],feature))

            elif dist.startswith('cat'):

                for k0 in range(self.K[0]):
                    for k1 in range(self.K[1]):

                        # extract parameters
                        ncats = int(dist[3:])
                        psis = [0.0]+[theta[k0][feature]["psi%d"%i][k1] for i in range(1,ncats)]
                        p = np.exp(psis - logsumexp(psis))

                        # extract gradient
                        for i in range(1,ncats):

                            if y[feature] == i:
                                grad_log_f[k0][feature]['psi%d'%i][k1] = 1.0-p[i]
                            else:
                                grad_log_f[k0][feature]['psi%d'%i][k1] = -p[i]

            # error checking
            else:
                print("unidentified emission distribution %s for %s"%(dist,feature))
                return

        # set gradients of fixed values to zero
        for k0 in range(self.K[0]):
            for feature in self.fix_theta[k0]:
                for param in self.fix_theta[k0][feature]:
                    for k1 in range(self.K[1]):
                        if not (self.fix_theta[k0][feature][param][k1] is None):
                            grad_log_f[k0][feature][param][k1] = 0.0

        return grad_log_f

    def get_grad_log_delta(self,eta0=None,eta=None):

        # set eta0 if not given
        if eta0 is None:
            eta0 = self.eta0

        # convert eta0 to delta
        log_delta = eta0_2_log_delta(eta0)
        coarse_delta = np.exp(log_delta[0])
        fine_deltas = [np.exp(log_delta1) for log_delta1 in log_delta[1]]

        # extract gradient of log_delta wrt eta (it is always zero)
        grad_eta_log_delta = [np.zeros((self.K[0],self.K[0],self.K[0],self.K[0])),
                              [np.zeros((self.K[1],self.K[1],self.K[1],self.K[1])) for _ in range(self.K[0])]]

        # extract gradient of log_delta wrt eta
        grad_eta0_log_delta = [np.eye(self.K[0]) - np.tile(coarse_delta,[self.K[0],1]),
                               [np.eye(self.K[1]) - np.tile(fine_delta,[self.K[1],1]) \
                                for fine_delta in fine_deltas]]

        # set some gradients to zero for identifiability
        grad_eta0_log_delta[0][:,0] = 0
        for k0 in range(self.K[0]):
            grad_eta0_log_delta[1][k0][:,0] = 0

        # set gradients to zero that are fixed
        for k0 in range(self.K[0]):

            # coarse-scale delta
            if not (self.fix_eta0[0][k0] is None):
                grad_eta0_log_delta[0][:,k0] = 0.0

            # fine-scale deltas
            for k1 in range(self.K[1]):
                if not (self.fix_eta0[1][k0][k1] is None):
                    grad_eta0_log_delta[1][k0][:,k1] = 0.0

        return grad_eta0_log_delta,grad_eta_log_delta

    def get_grad_log_Gamma(self,eta=None,eta0=None,jump=True):

        # set eta and eta0 if not given
        if eta is None:
            eta = self.eta

        if eta0 is None:
            eta0 = self.eta0

        # extract fine and coarse scale Gammas
        log_Gammas = eta_2_log_Gamma(eta)
        Gammas = [np.exp(log_Gammas[0]),
                  [np.exp(fine_log_Gamma) for fine_log_Gamma in log_Gammas[1]]]

        # extract fine-scale deltas
        log_deltas = eta0_2_log_delta(eta0)
        fine_deltas = [np.exp(log_delta1) for log_delta1 in log_deltas[1]]

        # construct log_Gamma
        K_total = self.K[0] * self.K[1]

        # get the gradient of log_Gamma wrt eta and eta0
        grad_eta_log_Gamma = [np.zeros((self.K[0],self.K[0],self.K[0],self.K[0])),
                              [np.zeros((self.K[1],self.K[1],self.K[1],self.K[1])) for _ in range(self.K[0])]]

        grad_eta0_log_delta = [np.zeros((self.K[0],self.K[0])),
                               [np.zeros((self.K[1],self.K[1])) for _ in range(self.K[0])]]

        # grad due to coarse_Gamma
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                for l in range(self.K[0]):
                    if not jump: # if we aren't jumping then the coarse-scale Gamma doesn't matter
                        pass
                    elif not (self.fix_eta[0][i,l] is None): # skip if eta_il is fixed
                        pass
                    elif i == l: # the diagonals are set to zero
                        pass
                    elif j == l: # d log_Gamma_ij / d eta_il = 1-Gamma_ij
                        grad_eta_log_Gamma[0][i,j,i,l] = 1.0-Gammas[0][i,l]
                    else: # d log_Gamma_ij / d eta_il = 1-Gamma_il
                        grad_eta_log_Gamma[0][i,j,i,l] = -Gammas[0][i,l]

        # grad due to fine_Gamma
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                for j in range(self.K[1]):
                    for l in range(self.K[1]):
                        if jump: # if we are jumping then the fine-scale Gamma doesn't matter
                            pass
                        elif not (self.fix_eta[1][k0][i,l] is None): # skip if eta_il is fixed
                            pass
                        elif i == l: # the diagonals are set to zero
                            pass
                        elif j == l: # d log_Gamma_ij / d eta_ij = 1-Gamma_ij
                            grad_eta_log_Gamma[1][k0][i,j,i,l] = 1.0-Gammas[1][k0][i,l]
                        else: # d log_Gamma_ij / d eta_il = -Gamma_il
                            grad_eta_log_Gamma[1][k0][i,j,i,l] = -Gammas[1][k0][i,l]

        # grad due to fine_delta
        for k0 in range(self.K[0]):
            for k1 in range(self.K[1]):
                if not jump: # if we aren't jumping then the coarse-scale Gamma doesn't matter
                    pass
                elif not (self.fix_eta0[1][k0][k1] is None): # fixed
                    pass
                elif k1 == 0: # first element is always zero
                    pass
                else:
                    grad_eta0_log_delta[1][k0][:,k1] = -fine_deltas[k0][k1]
                    grad_eta0_log_delta[1][k0][k1,k1] += 1.0

        return grad_eta0_log_delta,grad_eta_log_Gamma

    def get_grad_theta_t(self,t,grad_log_p_theta=None,grad_log_f=None,p_Xt=None):

        # initialize gradient
        grad_theta_t = deepcopy(self.grad_theta_t[t])

        # set grad_log_f and p_Xt if not given
        if grad_log_f is None:
            grad_log_f = self.get_grad_log_f(t)
        if p_Xt is None:
            p_Xt = self.p_Xt[t]

        # extract gradient
        for feature,settings in self.features.items():
            for param in self.grad_theta[0][feature]:
                for k0 in range(self.K[0]):
                    grad_theta_t[k0][feature][param] = p_Xt[(self.K[1]*k0):(self.K[1]*(k0+1))] * \
                                                       grad_log_f[k0][feature][param]

        # combine shared parameters
        for share_param in self.share_params:
            new_grad = 0.0
            for feature,param,k0,k1 in product(*share_param.values()):
                new_grad += grad_theta_t[k0][feature][param][k1]
            for feature,param,k0,k1 in product(*share_param.values()):
                grad_theta_t[k0][feature][param][k1] = np.copy(new_grad)

        return grad_theta_t

    def get_grad_eta_t(self,t,grad_eta0_log_delta=None,grad_eta_log_Gamma=None,grad_log_p_eta0=None,grad_log_p_eta=None,p_Xtm1_Xt=None):

        # initialize gradients
        grad_eta_t = [np.zeros((self.K[0],self.K[0])),
                      [np.zeros((self.K[1],self.K[1])) for _ in range(self.K[0])]]

        grad_eta0_t = [np.zeros(self.K[0]),
                       [np.zeros(self.K[1]) for _ in range(self.K[0])]]

        # get initial and final indices
        seq_num = np.argmax(self.initial_ts > t)-1
        t0 = self.initial_ts[seq_num]
        tf = self.final_ts[seq_num]

        # if t == t0, we use a seperate gradient procedure:
        if t == t0:

            # get gradient of log delta wrt eta0
            if grad_eta0_log_delta is None:
                grad_eta0_log_delta,_ = self.get_grad_log_delta()

            p_Xt = self.p_Xt[t]

            # add coarse-scale delta
            p_Xt_coarse = [np.sum(p_Xt[(self.K[1]*k0):(self.K[1]*(k0+1))]) for k0 in range(self.K[0])]
            for k0 in range(self.K[0]):
                grad_eta0_t[0][k0] = np.sum(p_Xt_coarse * grad_eta0_log_delta[0][:,k0])

            # add fine-scale delta
            for k0 in range(self.K[0]):
                p_Xt_fine = p_Xt[(self.K[1]*k0):(self.K[1]*(k0+1))]
                for k1 in range(self.K[1]):
                    grad_eta0_t[1][k0][k1] += np.sum(p_Xt_fine * grad_eta0_log_delta[1][k0][:,k1])

            return grad_eta_t,grad_eta0_t


        # get gradients of log Gamma wrt eta and eta0
        if (grad_eta0_log_delta is None) or (grad_eta_log_Gamma is None):
            if t in self.jump_inds:
                grad_eta0_log_delta,grad_eta_log_Gamma = self.get_grad_log_Gamma(jump=True)
            else:
                grad_eta0_log_delta,grad_eta_log_Gamma = self.get_grad_log_Gamma(jump=False)

        # get p_Xtm1_Xt if not given
        if p_Xtm1_Xt is None:
            p_Xtm1_Xt = self.p_Xtm1_Xt[t]

        # get coarse-scale probs
        p_Xtm1_Xt_coarse = np.zeros((self.K[0],self.K[0]))
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                p_Xtm1_Xt_coarse[i,j] = np.sum(p_Xtm1_Xt[(self.K[1]*i):(self.K[1]*(i+1)),
                                                         (self.K[1]*j):(self.K[1]*(j+1))])

        # add gradient from fine-scale delta
        for k0 in range(self.K[0]):
            p_Xt_fine = np.sum(p_Xtm1_Xt[:,(self.K[1]*k0):(self.K[1]*(k0+1))],0)
            for k1 in range(self.K[1]):
                grad_eta0_t[1][k0][k1] += np.sum(p_Xt_fine * grad_eta0_log_delta[1][k0][:,k1])

        # add gradient from coarse-scale Gamma
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                grad_eta_t[0][i,j] += np.sum(p_Xtm1_Xt_coarse * grad_eta_log_Gamma[0][:,:,i,j])

        # add gradient from fine-scale Gamma
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                for j in range(self.K[1]):
                    grad_eta_t[1][k0][i,j] += np.sum(p_Xtm1_Xt[(self.K[1]*k0):(self.K[1]*(k0+1)),
                                                               (self.K[1]*k0):(self.K[1]*(k0+1))] * \
                                                     grad_eta_log_Gamma[1][k0][:,:,i,j])

        return grad_eta_t,grad_eta0_t

    def E_step(self):

        # update log_alphas
        for t in range(self.T):
            self.update_alpha(t)

        # update log_betas
        for t in reversed(range(self.T)):
            self.update_beta(t)

        # initialize gradients
        self.initialize_grads()

        # update probs and gradients
        for t in range(self.T):

            # update weights
            self.update_p_Xt(t)
            seq_num = np.argmax(self.initial_ts > t)-1
            t0 = self.initial_ts[seq_num]
            if t != t0:
                self.update_p_Xtm1_Xt(t)

            # update gradient at t wrt theta
            self.grad_theta_t[t] = self.get_grad_theta_t(t)

            # update full gradient wrt theta
            for k0 in range(self.K[0]):
                for feature in self.grad_theta[k0]:
                    for param in self.grad_theta[k0][feature]:
                        self.grad_theta[k0][feature][param] += \
                        self.grad_theta_t[t][k0][feature][param]

            # update gradient at t wrt eta
            self.grad_eta_t[t],self.grad_eta0_t[t] = self.get_grad_eta_t(t)

            # update full gradient wrt eta
            self.grad_eta[0] += self.grad_eta_t[t][0]
            self.grad_eta0[0] += self.grad_eta0_t[t][0]
            for k0 in range(self.K[0]):
                self.grad_eta[1][k0] += self.grad_eta_t[t][1][k0]
                self.grad_eta0[1][k0] += self.grad_eta0_t[t][1][k0]

        return

    def grad_params_2_xprime(self):

        # update parameters
        xprime = np.zeros_like(self.x)
        ind = 0

        # update theta
        for share_param in self.share_params:
            for feature,param,k0,k1 in product(*share_param.values()):
                xprime[ind] = np.copy(self.grad_theta[k0][feature][param][k1])
            ind += 1

        # update eta coarse
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                if i == j:
                    xprime[ind] = 0.0
                else:
                    xprime[ind] = self.grad_eta[0][i,j]
                ind += 1

        # update eta fine
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                for j in range(self.K[1]):
                    if i == j:
                        xprime[ind] = 0.0
                    else:
                        xprime[ind] = self.grad_eta[1][k0][i,j]
                    ind += 1

        # update eta0 coarse
        for i in range(self.K[0]):
            if i == 0:
                xprime[ind] = 0.0
            else:
                xprime[ind] = self.grad_eta0[0][i]
            ind += 1

        # update eta0 fine
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                if i == 0:
                    xprime[ind] = 0.0
                else:
                    xprime[ind] = self.grad_eta0[1][k0][i]
                ind += 1

        return xprime

    def callback(self, x):

        # callback to terminate if max time or max number of epochs exceeded
        if self.train_time >= self.max_time:
            raise RuntimeError("Terminating optimization: time limit reached")
        elif self.epoch_num >= self.num_epochs:
            raise RuntimeError("Terminating optimization: epoch limit reached")
        else:
            print("starting epoch %.1f" % (self.epoch_num))
            print("")

            print("%.3f hours elapsed" % (self.train_time / 3600))
            print("")

            # show log likelihood
            print("current log-likelihood:")
            print(self.ll)
            print("")

            # show current parameters
            print("current parameters:")
            print(self.theta)
            print(self.eta)
            print(self.eta0)
            print("")

            # show current gradients
            print("current gradients:")
            print(self.grad_theta)
            print(self.grad_eta)
            print(self.grad_eta0)
            print("")

            # record time and epoch from E step
            self.train_time += time.time() - self.start_time
            self.epoch_num += 1.0
            self.time_trace.append(self.train_time)
            self.epoch_trace.append(self.epoch_num)

            # record trace
            self.log_like_trace.append(self.ll / self.T)
            self.grad_norm_trace.append(np.linalg.norm(self.xprime / self.T))
            self.theta_trace.append(deepcopy(self.theta))
            self.eta_trace.append(deepcopy(self.eta))
            self.eta0_trace.append(deepcopy(self.eta0))

            # start timer back up
            self.start_time = time.time()

    def train_HHMM(self,num_epochs=None,max_time=np.infty,method="BFGS",tol=1e-16,gtol=1e-16):

        # set options for scipy optimizor
        options = {'maxiter':num_epochs,'disp':True}

        # set options for termination
        self.max_time = max_time
        self.num_epochs = num_epochs

        # record training time
        self.start_time = time.time()
        self.train_time = 0.0

        def loss_fn(x):

            # set parameters
            self.x_2_params(x)

            # do an E-step to get the likelihood and the gradient
            self.E_step()

            # get likelihood
            self.ll = logsumexp(self.log_alphas[self.T-1])
            print(self.ll)

            # get gradient
            self.xprime = self.grad_params_2_xprime()

            return (-self.ll/self.T,-self.xprime/self.T)

        # initialize x0
        x0 = self.params_2_x()
        print(x0)

        # fit x
        try:
            res = minimize(loss_fn, x0,
                           method=method, tol=tol,
                           options=options, jac=True,
                           callback=self.callback)
        except RuntimeError:
            print("Terminating optimization: time or epoch limit reached")

        return
