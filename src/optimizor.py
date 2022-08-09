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
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import LinearConstraint
from scipy.signal import convolve
from scipy.interpolate import interp1d

from copy import deepcopy

import time
import pickle

import sys

def eta_2_Gamma(eta):

    # get coarse-scale Gamma
    Gamma_coarse = np.exp(eta[0])
    Gamma_coarse = (Gamma_coarse.T/np.sum(Gamma_coarse,1)).T

    # get fine-scale Gammas
    Gammas_fine = []
    for eta_fine in eta[1]:
        Gamma_fine = np.exp(eta_fine)
        Gammas_fine.append((Gamma_fine.T/np.sum(Gamma_fine,1)).T)

    return [Gamma_coarse,Gammas_fine]

def Gamma_2_eta(Gamma):

    # get Coarse-scale eta
    eta_coarse = np.zeros_like(Gamma[0])
    N = len(Gamma[0])
    for i in range(N):
        eta_coarse[i] = np.log(Gamma[0][i]) - np.log(Gamma[0][i,i])

    # get fine-scale eta
    etas_fine = []
    N = len(Gamma[1][0])
    for Gamma_fine in Gamma[1]:
        eta_fine = np.zeros_like(Gamma_fine)
        for i in range(N):
            eta_fine[i] = np.log(Gamma_fine[i]) - np.log(Gamma_fine[i,i])
        etas_fine.append(eta_fine)

    return [eta_coarse,etas_fine]

def eta0_2_delta(eta0):

    # get coarse-scale delta
    delta_coarse = np.exp(eta0[0])
    delta_coarse = delta_coarse/np.sum(delta_coarse)

    # get fine-scale Gammas
    deltas_fine = []
    for eta0_fine in eta0[1]:
        delta_fine = np.exp(eta0_fine)
        deltas_fine.append(delta_fine/np.sum(delta_fine))

    return [delta_coarse,deltas_fine]

def delta_2_eta0(delta):

    # get coarse-scale eta0
    eta0_coarse = np.log(delta[0]) - np.log(delta[0][0])

    # get fine-scale eta
    eta0s_fine = []
    for delta_fine in delta[1]:
        eta0_fine = np.log(delta_fine) - np.log(delta_fine[0])
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

class optimizor:

    def __init__(self,data,features,K):

        '''
        constructor for optimizor class
        '''

        self.data = data

        # parameters
        self.K = K
        self.K_total = self.K[0]*self.K[1]
        self.T = len(data)
        self.jump_every = 1

        # define features
        self.features = features

        # thetas and etas
        self.initialize_theta(data)
        self.initialize_eta()
        self.initialize_eta0()

        # get Gamma and delta
        self.log_Gamma = self.get_log_Gamma(jump=False)[0]
        self.log_Gamma_jump = self.get_log_Gamma(jump=True)[0]
        self.log_delta = self.get_log_delta()[0]

        self.theta_tilde = deepcopy(self.theta)
        self.eta_tilde = deepcopy(self.eta)
        self.eta0_tilde = deepcopy(self.eta0)

        self.theta_trace = []
        self.eta_trace = []
        self.eta0_trace = []

        # log-likelihood traces
        self.log_like_trace = []
        self.grad_norm_trace = []

        # time traces
        self.time_trace = []
        self.epoch_trace = []

        # alpha and beta
        self.log_alphas = np.zeros((self.T,self.K_total))
        self.log_betas = np.zeros((self.T,self.K_total))

        # p_Xt and p_Xtm1_Xt
        self.p_Xt = np.zeros((self.T,self.K_total))
        self.p_Xtm1_Xt = np.zeros((self.T,self.K_total,self.K_total))

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

        # copies for SVRG
        self.grad_theta_tilde = deepcopy(self.grad_theta)
        self.grad_eta_tilde = deepcopy(self.grad_eta)
        self.grad_eta0_tilde = deepcopy(self.grad_eta0)

        self.p_Xt_tilde = deepcopy(self.p_Xt)
        self.p_Xtm1_Xt_tilde = deepcopy(self.p_Xtm1_Xt)

        self.theta_tilde = deepcopy(self.theta)
        self.eta_tilde = deepcopy(self.eta)
        self.eta0_tilde = deepcopy(self.eta0)

        # time to train
        self.train_time = None
        self.start_time = None
        self.epoch_num = None

        # initialize step sizes and parameter bounds
        self.step_size = None
        self.param_bounds = [{feature: {} for feature in self.theta[0]},
                             {feature: {} for feature in self.theta[1][0]}]
        self.initialize_step_size()

        # Lipshitz constants
        if self.step_size != 0:
            self.L_theta = 1.0/(3.0*self.step_size)
            self.L_eta = 1.0/(3.0*self.step_size)
        else:
            self.L_theta = np.infty
            self.L_eta = np.infty

        return

    def initialize_theta(self,data):

        theta = []

        # first fill in the coarse-scale stuff
        K = self.K[0]
        theta.append({})

        for feature,settings in self.features[0].items():

            # initialize values
            theta[0][feature] = {}

            if data is not None:

                feature_data = [datum[feature] for datum in data]

                if settings['f'] == 'normal':

                    theta[0][feature]['mu'] = np.ones(K)*np.nanmean(feature_data)
                    theta[0][feature]['log_sig'] = np.log(np.ones(K)*max(0.1,np.nanstd(feature_data)))

                    theta[0][feature]['mu'] += norm.rvs(np.zeros(K),np.exp(theta[0][feature]['log_sig']))
                    theta[0][feature]['log_sig'] += norm.rvs(0.0,1.0,size=K)

                elif settings['f'] == 'bern':

                    theta[0][feature]['logit_p'] = np.ones(K)*logit(np.nanmean(feature_data))
                    theta[0][feature]['logit_p'] += norm.rvs(0.0,1.0,size=K)

                else:
                    pass

        # then fill in the fine-scale stuff
        theta.append([{} for _ in range(K)])
        K = self.K[1]

        for feature,settings in self.features[1].items():
            for k0 in range(self.K[0]):

                # initialize values
                theta[1][k0][feature] = {}

                if data is not None:

                    feature_data = [datum[feature] for datum in data]

                    # add randomness in initialization
                    if settings['f'] == 'normal':

                        theta[1][k0][feature]['mu'] = np.nanmean(feature_data)*np.ones(K)
                        theta[1][k0][feature]['log_sig'] = np.log(np.nanstd(feature_data)*np.ones(K))

                        theta[1][k0][feature]['mu'] += norm.rvs(np.zeros(K),np.exp(theta[1][k0][feature]['log_sig']))
                        theta[1][k0][feature]['log_sig'] += norm.rvs(0.0,1.0,size=K)

                    elif settings['f'] == 'bern':

                        theta[1][k0][feature]['logit_p'] = np.ones(K)*logit(np.nanmean(feature_data))
                        theta[1][k0][feature]['logit_p'] += norm.rvs(0.0,1.0,size=K)

                    else:
                        pass

        self.theta = theta

        return

    def initialize_eta(self):

        # initialize eta
        self.eta = []

        # fill in coarse eta
        eta_crude = -1.0 + np.random.normal(size=(self.K[0],self.K[0]))
        for i in range(self.K[0]):
            eta_crude[i,i] = 0
        self.eta.append(eta_crude)

        # fill in fine eta
        eta_fine = []
        for _ in range(self.K[0]):
            eta_fine_k = -1.0 + np.random.normal(size=(self.K[1],self.K[1]))
            for i in range(self.K[1]):
                eta_fine_k[i,i] = 0
            eta_fine.append(eta_fine_k)

        self.eta.append(eta_fine)

        return

    def initialize_eta0(self):

        # initialize eta0
        self.eta0 = [np.random.normal(size=self.K[0]),
                     [np.random.normal(size=self.K[1]) for _ in range(self.K[0])]]

        return

    def initialize_grads(self):

        # overall gradient wrt theta
        for feature,dist in self.features[0].items():
            if dist['f'] == 'normal':

                self.grad_theta[0][feature]['mu'] = np.zeros(self.K[0])
                self.grad_theta[0][feature]['log_sig'] = np.zeros(self.K[0])

            elif dist['f'] == 'bern':

                self.grad_theta[0][feature]['logit_p'] = np.zeros(self.K[0])

            else:
                raise('only independent normal distributions supported at this time')


        for feature,dist in self.features[1].items():
            if dist['f'] == 'normal':
                for k0 in range(self.K[0]):

                    self.grad_theta[1][k0][feature]['mu'] = np.zeros(self.K[1])
                    self.grad_theta[1][k0][feature]['log_sig'] = np.zeros(self.K[1])

            elif dist['f'] == 'bern':
                for k0 in range(self.K[0]):

                    self.grad_theta[1][k0][feature]['logit_p'] = np.zeros(self.K[1])

            else:
                raise('only independent normal distributions supported at this time')

        # get overall gradient wrt eta
        self.grad_eta = [np.zeros((self.K[0],self.K[0])),
                         [np.zeros((self.K[1],self.K[1])) for _ in range(self.K[0])]]

        # get overall gradient wrt eta0
        self.grad_eta0 = [np.zeros(self.K[0]),
                          [np.zeros(self.K[1]) for _ in range(self.K[0])]]

        for t in range(self.T):

            # get table values wrt eta
            self.grad_eta_t[t] = [np.zeros((self.K[0],self.K[0])),
                                  [np.zeros((self.K[1],self.K[1])) for _ in range(self.K[0])]]

            # get table values wrt eta0
            self.grad_eta0_t[t] = [np.zeros(self.K[0]),
                                   [np.zeros(self.K[1]) for _ in range(self.K[0])]]

            # get table values wrt theta
            for feature,dist in self.features[0].items():
                if dist['f'] == 'normal':

                    self.grad_theta_t[t][0][feature]['mu'] = np.zeros(self.K[0])
                    self.grad_theta_t[t][0][feature]['log_sig'] = np.zeros(self.K[0])

                elif dist['f'] == 'bern':

                    self.grad_theta_t[t][0][feature]['logit_p'] = np.zeros(self.K[0])

                else:
                    raise('only independent normal distributions supported at this time')

            for feature,dist in self.features[1].items():
                if dist['f'] == 'normal':
                    for k0 in range(self.K[0]):

                        self.grad_theta_t[t][1][k0][feature]['mu'] = np.zeros(self.K[1])
                        self.grad_theta_t[t][1][k0][feature]['log_sig'] = np.zeros(self.K[1])

                elif dist['f'] == 'bern':
                    for k0 in range(self.K[0]):

                        self.grad_theta_t[t][1][k0][feature]['logit_p'] = np.zeros(self.K[1])

                else:
                    raise('only independent normal distributions supported at this time')

        return

    def initialize_step_size(self,min_group_size=None):

        self.step_size = 0.01

        return

    def get_log_f(self,t,theta=None):

        if theta is None:
            theta = self.theta

        # define the data point
        y = self.data[t]

        # initialize the log-likelihood
        log_f = np.zeros(self.K_total)

        # initialize the gradient
        grad_log_f = [{},[{} for _ in range(self.K[0])]]

        # go through each feature and add to log-likelihood
        for feature,value in y.items():

            # coarse-scale features
            if feature in self.features[0]:

                # define the feature
                dist = self.features[0][feature]['f']

                if dist == 'normal':

                    # store log-likelihood
                    mu = np.repeat(theta[0][feature]['mu'],self.K[1])
                    log_sig = np.repeat(theta[0][feature]['log_sig'],self.K[1])
                    sig = np.exp(log_sig)

                    a = self.features[0][feature]['lower_bound']
                    b = self.features[0][feature]['upper_bound']

                    if np.isnan(y[feature]):
                        pass
                    elif (a is None) or (b is None):
                        log_f += norm.logpdf(y[feature],loc=mu,scale=sig)
                    else:
                        a = np.repeat(a,self.K[1])
                        b = np.repeat(b,self.K[1])
                        log_f += truncnorm.logpdf(y[feature],
                                                  a=(a-mu)/sig,
                                                  b=(b-mu)/sig,
                                                  loc=mu,scale=sig)


                    # store gradient
                    mu = theta[0][feature]['mu']
                    sig = np.exp(theta[0][feature]['log_sig'])

                    if np.isnan(y[feature]):
                        grad_log_f[0][feature] = {'mu': 0, 'log_sig': 0}
                    else:
                        grad_log_f[0][feature] = {'mu': (y[feature]-mu)/(sig**2),
                                                  'log_sig': ((y[feature]-mu)/sig)**2 - 1}

                        # deal with truncated normals
                        a = self.features[0][feature]['lower_bound']
                        b = self.features[0][feature]['upper_bound']
                        if (a is not None) and (b is not None):
                            for k0 in range(self.K[0]):

                                if a[k0] > -np.infty and b[k0] < np.infty:
                                    print("cannot handle two-sided truncated normals at this time.")

                                elif a[k0] > -np.infty:

                                    ratio = norm.pdf(a,loc=mu,scale=sig)/norm.sf(a,loc=mu,scale=sig)

                                    if y[feature] >= np.array(a[k1]):
                                        grad_log_f[0][feature]['mu'][k0] += -ratio
                                        grad_log_f[0][feature]['log_sig'][k0] += -(a-mu)*(ratio)
                                    else:
                                        grad_log_f[0][feature]['mu'][k0] = 0.0
                                        grad_log_f[0][feature]['log_sig'][k0] = 0.0

                                elif b[k1] < np.infty:

                                    ratio = norm.pdf(b,loc=mu,scale=sig)/norm.cdf(b,loc=mu,scale=sig)

                                    if y[feature] <= np.array(b[k1]):
                                        grad_log_f[0][feature]['mu'][k0] += ratio
                                        grad_log_f[0][feature]['log_sig'][k0] += (b-mu)*(ratio)
                                    else:
                                        grad_log_f[0][feature]['mu'][k0] = 0.0
                                        grad_log_f[0][feature]['log_sig'][k0] = 0.0


                elif dist == 'bern':

                    # store log-likelihood
                    if np.isnan(y[feature]):
                        pass
                    elif y[feature] == 0:
                        log_p = np.log(expit(np.repeat(-theta[0][feature]['logit_p'],self.K[1])))
                        log_f += log_p
                    elif y[feature] == 1:
                        log_p = np.log(expit(np.repeat(theta[0][feature]['logit_p'],self.K[1])))
                        log_f += log_p
                    else:
                        print("invalid data point %s for %s, which is bernoulli"%(y[feature],feature))

                    # store gradient
                    logit_p = theta[0][feature]['logit_p']

                    if np.isnan(y[feature]):
                        grad_log_f[0][feature] = {'logit_p': 0}
                    elif y[feature] == 0:
                        grad_log_f[0][feature] = {'logit_p': -expit(logit_p)}
                    elif y[feature] == 1:
                        grad_log_f[0][feature] = {'logit_p': expit(-logit_p)}
                    else:
                        print("invalid data point %s for %s, which is bernoulli"%(y[feature],feature))

                else:
                    print("unidentified emission distribution %s for %s"%(dist,feature))
                    return

            elif feature in self.features[1]:

                # define the feature
                dist = self.features[1][feature]['f']

                if dist == 'normal':

                    # store log-likelihood
                    mu = np.concatenate([theta_fine[feature]['mu'] for theta_fine in theta[1]])
                    log_sig = np.concatenate([theta_fine[feature]['log_sig'] for theta_fine in theta[1]])
                    sig = np.exp(log_sig)

                    a = self.features[1][feature]['lower_bound']
                    b = self.features[1][feature]['upper_bound']

                    if np.isnan(y[feature]):
                        pass
                    elif (a is None) or (b is None):
                        log_f += norm.logpdf(y[feature],loc=mu,scale=sig)
                    else:
                        a = np.concatenate([a for _ in theta[1]])
                        b = np.concatenate([b for _ in theta[1]])

                        log_f += truncnorm.logpdf(y[feature],
                                                  a=(a-mu)/sig,
                                                  b=(b-mu)/sig,
                                                  loc=mu,scale=sig)


                    # store gradient
                    for k0 in range(self.K[0]):

                        mu = theta[1][k0][feature]['mu']
                        log_sig = theta[1][k0][feature]['log_sig']
                        sig = np.exp(log_sig)

                        if np.isnan(y[feature]):
                            grad_log_f[1][k0][feature] = {'mu': 0, 'log_sig': 0}
                        else:
                            grad_log_f[1][k0][feature] = {'mu': (y[feature]-mu)/(sig**2),
                                                          'log_sig': ((y[feature]-mu)/sig)**2 - 1}

                            # deal with truncated normals
                            a = self.features[1][feature]['lower_bound']
                            b = self.features[1][feature]['upper_bound']

                            if (a is not None) and (b is not None):

                                ratio_a = norm.pdf(a,loc=mu,scale=sig)/norm.sf(a,loc=mu,scale=sig)
                                ratio_b = norm.pdf(b,loc=mu,scale=sig)/norm.cdf(b,loc=mu,scale=sig)

                                for k1 in range(self.K[1]):

                                    if a[k1] > -np.infty and b[k1] < np.infty:
                                        print("cannot handle two-sided truncated normals at this time.")

                                    elif a[k1] > -np.infty:

                                        if y[feature] >= np.array(a[k1]):
                                            grad_log_f[1][k0][feature]['mu'][k1] += -ratio_a[k1]
                                            grad_log_f[1][k0][feature]['log_sig'][k1] += -(a[k1]-mu[k1])*ratio_a[k1]
                                        else:
                                            grad_log_f[1][k0][feature]['mu'][k1] = 0.0
                                            grad_log_f[1][k0][feature]['log_sig'][k1] = 0.0

                                    elif b[k1] < np.infty:

                                        if y[feature] <= np.array(b[k1]):
                                            grad_log_f[1][k0][feature]['mu'][k1] += ratio_b[k1]
                                            grad_log_f[1][k0][feature]['log_sig'][k1] += (b[k1]-mu[k1])*ratio_b[k1]
                                        else:
                                            grad_log_f[1][k0][feature]['mu'][k1] = 0.0
                                            grad_log_f[1][k0][feature]['log_sig'][k1] = 0.0

                elif dist == 'bern':

                    # store log-likelihood
                    if np.isnan(y[feature]):
                        pass
                    elif y[feature] == 0:
                        log_p = np.log(expit(np.concatenate([-theta_fine[feature]['logit_p'] for theta_fine in theta[1]])))
                        log_f += log_p
                    elif y[feature] == 1:
                        log_p = np.log(expit(np.concatenate([theta_fine[feature]['logit_p'] for theta_fine in theta[1]])))
                        log_f += log_p
                    else:
                        print("invalid data point %s for %s, which is bernoulli"%(y[feature],feature))

                    # store gradient
                    for k0 in range(self.K[0]):

                        logit_p = theta[1][k0][feature]['logit_p']

                        if np.isnan(y[feature]):
                            grad_log_f[1][k0][feature] = {'logit_p': 0}
                        elif y[feature] == 0:
                            grad_log_f[1][k0][feature] = {'logit_p': -expit(logit_p)}
                        elif y[feature] == 1:
                            grad_log_f[1][k0][feature] = {'logit_p': expit(-logit_p)}
                        else:
                            print("invalid data point %s for %s, which is bernoulli"%(y[feature],feature))

                else:
                    print("unidentified emission distribution %s for %s"%(dist,feature))
                    return

            else:
                print("unidentified feature in y: %s" % feature)
                return

        # return the result
        return log_f,grad_log_f

    def get_log_delta(self,eta0=None):

        if eta0 is None:
            eta0 = self.eta0

        # get delta
        log_coarse_delta = np.repeat(eta0[0] - logsumexp(eta0[0]),self.K[1])
        log_fine_delta = np.concatenate([eta01 for eta01 in eta0[1]])
        log_delta = log_coarse_delta + log_fine_delta
        self.log_delta = np.copy(log_delta)

        # get the gradient
        coarse_delta = np.exp(eta0[0]) / np.sum(np.exp(eta0[0]))
        fine_deltas = [np.exp(eta01) / np.sum(np.exp(eta01)) for eta01 in eta0[1]]

        grad_eta0_log_delta = [np.eye(self.K[0]) - np.tile(coarse_delta,[self.K[0],1]),
                               [np.eye(self.K[1]) - np.tile(fine_delta,[self.K[1],1]) \
                                for fine_delta in fine_deltas]]

        # set some gradients to zero for identifiability
        grad_eta0_log_delta[0][:,0] = 0
        for k0 in range(self.K[0]):
            grad_eta0_log_delta[1][k0][:,0] = 0

        return log_delta,grad_eta0_log_delta

    def get_log_Gamma(self,eta=None,eta0=None,jump=True):

        if eta is None:
            eta = self.eta

        if eta0 is None:
            eta0 = self.eta0

        # get fine and coarse scale Gammas
        Gammas = eta_2_Gamma(eta)
        log_coarse_Gamma = np.log(Gammas[0])
        log_fine_Gammas = [np.log(fine_Gamma) for fine_Gamma in Gammas[1]]

        # get the fine-scale deltas
        fine_deltas = [np.exp(eta0i) / np.sum(np.exp(eta0i)) for eta0i in eta0[1]]
        log_fine_deltas = [eta0i - logsumexp(eta0i) for eta0i in eta0[1]]

        # construct log_Gamma
        K_total = self.K[0] * self.K[1]
        log_Gamma = np.zeros((K_total,K_total))

        for i in range(self.K[0]):
            for j in range(self.K[0]):

                # add fine-scale Gamma
                if i == j:
                    log_Gamma[(self.K[1]*i):(self.K[1]*(i+1)),
                              (self.K[1]*j):(self.K[1]*(j+1))] += log_fine_Gammas[j]

                # add coarse-scale Gamma and fine-scale delta if k == 0
                if jump:
                    log_Gamma[(self.K[1]*i):(self.K[1]*(i+1)),
                              (self.K[1]*j):(self.K[1]*(j+1))] += log_coarse_Gamma[i,j]
                    if i != j:
                        log_Gamma[(self.K[1]*i):(self.K[1]*(i+1)),
                                  (self.K[1]*j):(self.K[1]*(j+1))] += np.tile(log_fine_deltas[j],[self.K[1],1])
                else:
                    if i != j:
                        log_Gamma[(self.K[1]*i):(self.K[1]*(i+1)),
                                  (self.K[1]*j):(self.K[1]*(j+1))] += -np.infty

        # get the gradient of log_Gamma wrt eta and eta0
        grad_eta_log_Gamma = [np.zeros((self.K[0],self.K[0],self.K[0],self.K[0])),
                              [np.zeros((self.K[1],self.K[1],self.K[1],self.K[1])) for _ in range(self.K[0])]]

        grad_eta0_log_delta = [np.zeros((self.K[0],self.K[0])),
                               [np.zeros((self.K[1],self.K[1])) for _ in range(self.K[0])]]

        # grad from coarse_Gamma
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                for l in range(self.K[0]):
                    if not jump:
                        pass
                    elif i == l:
                        pass
                    elif j == l:
                        grad_eta_log_Gamma[0][i,j,i,l] = 1.0-Gammas[0][i,l]
                    else:
                        grad_eta_log_Gamma[0][i,j,i,l] = -Gammas[0][i,l]

        # grad from fine_Gamma
        for n in range(self.K[0]):
            for i in range(self.K[1]):
                for j in range(self.K[1]):
                    for l in range(self.K[1]):
                        if i == l:
                            pass
                        elif j == l:
                            grad_eta_log_Gamma[1][n][i,j,i,l] = 1.0-Gammas[1][n][i,l]
                        else:
                            grad_eta_log_Gamma[1][n][i,j,i,l] = -Gammas[1][n][i,l]

        # grad from fine_delta
        for k0 in range(self.K[0]):
            if not jump:
                pass
            else:
                grad_eta0_log_delta[1][k0] = np.eye(self.K[1]) - np.tile(fine_deltas[k0],[self.K[1],1])
                grad_eta0_log_delta[1][k0][:,0] = 0

        return log_Gamma,grad_eta0_log_delta,grad_eta_log_Gamma

    def update_alpha(self,t):

        # get log_f, grad_log_f
        log_f = self.get_log_f(t)[0]

        # update log_alpha
        if t == 0:
            self.log_alphas[t] = self.log_delta + log_f
        elif t % self.jump_every == 0:
            self.log_alphas[t] = logdotexp(self.log_alphas[t-1],self.log_Gamma_jump) + log_f
        else:
            self.log_alphas[t] = logdotexp(self.log_alphas[t-1],self.log_Gamma) + log_f

        return

    def update_beta(self,t):

        # update log_beta
        if t == self.T-1:
            self.log_betas[t] = np.zeros(self.K_total)
        elif (t % self.jump_every == 1) or (self.jump_every == 1):
            log_tp1 = self.get_log_f(t+1)[0]
            self.log_betas[t] = logdotexp(self.log_betas[t+1] + log_tp1,
                                          np.transpose(self.log_Gamma_jump))
        else:
            log_tp1 = self.get_log_f(t+1)[0]
            self.log_betas[t] = logdotexp(self.log_betas[t+1] + log_tp1,
                                          np.transpose(self.log_Gamma))
        return

    def update_p_Xt(self,t):

        ll = logsumexp(self.log_alphas[t] + self.log_betas[t])
        self.p_Xt[t] = np.exp(self.log_alphas[t] + self.log_betas[t] - ll)

        return

    def update_p_Xtm1_Xt(self,t,log_t=None):

        if t == 0:
            raise("p_Xtm1_Xt not defined for t = 0")

        if log_t is None:
            log_t = self.get_log_f(t)[0]

        if t % self.jump_every == 0:
            log_Gamma = self.log_Gamma_jump
        else:
            log_Gamma = self.log_Gamma

        p_XX = np.zeros((self.K_total,self.K_total))
        for i in range(self.K_total):
            for j in range(self.K_total):
                p_XX[i,j] = self.log_alphas[t-1,i] \
                            + log_Gamma[i,j] \
                            + log_t[j] \
                            + self.log_betas[t,j]

        self.p_Xtm1_Xt[t] = np.exp(p_XX - logsumexp(p_XX))

        return

    def get_grad_theta_t(self,t,grad_log_f=None,p_Xt=None):

        # initialize gradient
        grad_theta_t = deepcopy(self.grad_theta_t[t])

        # get gradient and weights
        if grad_log_f is None:
            grad_log_f = self.get_log_f(t)[1]
        if p_Xt is None:
            p_Xt = self.p_Xt[t]

        # calculate coarse-scale gradient
        p_Xt_coarse = [np.sum(p_Xt[(self.K[1]*k0):(self.K[1]*(k0+1))]) for k0 in range(self.K[0])]
        for feature in grad_log_f[0]:
            for param in grad_log_f[0][feature]:
                grad_theta_t[0][feature][param] = p_Xt_coarse * grad_log_f[0][feature][param]

        # calculate fine-scale gradient
        for k0 in range(self.K[0]):
            for feature in grad_log_f[1][k0]:
                for param in grad_log_f[1][k0][feature]:
                    grad_theta_t[1][k0][feature][param] = p_Xt[(self.K[1]*k0):(self.K[1]*(k0+1))] * grad_log_f[1][k0][feature][param]

        return grad_theta_t

    def get_grad_eta_t(self,t,grad_eta0_log_delta=None,grad_eta_log_Gamma=None,p_Xtm1_Xt=None):

        # initialize gradients
        grad_eta_t = [np.zeros((self.K[0],self.K[0])),
                      [np.zeros((self.K[1],self.K[1])) for _ in range(self.K[0])]]

        grad_eta0_t = [np.zeros(self.K[0]),
                       [np.zeros(self.K[1]) for _ in range(self.K[0])]]

        # deal with t == 0:
        if t == 0:

            # get gradient of log delta wrt eta0
            if grad_eta0_log_delta is None:
                _,grad_eta0_log_delta = self.get_log_delta()

            # get probabilities
            p_Xt = self.p_Xt[t]
            p_Xt_coarse = [np.sum(p_Xt[(self.K[1]*k0):(self.K[1]*(k0+1))]) for k0 in range(self.K[0])]

            # add coarse-scale delta
            for i in range(self.K[0]):
                grad_eta0_t[0][i] = np.sum(p_Xt_coarse * grad_eta0_log_delta[0][:,i])

            # add fine-scale delta
            for k0 in range(self.K[0]):
                for i in range(self.K[1]):
                    grad_eta0_t[1][k0][i] += np.sum(p_Xt[(self.K[1]*k0):(self.K[1]*(k0+1))] * \
                                                    grad_eta0_log_delta[1][k0][:,i])

            return grad_eta_t,grad_eta0_t

        # get gradients of log Gamma wrt eta and eta0
        if (grad_eta0_log_delta is None) or (grad_eta_log_Gamma is None):
            if t % self.jump_every == 0:
                _,grad_eta0_log_delta,grad_eta_log_Gamma = self.get_log_Gamma(jump=True)
            else:
                _,grad_eta0_log_delta,grad_eta_log_Gamma = self.get_log_Gamma(jump=False)

        # get weights
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

            # get prob of moving from a different coarse state to given fine
            p_Xt_jump = np.sum(p_Xtm1_Xt[:,(self.K[1]*k0):(self.K[1]*(k0+1))],0)
            p_Xt_jump = p_Xt_jump - np.sum(p_Xtm1_Xt[(self.K[1]*k0):(self.K[1]*(k0+1)),
                                                     (self.K[1]*k0):(self.K[1]*(k0+1))],0)

            for i in range(self.K[1]):
                grad_eta0_t[1][k0][i] += np.sum(p_Xt_jump * grad_eta0_log_delta[1][k0][:,i])


        # add gradient from coarse-scale Gamma
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                grad_eta_t[0][i,j] = np.sum(p_Xtm1_Xt_coarse * grad_eta_log_Gamma[0][:,:,i,j])

        # add gradient from fine-scale Gamma
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                for j in range(self.K[1]):
                    grad_eta_t[1][k0][i,j] += np.sum(p_Xtm1_Xt[(self.K[1]*k0):(self.K[1]*(k0+1)),
                                                               (self.K[1]*k0):(self.K[1]*(k0+1))] * \
                                                       grad_eta_log_Gamma[1][k0][:,:,i,j])

        return grad_eta_t,grad_eta0_t

    def est_grad_norm(self):

        grad_norm2 = 0

        # add gradient wrt eta0
        grad_norm2 += np.sum(self.grad_eta0[0]**2)
        for k0 in range(self.K[0]):
             grad_norm2 += np.sum(self.grad_eta0[1][k0]**2)

        # add gradient wrt eta
        grad_norm2 += np.sum(self.grad_eta[0]**2)
        for k0 in range(self.K[0]):
             grad_norm2 += np.sum(self.grad_eta[1][k0]**2)

        # add gradient wrt theta
        for feature in self.grad_theta[0]:
            for param in self.grad_theta[0][feature]:
                grad_norm2 += np.sum(self.grad_theta[0][feature][param]**2)

        for feature in self.grad_theta[1][0]:
            for param in self.grad_theta[1][0][feature]:
                grad = np.zeros(self.K[1])
                for k0 in range(self.K[0]):

                    # if NOT sharing states
                    grad_norm2 += np.sum(self.grad_theta[1][k0][feature][param]**2)

                    # if sharing states
                    #grad += self.grad_theta[1][k0][feature][param]
                #grad_norm2 += np.sum(grad**2)

        return np.sqrt(grad_norm2)

    def get_log_like(self):

        # store old values
        log_alphas_old = deepcopy(self.log_alphas)
        log_betas_old = deepcopy(self.log_betas)

        p_Xt_old = deepcopy(self.p_Xt)
        p_Xtm1_Xt_old = deepcopy(self.p_Xtm1_Xt)

        grad_theta_old = deepcopy(self.grad_theta)
        grad_eta_old = deepcopy(self.grad_eta)
        grad_eta0_old = deepcopy(self.grad_eta0)

        grad_theta_t_old = deepcopy(self.grad_theta_t)
        grad_eta_t_old = deepcopy(self.grad_eta_t)
        grad_eta0_t_old = deepcopy(self.grad_eta0_t)

        p_Xt_tilde_old = deepcopy(self.p_Xt_tilde)
        p_Xtm1_Xt_tilde_old = deepcopy(self.p_Xtm1_Xt_tilde)

        grad_theta_tilde_old = deepcopy(self.grad_theta_tilde)
        grad_eta_tilde_old = deepcopy(self.grad_eta_tilde)
        grad_eta0_tilde_old = deepcopy(self.grad_eta0_tilde)

        theta_tilde_old = deepcopy(self.theta_tilde)
        eta_tilde_old = deepcopy(self.eta_tilde)
        eta0_tilde_old = deepcopy(self.eta0_tilde)

        # get new likelihood and gradient
        self.E_step(update_tilde = False)
        ll = logsumexp(self.log_alphas[self.T-1])
        grad_norm = self.est_grad_norm()

        # return values to old state
        self.log_alphas = log_alphas_old
        self.log_betas = log_betas_old

        self.p_Xt = p_Xt_old
        self.p_Xtm1_Xt = p_Xtm1_Xt_old

        self.grad_theta = grad_theta_old
        self.grad_eta = grad_eta_old
        self.grad_eta0 = grad_eta0_old

        self.grad_theta_t = grad_theta_t_old
        self.grad_eta_t = grad_eta_t_old
        self.grad_eta0_t = grad_eta0_t_old

        self.p_Xt_tilde = p_Xt_tilde_old
        self.p_Xtm1_Xt_tilde = p_Xtm1_Xt_tilde_old

        self.grad_theta_tilde = grad_theta_tilde_old
        self.grad_eta_tilde = grad_eta_tilde_old
        self.grad_eta0_tilde = grad_eta0_tilde_old

        self.theta_tilde = theta_tilde_old
        self.eta_tilde = eta_tilde_old
        self.eta0_tilde = eta0_tilde_old

        self.log_Gamma = self.get_log_Gamma(jump=False)[0]
        self.log_Gamma_jump = self.get_log_Gamma(jump=True)[0]
        self.log_delta = self.get_log_delta()[0]

        return ll, grad_norm

    def check_L_eta(self,t):

        # get gradient and its norm
        grad_G_eta_t,grad_G_eta0_t = self.get_grad_eta_t(t)
        grad_G_t_norm2 = np.sum(grad_G_eta_t[0]**2)
        for k0 in range(self.K[0]):
            grad_G_t_norm2 += np.sum(grad_G_eta_t[1][k0]**2)
        grad_G_t_norm2 += np.sum(grad_G_eta0_t[0]**2)
        for k0 in range(self.K[0]):
            grad_G_t_norm2 += np.sum(grad_G_eta0_t[1][k0]**2)

        # get new value of eta
        eta_new = deepcopy(self.eta)
        eta_new[0] += grad_G_eta_t[0] / self.L_eta
        for k0 in range(self.K[0]):
            eta_new[1][k0] += grad_G_eta_t[1][k0] / self.L_eta

        # get new value of eta0
        eta0_new = deepcopy(self.eta0)
        eta0_new[0] += grad_G_eta0_t[0] / self.L_eta
        for k0 in range(self.K[0]):
            eta0_new[1][k0] += grad_G_eta0_t[1][k0] / self.L_eta

        # Evaluate G for eta and eta_new
        if t == 0:
            G_t  = -np.sum(self.p_Xt[t] * self.get_log_delta(eta0=self.eta0)[0])
            G_t_new = -np.sum(self.p_Xt[t] * self.get_log_delta(eta0=eta0_new)[0])
        else:
            G_t  = -np.sum(self.p_Xtm1_Xt[t] * self.get_log_Gamma(eta=self.eta,eta0=self.eta0)[0])
            G_t_new = -np.sum(self.p_Xtm1_Xt[t] * self.get_log_Gamma(eta=eta_new,eta0=eta0_new)[0])

        # check inequality
        if grad_G_t_norm2 < 1e-8:
            pass
        elif G_t_new > G_t - grad_G_t_norm2 / (2*self.L_eta):
            self.L_eta *= 2
        else:
            self.L_eta *= 2**(-1/self.T)

        return

    def check_L_theta(self,t):

        # get the gradients at the given time points
        grad_F_t = self.get_grad_theta_t(t)

        # initialize gradient norm and new theta
        grad_F_t_norm2 = 0
        theta_new = deepcopy(self.theta)

        # get new theta and gradient norm
        for feature in grad_F_t[0]:
            for param in grad_F_t[0][feature]:
                theta_new[0][feature][param] += grad_F_t[0][feature][param] / self.L_theta
                grad_F_t_norm2 += np.sum(grad_F_t[0][feature][param]**2)

        for k0 in range(self.K[0]):
            for feature in grad_F_t[1][k0]:
                for param in grad_F_t[1][k0][feature]:
                    theta_new[1][k0][feature][param] += grad_F_t[1][k0][feature][param] / self.L_theta
                    grad_F_t_norm2 += np.sum(grad_F_t[1][k0][feature][param]**2)

        # evaluate F for theta and theta_new
        F_t  = -np.sum(self.p_Xt[t] * self.get_log_f(t,theta=self.theta)[0])
        F_t_new = -np.sum(self.p_Xt[t] * self.get_log_f(t,theta=theta_new)[0])

        # check for inequality
        if grad_F_t_norm2 < 1e-8:
            pass
        elif F_t_new > F_t - grad_F_t_norm2 / (2*self.L_theta):
            self.L_theta *= 2
        else:
            self.L_theta *= 2**(-1/self.T)

        return

    def E_step(self,update_tilde=True,update_probs=True):

        if update_probs:

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

            if update_probs:

                # update theta
                self.update_p_Xt(t)

            self.grad_theta_t[t] = self.get_grad_theta_t(t)

            for feature in self.grad_theta[0]:
                for param in self.grad_theta[0][feature]:
                    self.grad_theta[0][feature][param] += \
                    self.grad_theta_t[t][0][feature][param]

            for k0 in range(self.K[0]):
                for feature in self.grad_theta[1][k0]:
                    for param in self.grad_theta[1][k0][feature]:
                        self.grad_theta[1][k0][feature][param] += \
                        self.grad_theta_t[t][1][k0][feature][param]

            # update eta
            if update_probs and t != 0:
                self.update_p_Xtm1_Xt(t)

            self.grad_eta_t[t],self.grad_eta0_t[t] = self.get_grad_eta_t(t)

            self.grad_eta[0] += self.grad_eta_t[t][0]
            self.grad_eta0[0] += self.grad_eta0_t[t][0]

            for k0 in range(self.K[0]):
                self.grad_eta[1][k0] += self.grad_eta_t[t][1][k0]
                self.grad_eta0[1][k0] += self.grad_eta0_t[t][1][k0]

        # record gradients, weights, and parameters for SVRG
        if update_tilde:
            self.grad_theta_tilde = deepcopy(self.grad_theta)
            self.grad_eta_tilde = deepcopy(self.grad_eta)
            self.grad_eta0_tilde = deepcopy(self.grad_eta0)

            self.p_Xt_tilde = deepcopy(self.p_Xt)
            self.p_Xtm1_Xt_tilde = deepcopy(self.p_Xtm1_Xt)

            self.theta_tilde = deepcopy(self.theta)
            self.eta_tilde = deepcopy(self.eta)
            self.eta0_tilde = deepcopy(self.eta0)

        return

    def M_step(self,max_iters=None,alpha_theta=None,alpha_eta=None,method="EM",partial_E=False,tol=1e-5,record_like=False):

        if max_iters is None:
            max_iters = self.T

        if alpha_theta is None:
            if self.L_theta != 0:
                alpha_theta = 1.0/(3.0*self.L_theta)
            else:
                alpha_theta = np.infty
        else:
            self.L_theta = 1.0/(3.0*alpha_theta)

        if alpha_eta is None:
            if self.L_eta != 0:
                alpha_eta = 1.0/(3.0*self.L_eta)
            else:
                alpha_eta = np.infty
        else:
            self.L_eta = 1.0/(3.0*alpha_eta)

        # update parameters
        if method == "EM":

            # update Gamma
            num = np.sum(self.p_Xt_Xtp1,axis=0)
            denom = np.sum(self.p_Xt[:-1],axis=0)
            self.Gamma[0] = num / denom[:,None]
            self.eta = Gamma_2_eta(self.Gamma)

            # update theta
            for feature in self.theta[0]:

                # get denominator
                denom = np.sum(self.p_Xt,axis=0)

                # update log-sig
                num = np.sum(np.array([self.p_Xt[t]*(self.data[t][feature]-self.theta[0][feature]["mu"])**2 for t in range(self.T)]),axis=0)
                var = num / denom
                self.theta[0][feature]["log_sig"] = np.log(np.sqrt(var))

                # update mu
                num = np.sum(np.array([self.p_Xt[t]*self.data[t][feature] for t in range(self.T)]),axis=0)
                self.theta[0][feature]["mu"] = num / denom

            return

        if method == "GD":

            # update eta0
            delta = alpha_eta * self.grad_eta0[0]
            self.eta0[0] += delta
            for k0 in range(self.K[0]):
                delta = alpha_eta * self.grad_eta0[1][k0]
                self.eta0[1][k0] += delta

            # update eta
            delta = alpha_eta * self.grad_eta[0]
            self.eta[0] += delta
            for k0 in range(self.K[0]):
                delta = alpha_eta * self.grad_eta[1][k0]
                self.eta[1][k0] += delta

            # update theta
            for feature in self.grad_theta[0]:
                for param in ['mu','log_sig']:
                    delta = alpha_theta * self.grad_theta[0][feature][param]
                    self.theta[0][feature][param] += delta

            for k0 in range(self.K[0]):
                for feature in self.grad_theta[1][k0]:
                    for param in ['mu','log_sig']:
                        delta = alpha_theta * self.grad_theta[1][k0][feature][param]
                        self.theta[1][k0][feature][param] += delta

            return

        if method == "GD":
            max_iters = 10

        for iter in range(max_iters):

            if method == "GD":

                self.E_step(update_probs=False)
                #print(self.grad_eta)
                #print(self.grad_eta0)
                #print(np.sum([self.p_Xt[t] * self.get_log_f(t)[0] for t in range(self.T)]))
                #print(np.sum([self.p_Xtm1_Xt[t] * self.get_log_Gamma()[0] for t in range(1,self.T)]))
                #print("")

                # update eta0
                delta = alpha_eta * self.grad_eta0[0]
                self.eta0[0] += delta
                for k0 in range(self.K[0]):
                    delta = alpha_eta * self.grad_eta0[1][k0]
                    self.eta0[1][k0] += delta

                # update eta
                delta = alpha_eta * self.grad_eta[0]
                self.eta[0] += delta
                for k0 in range(self.K[0]):
                    delta = alpha_eta * self.grad_eta[1][k0]
                    self.eta[1][k0] += delta

                # update theta
                for feature in self.grad_theta[0]:
                    for param in ['mu','log_sig']:
                        delta = alpha_theta * self.grad_theta[0][feature][param]
                        self.theta[0][feature][param] += delta

                for k0 in range(self.K[0]):
                    for feature in self.grad_theta[1][k0]:
                        for param in ['mu','log_sig']:
                            delta = alpha_theta * self.grad_theta[1][k0][feature][param]
                            self.theta[1][k0][feature][param] += delta

                # record trace
                #self.theta_trace.append(deepcopy(self.theta))
                #self.eta_trace.append(deepcopy(self.eta))
                #self.eta0_trace.append(deepcopy(self.eta0))

                # record log likelihood
                #ll, grad_norm = self.get_log_like()
                #self.grad_norm_trace.append(grad_norm / self.T)
                #self.log_like_trace.append(ll / self.T)

                continue

            # pick index
            t = np.random.choice(self.T)

            # get old gradient
            if method == "SVRG":

                # get old gradients
                grad_log_f = self.get_log_f(t,theta=self.theta_tilde)[1]

                if t == 0:
                    _,grad_eta0_log_delta = self.get_log_delta(eta0=self.eta0_tilde)
                    grad_eta_log_Gamma = [np.zeros((self.K[0],self.K[0],self.K[0],self.K[0])),
                                          [np.zeros((self.K[1],self.K[1],self.K[1],self.K[1])) for _ in range(self.K[0])]]
                else:
                    _,grad_eta0_log_delta,grad_eta_log_Gamma = self.get_log_Gamma(eta=self.eta_tilde,eta0=self.eta0_tilde)

                # get old weights
                p_Xt = self.p_Xt_tilde[t]
                p_Xtm1_X1 = self.p_Xtm1_Xt_tilde[t]

                # calculate old gradient
                old_grad_theta_t = self.get_grad_theta_t(t,
                                                         grad_log_f=grad_log_f,
                                                         p_Xt=p_Xt)

                old_grad_eta_t,old_grad_eta0_t = self.get_grad_eta_t(t,
                                                                     grad_eta0_log_delta=grad_eta0_log_delta,
                                                                     grad_eta_log_Gamma=grad_eta_log_Gamma,
                                                                     p_Xtm1_Xt=p_Xtm1_X1)

            else:

                # get old gradient at index
                old_grad_theta_t = deepcopy(self.grad_theta_t[t])
                old_grad_eta0_t= deepcopy(self.grad_eta0_t[t])
                old_grad_eta_t= deepcopy(self.grad_eta_t[t])

            # update alpha, beta, p_Xt, p_Xt_Xtp1
            if partial_E:

                self.update_alpha(t)
                self.update_beta(t)
                self.update_p_Xt(t)
                if t != 0:
                    self.update_p_Xtm1_Xt(t)
                if t != self.T-1:
                    self.update_p_Xtm1_Xt(t+1)

            # get new gradient at index
            new_grad_theta_t = self.get_grad_theta_t(t)
            new_grad_eta_t,new_grad_eta0_t = self.get_grad_eta_t(t)

            if method == "SGD":

                # update step size
                alpha_eta_m = alpha_eta / np.sqrt(iter+1)
                alpha_theta_m = alpha_theta / np.sqrt(iter+1)

                # update eta0
                delta = alpha_eta_m * new_grad_eta0_t[0]
                self.eta0[0] += delta
                for k0 in range(self.K[0]):
                    delta = alpha_eta_m * new_grad_eta0_t[1][k0]
                    self.eta0[1][k0] += delta

                # update eta
                delta = alpha_eta_m * new_grad_eta_t[0]
                self.eta[0] += delta
                for k0 in range(self.K[0]):
                    delta = alpha_eta_m * new_grad_eta_t[1][k0]
                    self.eta[1][k0] += delta

                # update theta
                for feature in new_grad_theta_t[0]:
                    for param in ['mu','log_sig']:
                        delta = alpha_theta_m * new_grad_theta_t[0][feature][param]
                        self.theta[0][feature][param] += delta

                for k0 in range(self.K[0]):
                    for feature in new_grad_theta_t[1][k0]:
                        for param in ['mu','log_sig']:
                            delta = alpha_theta_m * new_grad_theta_t[1][k0][feature][param]
                            self.theta[1][k0][feature][param] += delta

            elif method == "SAG":

                # update eta0
                delta = alpha_eta * (new_grad_eta0_t[0] \
                                   - old_grad_eta0_t[0] \
                                   + self.grad_eta0[0])/self.T
                self.eta0[0] += delta
                for k0 in range(self.K[0]):
                    delta = alpha_eta * (new_grad_eta0_t[1][k0] \
                                       - old_grad_eta0_t[1][k0]
                                       + self.grad_eta0[1][k0])/self.T
                    self.eta0[1][k0] += delta

                # update eta
                delta = alpha_eta * (new_grad_eta_t[0] \
                                   - old_grad_eta_t[0] \
                                   + self.grad_eta[0])/self.T
                self.eta[0] += delta
                for k0 in range(self.K[0]):
                    delta = alpha_eta * (new_grad_eta_t[1][k0] \
                                       - old_grad_eta_t[1][k0] \
                                       + self.grad_eta[1][k0])/self.T
                    self.eta[1][k0] += delta

                # update theta
                for feature in new_grad_theta_t[0]:
                    for param in ['mu','log_sig']:
                        delta = alpha_theta * (new_grad_theta_t[0][feature][param] \
                                             - old_grad_theta_t[0][feature][param] \
                                             + self.grad_theta[0][feature][param])/self.T
                        self.theta[0][feature][param] += delta

                for k0 in range(self.K[0]):
                    for feature in new_grad_theta_t[1][k0]:
                        for param in ['mu','log_sig']:
                            delta = alpha_theta * (new_grad_theta_t[1][k0][feature][param] \
                                                 - old_grad_theta_t[1][k0][feature][param] \
                                                 + self.grad_theta[1][k0][feature][param])/self.T
                            self.theta[1][k0][feature][param] += delta

            elif method == "SVRG":

                # update eta0
                delta = alpha_eta * (new_grad_eta0_t[0] \
                                   - old_grad_eta0_t[0] \
                                   + self.grad_eta0_tilde[0]/self.T)
                self.eta0[0] += delta
                for k0 in range(self.K[0]):
                    delta = alpha_eta * (new_grad_eta0_t[1][k0] \
                                       - old_grad_eta0_t[1][k0]
                                       + self.grad_eta0_tilde[1][k0]/self.T)
                    self.eta0[1][k0] += delta

                # update eta
                delta = alpha_eta * (new_grad_eta_t[0] \
                                   - old_grad_eta_t[0] \
                                   + self.grad_eta_tilde[0]/self.T)
                self.eta[0] += delta
                for k0 in range(self.K[0]):
                    delta = alpha_eta * (new_grad_eta_t[1][k0] \
                                       - old_grad_eta_t[1][k0] \
                                       + self.grad_eta_tilde[1][k0]/self.T)
                    self.eta[1][k0] += delta

                # update theta
                for feature in new_grad_theta_t[0]:
                    for param in ['mu','log_sig']:
                        delta = alpha_theta * (new_grad_theta_t[0][feature][param] \
                                             - old_grad_theta_t[0][feature][param] \
                                             + self.grad_theta_tilde[0][feature][param]/self.T)
                        self.theta[0][feature][param] += delta

                for k0 in range(self.K[0]):
                    for feature in new_grad_theta_t[1][k0]:
                        for param in ['mu','log_sig']:
                            delta = alpha_theta * (new_grad_theta_t[1][k0][feature][param] \
                                                 - old_grad_theta_t[1][k0][feature][param] \
                                                 + self.grad_theta_tilde[1][k0][feature][param]/self.T)
                            self.theta[1][k0][feature][param] += delta

            elif method == "SAGA":

                # update eta0
                delta = alpha_eta * (new_grad_eta0_t[0] \
                                   - old_grad_eta0_t[0] \
                                   + self.grad_eta0[0]/self.T)
                self.eta0[0] += delta
                for k0 in range(self.K[0]):
                    delta = alpha_eta * (new_grad_eta0_t[1][k0] \
                                       - old_grad_eta0_t[1][k0]
                                       + self.grad_eta0[1][k0]/self.T)
                    self.eta0[1][k0] += delta

                # update eta
                delta = alpha_eta * (new_grad_eta_t[0] \
                                   - old_grad_eta_t[0] \
                                   + self.grad_eta[0]/self.T)
                self.eta[0] += delta
                for k0 in range(self.K[0]):
                    delta = alpha_eta * (new_grad_eta_t[1][k0] \
                                       - old_grad_eta_t[1][k0] \
                                       + self.grad_eta[1][k0]/self.T)
                    self.eta[1][k0] += delta

                # update theta
                for feature in new_grad_theta_t[0]:
                    for param in ['mu','log_sig']:
                        delta = alpha_theta * (new_grad_theta_t[0][feature][param] \
                                             - old_grad_theta_t[0][feature][param] \
                                             + self.grad_theta[0][feature][param]/self.T)
                        self.theta[0][feature][param] += delta

                for k0 in range(self.K[0]):
                    for feature in new_grad_theta_t[1][k0]:
                        for param in ['mu','log_sig']:
                            delta = alpha_theta * (new_grad_theta_t[1][k0][feature][param] \
                                                 - old_grad_theta_t[1][k0][feature][param] \
                                                 + self.grad_theta[1][k0][feature][param]/self.T)
                            self.theta[1][k0][feature][param] += delta

            else:
                raise("method %s not recognized" % method)

            # check the Lipshitz constants
            if method != "SGD":
                self.check_L_theta(t)
                alpha_theta = 1.0/(3.0*self.L_theta)

                self.check_L_eta(t)
                alpha_eta = 1.0/(3.0*self.L_eta)

            # clip values
            self.eta[0] = np.clip(self.eta[0],-10,10)
            for feature in self.theta[0]:
                for param in self.theta[1][k0][feature]:
                    self.theta[0][feature][param] = np.clip(self.theta[0][feature][param],
                                                            self.param_bounds[0][feature][param][0],
                                                            self.param_bounds[0][feature][param][1])

            for k0 in range(self.K[0]):
                for feature in self.theta[1][k0]:
                    for param in self.theta[1][k0][feature]:
                        self.theta[1][k0][feature][param] = np.clip(self.theta[1][k0][feature][param],
                                                                    self.param_bounds[1][feature][param][0],
                                                                    self.param_bounds[1][feature][param][1])

            # update average gradient and table of gradients
            self.grad_eta0[0] += new_grad_eta0_t[0] - old_grad_eta0_t[0]
            self.grad_eta0_t[t][0] = deepcopy(new_grad_eta0_t[0])
            for k0 in range(self.K[0]):
                self.grad_eta0[1][k0] += new_grad_eta0_t[1][k0] - old_grad_eta0_t[1][k0]
                self.grad_eta0_t[t][1][k0] = deepcopy(new_grad_eta0_t[1][k0])

            self.grad_eta[0] += new_grad_eta_t[0] - old_grad_eta_t[0]
            self.grad_eta_t[t][0] = deepcopy(new_grad_eta_t[0])
            for k0 in range(self.K[0]):
                self.grad_eta[1][k0] += new_grad_eta_t[1][k0] - old_grad_eta_t[1][k0]
                self.grad_eta_t[t][1][k0] = deepcopy(new_grad_eta_t[1][k0])

            for feature in new_grad_theta_t[0]:
                for param in ['mu','log_sig']:
                    self.grad_theta[0][feature][param] += new_grad_theta_t[0][feature][param]
                    self.grad_theta[0][feature][param] -= old_grad_theta_t[0][feature][param]
                    self.grad_theta_t[t][0][feature][param] = deepcopy(new_grad_theta_t[0][feature][param])

            for k0 in range(self.K[0]):
                for feature in new_grad_theta_t[1][k0]:
                    for param in ['mu','log_sig']:
                        self.grad_theta[1][k0][feature][param] += new_grad_theta_t[1][k0][feature][param]
                        self.grad_theta[1][k0][feature][param] -= old_grad_theta_t[1][k0][feature][param]
                        self.grad_theta_t[t][1][k0][feature][param] = deepcopy(new_grad_theta_t[1][k0][feature][param])

            # update Gamma and delta
            self.log_Gamma = self.get_log_Gamma(jump=False)[0]
            self.log_Gamma_jump = self.get_log_Gamma(jump=True)[0]
            self.log_delta = self.get_log_delta()[0]

            # record likelihood and check for convergence every T iterations
            if (iter % (2*self.T) == (2*self.T-1)):

                # update epoch
                if partial_E:
                    self.epoch_num += 1.0
                else:
                    self.epoch_num += 1.0

                if record_like:

                    # record time trace and stop timer
                    self.train_time += time.time() - self.start_time
                    self.time_trace.append(self.train_time)
                    self.epoch_trace.append(self.epoch_num)

                    # record parameter traces
                    self.theta_trace.append(deepcopy(self.theta))
                    self.eta_trace.append(deepcopy(self.eta))
                    self.eta0_trace.append(deepcopy(self.eta0))

                    # record log likelihood
                    ll, grad_norm = self.get_log_like()
                    self.grad_norm_trace.append(grad_norm / self.T)
                    self.log_like_trace.append(ll / self.T)

                    # start timer back up
                    self.start_time = time.time()

                grad_norm = self.est_grad_norm() / self.T
                print(grad_norm)
                if (grad_norm < tol):
                    print("M-step sucesssfully converged")
                    return

        print("M-step failed to converge: maximum number of iterations reached")

        return

    def baseline_optim(self,num_epochs=None,method="CG",tol=1e-5):

        options = {'maxiter':num_epochs,'disp':True}

        def loss_fn(x):

            # hold on to backups
            theta_backup = deepcopy(self.theta)
            eta_backup = deepcopy(self.eta)
            eta0_backup = deepcopy(self.eta0)

            # update parameters
            ind = 0

            # update theta coarse
            for feature in self.theta[0]:
                if feature == 'broadDiveType':
                    continue
                for param in self.theta[0][feature]:
                    for k0 in range(self.K[0]):
                        self.theta[0][feature][param][k0] = x[ind]
                        ind += 1

            # update theta fine
            for feature in self.theta[1][0]:

                # skip features with fixed parameters
                if feature == 'broadDiveType':
                    continue

                for param in self.theta[1][0][feature]:
                    for k1 in range(self.K[1]):
                        for k0 in range(self.K[0]):

                            self.theta[1][k0][feature][param][k1] = x[ind]
                            #if k0 == (self.K[0]-1) or k1 != 0:
                            #if k0 == (self.K[0]-1):
                            ind += 1

            # update eta coarse
            for i in range(self.K[0]):
                for j in range(self.K[0]):
                    if i != j:
                        self.eta[0][i,j] = x[ind]
                        ind += 1

            # update eta fine
            for k0 in range(self.K[0]):
                for i in range(self.K[1]):
                    for j in range(self.K[1]):
                        if i != j:
                            self.eta[1][k0][i,j] = x[ind]
                            ind += 1

            # update eta0 coarse
            for i in range(self.K[0]):
                if i != 0:
                    self.eta0[0][i] = x[ind]
                    ind += 1

            # update eta0 fine
            for k0 in range(self.K[0]):
                for i in range(self.K[1]):
                    if i != 0:
                        self.eta0[1][k0][i] = x[ind]
                        ind += 1

            # update log delta and log Gamma
            self.log_Gamma = self.get_log_Gamma(jump=False)[0]
            self.log_Gamma_jump = self.get_log_Gamma(jump=True)[0]
            self.log_delta = self.get_log_delta()[0]

            # calculate likelihood
            self.E_step()

            # get likelihood
            ll = logsumexp(self.log_alphas[self.T-1])

            # initialize gradient
            grad_ll = np.zeros_like(x)
            ind = 0

            # gradient wrt theta coarse
            for feature in self.theta[0]:
                if feature == 'broadDiveType':
                    continue
                for param in self.theta[0][feature]:
                    for k0 in range(self.K[0]):
                        grad_ll[ind] = -self.grad_theta[0][feature][param][k0]
                        ind += 1

            # gradient wrt theta fine
            for feature in self.theta[1][k0]:

                if feature == 'broadDiveType':
                    continue

                for param in self.theta[1][k0][feature]:
                    for k1 in range(self.K[1]):
                        grad_ll[ind] = 0
                        for k0 in range(self.K[0]):

                            grad_ll[ind] += -self.grad_theta[1][k0][feature][param][k1]

                            # make shalow dives share parameters
                            #if k0 == (self.K[0]-1) or k1 != 0:
                            #if k0 == (self.K[0]-1):
                            ind += 1
                            grad_ll[ind] = 0

            # gradient wrt eta coarse
            for i in range(self.K[0]):
                for j in range(self.K[0]):
                    if i != j:
                        grad_ll[ind] = -self.grad_eta[0][i,j]
                        ind += 1

            # gradient wrt eta fine
            for k0 in range(self.K[0]):
                for i in range(self.K[1]):
                    for j in range(self.K[1]):
                        if i != j:
                            grad_ll[ind] = -self.grad_eta[1][k0][i,j]
                            ind += 1

            # gradient wrt eta0 coarse
            for i in range(self.K[0]):
                if i != 0:
                    grad_ll[ind] = -self.grad_eta0[0][i]
                    ind += 1

            # update eta0 fine
            for k0 in range(self.K[0]):
                for i in range(self.K[1]):
                    if i != 0:
                        grad_ll[ind] = -self.grad_eta0[1][k0][i]
                        ind += 1

            # record stuff
            self.epoch_num += 1.0
            self.epoch_trace.append(self.epoch_num)
            self.time_trace.append(time.time() - self.start_time)
            self.log_like_trace.append(ll / self.T)
            self.theta_trace.append(deepcopy(self.theta))
            self.eta_trace.append(deepcopy(self.eta))

            # return parameters
            self.theta = theta_backup
            self.eta = eta_backup
            self.eta0 = eta0_backup

            # return Gamma and delta
            self.log_Gamma = self.get_log_Gamma(jump=False)[0]
            self.log_Gamma_jump = self.get_log_Gamma(jump=True)[0]
            self.log_delta = self.get_log_delta()[0]

            print(ll/self.T)
            print(grad_ll/self.T)
            print("")

            return (-ll/self.T,grad_ll/self.T)

        # initialize x0
        x0 = []
        bounds = []

        # set theta coarse
        for feature in self.theta[0]:
            #print(self.theta[0])
            #print(feature)
            if feature == 'broadDiveType':
                continue
            for param in self.theta[0][feature]:
                for k0 in range(self.K[0]):
                    x0.append(self.theta[0][feature][param][k0])
                    bounds.append((None,None))

        # set theta fine
        for feature in self.theta[1][0]:

            if feature == 'broadDiveType':
                continue

            for param in self.theta[1][0][feature]:
                for k1 in range(self.K[1]):
                    for k0 in range(self.K[0]):

                        # skip if k0 != K0 and k1 = 0
                        #if k0 == (self.K[0]-1) or k1 != 0:
                        #if k0 == (self.K[0]-1):
                        x0.append(self.theta[1][k0][feature][param][k1])
                        bounds.append((None,None))

        # set eta coarse
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                if i != j:
                    x0.append(self.eta[0][i,j])
                    bounds.append((None,-np.log(50)))

        # set eta fine
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                for j in range(self.K[1]):
                    if i != j:
                        x0.append(self.eta[1][k0][i,j])
                        bounds.append((None,None))

        # set eta0 coarse
        for i in range(self.K[0]):
            if i != 0:
                x0.append(self.eta0[0][i])
                bounds.append((None,None))

        # set eta0 fine
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                if i != 0:
                    x0.append(self.eta0[1][k0][i])
                    bounds.append((None,None))

        # fit x
        print(x0)
        res = minimize(loss_fn, x0, bounds=bounds,
                       method=method, tol=tol,
                       options=options, jac=True)
        x = res["x"]

        # finalize parameters
        ind = 0

        # finalize theta coarse
        for feature in self.theta[0]:
            if feature == 'broadDiveType':
                continue
            for param in self.theta[0][feature]:
                for k0 in range(self.K[0]):
                    self.theta[0][feature][param][k0] = x[ind]
                    ind += 1

        # finalize theta fine
        for feature in self.theta[1][k0]:

            if feature == 'broadDiveType':
                continue

            for param in self.theta[1][k0][feature]:
                for k1 in range(self.K[1]):
                    for k0 in range(self.K[0]):
                        self.theta[1][k0][feature][param][k1] = x[ind]
                        #if k0 == (self.K[0]-1) or k1 != 0:
                        #if k0 == (self.K[0]-1):
                        ind += 1

        # finalize eta coarse
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                if i != j:
                    self.eta[0][i,j] = x[ind]
                    ind += 1

        # finalize eta fine
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                for j in range(self.K[1]):
                    if i != j:
                        self.eta[1][k0][i,j] = x[ind]
                        ind += 1

        # finalize eta0 coarse
        for i in range(self.K[0]):
            if i != 0:
                self.eta0[0][i] = x[ind]
                ind += 1

        # finalize eta0 fine
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                if i != 0:
                    self.eta0[1][k0][i] = x[ind]
                    ind += 1

        # finalize log Gamma and log delta
        self.log_Gamma = self.get_log_Gamma(jump=False)[0]
        self.log_Gamma_jump = self.get_log_Gamma(jump=True)[0]
        self.log_delta = self.get_log_delta()[0]

        return res

    def train_HHMM(self,num_epochs=10,max_time=np.infty,max_iters=None,alpha_theta=None,alpha_eta=None,tol=1e-5,grad_tol=1e-3,method="EM",partial_E=False,record_like=False):

        # fill in keyword args
        if max_iters is None:
            max_iters = self.T

        if alpha_theta is None:
            alpha_theta = self.step_size

        if alpha_eta is None:
            alpha_eta = self.step_size

        # check that the method makes sense
        if method not in ["EM","BFGS","Nelder-Mead","CG",
                          "GD","SGD","SAG","SVRG","SAGA"]:
            print("method %s not recognized" % method)
            return

        # start timer
        self.epoch_num = 0.0
        self.train_time = 0.0
        self.start_time = time.time()

        # do direct likelihood maximization
        if method in ["Nelder-Mead","BFGS","CG"]:
            res = self.baseline_optim(num_epochs=num_epochs,
                                      method=method,tol=tol)
            print(res)

            # record time
            self.train_time += time.time()-self.start_time

            return

        # initialize old values
        ll_old = -np.infty
        theta_old = deepcopy(self.theta)
        eta_old = deepcopy(self.eta)
        eta0_old = deepcopy(self.eta0)

        while (self.epoch_num < num_epochs) and (self.train_time < max_time):

            print("starting epoch %.1f" % (self.epoch_num))
            print("")

            print("%.3f hours elapsed" % (self.train_time / 3600))

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

            # do E-step
            print("starting E-step...")
            self.E_step()
            print("...done")
            print("")

            # record log-likelihood
            ll_new = logsumexp(self.log_alphas[self.T-1])
            print("current log likelihood: %f" % ll_new)
            print("")

            # check for convergence
            if (ll_new < ll_old):

                print("log likelihood decreased. Trying again...")
                print("")
                ll_old = ll_new

                # return old parameters
                self.theta = theta_old
                self.eta = eta_old
                self.eta0 = eta0_old

                # return Gamma and delta
                self.log_Gamma = self.get_log_Gamma(jump=False)[0]
                self.log_Gamma_jump = self.get_log_Gamma(jump=True)[0]
                self.log_delta = self.get_log_delta()[0]

                # return old gradients and weights
                self.E_step()

            elif ((ll_new - ll_old)/np.abs(ll_old)) < tol:

                print("relative change of log likelihood is less than %.1E. returning..." % tol)
                self.train_time += time.time() - self.start_time
                return

            else:
                ll_old = ll_new

                theta_old = deepcopy(self.theta)
                eta_old = deepcopy(self.eta)
                eta0_old = deepcopy(self.eta0)

            # record time and epoch from E step
            self.train_time += time.time() - self.start_time
            self.epoch_num += 1.0
            self.time_trace.append(self.train_time)
            self.epoch_trace.append(self.epoch_num)

            # record trace
            self.log_like_trace.append(ll_new / self.T)
            self.grad_norm_trace.append(self.est_grad_norm() / self.T)
            self.theta_trace.append(deepcopy(self.theta))
            self.eta_trace.append(deepcopy(self.eta))
            self.eta0_trace.append(deepcopy(self.eta))

            # start timer back up
            self.start_time = time.time()

            # do M-step
            print("starting M-step...")
            self.M_step(max_iters=max_iters,
                        method=method,
                        #alpha_theta=alpha_theta,
                        #alpha_eta=alpha_eta,
                        partial_E=partial_E,
                        tol=grad_tol,
                        record_like=record_like)
            print("...done")
            print("")
            print("L_theta: ",self.L_theta)
            print("L_eta: ",self.L_eta)
            print("")


        if self.train_time > max_time:
            print("Maximum training time (%.3f hrs) reached. Returning..." % (max_time / 3600))
        else:
            print("Maximum number of epochs (%.1f) reached. Returning..." % self.epoch_num)

        return

    def save(self,file):

        with open(file, 'wb') as f:
            pickle.dump(self, f)

        return
