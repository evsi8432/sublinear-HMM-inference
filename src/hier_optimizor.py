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

from copy import deepcopy

import time
import pickle

import sys
import Preprocessor
import Parameters
import HHMM
import Visualisor

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

    def __init__(self,pars,data):

        '''
        constructor for optimizor class
        '''

        self.data = data
        self.pars = pars

        self.K = pars.K
        self.K_total = self.K[0]*self.K[1]
        self.T = len(data)

        # thetas and etas
        self.initialize_theta(data)
        self.initialize_eta()
        self.initialize_eta0()

        # get Gamma and delta
        self.Gamma = eta_2_Gamma(self.eta)
        self.delta = eta0_2_delta(self.eta0)

        self.log_Gamma_full = self.get_log_Gamma_full()[0]
        self.log_delta_full = self.get_log_delta_full()[0]

        # store for SVRG
        self.theta_tilde = deepcopy(self.theta)
        self.eta_tilde = deepcopy(self.eta)
        self.eta0_tilde = deepcopy(self.eta0)

        # traces of parameters
        self.theta_trace = []
        self.eta_trace = []
        self.eta0_trace = []

        # traces of log-likelihood
        self.log_like_trace = []
        self.grad_norm_trace = []

        # traces of time
        self.time_trace = []
        self.epoch_trace = []

        # alpha and beta
        self.log_alphas = np.zeros((self.T,self.K_total))
        self.log_betas = np.zeros((self.T,self.K_total))

        # p_Xt and p_Xtm1_Xt
        self.p_Xt = np.zeros((self.T,self.K_total))
        self.p_Xtm1_Xt = np.zeros((self.T,self.K_total,self.K_total))
        self.p_Xt_tilde = np.zeros((self.T,self.K_total))
        self.p_Xtm1_Xt_tilde = np.zeros((self.T-1,self.K_total,self.K_total))

        # gradients wrt theta
        self.grad_theta_F_t = [deepcopy(self.theta) for _ in range(self.T)]
        self.grad_theta_F = deepcopy(self.theta)
        self.grad_theta_F_tilde = deepcopy(self.theta)

        # gradients wrt eta
        self.grad_eta_G_t = [deepcopy(self.eta) for _ in range(self.T)]
        self.grad_eta_G = deepcopy(self.eta)
        self.grad_eta_G_tilde = deepcopy(self.eta)

        # gradients wrt eta0
        self.grad_eta0_G_t = [deepcopy(self.eta0) for _ in range(self.T)]
        self.grad_eta0_G = deepcopy(self.eta0)
        self.grad_eta0_G_tilde = deepcopy(self.eta0)

        # time to train
        self.train_time = None
        self.start_time = None
        self.epoch_num = None

        # initialize gradients
        self.initialize_grads()

        # initialize step sizes and parameter bounds
        self.step_size = None
        self.param_bounds = {feature: {} for feature in self.theta[0]}
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
        K = self.pars.K[0]
        theta.append({})

        for feature,settings in self.pars.features[0].items():

            # initialize values
            theta[0][feature] = {'mu': np.zeros(K),
                                 'log_sig': np.zeros(K),
                                 'corr': np.zeros(K)}

            if data is not None:

                feature_data = [datum[feature] for datum in data]

                # first find basic parameter estimates
                theta[0][feature]['mu'] = np.ones(K)*np.mean(feature_data)
                theta[0][feature]['log_sig'] = np.log(np.ones(K)*max(0.1,np.std(feature_data)))
                theta[0][feature]['corr'] = -1.0*np.ones(K)

                # add randomness in initialization
                if settings['f'] == 'normal':
                    theta[0][feature]['mu'] += norm.rvs(np.zeros(K),0.1*np.exp(theta[0][feature]['log_sig']))
                    theta[0][feature]['log_sig'] += norm.rvs(0.0,0.1,size=K)
                    theta[0][feature]['corr'] += norm.rvs(0.0,0.1,size=K)
                else:
                    pass

        # then fill in the fine-scale stuff
        theta.append([{} for _ in range(K)])
        K = self.pars.K[1]

        for feature,settings in self.pars.features[1].items():
            for k0 in range(self.pars.K[0]):

                # initialize values
                theta[1][k0][feature] = {'mu': np.zeros(K),
                                         'log_sig': np.zeros(K),
                                         'corr': np.zeros(K)}

                if data is not None:

                    feature_data = [datum[feature] for datum in data]

                    # first find mu
                    theta[1][k0][feature]['mu'] = np.mean(feature_data)*np.ones(K)
                    theta[1][k0][feature]['log_sig'] = np.log(np.std(feature_data)*np.ones(K))
                    theta[1][k0][feature]['corr'] = -1.0 * np.ones(K)

                    # add randomness in initialization
                    if settings['f'] == 'normal':
                        theta[1][k0][feature]['mu'] += norm.rvs(np.zeros(K),0.1*np.exp(theta[1][k0][feature]['log_sig']))
                        theta[1][k0][feature]['log_sig'] += norm.rvs(0.0,0.1,size=K)
                        theta[1][k0][feature]['corr'] += norm.rvs(0.0,0.1,size=K)
                    else:
                        pass

        self.theta = theta

        return

    def initialize_eta(self):

        # initialize eta
        self.eta = []

        # fill in coarse eta
        eta_crude = 0.0 + 0.1*np.random.normal(size=(self.K[0],self.K[0]))
        for i in range(self.K[0]):
            eta_crude[i,i] = 0
        self.eta.append(eta_crude)

        # fill in fine eta
        eta_fine = []
        for _ in range(self.pars.K[0]):
            eta_fine_k = 0.0 + 0.1*np.random.normal(size=(self.K[1],self.K[1]))
            for i in range(self.pars.K[1]):
                eta_fine_k[i,i] = 0
            eta_fine.append(eta_fine_k)

        self.eta.append(eta_fine)

        return

    def initialize_eta0(self):

        # initialize eta0
        self.eta0 = [np.random.normal(size=self.K[0]),
                     [np.random.normal(size=self.K[1]) for _ in range(self.K[0])]]

        return

    def initialize_step_size(self,min_group_size=None):

        if min_group_size is None:
            min_group_size = max(int(0.01*self.T),3)

        self.step_size = np.infty
        t = self.T-1

        # get grad w.r.t theta
        for feature in self.grad_theta_F[0]:

            # set the parameter bounds for that feature
            feature_data = [self.data[s][feature] for s in range(self.T)]
            sorted_feature_data = np.sort(feature_data)

            # get range of mu
            min_mu = sorted_feature_data[0]
            max_mu = sorted_feature_data[-1]
            self.param_bounds[feature]["mu"] = [min_mu,max_mu]

            # get range of log_sig
            min_log_sig = np.log(min(sorted_feature_data[min_group_size:] - \
                                 sorted_feature_data[:-min_group_size]))
            max_log_sig = np.log(np.std(feature_data))
            self.param_bounds[feature]["log_sig"] = [min_log_sig,max_log_sig]

            # never make the step size larger than 10% of the range of any parameter
            mu_ss = 0.1*min(abs((max_mu-min_mu)/self.grad_theta_F[0][feature]["mu"]))
            log_sig_ss = 0.1*min(abs((max_log_sig-min_log_sig)/self.grad_theta_F[0][feature]["log_sig"]))
            param_ss = min(mu_ss,log_sig_ss)

            if param_ss < self.step_size:
                self.step_size = np.copy(param_ss)

        # get grad w.r.t eta
        for i in range(self.K[0]):
            for j in range(self.K[0]):

                if i != j:
                    continue

                # never make the step size larger than 1 (since |eta| < 15)
                param_ss = 1.0/abs(self.grad_eta_G[0][i,j])

                if param_ss < self.step_size:
                    self.step_size = np.copy(param_ss)

        return

    def initialize_grads(self):

        # overall gradient wrt theta
        for feature,dist in self.pars.features[0].items():
            if dist['f'] == 'normal' and not dist['corr']:

                self.grad_theta_F[0][feature]['mu'] = np.zeros(self.K[0])
                self.grad_theta_F[0][feature]['log_sig'] = np.zeros(self.K[0])
                self.grad_theta_F[0][feature]['corr'] = np.zeros(self.K[0])

                self.grad_theta_F_tilde[0][feature]['mu'] = np.zeros(self.K[0])
                self.grad_theta_F_tilde[0][feature]['log_sig'] = np.zeros(self.K[0])
                self.grad_theta_F_tilde[0][feature]['corr'] = np.zeros(self.K[0])

            else:
                raise('only independent normal distributions supported at this time')


        for feature,dist in self.pars.features[1].items():
            if dist['f'] == 'normal' and not dist['corr']:
                for k0 in range(self.K[0]):

                    self.grad_theta_F[1][k0][feature]['mu'] = np.zeros(self.K[1])
                    self.grad_theta_F[1][k0][feature]['log_sig'] = np.zeros(self.K[1])
                    self.grad_theta_F[1][k0][feature]['corr'] = np.zeros(self.K[1])

                    self.grad_theta_F_tilde[1][k0][feature]['mu'] = np.zeros(self.K[1])
                    self.grad_theta_F_tilde[1][k0][feature]['log_sig'] = np.zeros(self.K[1])
                    self.grad_theta_F_tilde[1][k0][feature]['corr'] = np.zeros(self.K[1])

            else:
                raise('only independent normal distributions supported at this time')

        # get overall gradient wrt eta
        self.grad_eta_G = [np.zeros((self.K[0],self.K[0])),
                           [np.zeros((self.K[1],self.K[1])) for _ in range(self.K[0])]]

        self.grad_eta_G_tilde = [np.zeros((self.K[0],self.K[0])),
                                 [np.zeros((self.K[1],self.K[1])) for _ in range(self.K[0])]]

        # get overall gradient wrt eta0
        self.grad_eta0_G = [np.zeros(self.K[0]),
                            [np.zeros(self.K[1]) for _ in range(self.K[0])]]

        self.grad_eta0_G_tilde = [np.zeros(self.K[0]),
                                  [np.zeros(self.K[1]) for _ in range(self.K[0])]]

        for t in range(self.T):

            # get table values wrt eta
            self.grad_eta_G_t[t] = [np.zeros((self.K[0],self.K[0])),
                                    [np.zeros((self.K[1],self.K[1])) for _ in range(self.K[0])]]

            # get table values wrt eta0
            self.grad_eta0_G_t[t] = [np.zeros(self.K[0]),
                                     [np.zeros(self.K[1]) for _ in range(self.K[0])]]

            # get table values wrt theta
            for feature,dist in self.pars.features[0].items():
                if dist['f'] == 'normal' and not dist['corr']:

                    self.grad_theta_F_t[t][0][feature]['mu'] = np.zeros(self.K[0])
                    self.grad_theta_F_t[t][0][feature]['log_sig'] = np.zeros(self.K[0])
                    self.grad_theta_F_t[t][0][feature]['corr'] = np.zeros(self.K[0])

                else:
                    raise('only independent normal distributions supported at this time')

            for feature,dist in self.pars.features[1].items():
                if dist['f'] == 'normal' and not dist['corr']:
                    for k0 in range(self.K[0]):

                        self.grad_theta_F_t[t][1][k0][feature]['mu'] = np.zeros(self.K[1])
                        self.grad_theta_F_t[t][1][k0][feature]['log_sig'] = np.zeros(self.K[1])
                        self.grad_theta_F_t[t][1][k0][feature]['corr'] = np.zeros(self.K[1])

                else:
                    raise('only independent normal distributions supported at this time')

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
            if feature in self.pars.features[0]:

                # define the feature
                dist = self.pars.features[0][feature]['f']

                if dist == 'normal':

                    # store log-likelihood
                    mu = np.repeat(theta[0][feature]['mu'],self.K[1])
                    log_sig = np.repeat(theta[0][feature]['log_sig'],self.K[1])
                    sig = np.exp(log_sig)

                    log_f += norm.logpdf(y[feature],
                                         loc=mu,
                                         scale=sig)

                    # store gradient
                    mu = theta[0][feature]['mu']
                    sig = np.exp(theta[0][feature]['log_sig'])
                    grad_log_f[0][feature] = {'mu': (y[feature]-mu)/(sig**2),
                                              'log_sig': ((y[feature]-mu)/sig)**2 - 1}

                else:
                    print("unidentified emission distribution %s for %s"%(dist,feature))
                    return

            elif feature in self.pars.features[1]:

                # define the feature
                dist = self.pars.features[1][feature]['f']

                if dist == 'normal':

                    # store log-likelihood
                    mu = np.concatenate([theta_fine[feature]['mu'] for theta_fine in theta[1]])
                    log_sig = np.concatenate([theta_fine[feature]['log_sig'] for theta_fine in theta[1]])
                    sig = np.exp(log_sig)

                    log_f += norm.logpdf(y[feature],
                                         loc=mu,
                                         scale=sig)

                    # store gradient
                    for k0 in range(self.K[0]):

                        mu = theta[1][k0][feature]['mu']
                        log_sig = theta[1][k0][feature]['log_sig']
                        sig = np.exp(log_sig)

                        grad_log_f[1][k0][feature] = {'mu': (y[feature]-mu)/(sig**2),
                                                      'log_sig': ((y[feature]-mu)/sig)**2 - 1}

                else:
                    print("unidentified emission distribution %s for %s"%(dist,feature))
                    return

            else:
                print("unidentified feature in y: %s" % feature)
                return

        # return the result
        return log_f,grad_log_f

    def get_log_delta_full(self,eta0=None):

        if eta0 is None:
            eta0 = self.eta0

        # get delta
        log_coarse_delta = np.repeat(eta0[0] - logsumexp(eta0[0]),self.K[1])
        log_fine_delta = np.concatenate([eta01 for eta01 in eta0[1]])
        log_delta = log_coarse_delta + log_fine_delta

        # get the gradient
        coarse_delta = np.exp(eta0[0]) / np.sum(np.exp(eta0[0]))
        fine_deltas = [np.exp(eta01) / np.sum(np.exp(eta01)) for eta01 in eta0[1]]

        grad_eta0_log_delta = [np.eye(self.K[0]) - np.tile(coarse_delta,[self.K[0],1]),
                               [np.eye(self.K[1]) - np.tile(fine_delta,[self.K[1],1]) \
                                for fine_delta in fine_deltas]]

        return log_delta,grad_eta0_log_delta

    def get_log_Gamma_full(self,eta=None,eta0=None):

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

        # construct log_Gamma_full
        K_total = self.K[0] * self.K[1]
        log_Gamma_full = np.zeros((K_total,K_total))

        for i in range(self.K[0]):
            for j in range(self.K[0]):

                # coarse-scale Gamma
                log_Gamma_full[(self.K[1]*i):(self.K[1]*(i+1)),
                               (self.K[1]*j):(self.K[1]*(j+1))] += log_coarse_Gamma[i,j]

                # fine-scale Gamma or delta
                if i == j:
                    log_Gamma_full[(self.K[1]*i):(self.K[1]*(i+1)),
                                   (self.K[1]*j):(self.K[1]*(j+1))] += log_fine_Gammas[j]
                else:
                    log_Gamma_full[(self.K[1]*i):(self.K[1]*(i+1)),
                                   (self.K[1]*j):(self.K[1]*(j+1))] += np.tile(log_fine_deltas[j],[self.K[1],1])

        grad_eta_log_Gamma = [np.zeros((self.K[0],self.K[0],self.K[0],self.K[0])),
                              [np.zeros((self.K[1],self.K[1],self.K[1],self.K[1])) for _ in range(self.K[0])]]

        grad_eta0_log_delta = [np.zeros((self.K[0],self.K[0])),
                               [np.zeros((self.K[1],self.K[1])) for _ in range(self.K[0])]]

        # grad from coarse_Gamma
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                for l in range(self.K[0]):
                    if i == l:
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
        for n in range(self.K[0]):
            grad_eta0_log_delta[1][n] = np.eye(self.K[1]) - np.tile(fine_deltas[n],[self.K[1],1])

        return log_Gamma_full,grad_eta0_log_delta,grad_eta_log_Gamma

    def update_alpha(self,t):

        # get log_f, grad_log_f
        log_f = self.get_log_f(t)[0]

        # update log_alpha
        if t == 0:
            self.log_alphas[t] = self.log_delta_full + log_f
        else:
            self.log_alphas[t] = logdotexp(self.log_alphas[t-1],self.log_Gamma_full) + log_f

        return

    def update_beta(self,t):

        # update log_beta
        if t == self.T-1:
            self.log_betas[t] = np.zeros(self.K_total)
        else:
            log_f_tp1 = self.get_log_f(t+1)[0]
            self.log_betas[t] = logdotexp(self.log_betas[t+1] + log_f_tp1,
                                            np.transpose(self.log_Gamma_full))

        return

    def est_log_like(self,t):
        return logsumexp(self.log_alphas[t] + self.log_betas[t])

    def est_grad_norm(self):

        norm_squared = 0

        # gradient wrt eta0
        norm_squared += np.sum(self.grad_eta_G[0]**2)
        for k0 in range(self.K[0]):
            norm_squared += np.sum(self.grad_eta_G[1][k0]**2)

        # gradient wrt eta
        norm_squared += np.sum(self.grad_eta0_G[0]**2)
        for k0 in range(self.K[0]):
            norm_squared += np.sum(self.grad_eta0_G[1][k0]**2)

        # gradient wrt theta
        for feature in self.grad_theta_F[0]:
            for param in self.grad_theta_F[0][feature]:
                norm_squared += np.sum(self.grad_theta_F[0][feature][param]**2)

        for k0 in range(self.K[0]):
            for feature in self.grad_theta_F[1][k0]:
                for param in self.grad_theta_F[1][k0][feature]:
                    norm_squared += np.sum(self.grad_theta_F[1][k0][feature][param]**2)

        return np.sqrt(norm_squared)

    def get_log_like(self):

        # store old values
        log_alphas0 = deepcopy(self.log_alphas)
        log_betas0 = deepcopy(self.log_betas)

        grad_theta_log_like0 = deepcopy(self.grad_theta_F)
        grad_eta_log_like0 = deepcopy(self.grad_eta_G)

        p_Xt0 = deepcopy(self.p_Xt)
        p_Xtm1_Xt0 = deepcopy(self.p_Xtm1_Xt)

        E_grad_log_f0 = deepcopy(self.grad_theta_F_t)
        E_grad_log_Gamma0 = deepcopy(self.grad_eta_G_t)

        grad_theta_log_like_tilde0 = deepcopy(self.grad_theta_F_tilde)
        grad_eta_log_like_tilde0 = deepcopy(self.grad_eta_G_tilde)

        p_Xt_tilde0 = deepcopy(self.p_Xt_tilde)
        p_Xtm1_Xt_tilde0 = deepcopy(self.p_Xtm1_Xt_tilde)

        theta_tilde0 = deepcopy(self.theta_tilde)
        eta_tilde0 = deepcopy(self.eta_tilde)

        # get new likelihood and gradient
        self.E_step()
        ll = self.est_log_like(0)
        grad_norm = self.est_grad_norm()

        # return values to old state
        self.log_alphas = log_alphas0
        self.log_betas = log_betas0

        self.grad_theta_F = grad_theta_log_like0
        self.grad_eta_G = grad_eta_log_like0

        self.p_Xt = p_Xt0
        self.p_Xtm1_Xt = p_Xtm1_Xt0

        self.grad_theta_F_t = E_grad_log_f0
        self.grad_eta_G_t = E_grad_log_Gamma0

        self.grad_theta_F_tilde = grad_theta_log_like_tilde0
        self.grad_eta_G_tilde = grad_eta_log_like_tilde0

        self.p_Xt_tilde = p_Xt_tilde0
        self.p_Xtm1_Xt_tilde = p_Xtm1_Xt_tilde0

        self.theta_tilde = theta_tilde0
        self.eta_tilde = eta_tilde0

        return ll, grad_norm

    def update_p_Xt(self,t):

        ll = self.est_log_like(t)
        self.p_Xt[t] = np.exp(self.log_alphas[t] + self.log_betas[t] - ll)

        return

    def update_p_Xtm1_Xt(self,t,log_f_t=None):

        if t == 0:
            raise("p_Xtm1_Xt not defined for t = 0")

        if log_f_t is None:
            log_f_t = self.get_log_f(t)[0]

        p_XX = np.zeros((self.K_total,self.K_total))
        for i in range(self.K_total):
            for j in range(self.K_total):
                p_XX[i,j] = self.log_alphas[t-1,i] \
                            + self.log_Gamma_full[i,j] \
                            + log_f_t[j] \
                            + self.log_betas[t,j]

        self.p_Xtm1_Xt[t] = np.exp(p_XX - logsumexp(p_XX))

        return

    def get_p_Xt_coarse(self,t):
        return np.array([np.sum(self.p_Xt[t][(self.K[1]*k0):(self.K[1]*(k0+1))]) \
                         for k0 in range(self.K[0])])

    def get_p_Xtm1_Xt_coarse(self,t):
        p_Xtm1_Xt_coarse = np.zeros((self.K[0],self.K[0]))
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                p_Xtm1_Xt_coarse[i,j] = np.sum(self.p_Xtm1_Xt[t][(self.K[1]*i):(self.K[1]*(i+1)),
                                                                 (self.K[1]*j):(self.K[1]*(j+1))])
        return p_Xtm1_Xt_coarse

    def get_p_Xt_fine_jump(self,k0,t):

        # get probability of arriving at fine-scale states
        p_Xt_fine = self.p_Xt[t][(self.K[1]*k0):(self.K[1]*(k0+1))]

        # subtract probability of arriving from same coarse-scale state
        if t != 0:
            p_Xt_fine_jump = p_Xt_fine - np.sum(self.get_p_Xtm1_Xt_fine_stay(k0,t),0)
        else:
            p_Xt_fine_jump = p_Xt_fine

        return p_Xt_fine_jump

    def get_p_Xtm1_Xt_fine_stay(self,k0,t):
        return self.p_Xtm1_Xt[t][(self.K[1]*k0):(self.K[1]*(k0+1)),
                                 (self.K[1]*k0):(self.K[1]*(k0+1))]

    def get_grad_theta_F_t(self,t,grad_log_f_t=None,p_Xt=None):

        # initialize gradient
        grad_theta_F_t = deepcopy(self.grad_theta_F_t[t])

        # get gradient and weights
        if grad_log_f_t is None:
            grad_log_f_t = self.get_log_f(t)[1]
        if p_Xt is None:
            p_Xt = self.p_Xt[t]

        # calculate coarse-scale gradient
        p_Xt_coarse = [np.sum(p_Xt[(self.K[1]*k0):(self.K[1]*(k0+1))]) for k0 in range(self.K[0])]
        for feature in grad_log_f_t[0]:
            for param in grad_log_f_t[0][feature]:
                grad_theta_F_t[0][feature][param] = p_Xt_coarse * grad_log_f_t[0][feature][param]

        # calculate fine-scale gradient
        for k0 in range(self.K[0]):
            for feature in grad_log_f_t[1][k0]:
                for param in grad_log_f_t[1][k0][feature]:
                    grad_theta_F_t[1][k0][feature][param] = p_Xt[(self.K[1]*k0):(self.K[1]*(k0+1))] * grad_log_f_t[1][k0][feature][param]

        return grad_theta_F_t

    def get_grad_eta_G_t(self,t,grad_log_Gamma=None,p_Xtm1_Xt=None):

        # initialize gradients
        grad_eta_G_t = [np.zeros((self.K[0],self.K[0])),
                        [np.zeros((self.K[1],self.K[1])) for _ in range(self.K[0])]]

        grad_eta0_G_t = [np.zeros(self.K[0]),
                         [np.zeros(self.K[1]) for _ in range(self.K[0])]]

        # deal with t == 0:
        if t == 0:

            p_Xt = self.p_Xt[t]
            p_Xt_coarse = [np.sum(p_Xt[(self.K[1]*k0):(self.K[1]*(k0+1))]) for k0 in range(self.K[0])]
            _,grad_eta0_log_delta = self.get_log_delta_full()

            # add coarse-scale delta
            for i in range(self.K[0]):
                grad_eta0_G_t[0][i] = np.sum(p_Xt_coarse * grad_eta0_log_delta[0][:,i])

            # add fine-scale delta
            for k0 in range(self.K[0]):
                for i in range(self.K[1]):
                    grad_eta0_G_t[1][k0][i] += np.sum(p_Xt[(self.K[1]*k0):(self.K[1]*(k0+1))] * \
                                                      grad_eta0_log_delta[1][k0][:,i])

            return grad_eta_G_t,grad_eta0_G_t


        # get gradients and weights
        if grad_log_Gamma is None:
            _,grad_eta0_log_delta,grad_eta_log_Gamma = self.get_log_Gamma_full()

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
                grad_eta0_G_t[1][k0][i] += np.sum(p_Xt_jump * grad_eta0_log_delta[1][k0][:,i])


        # add gradient from coarse-scale Gamma
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                grad_eta_G_t[0][i,j] = np.sum(p_Xtm1_Xt_coarse * grad_eta_log_Gamma[0][:,:,i,j])

        # add gradient from fine-scale Gamma
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                for j in range(self.K[1]):
                    grad_eta_G_t[1][k0][i,j] += np.sum(p_Xtm1_Xt[(self.K[1]*k0):(self.K[1]*(k0+1)),
                                                                 (self.K[1]*k0):(self.K[1]*(k0+1))] * \
                                                       grad_eta_log_Gamma[1][k0][:,:,i,j])

        return grad_eta_G_t,grad_eta0_G_t

    def check_L_eta(self,t):

        # get gradient and its norm
        grad_eta_G_t,grad_eta0_G_t = self.get_grad_eta_G_t(t)

        grad_G_t_norm2 = np.sum(grad_eta_G_t[0]**2)
        for k0 in range(self.K[0]):
            grad_G_t_norm2 += np.sum(grad_eta_G_t[1][k0]**2)

        # get new value of eta
        eta_new = deepcopy(self.eta)
        eta_new[0] += grad_eta_G_t[0] / self.L_eta
        for k0 in range(self.K[0]):
            eta_new[1][k0] += grad_eta_G_t[1][k0] / self.L_eta

        # get new value of eta0
        eta0_new = deepcopy(self.eta0)
        eta0_new[0] += grad_eta0_G_t[0] / self.L_eta
        for k0 in range(self.K[0]):
            eta0_new[1][k0] += grad_eta0_G_t[1][k0] / self.L_eta

        # Evaluate G for eta and eta0
        if t == 0:
            G_t = -np.sum(self.p_Xt[t] * self.get_log_delta_full(t,eta0=self.eta0)[0])
            G_t_new = -np.sum(self.p_Xt[t] * self.get_log_delta_full(t,eta0=eta0_new)[0])
        else:
            G_t  = -np.sum(self.p_Xtm1_Xt[t] * self.get_log_Gamma_full(t,eta=self.eta)[0])
            G_t_new = -np.sum(self.p_Xtm1_Xt[t] * self.get_log_Gamma_full(t,eta=eta_new)[0])

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
        grad_F_t = self.get_grad_theta_F_t(t)

        # initialize gradient norms
        grad_F_t_norm2 = 0

        # get new theta and gradient norm
        theta0 = deepcopy(self.theta)
        for feature in grad_F_t[0]:
            for param in grad_F_t[0][feature]:
                theta0[0][feature][param] += grad_F_t[0][feature][param] / self.L_theta
                grad_F_t_norm2 += np.sum(grad_F_t[0][feature][param]**2)

        # evaluate F for theta and theta0
        F_t  = -np.sum(self.p_Xt[t] * self.get_log_f(t,theta=self.theta)[0])
        F_t0 = -np.sum(self.p_Xt[t] * self.get_log_f(t,theta=theta0)[0])

        # check for inequality
        if grad_F_t_norm2 < 1e-8:
            pass
        elif F_t0 > F_t - grad_F_t_norm2 / (2*self.L_theta):
            self.L_theta *= 2
        else:
            self.L_theta *= 2**(-1/self.T)

        return

    def E_step(self,update_tilde=True):

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

            # update theta
            self.update_p_Xt(t)
            self.grad_theta_F_t[t] = self.get_grad_theta_F_t(t)

            for feature in self.grad_theta_F[0]:
                for param in self.grad_theta_F[0][feature]:
                    self.grad_theta_F[0][feature][param] += \
                    self.grad_theta_F_t[t][0][feature][param]

            for k0 in range(self.K[0]):
                for feature in self.grad_theta_F[1][k0]:
                    for param in self.grad_theta_F[1][k0][feature]:
                        self.grad_theta_F[1][k0][feature][param] += \
                        self.grad_theta_F_t[t][1][k0][feature][param]

            # update eta
            if t != 0:
                self.update_p_Xtm1_Xt(t)

            self.grad_eta_G_t[t],self.grad_eta0_G_t[t] = self.get_grad_eta_G_t(t)

            self.grad_eta_G[0] += self.grad_eta_G_t[t][0]
            self.grad_eta0_G[0] += self.grad_eta0_G_t[t][0]

            for k0 in range(self.K[0]):
                self.grad_eta_G[1][k0] += self.grad_eta_G_t[t][1][k0]
                self.grad_eta0_G[1][k0] += self.grad_eta0_G_t[t][1][k0]

        # record gradients, weights, and parameters for SVRG
        if update_tilde:
            self.grad_theta_F_tilde = deepcopy(self.grad_theta_F)
            self.grad_eta_G_tilde = deepcopy(self.grad_eta_G)
            self.grad_eta0_G_tilde = deepcopy(self.grad_eta0_G)

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

            # update coarse-scale delta
            self.delta[0] = self.get_p_Xt_coarse(0)

            # update fine-scale delta
            for k0 in range(self.K[0]):
                num = np.zeros(self.K[1])
                for t in range(self.T):
                    num += self.get_p_Xt_fine_jump(k0,t)
                self.delta[1][k0] = num/np.sum(num)

            # update coarse-scale Gamma
            num = np.zeros((self.K[0],self.K[0]))
            for t in range(1,self.T):
                num += self.get_p_Xtm1_Xt_coarse(t)
            self.Gamma[0] = num / np.sum(num,1)[:,None]

            # update fine-scale Gamma
            for k0 in range(self.K[0]):
                num = np.zeros((self.K[1],self.K[1]))
                for t in range(self.T):
                    num += self.get_p_Xtm1_Xt_fine_stay(k0,t)
                self.Gamma[1][k0] = num / np.sum(num,1)[:,None]

            # set eta and eta0
            self.eta = Gamma_2_eta(self.Gamma)
            self.eta0 = delta_2_eta0(self.delta)

            # update theta
            for feature in self.theta[0]:

                # get denominator
                denom = np.sum([self.get_p_Xt_coarse(t) for t in range(self.T)],axis=0)

                # update log-sig
                num = np.sum(np.array([self.get_p_Xt_coarse(t)*(self.data[t][feature]-self.theta[0][feature]["mu"])**2 for t in range(self.T)]),axis=0)
                var = num / denom
                self.theta[0][feature]["log_sig"] = np.log(np.sqrt(var))

                # update mu
                num = np.sum(np.array([self.get_p_Xt_coarse(t)*self.data[t][feature] for t in range(self.T)]),axis=0)
                self.theta[0][feature]["mu"] = num / denom

            for k0 in range(self.K[0]):
                for feature in self.theta[1][k0]:

                    # get denominator
                    denom = np.sum([self.p_Xt[t][(self.K[1]*k0):(self.K[1]*(k0+1))] \
                                    for t in range(self.T)],axis=0)

                    # update log-sig
                    num = np.sum(np.array([self.p_Xt[t][(self.K[1]*k0):(self.K[1]*(k0+1))] * \
                                          (self.data[t][feature]-self.theta[1][k0][feature]["mu"])**2 \
                                           for t in range(self.T)]),axis=0)
                    var = num / denom
                    self.theta[1][k0][feature]["log_sig"] = np.log(np.sqrt(var))

                    # update mu
                    num = np.sum(np.array([self.p_Xt[t][(self.K[1]*k0):(self.K[1]*(k0+1))] * \
                                           self.data[t][feature] \
                                           for t in range(self.T)]),axis=0)
                    self.theta[1][k0][feature]["mu"] = num / denom

            return

        if method == "GD":

            # update coarse-eta
            delta = alpha_eta * self.grad_eta_G[0] / self.T
            self.eta[0] += delta

            # update coarse-theta
            for feature in self.theta[0]:
                for param in ['mu','log_sig']:
                    delta = alpha_theta * self.grad_theta_F[0][feature][param] / self.T
                    self.theta[0][feature][param] += delta

            for k0 in range(self.K[0]):

                # update fine-eta
                delta = alpha_eta * self.grad_eta_G[1][k0] / self.T
                self.eta[1][k0] += delta

                # update fine-theta
                for feature in self.theta[0]:
                    for param in ['mu','log_sig']:
                        delta = alpha_theta * self.grad_theta_F[1][k0][feature][param] / self.T
                        self.theta[1][k0][feature][param] += delta

            return

        for iter in range(max_iters):

            # pick index
            t = np.random.choice(self.T)

            # get old gradient
            if method == "SVRG":

                # get old parameters and weights
                grad_log_f_t = self.get_log_f(t,theta=self.theta_tilde)[1]
                grad_log_Gamma = self.get_log_Gamma_full(t,eta=self.eta_tilde)[1]

                p_Xt = self.p_Xt_tilde[t]
                if t != self.T - 1:
                    p_Xtm1_Xt = self.p_Xtm1_Xt_tilde[t]

                # calculate old gradient
                old_grad_theta_F_t = self.get_grad_theta_F_t(t,
                                                             grad_log_f_t=grad_log_f_t,
                                                             p_Xt=p_Xt)
                if t != self.T-1:
                    old_grad_eta_G_t = self.get_grad_eta_G_t(t,
                                                             grad_log_Gamma=grad_log_Gamma,
                                                             p_Xtm1_Xt=p_Xtm1_Xt)

            else:

                # get old gradient at index
                old_grad_theta_F_t = deepcopy(self.grad_theta_F_t[t])
                if t != self.T-1:
                    old_grad_eta_G_t = deepcopy(self.grad_eta_G_t[t])

            # update alpha, beta, p_Xt, p_Xtm1_Xt
            if partial_E:

                self.update_alpha(t)
                self.update_beta(t)
                self.update_p_Xt(t)
                if t != 0:
                    self.update_p_Xtm1_Xt(t)
                if t != self.T-1:
                    self.update_p_Xtm1_Xt(t+1)

            # get new gradient at index
            new_grad_theta_F_t = self.get_grad_theta_F_t(t)
            new_grad_eta_G_t = self.get_grad_eta_G_t(t)

            # update parameters
            if method == "EM":

                # TODO: impletement a partial EM algorithm
                if not partial_E:
                    break

            elif method == "SGD":

                # update step size
                alpha_eta0 = alpha_eta / np.sqrt(iter+1)
                alpha_theta0 = alpha_theta / np.sqrt(iter+1)

                # update eta
                if t != self.T-1:
                    delta = alpha_eta0 * new_grad_eta_G_t
                    self.eta[0] += delta

                # update theta
                for feature in new_grad_theta_F_t[0]:
                    for param in ['mu','log_sig']:
                        delta = alpha_theta0 * new_grad_theta_F_t[0][feature][param]
                        self.theta[0][feature][param] += delta

            elif method == "SAG":

                # update eta
                if t != self.T-1:
                    delta = alpha_eta * (new_grad_eta_G_t \
                                   - old_grad_eta_G_t \
                                   + self.grad_eta_G)/self.T
                    self.eta[0] += delta

                # update theta
                for feature in self.theta[0]:
                    for param in ['mu','log_sig']:
                        delta = alpha_theta * (new_grad_theta_F_t[0][feature][param] \
                                       - old_grad_theta_F_t[0][feature][param] \
                                       + self.grad_theta_F[0][feature][param])/self.T
                        self.theta[0][feature][param] += delta

            elif method == "SVRG":

                # update eta
                if t != self.T-1:
                    delta = alpha_eta * (new_grad_eta_G_t \
                                   - old_grad_eta_G_t \
                                   + self.grad_eta_G_tilde/self.T)
                    self.eta[0] += delta

                # update theta
                for feature in self.theta[0]:
                    for param in ['mu','log_sig']:
                        delta = alpha_theta * (new_grad_theta_F_t[0][feature][param] \
                                       - old_grad_theta_F_t[0][feature][param] \
                                       + self.grad_theta_F_tilde[0][feature][param]/self.T)
                        self.theta[0][feature][param] += delta

            elif method == "SAGA":

                # update eta
                if t != self.T-1:
                    delta = alpha_eta * (new_grad_eta_G_t \
                                       - old_grad_eta_G_t \
                                       + self.grad_eta_G/self.T)
                    self.eta[0] += delta

                # update theta
                for feature in self.theta[0]:
                    for param in ['mu','log_sig']:
                        delta = alpha_theta * (new_grad_theta_F_t[0][feature][param] \
                                       - old_grad_theta_F_t[0][feature][param] \
                                       + self.grad_theta_F[0][feature][param]/self.T)
                        self.theta[0][feature][param] += delta

            else:
                raise("method %s not recognized" % method)

            # check the Lipshitz constants
            self.check_L_theta(t)
            alpha_theta = 1.0/(3.0*self.L_theta)

            self.check_L_eta(t)
            alpha_eta = 1.0/(3.0*self.L_eta)

            # clip values
            self.eta[0] = np.clip(self.eta[0],-10,10)
            for feature in self.theta[0]:
                for param in ['mu','log_sig']:
                    self.theta[0][feature][param] = np.clip(self.theta[0][feature][param],
                                                            self.param_bounds[feature][param][0],
                                                            self.param_bounds[feature][param][1])

            # update average gradient
            if t != self.T-1:
                self.grad_eta_G[0] += new_grad_eta_G_t[0] - old_grad_eta_G_t

            for feature in self.theta[0]:
                for param in ['mu','log_sig']:
                    self.grad_theta_F[0][feature][param] += \
                        new_grad_theta_F_t[0][feature][param] \
                      - old_grad_theta_F_t[0][feature][param]

            # update table of gradients
            self.grad_theta_F_t[t] = new_grad_theta_F_t
            if t != self.T-1:
                self.grad_eta_G_t[t] = new_grad_eta_G_t

            # record Gamma
            self.Gamma = eta_2_Gamma(self.eta)

            # record trace
            self.theta_trace.append(deepcopy(self.theta))
            self.eta_trace.append(deepcopy(self.eta))

            # record likelihood and check for convergence every T iterations
            if (iter % (2*self.T) == (2*self.T-1)):

                # update epoch
                if partial_E:
                    self.epoch_num += 1.0
                else:
                    self.epoch_num += 1.0

                if record_like:

                    # record time trace
                    self.train_time += time.time() - self.start_time
                    self.time_trace.append(self.train_time)
                    self.epoch_trace.append(self.epoch_num)

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

    def direct_maximization(self,num_epochs=None,method="CG",tol=1e-5):

        options = {'maxiter':num_epochs,'disp':True}

        def loss_fn(x):

            # hold on to backups
            theta_backup = deepcopy(self.theta)
            eta_backup = deepcopy(self.eta)

            # update parameters
            ind = 0

            # update theta
            for feature in self.theta[0]:
                for param in ['mu','log_sig']:
                    for k in range(self.K[0]):
                        self.theta[0][feature][param][k] = x[ind]
                        ind += 1

            # update eta
            for i in range(self.K[0]):
                for j in range(self.K[0]):
                    if i != j:
                        self.eta[0][i,j] = x[ind]
                        ind += 1

            # update Gamma
            self.Gamma = eta_2_Gamma(self.eta)

            # calculate likelihood
            self.E_step(update_tilde=False)

            # get likelihood
            ll = logsumexp(self.log_alphas[self.T-1])

            # get the gradient using Fischer
            grad_ll = np.zeros_like(x)
            ind = 0

            # update theta
            for feature in self.theta[0]:
                for param in ['mu','log_sig']:
                    for k in range(self.K[0]):
                        grad_ll[ind] = -self.grad_theta_F[0][feature][param][k]
                        ind += 1

            # update eta
            for i in range(self.K[0]):
                for j in range(self.K[0]):
                    if i != j:
                        grad_ll[ind] = -self.grad_eta_G[i,j]
                        ind += 1

            # record stuff
            self.epoch_num += 1.0
            self.epoch_trace.append(self.epoch_num)
            self.time_trace.append(time.time() - self.start_time)
            self.log_like_trace.append(ll / self.T)
            print(ll)
            self.theta_trace.append(deepcopy(self.theta))
            self.eta_trace.append(deepcopy(self.eta))

            # return parameters
            self.theta = theta_backup
            self.eta = eta_backup

            return (-ll,grad_ll)

        # initialize x0
        x0 = []

        # set theta
        for feature in self.theta[0]:
            for param in ['mu','log_sig']:
                for k in range(self.K[0]):
                    x0.append(self.theta[0][feature][param][k])

        # set eta
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                if i != j:
                    x0.append(self.eta[0][i,j])

        # fit x
        res = minimize(loss_fn, x0,
                       method=method, tol=tol,
                       options=options, jac=True)
        x = res["x"]

        ind = 0

        # update theta
        for feature in self.theta[0]:
            for param in ['mu','log_sig']:
                for k in range(self.K[0]):
                    self.theta[0][feature][param][k] = x[ind]
                    ind += 1

        # update eta
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                if i != j:
                    self.eta[0][i,j] = x[ind]
                    ind += 1

        return res

    def train_HHMM(self,num_epochs=10,max_iters=None,alpha_theta=None,alpha_eta=None,tol=1e-5,grad_tol=1e-3,method="EM",partial_E=False,record_like=False):

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
            res = self.direct_maximization(num_epochs=num_epochs,
                                           method=method,tol=tol)
            print(res)

            # record time
            self.train_time += time.time()-self.start_time

            return

        ll_old = -np.infty

        while self.epoch_num < num_epochs:

            print("starting epoch %.1f" % (self.epoch_num))
            print("")

            # do E-step
            print("starting E-step...")
            self.E_step()
            print("...done")

            # record log-likelihood
            ll_new = self.est_log_like(self.T-1)
            print(ll_new)

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

            # start timer back up
            self.start_time = time.time()

            # check for convergence
            if ((ll_new - ll_old)/np.abs(ll_old)) < tol:
                print("relative change of log likelihood is less than %.1E. returning..." % tol)
                self.train_time += time.time() - self.start_time
                return
            else:
                ll_old = ll_new

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


        print("maximum number of epochs (%.1f) reached. Returning..." % self.epoch_num)

        return
