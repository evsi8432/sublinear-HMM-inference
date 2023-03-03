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

from itertools import product

from copy import deepcopy

import warnings
import time
import pickle
import sys

from helper_funcs import eta_2_log_Gamma
from helper_funcs import log_Gamma_2_eta
from helper_funcs import eta0_2_log_delta
from helper_funcs import log_delta_2_eta0
from helper_funcs import logdotexp

class HHMM:

    def __init__(self,data,features,K):

        '''
        constructor for HHMM class
        '''

        self.data = data

        # parameters
        self.K = K
        self.K_total = self.K[0]*self.K[1]
        self.T = len(data)
        self.jump_every = 1
        self.features = features
        self.stationary_delta = False

        # indices where sequences start and end
        self.initial_ts = np.array([0])
        self.final_ts = np.array([self.T-1])

        # get log likelihood and grad ll
        self.ll = None

        # thetas and etas
        self.initialize_theta(data)
        self.initialize_eta()
        self.initialize_eta0()

        # get Gamma and delta
        self.get_log_Gamma(jump=False)
        self.get_log_Gamma(jump=True)
        self.get_log_delta()

        # alpha and beta
        self.log_alphas = np.zeros((self.T,self.K_total))
        self.log_betas = np.zeros((self.T,self.K_total))

        # p_Xt and p_Xtm1_Xt
        self.p_Xt = np.zeros((self.T,self.K_total))
        self.p_Xtm1_Xt = np.zeros((self.T,self.K_total,self.K_total))

        # initialize x
        self.initialize_nparams()
        self.x = self.params_2_x()
        self.x_2_params(self.x)

        return

    def initialize_theta(self,data):

        if data is None:
            print("no data")
            return

        theta = [{} for _ in range(self.K[0])]
        theta_mus = [{} for _ in range(self.K[0])]
        theta_sigs = [{} for _ in range(self.K[0])]

        param_bounds = {}

        initial_inds = np.random.choice(len(data),size=(self.K[0],self.K[1]))

        for feature,settings in self.features.items():

            feature_data = [datum[feature] for datum in data]

            param_bounds[feature] = {}

            for k0 in range(self.K[0]):

                # initialize values
                theta[k0][feature] = {}
                theta_mus[k0][feature] = {}
                theta_sigs[k0][feature] = {}

                if settings['f'] == 'normal':

                    #theta[k0][feature]['mu'] = np.array([feature_data[initial_inds[k0,k1]] for k1 in range(self.K[1])])
                    theta[k0][feature]['mu'] = np.nanmean(feature_data)*np.ones(self.K[1])
                    theta[k0][feature]['log_sig'] = np.ones(self.K[1])*np.log(np.nanstd(feature_data))

                    #theta[k0][feature]['mu'] += norm.rvs(np.zeros(self.K[1]),0.1*np.exp(theta[k0][feature]['log_sig']))
                    theta[k0][feature]['mu'] += norm.rvs(np.zeros(self.K[1]),np.exp(theta[k0][feature]['log_sig']))
                    #theta[k0][feature]['log_sig'] += norm.rvs(0.0,0.25,size=self.K[1])
                    theta[k0][feature]['log_sig'] = np.maximum(theta[k0][feature]['log_sig'],np.log(0.001))

                    # define priors
                    theta_mus[k0][feature]['mu'] = np.nanmean(feature_data) * np.ones(self.K[1])
                    theta_sigs[k0][feature]['mu'] = (np.nanmax(feature_data)-np.nanmin(feature_data)) * np.ones(self.K[1])

                    theta_mus[k0][feature]['log_sig'] = np.log(np.nanstd(feature_data)) - np.log(self.K_total + 1) * np.ones(self.K[1])
                    theta_sigs[k0][feature]['log_sig'] = np.log(self.K_total + 1) * np.ones(self.K[1])

                    if k0 == 0:
                        param_bounds[feature]['mu'] = [min(feature_data),
                                                       max(feature_data)]

                        param_bounds[feature]['log_sig'] = [np.log(np.nanmin(np.diff(sorted(set(feature_data))))/2),
                                                            np.log(np.nanmax(feature_data) -
                                                                   np.nanmin(feature_data))]

                elif settings['f'] == 'bern':

                    theta[k0][feature]['logit_p'] = np.ones(self.K[1])*logit(np.nanmean(feature_data))
                    theta[k0][feature]['logit_p'] += norm.rvs(0.0,1.0,size=self.K[1])

                    if k0 == 0:
                        param_bounds[feature]['logit_p'] = [logit(0.1/self.T),logit(1.0-0.1/self.T)]

                else:
                    pass

        self.theta = theta
        self.theta_mus = theta_mus
        self.theta_sigs = theta_sigs
        self.param_bounds = param_bounds

        return

    def initialize_eta(self):

        # initialize eta
        self.eta = []
        self.eta_mus = []
        self.eta_sigs = []

        # fill in coarse eta
        eta_crude = -2.0 + 2.0*np.random.normal(size=(self.K[0],self.K[0]))
        for i in range(self.K[0]):
            eta_crude[i,i] = 0
        self.eta.append(eta_crude)

        # fill in coarse eta prior
        self.eta_mus.append(np.zeros((self.K[0],self.K[0])))
        self.eta_sigs.append((np.log(self.T)/3.0) * np.ones((self.K[0],self.K[0])))

        # fill in fine eta
        eta_fine = []
        eta_fine_mus = []
        eta_fine_sigs = []
        for _ in range(self.K[0]):
            eta_fine_k = -1.0 + np.random.normal(size=(self.K[1],self.K[1]))
            eta_fine_mu_k = np.zeros((self.K[1],self.K[1]))
            eta_fine_sig_k = (np.log(self.T)/3.0) * np.ones((self.K[1],self.K[1]))
            for i in range(self.K[1]):
                eta_fine_k[i,i] = 0
            eta_fine.append(eta_fine_k)
            eta_fine_mus.append(eta_fine_mu_k)
            eta_fine_sigs.append(eta_fine_sig_k)

        self.eta.append(eta_fine)
        self.eta_mus.append(eta_fine_mus)
        self.eta_sigs.append(eta_fine_sigs)

        return

    def initialize_eta0(self):

        # initialize eta0
        self.eta0 = [np.concatenate(([0.0],np.random.normal(size=(self.K[0]-1)))),
                     [np.concatenate(([0.0],np.random.normal(size=(self.K[1]-1)))) \
                      for _ in range(self.K[0])]]

        # initialize eta0 prior
        self.eta0_mus = [np.zeros(self.K[0]),
                         [np.zeros(self.K[1]) for _ in range(self.K[0])]]

        N_whales = len(self.initial_ts)
        self.eta0_sigs = [np.log(N_whales+1)/3.0 * np.ones((self.K[0])),
                          [np.log(self.T)/3.0 * np.ones((self.K[1])) for _ in range(self.K[0])]]

        return

    def initialize_nparams(self):

        self.nparams = 0

        # add params from theta
        for feature,settings in self.features.items():
            for param in self.theta[0][feature]:
                if settings['share_coarse'] and settings['share_fine']:
                    self.nparams += 1
                elif settings['share_coarse']:
                    self.nparams += self.K[1]
                elif settings['share_fine']:
                    self.nparams += self.K[0]
                else:
                    self.nparams += self.K[0]*self.K[1]

        self.nparams += self.K[0]**2 # eta-coarse
        self.nparams += self.K[0]*self.K[1]**2 # eta-fine

        self.nparams += self.K[0] # eta0-coarse
        self.nparams += self.K[0]*self.K[1] # eta0-fine

        return

    def get_log_f(self,t,theta=None):

        if theta is None:
            theta = deepcopy(self.theta)

        # define the data point
        y = self.data[t]

        # initialize the log-likelihood
        log_f = np.zeros(self.K_total)

        # go through each feature and add to log-likelihood
        for feature,value in y.items():

            if feature not in self.features:
                continue

            # define the feature
            dist = self.features[feature]['f']

            if dist == 'normal':

                mu = np.concatenate([theta_i[feature]['mu'] for theta_i in theta])
                log_sig = np.concatenate([theta_i[feature]['log_sig'] for theta_i in theta])
                sig = np.exp(log_sig)
                a = self.features[feature]['lower_bound']
                b = self.features[feature]['upper_bound']

                if np.isnan(y[feature]):
                    pass
                elif (a is None) or (b is None):
                    log_f += norm.logpdf(y[feature],loc=mu,scale=sig)
                else:
                    a = np.concatenate([a for _ in theta])
                    b = np.concatenate([b for _ in theta])

                    log_f += truncnorm.logpdf(y[feature],
                                              a=(a-mu)/sig,
                                              b=(b-mu)/sig,
                                              loc=mu,scale=sig)

            elif dist == "gamma":

                mu = np.concatenate([theta_i[feature]['mu'] for theta_i in theta])
                log_sig = np.concatenate([theta_i[feature]['log_sig'] for theta_i in theta])
                sig = np.exp(log_sig)

                shape = mu**2 / sig**2
                scale = sig**2 / mu

                if np.isnan(y[feature]):
                    pass
                else:
                    log_f += gamma.logpdf(y[feature],shape,scale=scale)    

            elif dist == 'bern':

                logit_p = np.concatenate([theta_i[feature]['logit_p'] for theta_i in theta])

                if np.isnan(y[feature]):
                    pass
                elif y[feature] == 0:
                    log_f += np.log(expit(-logit_p))
                elif y[feature] == 1:
                    log_f += np.log(expit(logit_p))
                else:
                    print("invalid data point %s for %s, which is bernoulli"%(y[feature],feature))

            elif dist.startswith("cat"):

                ncats = int(dist[3:])

                log_py = np.zeros(self.K_total)
                for i in range(self.K[0]):
                    for j in range(self.K[1]):
                        psis = [0.0]+[theta[i][feature]["psi%d"%cat_num][j] for cat_num in range(1,ncats)]
                        log_py[(i*K[1]) + j] = psis[y[feature]] - logsumexp(psis)

                log_f += log_py

            else:
                print("unidentified emission distribution %s for %s"%(dist,feature))
                return

        # return the result
        return log_f

    def get_log_delta(self,eta=None,eta0=None):

        # get delta
        if self.stationary_delta:

            if eta is None:
                eta = self.eta

            log_Gammas = eta_2_log_Gamma(eta)

            coarse_Gamma = np.exp(log_Gammas[0])
            fine_Gammas = [np.exp(log_fine_Gamma) for log_fine_Gamma in log_Gammas[1]]

            coarse_delta = np.linalg.solve((np.eye(self.K[0])-coarse_Gamma+1).transpose(),np.ones(self.K[0]))
            coarse_delta = np.repeat(coarse_delta,self.K[1])
            log_coarse_delta = np.log(coarse_delta)

            fine_deltas = []
            for fine_Gamma in fine_Gammas:
                fine_deltas.append(np.linalg.solve((np.eye(self.K[1])-fine_Gamma+1).transpose(),np.ones(self.K[1])))
            log_fine_delta = np.concatenate([np.log(fine_delta) for fine_delta in fine_deltas])

        else:
            if eta0 is None:
                eta0 = self.eta0

            log_coarse_delta = np.repeat(eta0[0] - logsumexp(eta0[0]),self.K[1])
            log_fine_delta = np.concatenate([(eta01 - logsumexp(eta01)) for eta01 in eta0[1]])

        log_delta = log_coarse_delta + log_fine_delta
        self.log_delta = np.copy(log_delta)

        return log_delta

    def get_log_Gamma(self,eta=None,eta0=None,jump=True):

        if eta is None:
            eta = self.eta

        if eta0 is None:
            eta0 = self.eta0

        # get fine and coarse scale Gammas
        log_Gammas = eta_2_log_Gamma(eta)
        log_coarse_Gamma = log_Gammas[0]
        log_fine_Gammas = [log_fine_Gamma for log_fine_Gamma in log_Gammas[1]]

        # get the fine-scale deltas
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

        if jump:
            self.log_Gamma_jump = np.copy(log_Gamma)
        else:
            self.log_Gamma = np.copy(log_Gamma)

        return log_Gamma

    def get_log_p_theta(self,theta=None):

        log_p = 0.0

        if theta is None:
            theta = self.theta

        for k0 in range(self.K[0]):
            for feature in theta[k0]:
                for param in theta[k0][feature]:
                    log_p += np.sum(norm.logpdf(theta[k0][feature][param],
                                                loc=self.theta_mus[k0][feature][param],
                                                scale=self.theta_sigs[k0][feature][param]))

        return log_p

    def get_log_p_eta(self,eta=None):

        if eta is None:
            eta = self.eta

        log_p = 0.0

        # add eta_coarse
        eta_coarse = eta[0]
        eta_coarse_mu = self.eta_mus[0]
        eta_coarse_sig = self.eta_sigs[0]
        log_p += np.sum(norm.logpdf(eta_coarse,
                                    loc = eta_coarse_mu,
                                    scale = eta_coarse_sig))

        # add eta_fine
        for k0 in range(self.K[0]):
            eta_fine = eta[1][k0]
            eta_fine_mu = self.eta_mus[1][k0]
            eta_fine_sig = self.eta_sigs[1][k0]
            log_p += np.sum(norm.logpdf(eta_fine,
                                        loc = eta_fine_mu,
                                        scale = eta_fine_sig))

        return log_p

    def get_log_p_eta0(self,eta0=None):

        if eta0 is None:
            eta0 = self.eta0

        log_p = 0.0

        # add eta_coarse
        eta0_coarse = eta0[0]
        eta0_coarse_mu = self.eta0_mus[0]
        eta0_coarse_sig = self.eta0_sigs[0]
        log_p += np.sum(norm.logpdf(eta0_coarse,
                                    loc = eta0_coarse_mu,
                                    scale = eta0_coarse_sig))

        # add eta_fine
        for k0 in range(self.K[0]):
            eta0_fine = eta0[1][k0]
            eta0_fine_mu = self.eta0_mus[1][k0]
            eta0_fine_sig = self.eta0_sigs[1][k0]
            log_p += np.sum(norm.logpdf(eta0_fine,
                                        loc = eta0_fine_mu,
                                        scale = eta0_fine_sig))

        return log_p

    def update_alpha(self,t):

        # get log_f
        log_f = self.get_log_f(t)

        # get initial index
        seq_num = np.argmax(self.initial_ts > t)-1
        t0 = self.initial_ts[seq_num]

        # update log_alpha
        if t == 0:
            self.log_alphas[t] = self.log_delta + log_f
        elif t == t0:
            self.log_alphas[t] = logsumexp(self.log_alphas[t-1]) + self.log_delta + log_f
        elif (t-t0) % self.jump_every == 0:
            self.log_alphas[t] = logdotexp(self.log_alphas[t-1],self.log_Gamma_jump) + log_f
        else:
            self.log_alphas[t] = logdotexp(self.log_alphas[t-1],self.log_Gamma) + log_f

        return

    def update_beta(self,t):

        # get initial and final indices
        seq_num = np.argmax(self.initial_ts > t)-1
        t0 = self.initial_ts[seq_num]
        tf = self.final_ts[seq_num]

        # update log_beta
        if t == self.T-1:
            self.log_betas[t] = np.zeros(self.K_total)
        elif t == tf:
            log_f_tp1 = self.get_log_f(t+1)
            self.log_betas[t] = np.ones(self.K_total) * \
                                logsumexp(self.log_betas[t+1] + log_f_tp1 + \
                                          self.log_delta)
        elif ((t-t0) % self.jump_every == 1) or (self.jump_every == 1):
            log_f_tp1 = self.get_log_f(t+1)
            self.log_betas[t] = logdotexp(self.log_betas[t+1] + log_f_tp1,
                                          np.transpose(self.log_Gamma_jump))
        else:
            log_f_tp1 = self.get_log_f(t+1)
            self.log_betas[t] = logdotexp(self.log_betas[t+1] + log_f_tp1,
                                          np.transpose(self.log_Gamma))
        return

    def update_p_Xt(self,t):

        ll = logsumexp(self.log_alphas[t] + self.log_betas[t])
        self.p_Xt[t] = np.exp(self.log_alphas[t] + self.log_betas[t] - ll)

        return

    def update_p_Xtm1_Xt(self,t,log_f=None):

        # get initial and final indices
        seq_num = np.argmax(self.initial_ts > t)-1
        t0 = self.initial_ts[seq_num]

        if t == t0:
            raise("p_Xtm1_Xt not defined for t = t0")

        if log_f is None:
            log_f = self.get_log_f(t)

        if (t-t0) % self.jump_every == 0:
            log_Gamma = self.log_Gamma_jump
        else:
            log_Gamma = self.log_Gamma

        p_XX = np.zeros((self.K_total,self.K_total))
        for i in range(self.K_total):
            for j in range(self.K_total):
                p_XX[i,j] = self.log_alphas[t-1,i] \
                            + log_Gamma[i,j] \
                            + log_f[j] \
                            + self.log_betas[t,j]

        self.p_Xtm1_Xt[t] = np.exp(p_XX - logsumexp(p_XX))
        return

    def x_2_params(self,x):

        '''
        set parameters of HHMM with "x" output from optimizors
        '''

        # update parameters
        ind = 0

        # update theta
        for feature,settings in self.features.items():

            if settings['share_coarse'] and settings['share_fine']:
                for param in self.theta[0][feature]:
                    for k0,k1 in product(range(self.K[0]),range(self.K[1])):
                        self.theta[k0][feature][param][k1] = x[ind]
                    ind += 1

            elif settings['share_fine']:
                for param in self.theta[0][feature]:
                    for k0 in range(self.K[0]):
                        for k1 in range(self.K[1]):
                            self.theta[k0][feature][param][k1] = x[ind]
                        ind += 1

            elif settings['share_coarse']:
                for param in self.theta[0][feature]:
                    for k1 in range(self.K[1]):
                        for k0 in range(self.K[0]):
                            self.theta[k0][feature][param][k1] = x[ind]
                        ind += 1

            else:
                for param in self.theta[0][feature]:
                    for k0,k1 in product(range(self.K[0]),range(self.K[1])):
                        self.theta[k0][feature][param][k1] = x[ind]
                        ind += 1

        # update eta coarse
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                if i == j:
                    self.eta[0][i,j] = 0.0
                else:
                    self.eta[0][i,j] = x[ind]
                ind += 1

        # update eta fine
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                for j in range(self.K[1]):
                    if i == j:
                        self.eta[1][k0][i,j] = 0.0
                    else:
                        self.eta[1][k0][i,j] = x[ind]
                    ind += 1

        # update eta0 coarse
        for i in range(self.K[0]):
            if i == 0:
                self.eta0[0][i] = 0.0
            else:
                self.eta0[0][i] = x[ind]
            ind += 1

        # update eta0 fine
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                if i == 0:
                    self.eta0[1][k0][i] = 0.0
                else:
                    self.eta0[1][k0][i] = x[ind]
                ind += 1

        # update log Gamma and log delta
        self.get_log_Gamma(jump=True)
        self.get_log_Gamma(jump=False)
        self.get_log_delta()

        return

    def params_2_x(self):

        '''
        set parameters of HHMM with "x" output from optimizors
        '''

        # update parameters
        x = np.zeros(self.nparams)
        ind = 0

        # update theta
        for feature,settings in self.features.items():

            if settings['share_coarse'] and settings['share_fine']:
                for param in self.theta[0][feature]:
                    for k0,k1 in product(range(1),range(1)):
                        x[ind] = self.theta[k0][feature][param][k1]
                    ind += 1

            elif settings['share_fine']:
                for param in self.theta[0][feature]:
                    for k0 in range(self.K[0]):
                        for k1 in range(1):
                            x[ind] = self.theta[k0][feature][param][k1]
                        ind += 1

            elif settings['share_coarse']:
                for param in self.theta[0][feature]:
                    for k1 in range(self.K[1]):
                        for k0 in range(1):
                            x[ind] = self.theta[k0][feature][param][k1]
                        ind += 1

            else:
                for param in self.theta[0][feature]:
                    for k0,k1 in product(range(self.K[0]),range(self.K[1])):
                        x[ind] = self.theta[k0][feature][param][k1]
                        ind += 1

        # update eta coarse
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                if i == j:
                    x[ind] = 0.0
                else:
                    x[ind] = self.eta[0][i,j]
                ind += 1

        # update eta fine
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                for j in range(self.K[1]):
                    if i == j:
                        x[ind] = 0.0
                    else:
                        x[ind] = self.eta[1][k0][i,j]
                    ind += 1

        # update eta0 coarse
        for i in range(self.K[0]):
            if i == 0:
                x[ind] = 0.0
            else:
                x[ind] = self.eta0[0][i]
            ind += 1

        # update eta0 fine
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                if i == 0:
                    x[ind] = 0.0
                else:
                    x[ind] = self.eta0[1][k0][i]
                ind += 1

        self.x = x

        return x

    def get_log_like(self):

        for t in range(self.T):
            self.update_alpha(t)

        return logsumexp(self.log_alphas[self.T-1])

    def get_mixing_time(self,Gamma,max_t=None,buffer_eps=1e-3):

        if max_t is None:
            max_t = int(np.sqrt(self.T)/2.0)

        # get stationary distribution
        evals,evecs = np.linalg.eig(Gamma.T)
        ind = np.where(np.isclose(evals,1))[0][0]
        pi = evecs[:,ind] / np.sum(evecs[:,ind])

        # get matrix to multiply
        Gamma0 = np.eye(Gamma.shape[0])

        for t in range(max_t):

            Gamma0 = np.matmul(Gamma0,Gamma)
            tvd = max(np.sum(np.abs(Gamma0 - pi),1))

            if tvd <= buffer_eps:
                return max(t,1)

        return max_t
