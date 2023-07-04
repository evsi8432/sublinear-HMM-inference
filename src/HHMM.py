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

    def __init__(self,data,features,share_params,K,fix_theta=None,fix_eta=None,fix_eta0=None):

        '''
        constructor for HHMM class
        '''

        self.data = data

        # parameters
        self.K = K
        self.K_total = self.K[0]*self.K[1]
        self.T = len(data)
        self.jump_inds = range(self.T)
        self.features = features

        # indices where sequences start and end
        self.initial_ts = np.array([0])
        self.final_ts = np.array([self.T-1])

        # shared params
        self.share_params = share_params

        # get log likelihood and grad ll
        self.ll = None

        # thetas and etas
        self.initialize_theta(data,fix_theta)
        self.initialize_eta(fix_eta)
        self.initialize_eta0(fix_eta0)

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

    def initialize_theta(self,data,fix_theta):

        if data is None:
            print("no data")
            return

        theta = [{} for _ in range(self.K[0])]

        if fix_theta is None:
            build_fix_theta = True
            fix_theta = [{} for _ in range(self.K[0])]
        else:
            build_fix_theta = False

        param_bounds = {}

        for feature,settings in self.features.items():

            feature_data = [datum[feature] for datum in data]
            param_bounds[feature] = {}

            for k0 in range(self.K[0]):

                # initialize values
                theta[k0][feature] = {}
                if build_fix_theta:
                    fix_theta[k0][feature] = {}

                dist = settings['f']

                if dist == 'normal':

                    theta[k0][feature]['mu'] = np.nanmean(feature_data)*np.ones(self.K[1])
                    theta[k0][feature]['log_sig'] = np.ones(self.K[1])*np.log(np.nanstd(feature_data)) - np.log(self.K[1])

                    theta[k0][feature]['mu'] += norm.rvs(np.zeros(self.K[1]),np.exp(theta[k0][feature]['log_sig']))
                    theta[k0][feature]['log_sig'] += norm.rvs(np.zeros(self.K[1]),1.0)
                    theta[k0][feature]['log_sig'] = np.maximum(theta[k0][feature]['log_sig'],np.log(0.001))

                    # deal with fixed parameters
                    if build_fix_theta:
                        fix_theta[k0][feature]['mu'] = np.array([None]*self.K[1])
                        fix_theta[k0][feature]['log_sig'] = np.array([None]*self.K[1])

                    if k0 == 0:
                        param_bounds[feature]['mu'] = [min(feature_data),
                                                       max(feature_data)]

                        param_bounds[feature]['log_sig'] = [-5.0,#np.log(np.nanmin(np.diff(sorted(set(feature_data))))/2),
                                                            np.log(np.nanmax(feature_data) -
                                                                   np.nanmin(feature_data))]

                elif dist == 'normal_AR':

                    theta[k0][feature]['mu'] = np.nanmean(feature_data)*np.ones(self.K[1])
                    theta[k0][feature]['log_sig'] = np.ones(self.K[1])*np.log(np.nanstd(feature_data))
                    theta[k0][feature]['logit_rho'] = -1.0*np.ones(self.K[1])

                    theta[k0][feature]['mu'] += norm.rvs(np.zeros(self.K[1]),np.exp(theta[k0][feature]['log_sig']))
                    theta[k0][feature]['log_sig'] = np.maximum(theta[k0][feature]['log_sig'],np.log(0.001))
                    theta[k0][feature]['logit_rho'] += norm.rvs(np.zeros(self.K[1]),1.0)

                    if build_fix_theta:
                        fix_theta[k0][feature]['mu'] = np.array([None]*self.K[1])
                        fix_theta[k0][feature]['log_sig'] = np.array([None]*self.K[1])
                        fix_theta[k0][feature]['logit_rho'] = np.array([None]*self.K[1])

                    if k0 == 0:
                        param_bounds[feature]['mu'] = [min(feature_data),
                                                       max(feature_data)]

                        param_bounds[feature]['log_sig'] = [-5.0,#np.log(np.nanmin(np.diff(sorted(set(feature_data))))/2),
                                                            np.log(np.nanmax(feature_data) -
                                                                   np.nanmin(feature_data))]

                        param_bounds[feature]['logit_rho'] = [-10.0,10.0]


                elif dist == 'gamma':

                    theta[k0][feature]['log_mu'] = np.log(np.nanmean(feature_data))*np.ones(self.K[1])
                    theta[k0][feature]['log_sig'] = np.ones(self.K[1])*np.log(np.nanstd(feature_data))

                    theta[k0][feature]['log_mu'] *= norm.rvs(np.ones(self.K[1]),np.log(1.1))
                    theta[k0][feature]['log_sig'] = np.maximum(theta[k0][feature]['log_sig'],np.log(0.001))

                    if build_fix_theta:
                        fix_theta[k0][feature]['log_mu'] = np.array([None]*self.K[1])
                        fix_theta[k0][feature]['log_sig'] = np.array([None]*self.K[1])

                    if k0 == 0:
                        param_bounds[feature]['log_mu'] = [np.log(min(feature_data)),
                                                           np.log(max(feature_data))]

                        param_bounds[feature]['log_sig'] = [-5.0,#np.log(np.nanmin(np.diff(sorted(set(feature_data))))/2),
                                                            np.log(np.nanmax(feature_data) -
                                                                   np.nanmin(feature_data))]

                elif dist == 'bern':

                    theta[k0][feature]['logit_p'] = np.ones(self.K[1])*logit(np.nanmean(feature_data))
                    theta[k0][feature]['logit_p'] += norm.rvs(0.0,1.0,size=self.K[1])

                    if build_fix_theta:
                        fix_theta[k0][feature]['logit_p'] = np.array([None]*self.K[1])

                    if k0 == 0:
                        param_bounds[feature]['logit_p'] = [-np.infty,np.infty]

                else:
                    pass

                # set parameters equal within fix_theta (if you have it)
                if not build_fix_theta:
                    for param in fix_theta[k0][feature]:
                        for k1 in range(self.K[1]):
                            if not (fix_theta[k0][feature][param][k1] is None):
                                theta[k0][feature][param][k1] = fix_theta[k0][feature][param][k1]

        self.theta = theta
        self.fix_theta = fix_theta
        self.param_bounds = param_bounds

        return

    def initialize_eta(self,fix_eta):

        # initialize eta
        self.eta = []

        # determine if we need to build fix_eta
        if fix_eta is None:
            self.fix_eta = []
            build_fix_eta = True
        else:
            self.fix_eta = fix_eta
            build_fix_eta = False

        # fill in coarse eta
        eta_crude = -3.0 + 1.0*np.random.normal(size=(self.K[0],self.K[0]))
        for i in range(self.K[0]):
            eta_crude[i,i] = 0
        self.eta.append(eta_crude)

        # fill in coarse fix_eta
        if build_fix_eta:
            self.fix_eta.append(np.array([[None for _ in range(self.K[0])] for _ in range(self.K[0])]))
        else:
            for i in range(self.K[0]):
                for j in range(self.K[0]):
                    if not (self.fix_eta[0][i,j] is None):
                        self.eta[0][i,j] = self.fix_eta[0][i,j]

        # fill in fine eta
        eta_fine = []
        fix_eta_fine = []
        for _ in range(self.K[0]):
            eta_fine_k = -1.0 + 1.0*np.random.normal(size=(self.K[1],self.K[1]))
            for i in range(self.K[1]):
                eta_fine_k[i,i] = 0
            eta_fine.append(eta_fine_k)
            fix_eta_fine.append(np.array([[None for _ in range(self.K[1])] for _ in range(self.K[1])]))

        # fill in fine eta
        self.eta.append(eta_fine)

        # fill in fine fix_eta
        if build_fix_eta:
            self.fix_eta.append(fix_eta_fine)
        else:
            for k0 in range(self.K[0]):
                for i in range(self.K[1]):
                    for j in range(self.K[1]):
                        if not (self.fix_eta[1][k0][i,j] is None):
                            self.eta[1][k0][i,j] = self.fix_eta[1][k0][i,j]

        return

    def initialize_eta0(self,fix_eta0):

        # initialize eta0
        self.eta0 = [np.concatenate(([0.0],np.random.normal(size=(self.K[0]-1)))),
                     [np.concatenate(([0.0],np.random.normal(size=(self.K[1]-1)))) \
                      for _ in range(self.K[0])]]

        # determine if we need to build fix_eta0
        if fix_eta0 is None:
            self.fix_eta0 = [np.array([None for _ in range(self.K[0])]),
                             [np.array([None for _ in range(self.K[1])]) \
                              for _ in range(self.K[0])]]
        else:
            self.fix_eta0 = fix_eta0

            for k0 in range(self.K[0]):

                if not (self.fix_eta0[0][k0] is None):
                    self.eta0[0][k0] = self.fix_eta0[0][k0]

                for k1 in range(self.K[1]):
                    if not (self.fix_eta0[1][k0][k1] is None):
                        self.eta0[1][k0][k1] = self.fix_eta0[1][k0][k1]

        return

    def initialize_nparams(self):

        self.nparams = len(self.share_params)

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


            elif dist == 'normal_AR':

                mu = np.concatenate([theta_i[feature]['mu'] for theta_i in theta])
                log_sig = np.concatenate([theta_i[feature]['log_sig'] for theta_i in theta])
                sig = np.exp(log_sig)
                logit_rho = np.concatenate([theta_i[feature]['logit_rho'] for theta_i in theta])
                rho = expit(logit_rho)

                if t == 0 or np.isnan(self.data[t-1][feature]):
                    y_tm1 = mu
                else:
                    y_tm1 = self.data[t-1][feature]

                if np.isnan(y[feature]):
                    pass
                else:
                    mu_t = rho*y_tm1 + (1.0-rho)*mu
                    log_f += norm.logpdf(y[feature],loc=mu_t,scale=sig)

            elif dist == "gamma":

                log_mu = np.concatenate([theta_i[feature]['log_mu'] for theta_i in theta])
                mu = np.exp(log_mu)
                log_sig = np.concatenate([theta_i[feature]['log_sig'] for theta_i in theta])
                sig = np.exp(log_sig)

                shape = mu**2 / sig**2
                scale = sig**2 / mu

                if np.isnan(y[feature]):
                    pass
                elif y[feature] <= 0:
                    raise("data to fit Gamma distribution has negative value")
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

                if jump:
                    log_Gamma[(self.K[1]*i):(self.K[1]*(i+1)),
                              (self.K[1]*j):(self.K[1]*(j+1))] += log_coarse_Gamma[i,j]

                    log_Gamma[(self.K[1]*i):(self.K[1]*(i+1)),
                              (self.K[1]*j):(self.K[1]*(j+1))] += np.tile(log_fine_deltas[j],[self.K[1],1])
                else:
                    if i == j:
                        log_Gamma[(self.K[1]*i):(self.K[1]*(i+1)),
                                  (self.K[1]*j):(self.K[1]*(j+1))] += log_fine_Gammas[j]
                    else:
                        log_Gamma[(self.K[1]*i):(self.K[1]*(i+1)),
                                  (self.K[1]*j):(self.K[1]*(j+1))] += -np.infty

        if jump:
            self.log_Gamma_jump = np.copy(log_Gamma)
        else:
            self.log_Gamma = np.copy(log_Gamma)

        return log_Gamma

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
        elif t in self.jump_inds:
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
        elif t+1 in self.jump_inds:
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

        if t in self.jump_inds:
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
        for share_param in self.share_params:
            for feature,param,k0,k1 in product(*share_param.values()):
                self.theta[k0][feature][param][k1] = np.copy(x[ind])
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
        for share_param in self.share_params:
            for feature,param,k0,k1 in product(*share_param.values()):
                x[ind] = np.copy(self.theta[k0][feature][param][k1])
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
