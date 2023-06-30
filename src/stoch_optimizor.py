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

from HHMM import HHMM
from optimizor import Optimizor

from helper_funcs import eta_2_log_Gamma
from helper_funcs import log_Gamma_2_eta
from helper_funcs import eta0_2_log_delta
from helper_funcs import log_delta_2_eta0

class StochOptimizor(Optimizor):

    def __init__(self,data,features,share_params,K,
                 fix_theta=None,fix_eta=None,fix_eta0=None):

        '''
        constructor for optimizor class
        '''

        # get all of the stuff from optim
        super(StochOptimizor, self).__init__(data,features,share_params,K,
                                             fix_theta=fix_theta,
                                             fix_eta=fix_eta,
                                             fix_eta0=fix_eta0)

        # copies for SVRG
        self.theta_tilde = deepcopy(self.theta)
        self.eta_tilde = deepcopy(self.eta)
        self.eta0_tilde = deepcopy(self.eta0)

        self.grad_theta_tilde = deepcopy(self.grad_theta)
        self.grad_eta_tilde = deepcopy(self.grad_eta)
        self.grad_eta0_tilde = deepcopy(self.grad_eta0)

        self.p_Xt_tilde = deepcopy(self.p_Xt)
        self.p_Xtm1_Xt_tilde = deepcopy(self.p_Xtm1_Xt)

        self.theta_tilde = deepcopy(self.theta)
        self.eta_tilde = deepcopy(self.eta)
        self.eta0_tilde = deepcopy(self.eta0)

        # initialize step sizes and parameter bounds
        self.step_size = 0.01

        # Lipshitz constants
        if self.step_size != 0:
            self.L_theta = 1.0/(3.0*self.step_size) #* np.ones(self.K_total)
            self.L_eta = 1.0/(3.0*self.step_size) #* np.ones((self.K_total,self.K_total))
        else:
            self.L_theta = np.infty #* np.ones(self.K_total)
            self.L_eta = np.infty #* np.ones((self.K_total,self.K_total))

        # divider of Lipshitz constant
        self.divider = 3.0

        # stuff for adam to work
        self.eta0_m = deepcopy(self.grad_eta0)
        self.eta0_v = deepcopy(self.grad_eta0)

        self.eta_m = deepcopy(self.grad_eta)
        self.eta_v = deepcopy(self.grad_eta)

        self.theta_m = deepcopy(self.grad_eta)
        self.theta_v = deepcopy(self.grad_theta)

        self.beta1 = 0.9#1.0 - np.sqrt(1.0/self.T)
        self.beta2 = 0.999#1.0 - (1.0/self.T)

        self.epsilon = 1e-8

        return

    def get_ll_keep_params(self):

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
        self.E_step()
        #self.update_tilde()
        ll = logsumexp(self.log_alphas[self.T-1])
        #ll += self.get_log_p_theta()
        #ll += self.get_log_p_eta()
        #ll += self.get_log_p_eta0()
        grad_norm = np.linalg.norm(self.grad_params_2_xprime())

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

        self.get_log_Gamma(jump=False)
        self.get_log_Gamma(jump=True)
        self.get_log_delta()

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

        # get initial index
        seq_num = np.argmax(self.initial_ts > t)-1
        t0 = self.initial_ts[seq_num]

        # Evaluate G for eta and eta_new
        if t == t0:
            G_t_new = -np.sum(np.nan_to_num(self.p_Xt[t] * self.get_log_delta(eta0=eta0_new)))
            G_t  = -np.sum(np.nan_to_num(self.p_Xt[t] * self.get_log_delta(eta0=self.eta0)))
        elif t in self.jump_inds:
            G_t_new = -np.sum(np.nan_to_num(self.p_Xtm1_Xt[t] * self.get_log_Gamma(eta=eta_new,eta0=eta0_new,jump=True)))
            G_t  = -np.sum(np.nan_to_num(self.p_Xtm1_Xt[t] * self.get_log_Gamma(eta=self.eta,eta0=self.eta0,jump=True)))
        else:
            G_t_new = -np.sum(np.nan_to_num(self.p_Xtm1_Xt[t] * self.get_log_Gamma(eta=eta_new,eta0=eta0_new,jump=False)))
            G_t  = -np.sum(np.nan_to_num(self.p_Xtm1_Xt[t] * self.get_log_Gamma(eta=self.eta,eta0=self.eta0,jump=False)))

        # add priors
        #G_t -= self.get_log_p_eta() / self.T + \
        #      self.get_log_p_eta0() / self.T
        #G_t_new -= self.get_log_p_eta(eta=eta_new) / self.T + \
        #          self.get_log_p_eta0(eta0=eta0_new) / self.T

        # check inequality
        check_again = False

        if grad_G_t_norm2 < 1e-8:
            pass
        elif G_t_new > G_t - grad_G_t_norm2 / (2*self.L_eta):
            self.L_eta *= 2
            check_again = True
        else:
            self.L_eta *= 2**(-1/(self.T-1))

        if check_again:
            self.check_L_theta(t)

        return

    def check_L_theta(self,t):

        # get the gradients at the given time points
        grad_F_t = self.get_grad_theta_t(t)

        # get the new theta
        theta_new = deepcopy(self.theta)
        for feature,settings in self.features.items():
            for param in grad_F_t[0][feature]:
                for k0 in range(self.K[0]):
                    for k1 in range(self.K[1]):
                        theta_new[k0][feature][param][k1] += grad_F_t[k0][feature][param][k1] / self.L_theta

        # find gradient norm (don't double count shared parameters)
        grad_F_t_norm2 = 0.0
        for share_param in self.share_params:
            k0 = share_param['K_coarse'][0]
            feature = share_param['features'][0]
            param = share_param['params'][0]
            k1 = share_param['K_fine'][0]
            grad_F_t_norm2 += grad_F_t[k0][feature][param][k1]**2

        F_t  = np.sum(-np.nan_to_num(self.p_Xt[t] * self.get_log_f(t,theta=deepcopy(self.theta))))
        F_t_new = np.sum(-np.nan_to_num(self.p_Xt[t] * self.get_log_f(t,theta=theta_new)))

        # add priors
        #F_t -= self.get_log_p_theta() / self.T
        #F_t_new -= self.get_log_p_theta(theta=theta_new) / self.T

        # check for inequality
        check_again = False

        if grad_F_t_norm2 < 1e-8:
            pass
        elif F_t_new > F_t - grad_F_t_norm2 / (2*self.L_theta):
            self.L_theta *= 2
            check_again = True
        else:
            self.L_theta *= 2**(-1/self.T)

        #for k0 in range(self.K[0]):
        #    for k1 in range(self.K[1]):
        #        ind = k0*self.K[1] + k1
        #        if grad_F_t_norm2[ind] < 1e-8:
        #            pass
        #        elif F_t_new[ind] > F_t[ind] - grad_F_t_norm2[ind] / (2*self.L_theta[ind]):
        #            self.L_theta[ind] *= 2
        #            check_again = True
        #        else:
        #            self.L_theta[ind] *= 2**(-1/self.T)

        #print(self.L_theta)

        if check_again:
            self.check_L_theta(t)

        return

    def update_tilde(self):

        self.grad_theta_tilde = deepcopy(self.grad_theta)
        self.grad_eta_tilde = deepcopy(self.grad_eta)
        self.grad_eta0_tilde = deepcopy(self.grad_eta0)

        self.p_Xt_tilde = deepcopy(self.p_Xt)
        self.p_Xtm1_Xt_tilde = deepcopy(self.p_Xtm1_Xt)

        self.theta_tilde = deepcopy(self.theta)
        self.eta_tilde = deepcopy(self.eta)
        self.eta0_tilde = deepcopy(self.eta0)

        return

    def update_adam(self):

        self.eta0_m = deepcopy(self.grad_eta0)
        self.eta_m = deepcopy(self.grad_eta)
        self.theta_m = deepcopy(self.grad_theta)

        # initialize theta_v
        for feature,dist in self.features.items():
            if dist['f'] == 'normal':
                for k0 in range(self.K[0]):

                    self.theta_v[k0][feature]['mu'] = 0
                    self.theta_v[k0][feature]['log_sig'] = 0

            elif dist['f'] == 'normal_AR':
                for k0 in range(self.K[0]):

                    self.theta_v[k0][feature]['mu'] = 0
                    self.theta_v[k0][feature]['log_sig'] = 0
                    self.theta_v[k0][feature]['logit_rho'] = 0

            elif dist['f'] == 'gamma':
                for k0 in range(self.K[0]):

                    self.theta_v[k0][feature]['log_mu'] = 0
                    self.theta_v[k0][feature]['log_sig'] = 0

            elif dist['f'] == 'bern':
                for k0 in range(self.K[0]):

                    self.theta_v[k0][feature]['logit_p'] = 0

            else:
                raise('only independent normal distributions supported at this time')

        # initialize eta_v
        self.eta_v = [np.zeros_like(self.eta_m[0]),
                      [np.zeros_like(self.eta_m[1][k0]) for k0 in range(self.K[0])]]

        # initialize eta0_v
        self.eta0_v = [np.zeros_like(self.eta0_m[0]),
                       [np.zeros_like(self.eta0_m[1][k0]) for k0 in range(self.K[0])]]

        # iterate through all data points
        for t in range(self.T):

            # update theta_v
            for feature,dist in self.features.items():
                if dist['f'] == 'normal':
                    for k0 in range(self.K[0]):

                        self.theta_v[k0][feature]['mu'] += \
                            self.grad_theta_t[t][k0][feature]['mu']**2

                        self.theta_v[k0][feature]['log_sig'] += \
                            self.grad_theta_t[t][k0][feature]['log_sig']**2

                elif dist['f'] == 'normal_AR':
                    for k0 in range(self.K[0]):

                        self.theta_v[k0][feature]['mu'] += \
                            self.grad_theta_t[t][k0][feature]['mu']**2

                        self.theta_v[k0][feature]['log_sig'] += \
                            self.grad_theta_t[t][k0][feature]['log_sig']**2

                        self.theta_v[k0][feature]['logit_rho'] += \
                            self.grad_theta_t[t][k0][feature]['logit_rho']**2

                elif dist['f'] == 'gamma':
                    for k0 in range(self.K[0]):

                        self.theta_v[k0][feature]['log_mu'] += \
                            self.grad_theta_t[t][k0][feature]['log_mu']**2

                        self.theta_v[k0][feature]['log_sig'] += \
                            self.grad_theta_t[t][k0][feature]['log_sig']**2

                elif dist['f'] == 'bern':
                    for k0 in range(self.K[0]):

                        self.theta_v[k0][feature]['logit_p'] += \
                            self.grad_theta_t[t][k0][feature]['logit_p']**2

                else:
                    raise('only independent normal distributions supported at this time')

            # update eta_v, eta0_v
            self.eta_v[0] += self.grad_eta_t[t][0]**2
            self.eta0_v[0] += self.grad_eta0_t[t][0]**2

            for k0 in range(k0):
                self.eta_v[1][k0] += self.grad_eta_t[t][1][k0]**2
                self.eta0_v[1][k0] += self.grad_eta0_t[t][1][k0]**2

        return

    def M_step(self,max_epochs=1,max_time=np.infty,alpha_theta=None,alpha_eta=None,method="EM",partial_E=False,tol=1e-5,record_like=False,weight_buffer="none",grad_buffer="none",buffer_eps=1e-3):

        if not weight_buffer in ["coarse","fine","none"]:
            print("buffer type not understood. Setting to 'none'")
            buffer = "none"

        if not grad_buffer in ["coarse","fine","none"]:
            print("buffer type not understood. Setting to 'none'")
            buffer = "none"

        if alpha_theta is None:
            alpha_theta = 1.0 / self.L_theta #np.divide(1.0,self.divider*self.L_theta,where=self.L_theta!=0.0)

        if alpha_eta is None:
            alpha_eta = np.divide(1.0,self.divider*self.L_eta,where=self.L_eta!=0.0)

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
            for k0 in range(self.K[0]):
                inds = slice(self.K[1]*k0,self.K[1]*(k0+1))
                for feature in self.grad_theta[k0]:
                    for param in self.grad_theta[k0][feature]:
                        delta = alpha_theta * self.grad_theta[k0][feature][param]
                        self.theta[k0][feature][param] += delta

            # clip values
            self.eta0[0] = np.clip(self.eta0[0],-1-np.log(self.T),1+np.log(self.T))
            self.eta[0]  = np.clip(self.eta[0], -1-np.log(self.T),1+np.log(self.T))
            for k0 in range(self.K[0]):
                self.eta0[1][k0] = np.clip(self.eta0[1][k0],-1-np.log(self.T),1+np.log(self.T))
                self.eta[1][k0]  = np.clip(self.eta[1][k0], -1-np.log(self.T),1+np.log(self.T))
                for feature in self.theta[k0]:
                    for param in self.theta[k0][feature]:
                        self.theta[k0][feature][param] = np.clip(self.theta[k0][feature][param],
                                                                 self.param_bounds[feature][param][0],
                                                                 self.param_bounds[feature][param][1])

            # update log_Gamma and log_delta
            self.get_log_Gamma(jump=False)
            self.get_log_Gamma(jump=True)
            self.get_log_delta()

            return

        for epoch_num in range(max_epochs):

            # pick grad_buffer size (minibatch size pretty much)
            if grad_buffer == "none":
                grad_buffer_size = 1
            elif grad_buffer == "fine":
                log_Gammas = eta_2_log_Gamma(self.eta)[1]
                grad_buffer_size = 2*max([self.get_mixing_time(np.exp(log_Gamma),buffer_eps=buffer_eps) for log_Gamma in log_Gammas])
            elif grad_buffer == "coarse":
                log_Gamma = eta_2_log_Gamma(self.eta)[0]
                grad_buffer_size = 2*self.get_mixing_time(np.exp(log_Gamma),buffer_eps=buffer_eps)
            else:
                print("unknown grad buffer scale: %d" % grad_buffer)
                grad_buffer_size = 1

            #print("grad buffer size: %d" % grad_buffer_size)
            #print("")

            # pick weight buffer size
            if weight_buffer == "none":
                weight_buffer_size = 0
            elif weight_buffer == "fine":
                log_Gammas = eta_2_log_Gamma(self.eta)[1]
                weight_buffer_size = max([self.get_mixing_time(np.exp(log_Gamma),buffer_eps=buffer_eps) for log_Gamma in log_Gammas])
            elif weight_buffer == "coarse":
                log_Gamma = eta_2_log_Gamma(self.eta)[0]
                weight_buffer_size = self.get_mixing_time(np.exp(log_Gamma),buffer_eps=buffer_eps)
            else:
                print("unknown weight buffer scale: %s" % weight_buffer)
                weight_buffer_size = 0

            #print("weight buffer size: %d" % weight_buffer_size)
            #print("")

            nbatches = int(self.T / grad_buffer_size)
            minibatches = [range(grad_buffer_size*i,min(grad_buffer_size*(i+1),self.T)) for i in range(nbatches)]
            batch_order = np.random.permutation(range(nbatches))
            iter = 0

            for batch_num,batch_ind in enumerate(batch_order):

                # pick index
                ts = minibatches[batch_ind]

                # get old gradient
                old_grad_thetas = []
                old_grad_etas = []
                old_grad_eta0s = []

                for t in ts:

                    if method == "SVRG":

                        # get old gradients
                        grad_log_f = self.get_grad_log_f(t,theta=deepcopy(self.theta_tilde))
                        #grad_log_p_theta = self.get_grad_log_p_theta(theta=deepcopy(self.theta_tilde))

                        # get initial index
                        seq_num = np.argmax(self.initial_ts > t)-1
                        t0 = self.initial_ts[seq_num]

                        if t == t0:
                            grad_eta0_log_delta,_ = self.get_grad_log_delta(eta0=self.eta0_tilde)
                            grad_eta_log_Gamma = [np.zeros((self.K[0],self.K[0],self.K[0],self.K[0])),
                                                  [np.zeros((self.K[1],self.K[1],self.K[1],self.K[1])) for _ in range(self.K[0])]]
                        elif t in self.jump_inds:
                            grad_eta0_log_delta,grad_eta_log_Gamma = self.get_grad_log_Gamma(eta=self.eta_tilde,
                                                                                             eta0=self.eta0_tilde,
                                                                                             jump=True)
                        else:
                            grad_eta0_log_delta,grad_eta_log_Gamma = self.get_grad_log_Gamma(eta=self.eta_tilde,
                                                                                             eta0=self.eta0_tilde,
                                                                                             jump=False)

                        #grad_log_p_eta = self.get_grad_log_p_eta(eta=deepcopy(self.eta_tilde))
                        #grad_log_p_eta0 = self.get_grad_log_p_eta0(eta0=deepcopy(self.eta0_tilde))

                        # get old weights
                        p_Xt = self.p_Xt_tilde[t]
                        p_Xtm1_X1 = self.p_Xtm1_Xt_tilde[t]

                        # calculate old gradient
                        old_grad_theta_t = self.get_grad_theta_t(t,
                                                                 grad_log_f=grad_log_f,
                                                                 #grad_log_p_theta=grad_log_p_theta,
                                                                 grad_log_p_theta=None,
                                                                 p_Xt=p_Xt)

                        old_grad_eta_t,old_grad_eta0_t = self.get_grad_eta_t(t,
                                                                             grad_eta0_log_delta=grad_eta0_log_delta,
                                                                             grad_eta_log_Gamma=grad_eta_log_Gamma,
                                                                             grad_log_p_eta=None,
                                                                             grad_log_p_eta0=None,
                                                                             #grad_log_p_eta=grad_log_p_eta,
                                                                             #grad_log_p_eta0=grad_log_p_eta0,
                                                                             p_Xtm1_Xt=p_Xtm1_X1)

                    else:

                        # get old gradient at index
                        old_grad_theta_t = deepcopy(self.grad_theta_t[t])
                        old_grad_eta0_t = deepcopy(self.grad_eta0_t[t])
                        old_grad_eta_t = deepcopy(self.grad_eta_t[t])

                    old_grad_thetas.append(old_grad_theta_t)
                    old_grad_etas.append(old_grad_eta_t)
                    old_grad_eta0s.append(old_grad_eta0_t)

                weight_change = 0.0

                # update weights
                if partial_E:

                    # pick index
                    weight_ts = range(max(0,min(ts)-weight_buffer_size), \
                                      min(self.T,max(ts)+weight_buffer_size+1))

                    for t in weight_ts:

                        old_p_Xt = np.copy(self.p_Xt[t])
                        old_p_Xtm1_Xt = np.copy(self.p_Xtm1_Xt[t])

                        self.update_alpha(t)
                        self.update_beta(t)
                        self.update_p_Xt(t)

                        seq_num = np.argmax(self.initial_ts > t)-1
                        t0 = self.initial_ts[seq_num]
                        tf = self.final_ts[seq_num]

                        if t != t0:
                            self.update_p_Xtm1_Xt(t)
                        if t != tf:
                            self.update_p_Xtm1_Xt(t+1)

                        weight_change += np.sum(np.abs(self.p_Xt[t]-old_p_Xt))
                        weight_change += np.sum(np.abs(self.p_Xtm1_Xt[t]-old_p_Xtm1_Xt))

                # update parameters
                for i,t in enumerate(ts):

                    # check Lipshitz constants
                    self.check_L_eta(t)
                    alpha_eta = 1.0/(self.divider*self.L_eta)

                    self.check_L_theta(t)
                    alpha_theta = 1.0/(self.divider*self.L_theta)

                    # get old gradients
                    old_grad_theta_t = old_grad_thetas[i]
                    old_grad_eta_t = old_grad_etas[i]
                    old_grad_eta0_t = old_grad_eta0s[i]

                    # get new gradients
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
                        for k0 in range(self.K[0]):
                            inds = slice(self.K[1]*k0,self.K[1]*(k0+1))
                            for feature in new_grad_theta_t[k0]:
                                for param in new_grad_theta_t[k0][feature]:
                                    delta = alpha_theta_m * new_grad_theta_t[k0][feature][param]
                                    self.theta[k0][feature][param] += delta

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
                        for k0 in range(self.K[0]):
                            inds = slice(self.K[1]*k0,self.K[1]*(k0+1))
                            for feature in new_grad_theta_t[k0]:
                                for param in new_grad_theta_t[k0][feature]:
                                    delta = alpha_theta * (new_grad_theta_t[k0][feature][param] \
                                                         - old_grad_theta_t[k0][feature][param] \
                                                         + self.grad_theta[k0][feature][param])/self.T
                                    self.theta[k0][feature][param] += delta

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
                        for k0 in range(self.K[0]):
                            inds = slice(self.K[1]*k0,self.K[1]*(k0+1))
                            for feature in new_grad_theta_t[k0]:
                                for param in new_grad_theta_t[k0][feature]:
                                    delta = alpha_theta * (new_grad_theta_t[k0][feature][param] \
                                                           - old_grad_theta_t[k0][feature][param] \
                                                           + self.grad_theta_tilde[k0][feature][param]/self.T)
                                    self.theta[k0][feature][param] += delta

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
                        for k0 in range(self.K[0]):
                            inds = slice(self.K[1]*k0,self.K[1]*(k0+1))
                            for feature in new_grad_theta_t[k0]:
                                for param in new_grad_theta_t[k0][feature]:
                                    delta = alpha_theta * (new_grad_theta_t[k0][feature][param] \
                                                               - old_grad_theta_t[k0][feature][param] \
                                                               + self.grad_theta[k0][feature][param]/self.T)
                                    self.theta[k0][feature][param] += delta

                    elif method == "Adam":

                        # update eta0
                        self.eta0_m[0] = self.beta1*self.eta0_m[0] + (1-self.beta1)*new_grad_eta0_t[0]
                        self.eta0_v[0] = self.beta2*self.eta0_v[0] + (1-self.beta2)*new_grad_eta0_t[0]**2
                        delta = self.step_size[0]*(self.eta0_m[0]/(np.sqrt(self.eta0_v[0]) + self.epsilon))
                        self.eta0[0] += delta

                        for k0 in range(self.K[0]):
                            self.eta0_m[1][k0] = self.beta1*self.eta0_m[1][k0] + (1-self.beta1)*new_grad_eta0_t[1][k0]
                            self.eta0_v[1][k0] = self.beta2*self.eta0_v[1][k0] + (1-self.beta2)*new_grad_eta0_t[1][k0]**2
                            delta = self.step_size[0]*(self.eta0_m[1][k0]/(np.sqrt(self.eta0_v[1][k0]) + self.epsilon))
                            self.eta0[1][k0] += delta


                        # update eta
                        self.eta_m[0] = self.beta1*self.eta_m[0] + (1-self.beta1)*new_grad_eta_t[0]
                        self.eta_v[0] = self.beta2*self.eta_v[0] + (1-self.beta2)*new_grad_eta_t[0]**2
                        delta = self.step_size[0]*(self.eta_m[0]/(np.sqrt(self.eta_v[0]) + self.epsilon))
                        self.eta[0] += delta

                        for k0 in range(self.K[0]):
                            self.eta_m[1][k0] = self.beta1*self.eta_m[1][k0] + (1-self.beta1)*new_grad_eta_t[1][k0]
                            self.eta_v[1][k0] = self.beta2*self.eta_v[1][k0] + (1-self.beta2)*new_grad_eta_t[1][k0]**2
                            delta = self.step_size[0]*(self.eta_m[1][k0]/(np.sqrt(self.eta_v[1][k0]) + self.epsilon))
                            self.eta[1][k0] += delta


                        # update theta
                        for k0 in range(self.K[0]):
                            for feature in new_grad_theta_t[k0]:
                                for param in new_grad_theta_t[k0][feature]:

                                    self.theta_m[k0][feature][param] = self.beta1*self.theta_m[k0][feature][param] + \
                                                                       (1-self.beta1)*new_grad_theta_t[k0][feature][param]

                                    self.theta_v[k0][feature][param] = self.beta2*self.theta_v[k0][feature][param] + \
                                                                       (1-self.beta2)*new_grad_theta_t[k0][feature][param]**2

                                    delta = self.step_size[1]*(self.theta_m[k0][feature][param]/(np.sqrt(self.theta_v[k0][feature][param]) + self.epsilon))
                                    self.theta[k0][feature][param] += delta

                        #print("t:")
                        #print(batch_num)
                        #print("")
                        #print("parameters:")
                        #print(self.theta)
                        #print("")
                        #print("gradients:")
                        #print(self.theta_m)
                        #print("")
                        #print("variances:")
                        #print(self.theta_v)
                        #print("")
                        #print("")
                    else:
                        raise("method %s not recognized" % method)

                    # clip values
                    #self.eta0[0] = np.clip(self.eta0[0],-1-np.log(self.T),1+np.log(self.T))
                    #self.eta[0]  = np.clip(self.eta[0], -1-np.log(self.T),1+np.log(self.T))
                    for k0 in range(self.K[0]):
                        #self.eta0[1][k0] = np.clip(self.eta0[1][k0],-1-np.log(self.T),1+np.log(self.T))
                        #self.eta[1][k0]  = np.clip(self.eta[1][k0], -1-np.log(self.T),1+np.log(self.T))
                        for feature in self.theta[k0]:
                            for param in self.theta[k0][feature]:
                                self.theta[k0][feature][param] = np.clip(self.theta[k0][feature][param],
                                                                         self.param_bounds[feature][param][0],
                                                                         self.param_bounds[feature][param][1])

                    # update Gamma and delta
                    self.get_log_Gamma(jump=False)
                    self.get_log_Gamma(jump=True)
                    self.get_log_delta()

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

                    for k0 in range(self.K[0]):
                        for feature in new_grad_theta_t[k0]:
                            for param in new_grad_theta_t[k0][feature]:
                                self.grad_theta[k0][feature][param] += new_grad_theta_t[k0][feature][param]
                                self.grad_theta[k0][feature][param] -= old_grad_theta_t[k0][feature][param]
                                self.grad_theta_t[t][k0][feature][param] = deepcopy(new_grad_theta_t[k0][feature][param])

                # print iteration
                if False:
                    print("batch number %d of %d" % (iter,nbatches))
                    print("")

                    # show current parameters
                    print("current parameters:")
                    print(self.theta)
                    print(self.eta)
                    print(self.eta0)
                    print("")

                    if method == "SVRG":
                        print("table averages:")
                        print(self.grad_theta_tilde)
                        print(self.grad_eta_tilde)
                        print(self.grad_eta0_tilde)
                        print("")
                    elif method == "Adam":
                        print("gradient means:")
                        print(self.theta_m)
                        print(self.eta_m)
                        print(self.eta0_m)
                        print("")
                        print("gradient varainces:")
                        print(self.theta_v)
                        print(self.eta_v)
                        print(self.eta0_v)
                        print("")
                        print("")
                    else:
                        print("table averages:")
                        print(self.grad_theta)
                        print(self.grad_eta)
                        print(self.grad_eta0)
                        print("")

                    print("average weight change:")
                    print(weight_change / (grad_buffer_size*self.K[0]*self.K[1]))
                    print("")

                    print("L_theta: ",self.L_theta)
                    print("alpha_theta: ",1.0 / (self.divider*self.L_theta))
                    print("L_eta: ",self.L_eta)
                    print("alpha_eta: ",1.0 / (self.divider*self.L_eta))
                    print("")

                # update iteration number
                iter += 1

            # record likelihood and check for convergence every epoch
            if epoch_num != 0.0:

                # update epoch
                if partial_E:
                    self.epoch_num += 2.0
                else:
                    self.epoch_num += 1.0

                print("starting epoch %.1f" % (self.epoch_num))
                print("")

                print("%.3f hours elapsed" % (self.train_time / 3600))
                print("")

                # show current parameters
                print("current parameters:")
                print(self.theta)
                print(self.eta)
                print(self.eta0)
                print("")

                if method == "Adam":
                    print("gradient means:")
                    print(self.theta_m)
                    print(self.eta_m)
                    print(self.eta0_m)
                    print("")
                    print("gradient varainces:")
                    print(self.theta_v)
                    print(self.eta_v)
                    print(self.eta0_v)
                    print("")
                    print("")
                else:
                    print("current table averages:")
                    print(self.grad_theta)
                    print(self.grad_eta)
                    print(self.grad_eta0)
                    print("")

                print("L_theta: ",self.L_theta)
                print("alpha_theta: ",1.0 / (self.divider*self.L_theta))
                print("L_eta: ",self.L_eta)
                print("alpha_eta: ",1.0 / (self.divider*self.L_eta))
                print("")

                # update grad tilde if using SVRG:
                if method == "SVRG":

                    # initialize gradients
                    self.initialize_grads()

                    # update probs and gradients
                    for t in range(self.T):

                        # update grad theta
                        self.grad_theta_t[t] = self.get_grad_theta_t(t)

                        for k0 in range(self.K[0]):
                            for feature in self.grad_theta[k0]:
                                for param in self.grad_theta[k0][feature]:
                                    self.grad_theta[k0][feature][param] += \
                                    self.grad_theta_t[t][k0][feature][param]

                        # update grad eta and eta0
                        self.grad_eta_t[t],self.grad_eta0_t[t] = self.get_grad_eta_t(t)

                        self.grad_eta[0] += self.grad_eta_t[t][0]
                        self.grad_eta0[0] += self.grad_eta0_t[t][0]

                        for k0 in range(self.K[0]):
                            self.grad_eta[1][k0] += self.grad_eta_t[t][1][k0]
                            self.grad_eta0[1][k0] += self.grad_eta0_t[t][1][k0]

                    self.update_tilde()

                    # show current gradients
                    print("new table averages:")
                    print(self.grad_theta_tilde)
                    print(self.grad_eta_tilde)
                    print(self.grad_eta0_tilde)
                    print("")

                if record_like:

                    # record time trace and stop timer
                    self.train_time += time.time() - self.start_time
                    self.time_trace.append(self.train_time)
                    self.epoch_trace.append(self.epoch_num)

                    # record parameter traces
                    self.theta_trace.append(deepcopy(self.theta))
                    self.eta_trace.append(deepcopy(self.eta))
                    self.eta0_trace.append(deepcopy(self.eta0))

                    # record log-likelihood
                    ll, grad_norm = self.get_ll_keep_params()
                    #print("current log-likelihood: %f" % ll)
                    # print("")
                    self.grad_norm_trace.append(grad_norm / self.T)
                    self.log_like_trace.append(ll / self.T)

                    # start timer back up
                    self.start_time = time.time()

                grad_norm = np.linalg.norm(self.grad_params_2_xprime()) / self.T
                if (grad_norm < tol):
                    print("M-step sucesssfully converged")
                    return
                elif (self.train_time > max_time):
                    print("Time limit reached within M step")
                    return

        print("M-step failed to converge: maximum number of iterations reached")

        return

    def train_HHMM_stoch(self,num_epochs=10,max_time=np.infty,max_epochs=1,alpha_theta=None,alpha_eta=None,tol=1e-5,grad_tol=1e-5,method="EM",partial_E=False,record_like=False,weight_buffer="none",grad_buffer="none",buffer_eps=1e-3):

        # check that the method makes sense
        if method not in ["EM","BFGS","Nelder-Mead","CG",
                          "GD","SGD","SAG","SVRG","SAGA","Adam"]:
            print("method %s not recognized" % method)
            return

        # start timer
        self.epoch_num = 0.0
        self.train_time = 0.0
        self.start_time = time.time()

        # do direct likelihood maximization
        if method in ["Nelder-Mead","BFGS","CG"]:
            res = self.train_HHMM(num_epochs=num_epochs,
                                  method=method,tol=tol,gtol=grad_tol,
                                  max_time=max_time)
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
            print("")

            # do E-step
            print("starting E-step...")
            self.E_step()
            self.update_tilde()
            self.update_adam()
            print("...done")
            print("")

            # show current parameters
            print("current parameters:")
            print(self.theta)
            print(self.eta)
            print(self.eta0)
            print("")

            if method == "Adam":
                print("gradient means:")
                print(self.theta_m)
                print(self.eta_m)
                print(self.eta0_m)
                print("")
                print("gradient varainces:")
                print(self.theta_v)
                print(self.eta_v)
                print(self.eta0_v)
                print("")
                print("")
            else:
                print("table averages:")
                print(self.grad_theta)
                print(self.grad_eta)
                print(self.grad_eta0)
                print("")

            print("L_theta: ",self.L_theta)
            print("alpha_theta: ",1.0 / (self.divider*self.L_theta))
            print("L_eta: ",self.L_eta)
            print("alpha_eta: ",1.0 / (self.divider*self.L_eta))
            print("")

            # record log-likelihood
            ll_new = logsumexp(self.log_alphas[self.T-1])
            #ll_new += self.get_log_p_theta()
            #ll_new += self.get_log_p_eta()
            #ll_new += self.get_log_p_eta0()
            print("current log-likelihood: %f" % ll_new)
            print("")

            # check for convergence
            if (ll_new < ll_old) or np.isnan(ll_new):

                print("log-likelihood decreased.")
                print("old log-likelihood: %f" % ll_old)
                print("new log-likelihood: %f" % ll_new)

                if method == "GD":
                    self.L_theta *= 2.0
                    self.L_eta *= 2.0
                elif method == "Adam":
                    self.step_size[0] *= 0.5
                    self.step_size[1] *= 0.5
                    print("step size decreased: [%.3f,%.3f]" % (self.step_size[0],
                                                                self.step_size[1]))
                else:
                    self.divider *= 2.0
                    print("step size decreased: 1/(%.3f * L)" % self.divider)

                # return old parameters
                self.theta = deepcopy(theta_old)
                self.eta = deepcopy(eta_old)
                self.eta0 = deepcopy(eta0_old)

                # return Gamma and delta
                self.get_log_Gamma(jump=False)
                self.get_log_Gamma(jump=True)
                self.get_log_delta()

                # return old gradients, weights, and likleihood
                self.E_step()
                self.update_tilde()
                self.update_adam()
                ll_new = logsumexp(self.log_alphas[self.T-1])
                #ll_new += self.get_log_p_theta()
                #ll_new += self.get_log_p_eta()
                #ll_new += self.get_log_p_eta0()

                print("returned log-likelihood: %f" % ll_new)
                print("Trying again...")

            elif ((ll_new - ll_old)/np.abs(ll_old)) < tol:

                print("relative change of log-likelihood is less than %.1E. returning..." % tol)
                self.train_time += time.time() - self.start_time
                return

            else:
                ll_old = ll_new

                if method == "GD":
                    self.L_theta /= 1.05
                    self.L_eta /= 1.05

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
            self.grad_norm_trace.append(np.linalg.norm(self.grad_params_2_xprime()) / self.T)
            self.theta_trace.append(deepcopy(self.theta))
            self.eta_trace.append(deepcopy(self.eta))
            self.eta0_trace.append(deepcopy(self.eta0))

            # start timer back up
            self.start_time = time.time()

            # do M-step
            print("starting M-step...")
            print("")
            self.M_step(max_epochs=max_epochs,
                        max_time=max_time,
                        method=method,
                        alpha_theta=alpha_theta,
                        alpha_eta=alpha_eta,
                        partial_E=partial_E,
                        tol=grad_tol,
                        record_like=record_like,
                        weight_buffer=weight_buffer,
                        grad_buffer=grad_buffer,
                        buffer_eps=buffer_eps)

            # record epoch from M-step
            if partial_E:
                self.epoch_num += 2.0
            else:
                self.epoch_num += 1.0

            print("...done")
            print("")
            print("L_theta: ",self.L_theta)
            print("alpha_theta: ",1.0 / (self.divider*self.L_theta))
            print("L_eta: ",self.L_eta)
            print("alpha_eta: ",1.0 / (self.divider*self.L_eta))
            print("")
            print("")


        if self.train_time > max_time:
            print("Maximum training time (%.3f hrs) reached. Returning..." % (max_time / 3600))
        else:
            print("Maximum number of epochs (%.1f) reached. Returning..." % self.epoch_num)

        return
