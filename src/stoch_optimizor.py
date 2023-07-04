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

    def __init__(self,data,features,share_params,K,fix_theta=None,fix_eta=None,fix_eta0=None):

        '''
        constructor for optimizor class
        '''

        # get everything from optim
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
            self.L_theta = 1.0/(3.0*self.step_size)
            self.L_eta = 1.0/(3.0*self.step_size)
        else:
            self.L_theta = np.infty
            self.L_eta = np.infty

        # divider of Lipshitz constant
        self.divider = 3.0

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
        ll = logsumexp(self.log_alphas[self.T-1])
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

        # check for inequality
        check_again = False

        if grad_F_t_norm2 < 1e-8:
            pass
        elif F_t_new > F_t - grad_F_t_norm2 / (2*self.L_theta):
            self.L_theta *= 2
            check_again = True
        else:
            self.L_theta *= 2**(-1/self.T)

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

    def M_step(self,max_epochs=1,max_time=np.infty,alpha_theta=None,alpha_eta=None,method="EM",partial_E=False,tol=1e-5,record_like=False,weight_buffer="none",grad_buffer="none",buffer_eps=1e-3):

        if alpha_theta is None:
            alpha_theta = 1.0 / self.L_theta

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

            ts = np.random.permutation(range(self.T))

            for m,t in enumerate(ts):

                # get old gradient
                if method == "SVRG":

                    # get old gradients
                    grad_log_f = self.get_grad_log_f(t,theta=deepcopy(self.theta_tilde))

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

                    # get old weights
                    p_Xt = self.p_Xt_tilde[t]
                    p_Xtm1_X1 = self.p_Xtm1_Xt_tilde[t]

                    # calculate old gradient
                    old_grad_theta_t = self.get_grad_theta_t(t,
                                                             grad_log_f=grad_log_f,
                                                             grad_log_p_theta=None,
                                                             p_Xt=p_Xt)

                    old_grad_eta_t,old_grad_eta0_t = self.get_grad_eta_t(t,
                                                                         grad_eta0_log_delta=grad_eta0_log_delta,
                                                                         grad_eta_log_Gamma=grad_eta_log_Gamma,
                                                                         grad_log_p_eta=None,
                                                                         grad_log_p_eta0=None,
                                                                         p_Xtm1_Xt=p_Xtm1_X1)

                else:

                    # get old gradient at index
                    old_grad_theta_t = deepcopy(self.grad_theta_t[t])
                    old_grad_eta0_t = deepcopy(self.grad_eta0_t[t])
                    old_grad_eta_t = deepcopy(self.grad_eta_t[t])

                # update weights
                if partial_E:

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

                # check Lipshitz constants
                self.check_L_eta(t)
                alpha_eta = 1.0/(self.divider*self.L_eta)

                self.check_L_theta(t)
                alpha_theta = 1.0/(self.divider*self.L_theta)

                # get new gradients
                new_grad_theta_t = self.get_grad_theta_t(t)
                new_grad_eta_t,new_grad_eta0_t = self.get_grad_eta_t(t)

                # update parameters
                if method == "SVRG":

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

                else:
                    raise("method %s not recognized" % method)

                # clip values
                for k0 in range(self.K[0]):
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
                          "GD","SGD","SAG","SVRG","SAGA"]:
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
            print("...done")
            print("")

            # show current parameters
            print("current parameters:")
            print(self.theta)
            print(self.eta)
            print(self.eta0)
            print("")

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
                ll_new = logsumexp(self.log_alphas[self.T-1])

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
