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

    # get Coarse-scale Gamma
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

    for _ in range(100):
        for i in range(N):
            A = sum(np.exp(eta_coarse[i]))
            for j in range(N):
                if i != j:
                    A0 = A-np.exp(eta_coarse[i,j])
                    eta_coarse[i,j] = np.log((A0*Gamma[0][i,j])/(1-Gamma[0][i,j]))

    # get fine-scale eta
    etas_fine = []
    N = len(Gamma[1][0])
    for Gamma_fine in Gamma[1]:
        eta_fine = np.zeros_like(Gamma_fine)
        for _ in range(100):
            for i in range(N):
                A = sum(np.exp(eta_fine[i]))
                for j in range(N):
                    if i != j:
                        A0 = A-np.exp(eta_fine[i,j])
                        eta_fine[i,j] = np.log((A0*Gamma_fine[i,j])/(1-Gamma_fine[i,j]))

        etas_fine.append(eta_fine)


    return [eta_coarse,etas_fine]

def logdotexp(A, ptm):

    max_A = np.max(A)
    C = np.dot(np.exp(A - max_A), ptm)
    C = np.log(C)
    C += max_A
    return C

class optimizor:

    def __init__(self,hhmm):

        '''
        constructor for optimizor class
        '''

        self.hhmm = hhmm
        self.data = hhmm.data

        self.K = hhmm.pars.K
        self.T = len(hhmm.data)

        # thetas and etas
        self.theta = deepcopy(hhmm.theta)
        self.eta = deepcopy(hhmm.eta)
        self.eta0 = [np.zeros(self.K[0]),[np.zeros(self.K[1]) for _ in range(self.K[0])]]
        self.Gamma = eta_2_Gamma(self.eta)

        self.theta_tilde = deepcopy(hhmm.theta)
        self.eta_tilde = deepcopy(hhmm.eta)
        self.Gamma_tilde = eta_2_Gamma(self.eta_tilde)

        self.theta_trace = []
        self.eta_trace = []

        self.delta = np.ones(self.K[0])/self.K[0]
        for _ in range(100):
            self.delta = np.dot(self.delta,self.Gamma[0])

        # log-likelihood
        self.log_like_trace = []
        self.grad_norm_trace = []

        # time
        self.time_trace = []
        self.epoch_trace = []

        # alpha and beta
        self.log_alphas = np.zeros((self.T,self.K[0]))
        self.log_betas = np.zeros((self.T,self.K[0]))

        # p_Xt and p_Xt_Xtp1
        self.p_Xt = np.zeros((self.T,self.K[0]))
        self.p_Xt_Xtp1 = np.zeros((self.T-1,self.K[0],self.K[0]))
        self.p_Xt_tilde = np.zeros((self.T,self.K[0]))
        self.p_Xt_Xtp1_tilde = np.zeros((self.T-1,self.K[0],self.K[0]))

        # gradients wrt theta
        self.E_grad_log_f = [deepcopy(hhmm.theta) for _ in range(self.T)]

        self.grad_theta_log_like = deepcopy(hhmm.theta)
        self.grad_theta_log_like_tilde = deepcopy(hhmm.theta)

        self.grad_theta_trace = []

        # gradients wrt eta
        self.E_grad_log_Gamma = [np.zeros((self.K[0],self.K[0])) for _ in range(self.T)]
        self.E_grad_log_Gamma_tilde = [np.zeros((self.K[0],self.K[0])) for _ in range(self.T)]

        self.grad_eta_log_like = np.zeros((self.K[0],self.K[0]))
        self.grad_eta_log_like_tilde = np.zeros((self.K[0],self.K[0]))

        self.grad_eta_trace = []

        # time to train
        self.train_time = None
        self.start_time = None
        self.epoch_num = None

        # initialize gradients
        self.initialize_grads()

        # initialize step sizes and parameter bounds
        self.step_size = None
        self.param_bounds = {feature: {} for feature in hhmm.theta[0]}
        self.initialize_step_size()

        # Lipshitz constants
        if self.step_size != 0:
            self.L_theta = 1.0/(3.0*self.step_size)
            self.L_eta = 1.0/(3.0*self.step_size)
        else:
            self.L_theta = np.infty
            self.L_eta = np.infty

        return

    def initialize_step_size(self,min_group_size=None):

        if min_group_size is None:
            min_group_size = max(int(0.01*self.T),3)

        self.step_size = np.infty
        t = self.T-1

        # get grad w.r.t theta
        for feature in self.grad_theta_log_like[0]:

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
            mu_ss = 0.1*min(abs((max_mu-min_mu)/self.grad_theta_log_like[0][feature]["mu"]))
            log_sig_ss = 0.1*min(abs((max_log_sig-min_log_sig)/self.grad_theta_log_like[0][feature]["log_sig"]))
            param_ss = min(mu_ss,log_sig_ss)

            if param_ss < self.step_size:
                self.step_size = np.copy(param_ss)

        # get grad w.r.t eta
        for i in range(self.K[0]):
            for j in range(self.K[0]):

                if i != j:
                    continue

                # never make the step size larger than 1 (since |eta| < 15)
                param_ss = 1.0/abs(self.grad_eta_log_like[i,j])

                if param_ss < self.step_size:
                    self.step_size = np.copy(param_ss)

        return

    def initialize_grads(self):

        # get overal gradient wrt eta
        self.grad_eta_log_like = np.zeros((self.K[0],self.K[0]))

        # get SVRG control variate wrt eta
        self.grad_eta_log_like_tilde = np.zeros((self.K[0],self.K[0]))

        for t in range(self.T):

            # get table values wrt eta
            self.E_grad_log_Gamma[t] = np.zeros((self.K[0],self.K[0]))

            for feature,dist in self.hhmm.pars.features[0].items():
                if dist['f'] == 'normal' and not dist['corr']:
                    if t == 0:

                        # get overal gradient wrt theta
                        self.grad_theta_log_like[0][feature]['mu'] = np.zeros(self.K[0])
                        self.grad_theta_log_like[0][feature]['log_sig'] = np.zeros(self.K[0])
                        self.grad_theta_log_like[0][feature]['corr'] = np.zeros(self.K[0])

                        # get SVRG control variate wrt theta
                        self.grad_theta_log_like_tilde[0][feature]['mu'] = np.zeros(self.K[0])
                        self.grad_theta_log_like_tilde[0][feature]['log_sig'] = np.zeros(self.K[0])
                        self.grad_theta_log_like_tilde[0][feature]['corr'] = np.zeros(self.K[0])

                    # get table values wrt theta
                    self.E_grad_log_f[t][0][feature]['mu'] = np.zeros(self.K[0])
                    self.E_grad_log_f[t][0][feature]['log_sig'] = np.zeros(self.K[0])
                    self.E_grad_log_f[t][0][feature]['corr'] = np.zeros(self.K[0])

                else:
                    raise('only independent normal distributions supported at this time')

        return

    def log_f(self,t,theta=None):

        if theta is None:
            theta = self.theta

        # define the data point
        y = self.data[t]

        # initialize the log-likelihood
        log_f = np.zeros(self.K[0])

        # initialize the gradient
        grad_log_f = [{},{}]

        # go through each coarse-scale feature and add to log-likelihood
        for feature,value in y.items():

            # initialize gradient for this feature
            if feature == 'subdive_features':
                continue

            if feature not in self.hhmm.pars.features[0]:
                print("unidentified feature in y: %s" % feature)
                return

            dist = self.hhmm.pars.features[0][feature]['f']

            if dist == 'normal':

                mu = theta[0][feature]['mu']
                log_sig = theta[0][feature]['log_sig']
                sig = np.exp(log_sig)

                log_f += norm.logpdf(y[feature],
                                     loc=mu,
                                     scale=sig)

                grad_log_f[0][feature] = {'mu': (y[feature]-mu)/(sig**2),
                                          'log_sig': ((y[feature]-mu)/sig)**2 - 1}

            else:
                print("unidentified emission distribution %s for %s"%(dist,feature))
                return

        # return the result
        return log_f,grad_log_f

    def log_Gamma(self,t,eta=None,eta0=None):

        if eta is None:
            eta = self.eta

        if eta0 is None:
            eta0 = self.eta0

        # get fine and coarse scale Gammas
        Gammas = eta_2_Gamma(eta)
        log_coarse_Gamma = np.log(Gammas[0])
        log_fine_Gammas = [np.log(fine_Gamma) for fine_Gamma in Gammas[1]]

        # get the fine-scale Gammas
        fine_deltas = [np.exp(eta0i) / np.sum(np.exp(eta0i)) for eta0i in eta0]
        log_fine_deltas = [eta0i - np.logsumexp(eta0i) for eta0i in eta0]

        # construct Gamma
        K_total = self.K[0] * self.K[1]
        log_Gamma = np.zeros((K_total,K_total))

        for i in range(self.K[0]):
            for j in range(self.K[0]):

                # coarse-scale Gamma
                log_Gamma[(self.K[1]*i):(self.K[1]*(i+1)),
                          (self.K[1]*j):(self.K[1]*(j+1))] += log_coarse_Gamma[i,j]

                # fine-scale Gamma or delta
                if i == j:
                    log_Gamma[(self.K[1]*i):(self.K[1]*(i+1)),
                              (self.K[1]*j):(self.K[1]*(j+1))] += log_fine_Gammas[j]
                else:
                    log_Gamma[(self.K[1]*i):(self.K[1]*(i+1)),
                              (self.K[1]*j):(self.K[1]*(j+1))] += np.tile(log_fine_deltas[j],[self.K[1],1])

        grad_log_Gamma = [np.zeros((self.K[0],self.K[0],self.K[0],self.K[0])),
                          [np.zeros((self.K[1],self.K[1],self.K[1],self.K[1])) for _ in range(self.K[0])],
                          [np.zeros((self.K[1],self.K[1])) for _ in range(self.K[0])]]

        # grad from coarse_Gamma
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                for l in range(self.K[0]):
                    if i == l:
                        pass
                    elif j == l:
                        grad_log_Gamma[0][i,j,i,l] = 1.0-Gammas[0][i,l]
                    else:
                        grad_log_Gamma[0][i,j,i,l] = -Gammas[0][i,l]

        # grad from fine_Gamma
        for n in range(self.K[0]):
            for i in range(self.K[1]):
                for j in range(self.K[1]):
                    for l in range(self.K[1]):
                        if i == l:
                            pass
                        elif j == l:
                            grad_log_Gamma[1][n][i,j,i,l] = 1.0-Gammas[1][n][i,l]
                        else:
                            grad_log_Gamma[1][n][i,j,i,l] = -Gammas[1][n][i,l]

        # grad from fine_delta
        for n in range(self.K[0]):
            for i in range(self.K[1]):
                for j in range(self.K[1]):
                    if i == j:
                        grad_log_Gamma[2][n][i,j] = 1.0-fine_deltas[n][j]
                    else:
                        grad_log_Gamma[2][n][i,j] = -fine_deltas[n][j]

        return log_Gamma,grad_log_Gamma

    def log_delta(self,eta0=None):

        if eta0 is None:
            eta0 = self.eta0

        # get delta
        delta = np.exp(eta0[0]) / np.sum(eta0[0])
        log_delta = np.log(delta) - np.log(np.sum(delta))

        # get grad_log_delta
        grad_log_delta = np.zeros((self.K[0],self.K[0]))
        for i in range(self.K[1]):
            for j in range(self.K[1]):
                if i == j:
                    grad_log_delta[i,j] = 1.0-delta[j]
                else:
                    grad_log_delta[i,j] = -delta[j]

        return log_delta,grad_log_delta

    def update_alpha(self,t):

        # get log_f, grad_log_f
        log_f = self.log_f(t)[0]

        # update log_alpha
        if t == 0:
            self.log_alphas[t] = np.log(self.delta) + log_f
        else:
            self.log_alphas[t] = logdotexp(self.log_alphas[t-1],self.Gamma[0]) + log_f

        return

    def update_beta(self,t):

        # update log_beta
        if t == self.T-1:
            self.log_betas[t] = np.zeros(self.K[0])
        else:
            log_f_tp1 = self.log_f(t+1)[0]
            self.log_betas[t] = logdotexp(self.log_betas[t+1] + log_f_tp1,
                                            np.transpose(self.Gamma[0]))

        return

    def est_log_like(self,t):
        return logsumexp(self.log_alphas[t] + self.log_betas[t])

    def est_grad_norm(self):

        norm_squared = 0

        norm_squared += np.sum(self.grad_eta_log_like**2)

        for feature in self.grad_theta_log_like[0]:
            for param in self.grad_theta_log_like[0][feature]:
                norm_squared += np.sum(self.grad_theta_log_like[0][feature][param]**2)

        return np.sqrt(norm_squared)

    def get_log_like(self):

        # store old values
        log_alphas0 = deepcopy(self.log_alphas)
        log_betas0 = deepcopy(self.log_betas)

        grad_theta_log_like0 = deepcopy(self.grad_theta_log_like)
        grad_eta_log_like0 = deepcopy(self.grad_eta_log_like)

        p_Xt0 = deepcopy(self.p_Xt)
        p_Xt_Xtp10 = deepcopy(self.p_Xt_Xtp1)

        E_grad_log_f0 = deepcopy(self.E_grad_log_f)
        E_grad_log_Gamma0 = deepcopy(self.E_grad_log_Gamma)

        grad_theta_log_like_tilde0 = deepcopy(self.grad_theta_log_like_tilde)
        grad_eta_log_like_tilde0 = deepcopy(self.grad_eta_log_like_tilde)

        p_Xt_tilde0 = deepcopy(self.p_Xt_tilde)
        p_Xt_Xtp1_tilde0 = deepcopy(self.p_Xt_Xtp1_tilde)

        theta_tilde0 = deepcopy(self.theta_tilde)
        eta_tilde0 = deepcopy(self.eta_tilde)
        Gamma_tilde0 = eta_2_Gamma(eta_tilde0)

        # get new likelihood and gradient
        self.E_step()
        ll = self.est_log_like(0)
        grad_norm = self.est_grad_norm()

        # return values to old state
        self.log_alphas = log_alphas0
        self.log_betas = log_betas0

        self.grad_theta_log_like = grad_theta_log_like0
        self.grad_eta_log_like = grad_eta_log_like0

        self.p_Xt = p_Xt0
        self.p_Xt_Xtp1 = p_Xt_Xtp10

        self.E_grad_log_f = E_grad_log_f0
        self.E_grad_log_Gamma = E_grad_log_Gamma0

        self.grad_theta_log_like_tilde = grad_theta_log_like_tilde0
        self.grad_eta_log_like_tilde = grad_eta_log_like_tilde0

        self.p_Xt_tilde = p_Xt_tilde0
        self.p_Xt_Xtp1_tilde = p_Xt_Xtp1_tilde0

        self.theta_tilde = theta_tilde0
        self.eta_tilde = eta_tilde0
        self.Gamma_tilde = eta_2_Gamma(self.eta_tilde)

        return ll, grad_norm

    def update_p_Xt(self,t):

        ll = self.est_log_like(t)
        self.p_Xt[t] = np.exp(self.log_alphas[t] + self.log_betas[t] - ll)

        return

    def update_p_Xt_Xtp1(self,t,log_f_tp1=None):

        if log_f_tp1 is None:
            log_f_tp1 = self.log_f(t+1)[0]

        p_XX = np.zeros((self.K[0],self.K[0]))
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                p_XX[i,j] = self.log_alphas[t,i] \
                            + np.log(self.Gamma[0][i,j]) \
                            + log_f_tp1[j] \
                            + self.log_betas[t+1,j]

        self.p_Xt_Xtp1[t] = np.exp(p_XX - logsumexp(p_XX))

        return

    def get_E_grad_log_f(self,t,grad_log_f_t=None,p_Xt=None):

        # initialize E_grad
        E_grad = deepcopy(self.E_grad_log_f[t])

        # get gradient and weights
        if grad_log_f_t is None:
            grad_log_f_t = self.log_f(t)[1]
        if p_Xt is None:
            p_Xt = self.p_Xt[t]

        # calculate E_grad
        for feature in grad_log_f_t[0]:
            for param in grad_log_f_t[0][feature]:
                E_grad[0][feature][param] = p_Xt * grad_log_f_t[0][feature][param]

        return E_grad

    def get_E_grad_log_Gamma(self,t,grad_log_Gamma=None,p_Xt_Xtp1=None):

        # initialize new and old grad
        E_grad = deepcopy(self.E_grad_log_Gamma[t])

        # get gradients and weights
        if grad_log_Gamma is None:
            grad_log_Gamma = self.log_Gamma(t)[1]
        if p_Xt_Xtp1 is None:
            p_Xt_Xtp1 = self.p_Xt_Xtp1[t]

        # calculate E_grad
        for k in range(self.K[0]):
            for l in range(self.K[0]):
                E_grad[k,l] = np.sum(p_Xt_Xtp1 * grad_log_Gamma[:,:,k,l])

        return E_grad

    def check_L_eta(self,t):

        # get gradient and its norm
        if t == 1:
            grad_G_t = 0#np.zeros(self.K[0])
        else:
            grad_G_t = self.get_E_grad_log_Gamma(t-1)

        grad_G_t_norm2 = np.sum(grad_G_t**2)

        # get new value of eta
        eta0 = deepcopy(self.eta)
        eta0[0] += grad_G_t / self.L_eta

        # Evaluate G for eta and eta0
        if t == 1:
            G_t = 0
            G_t0 = 0
        else:
            G_t  = -np.sum(self.p_Xt_Xtp1[t-1] * self.log_Gamma(t,eta=self.eta)[0])
            G_t0 = -np.sum(self.p_Xt_Xtp1[t-1] * self.log_Gamma(t,eta=eta0)[0])

        # check inequality
        if grad_G_t_norm2 < 1e-8:
            pass
        elif G_t0 > G_t - grad_G_t_norm2 / (2*self.L_eta):
            self.L_eta *= 2
        else:
            self.L_eta *= 2**(-1/self.T)

        return

    def check_L_theta(self,t):

        # get the gradients at the given time points
        grad_F_t = self.get_E_grad_log_f(t)

        # initialize gradient norms
        grad_F_t_norm2 = 0

        # get new theta and gradient norm
        theta0 = deepcopy(self.theta)
        for feature in grad_F_t[0]:
            for param in grad_F_t[0][feature]:
                theta0[0][feature][param] += grad_F_t[0][feature][param] / self.L_theta
                grad_F_t_norm2 += np.sum(grad_F_t[0][feature][param]**2)

        # evaluate F for theta and theta0
        F_t  = -np.sum(self.p_Xt[t] * self.log_f(t,theta=self.theta)[0])
        F_t0 = -np.sum(self.p_Xt[t] * self.log_f(t,theta=theta0)[0])

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
        for feature in self.grad_theta_log_like[0]:
            for param in self.grad_theta_log_like[0][feature]:
                self.grad_theta_log_like[0][feature][param] = 0
        self.grad_eta_log_like = np.zeros((self.K[0],self.K[0]))

        # update probs and gradients
        for t in range(self.T):

            # update theta
            self.update_p_Xt(t)
            self.E_grad_log_f[t] = self.get_E_grad_log_f(t)
            for feature in self.grad_theta_log_like[0]:
                for param in self.grad_theta_log_like[0][feature]:
                    self.grad_theta_log_like[0][feature][param] += \
                    self.E_grad_log_f[t][0][feature][param]

            # update eta
            if t != self.T-1:
                self.update_p_Xt_Xtp1(t)
                self.E_grad_log_Gamma[t] = self.get_E_grad_log_Gamma(t)
                self.grad_eta_log_like += self.E_grad_log_Gamma[t]

        # record gradients, weights, and parameters for SVRG
        if update_tilde:
            self.grad_theta_log_like_tilde = deepcopy(self.grad_theta_log_like)
            self.grad_eta_log_like_tilde = deepcopy(self.grad_eta_log_like)

            self.p_Xt_tilde = deepcopy(self.p_Xt)
            self.p_Xt_Xtp1_tilde = deepcopy(self.p_Xt_Xtp1)

            self.theta_tilde = deepcopy(self.theta)
            self.eta_tilde = deepcopy(self.eta)
            self.Gamma_tilde = eta_2_Gamma(self.eta_tilde)

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

            # update eta
            delta = alpha_eta * self.grad_eta_log_like / self.T
            self.eta[0] += delta

            # update theta
            for feature in self.theta[0]:
                for param in ['mu','log_sig']:
                    delta = alpha_theta * self.grad_theta_log_like[0][feature][param] / self.T
                    self.theta[0][feature][param] += delta
            return

        for iter in range(max_iters):

            # pick index
            t = np.random.choice(self.T)

            # get old gradient
            if method == "SVRG":

                # get old parameters and weights
                grad_log_f_t = self.log_f(t,theta=self.theta_tilde)[1]
                grad_log_Gamma = self.log_Gamma(t,eta=self.eta_tilde)[1]
                p_Xt = self.p_Xt_tilde[t]
                if t != self.T - 1:
                    p_Xt_Xtp1 = self.p_Xt_Xtp1_tilde[t]

                # calculate old gradient
                old_E_grad_log_f = self.get_E_grad_log_f(t,
                                                         grad_log_f_t=grad_log_f_t,
                                                         p_Xt=p_Xt)
                if t != self.T-1:
                    old_E_grad_log_Gamma = self.get_E_grad_log_Gamma(t,
                                                                     grad_log_Gamma=grad_log_Gamma,
                                                                     p_Xt_Xtp1=p_Xt_Xtp1)

            else:

                # get old gradient at index
                old_E_grad_log_f = deepcopy(self.E_grad_log_f[t])
                if t != self.T-1:
                    old_E_grad_log_Gamma = deepcopy(self.E_grad_log_Gamma[t])

            # update alpha, beta, p_Xt, p_Xt_Xtp1
            if partial_E:

                self.update_alpha(t)
                self.update_beta(t)
                self.update_p_Xt(t)
                if t != 0:
                    self.update_p_Xt_Xtp1(t-1)
                if t != self.T-1:
                    self.update_p_Xt_Xtp1(t)

            # get new gradient at index
            new_E_grad_log_f = self.get_E_grad_log_f(t)
            if t != self.T-1:
                new_E_grad_log_Gamma = self.get_E_grad_log_Gamma(t)

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
                    delta = alpha_eta0 * new_E_grad_log_Gamma
                    self.eta[0] += delta

                # update theta
                for feature in new_E_grad_log_f[0]:
                    for param in ['mu','log_sig']:
                        delta = alpha_theta0 * new_E_grad_log_f[0][feature][param]
                        self.theta[0][feature][param] += delta

            elif method == "SAG":

                # update eta
                if t != self.T-1:
                    delta = alpha_eta * (new_E_grad_log_Gamma \
                                   - old_E_grad_log_Gamma \
                                   + self.grad_eta_log_like)/self.T
                    self.eta[0] += delta

                # update theta
                for feature in self.theta[0]:
                    for param in ['mu','log_sig']:
                        delta = alpha_theta * (new_E_grad_log_f[0][feature][param] \
                                       - old_E_grad_log_f[0][feature][param] \
                                       + self.grad_theta_log_like[0][feature][param])/self.T
                        self.theta[0][feature][param] += delta

            elif method == "SVRG":

                # update eta
                if t != self.T-1:
                    delta = alpha_eta * (new_E_grad_log_Gamma \
                                   - old_E_grad_log_Gamma \
                                   + self.grad_eta_log_like_tilde/self.T)
                    self.eta[0] += delta

                # update theta
                for feature in self.theta[0]:
                    for param in ['mu','log_sig']:
                        delta = alpha_theta * (new_E_grad_log_f[0][feature][param] \
                                       - old_E_grad_log_f[0][feature][param] \
                                       + self.grad_theta_log_like_tilde[0][feature][param]/self.T)
                        self.theta[0][feature][param] += delta

            elif method == "SAGA":

                # update eta
                if t != self.T-1:
                    delta = alpha_eta * (new_E_grad_log_Gamma \
                                   - old_E_grad_log_Gamma \
                                   + self.grad_eta_log_like/self.T)
                    self.eta[0] += delta

                # update theta
                for feature in self.theta[0]:
                    for param in ['mu','log_sig']:
                        delta = alpha_theta * (new_E_grad_log_f[0][feature][param] \
                                       - old_E_grad_log_f[0][feature][param] \
                                       + self.grad_theta_log_like[0][feature][param]/self.T)
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
                self.grad_eta_log_like += new_E_grad_log_Gamma - old_E_grad_log_Gamma

            for feature in self.theta[0]:
                for param in ['mu','log_sig']:
                    self.grad_theta_log_like[0][feature][param] += \
                        new_E_grad_log_f[0][feature][param] \
                      - old_E_grad_log_f[0][feature][param]

            # update table of gradients
            self.E_grad_log_f[t] = new_E_grad_log_f
            if t != self.T-1:
                self.E_grad_log_Gamma[t] = new_E_grad_log_Gamma

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
                        grad_ll[ind] = -self.grad_theta_log_like[0][feature][param][k]
                        ind += 1

            # update eta
            for i in range(self.K[0]):
                for j in range(self.K[0]):
                    if i != j:
                        grad_ll[ind] = -self.grad_eta_log_like[i,j]
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
