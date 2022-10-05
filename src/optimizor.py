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

from helper_funcs import eta_2_log_Gamma
from helper_funcs import log_Gamma_2_eta
from helper_funcs import eta0_2_log_delta
from helper_funcs import log_delta_2_eta0

class Optimizor(HHMM):

    def __init__(self,data,features,K):

        '''
        constructor for optimizor class
        '''

        # get all of the stuff from HHMM
        super().__init__(data,features,K)

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

        # overall gradient wrt theta
        for feature,dist in self.features.items():
            if dist['f'] == 'normal':
                for k0 in range(self.K[0]):

                    self.grad_theta[k0][feature]['mu'] = np.zeros(self.K[1])
                    self.grad_theta[k0][feature]['log_sig'] = np.zeros(self.K[1])

            elif dist['f'] == 'bern':
                for k0 in range(self.K[0]):

                    self.grad_theta[k0][feature]['logit_p'] = np.zeros(self.K[1])

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
            for feature,dist in self.features.items():
                if dist['f'] == 'normal':
                    for k0 in range(self.K[0]):

                        self.grad_theta_t[t][k0][feature]['mu'] = np.zeros(self.K[1])
                        self.grad_theta_t[t][k0][feature]['log_sig'] = np.zeros(self.K[1])

                elif dist['f'] == 'bern':
                    for k0 in range(self.K[0]):

                        self.grad_theta_t[t][k0][feature]['logit_p'] = np.zeros(self.K[1])

                else:
                    raise('only independent normal distributions supported at this time')

        return

    def get_grad_log_f(self,t,theta=None):

        if theta is None:
            theta = self.theta

        # define the data point
        y = self.data[t]

        # initialize the gradient
        grad_log_f = [{} for _ in range(self.K[0])]

        # go through each feature and add to log-likelihood
        for feature,value in y.items():

            if feature not in self.features:
                print("unidentified feature in y: %s" % feature)
                return

            if self.features[feature]['f'] == 'normal':

                # store gradient
                for k0 in range(self.K[0]):

                    mu = theta[k0][feature]['mu']
                    log_sig = theta[k0][feature]['log_sig']
                    sig = np.exp(log_sig)

                    if np.isnan(y[feature]):
                        grad_log_f[k0][feature] = {'mu': 0, 'log_sig': 0}
                    else:
                        grad_log_f[k0][feature] = {'mu': (y[feature]-mu)/(sig**2),
                                                   'log_sig': ((y[feature]-mu)/sig)**2 - 1}

                        # deal with truncated normals
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

            elif self.features[feature]['f'] == 'bern':

                # store gradient
                for k0 in range(self.K[0]):

                    logit_p = theta[k0][feature]['logit_p']

                    if np.isnan(y[feature]):
                        grad_log_f[k0][feature] = {'logit_p': 0}
                    elif y[feature] == 0:
                        grad_log_f[k0][feature] = {'logit_p': -expit(logit_p)}
                    elif y[feature] == 1:
                        grad_log_f[k0][feature] = {'logit_p': expit(-logit_p)}
                    else:
                        print("invalid data point %s for %s, which is bernoulli." % (y[feature],feature))

            else:
                print("unidentified emission distribution %s for %s"%(dist,feature))
                return

        return grad_log_f

    def get_grad_log_delta(self,eta0=None):

        if eta0 is None:
            eta0 = self.eta0

        log_delta = eta0_2_log_delta(eta0)
        coarse_delta = np.exp(log_delta[0])
        fine_deltas = [np.exp(log_delta1) for log_delta1 in log_delta[1]]

        grad_eta0_log_delta = [np.eye(self.K[0]) - np.tile(coarse_delta,[self.K[0],1]),
                               [np.eye(self.K[1]) - np.tile(fine_delta,[self.K[1],1]) \
                                for fine_delta in fine_deltas]]

        # set some gradients to zero for identifiability
        grad_eta0_log_delta[0][:,0] = 0
        for k0 in range(self.K[0]):
            grad_eta0_log_delta[1][k0][:,0] = 0

        return grad_eta0_log_delta

    def get_grad_log_Gamma(self,eta=None,eta0=None,jump=True):

        if eta is None:
            eta = self.eta

        if eta0 is None:
            eta0 = self.eta0

        # get fine and coarse scale Gammas
        log_Gammas = eta_2_log_Gamma(eta)
        Gammas = [np.exp(log_Gammas[0]),
                  [np.exp(fine_log_Gamma) for fine_log_Gamma in log_Gammas[1]]]

        # get the fine-scale deltas
        log_deltas = eta0_2_log_delta(eta0)
        fine_deltas = [np.exp(log_delta1) for log_delta1 in log_deltas[1]]

        # construct log_Gamma
        K_total = self.K[0] * self.K[1]

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
                grad_eta0_log_delta[1][k0][:,0] = 0.0

        return grad_eta0_log_delta,grad_eta_log_Gamma

    def get_grad_theta_t(self,t,grad_log_f=None,p_Xt=None):

        # initialize gradient
        grad_theta_t = deepcopy(self.grad_theta_t[t])

        # get gradient and weights
        if grad_log_f is None:
            grad_log_f = self.get_grad_log_f(t)
        if p_Xt is None:
            p_Xt = self.p_Xt[t]

        # calculate gradient
        for feature,settings in self.features.items():
            for param in self.grad_theta[0][feature]:

                if settings['share_coarse'] and settings['share_fine']:

                    # get gradient for first coarse state
                    grad_theta_t[0][feature][param] = np.zeros(self.K[1])
                    for k0 in range(self.K[0]):
                        grad_theta_t[0][feature][param] += np.sum(p_Xt[(self.K[1]*k0):(self.K[1]*(k0+1))] * \
                                                                  grad_log_f[k0][feature][param])
                    # set all other coarse scale states eqaul
                    for k0 in range(1,self.K[0]):
                        grad_theta_t[k0][feature][param] = np.copy(grad_theta_t[0][feature][param])

                elif settings['share_fine']:

                    # get gradient from each fine state and sum them
                    for k0 in range(self.K[0]):
                        grad_theta_t[k0][feature][param] = np.ones(self.K[1]) * \
                                                           np.sum(p_Xt[(self.K[1]*k0):(self.K[1]*(k0+1))] * \
                                                                  grad_log_f[k0][feature][param])

                elif settings['share_coarse']:

                    # get gradient for first coarse state
                    grad_theta_t[0][feature][param] = np.zeros(self.K[1])
                    for k0 in range(self.K[0]):
                        grad_theta_t[0][feature][param] += p_Xt[(self.K[1]*k0):(self.K[1]*(k0+1))] * \
                                                           grad_log_f[k0][feature][param]
                    # set all other coarse scale states equal
                    for k0 in range(1,self.K[0]):
                        grad_theta_t[k0][feature][param] = np.copy(grad_theta_t[0][feature][param])

                else:

                    # simply find the gradient at each state
                    for k0 in range(self.K[0]):
                        grad_theta_t[k0][feature][param] = p_Xt[(self.K[1]*k0):(self.K[1]*(k0+1))] * \
                                                           grad_log_f[k0][feature][param]

        return grad_theta_t

    def get_grad_eta_t(self,t,grad_eta0_log_delta=None,grad_eta_log_Gamma=None,p_Xtm1_Xt=None):

        # initialize gradients
        grad_eta_t = [np.zeros((self.K[0],self.K[0])),
                      [np.zeros((self.K[1],self.K[1])) for _ in range(self.K[0])]]

        grad_eta0_t = [np.zeros(self.K[0]),
                       [np.zeros(self.K[1]) for _ in range(self.K[0])]]

        # get initial and final indices
        seq_num = np.argmax(self.initial_ts > t)-1
        t0 = self.initial_ts[seq_num]
        tf = self.final_ts[seq_num]

        # deal with t == t0:
        if t == t0:

            # get gradient of log delta wrt eta0
            if grad_eta0_log_delta is None:
                grad_eta0_log_delta = self.get_grad_log_delta()

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
                grad_eta0_log_delta,grad_eta_log_Gamma = self.get_grad_log_Gamma(jump=True)
            else:
                grad_eta0_log_delta,grad_eta_log_Gamma = self.get_grad_log_Gamma(jump=False)

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

            # update theta
            self.grad_theta_t[t] = self.get_grad_theta_t(t)

            for k0 in range(self.K[0]):
                for feature in self.grad_theta[k0]:
                    for param in self.grad_theta[k0][feature]:
                        self.grad_theta[k0][feature][param] += \
                        self.grad_theta_t[t][k0][feature][param]

            # update eta
            self.grad_eta_t[t],self.grad_eta0_t[t] = self.get_grad_eta_t(t)

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
        for feature,settings in self.features.items():

            if settings['share_coarse'] and settings['share_fine']:
                for param in self.grad_theta[0][feature]:
                    xprime[ind] = self.grad_theta[0][feature][param][0]
                    ind += 1

            elif settings['share_fine']:
                for param in self.grad_theta[0][feature]:
                    for k0 in range(self.K[0]):
                        xprime[ind] = self.grad_theta[k0][feature][param][0]
                        ind += 1

            elif settings['share_coarse']:
                for param in self.grad_theta[0][feature]:
                    for k1 in range(self.K[1]):
                        xprime[ind] = self.grad_theta[0][feature][param][k1]
                        ind += 1

            else:
                for param in self.grad_theta[0][feature]:
                    for k0,k1 in product(range(self.K[0]),range(self.K[1])):
                        xprime[ind] = self.grad_theta[k0][feature][param][k1]
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

        # callback to terminate if max_sec exceeded
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

        options = {'maxiter':num_epochs,'disp':True}
        self.max_time = max_time
        self.num_epochs = num_epochs

        self.start_time = time.time()
        self.train_time = 0.0

        def loss_fn(x):

            # set parameters
            self.x_2_params(x)

            # calculate likelihood
            self.E_step()

            # get likelihood
            self.ll = logsumexp(self.log_alphas[self.T-1])

            # get gradient
            self.xprime = self.grad_params_2_xprime()

            return (-self.ll/self.T,-self.xprime/self.T)

        # initialize x0
        x0 = self.params_2_x()

        # fit x
        print(x0)

        try:
            res = minimize(loss_fn, x0,
                           method=method, tol=tol,
                           options=options, jac=True,
                           callback=self.callback)
        except RuntimeError:
            print("Terminating optimization: time or epoch limit reached")

        return
