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
#sys.path.insert(0,'/Users/evsi8432/Documents/Research/CarHHMM-DFT/Repository/Code')
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
        self.Gamma = eta_2_Gamma(self.eta)

        self.theta_trace = []
        self.eta_trace = []

        self.delta = np.ones(self.K[0])/self.K[0]
        for _ in range(100):
            self.delta = np.dot(self.delta,self.Gamma[0])

        # log-likelihood
        self.log_like = 0
        self.log_like_trace = []

        # alpha and beta
        self.log_alphas = np.zeros((self.K[0],self.T))
        self.log_betas = np.zeros((self.K[0],self.T))

        # gradients wrt theta
        self.d_log_alpha_d_theta = [[deepcopy(hhmm.theta) for _ in range(self.K[0])] for _ in range(self.T)]
        self.d_log_beta_d_theta =  [[deepcopy(hhmm.theta) for _ in range(self.K[0])] for _ in range(self.T)]
        self.d_log_like_d_theta = [deepcopy(hhmm.theta) for _ in range(self.T)]
        self.grad_theta_trace = []

        # gradients wrt eta
        self.d_log_alpha_d_eta = [[deepcopy(hhmm.eta) for _ in range(self.K[0])] for _ in range(self.T)]
        self.d_log_beta_d_eta =  [[deepcopy(hhmm.eta) for _ in range(self.K[0])] for _ in range(self.T)]
        self.d_log_like_d_eta = [deepcopy(hhmm.eta) for _ in range(self.T)]
        self.grad_eta_trace = []

        # initialize gradients
        self.initialize_grads()

        # do a forward and backward pass of the data
        self.fwd_pass()
        self.bwd_pass()

        # get the initial gradients
        for t in range(self.T):
            self.log_likelihood(t)
            self.grad_log_likelihood(t)

        # initialize step sizes and parameter bounds
        self.step_size = None
        self.param_bounds = {feature: {} for feature in hhmm.theta[0]}
        self.step_num = 1
        self.initialize_step_size()

        return

    def initialize_step_size(self,min_group_size=None):

        if min_group_size is None:
            min_group_size = max(int(0.01*self.T),3)

        self.step_size = np.infty
        t = self.T-1

        # get grad w.r.t theta
        for feature in self.d_log_like_d_theta[t][0]:

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
            mu_ss = 0.1*min(abs((max_mu-min_mu)/self.d_log_like_d_theta[t][0][feature]["mu"]))
            log_sig_ss = 0.1*min(abs((max_log_sig-min_log_sig)/self.d_log_like_d_theta[t][0][feature]["log_sig"]))
            param_ss = min(mu_ss,log_sig_ss)

            if param_ss < self.step_size:
                self.step_size = np.copy(param_ss)

        # get grad w.r.t eta
        for i in range(self.K[0]):
            for j in range(self.K[0]):

                if i != j:
                    continue

                # never make the step size larger than 1 (since |eta| < 15)
                param_ss = 1.0/abs(self.d_log_like_d_eta[t][i,j])

                if param_ss < self.step_size:
                    self.step_size = np.copy(param_ss)

        return

    def initialize_grads(self):

        for t in range(self.T):

            # get log likelihood gradients
            self.d_log_like_d_eta[t] = np.zeros((self.K[0],self.K[0]))

            for k in range(self.K[0]):

                self.d_log_alpha_d_eta[t][k] = np.zeros((self.K[0],self.K[0]))
                self.d_log_beta_d_eta[t][k] = np.zeros((self.K[0],self.K[0]))

                for feature,dist in self.hhmm.pars.features[0].items():
                    if dist['f'] == 'normal' and not dist['corr']:

                        if k == 0:
                            self.d_log_like_d_theta[t][0][feature]['mu'] = np.zeros(self.K[0])
                            self.d_log_like_d_theta[t][0][feature]['log_sig'] = np.zeros(self.K[0])
                            self.d_log_like_d_theta[t][0][feature]['corr'] = np.zeros(self.K[0])

                        self.d_log_alpha_d_theta[t][k][0][feature]['mu'] = np.zeros(self.K[0])
                        self.d_log_alpha_d_theta[t][k][0][feature]['log_sig'] = np.zeros(self.K[0])
                        self.d_log_alpha_d_theta[t][k][0][feature]['corr'] = np.zeros(self.K[0])

                        self.d_log_beta_d_theta[t][k][0][feature]['mu'] = np.zeros(self.K[0])
                        self.d_log_beta_d_theta[t][k][0][feature]['log_sig'] = np.zeros(self.K[0])
                        self.d_log_beta_d_theta[t][k][0][feature]['corr'] = np.zeros(self.K[0])

                    else:
                        raise('only independent normal distributions supported at this time')

    def log_f(self,t):

        # define the data point
        y = self.data[t]

        # initialize the log-likelihood
        log_f = np.zeros(self.K[0])

        # initialize the gradient of the log-likelihood
        grad_log_f = [{},{}] # one for coarse scale, one for fine-scale

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

                mu = self.theta[0][feature]['mu']
                log_sig = self.theta[0][feature]['log_sig']
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

    def fwd_step(self,t,grad=True):

        # get log_f, grad_log_f
        log_f,grad_log_f = self.log_f(t)

        # deal with t = 0
        if t == 0:

            # update log_alpha
            self.log_alphas[:,t] = np.log(self.delta) + log_f

            # update d_log_alpha_t/d_theta
            for k in range(self.K[0]):
                for feature in grad_log_f[0]:
                    for param in grad_log_f[0][feature]:
                        self.d_log_alpha_d_theta[t][k][0][feature][param][k] = grad_log_f[0][feature][param][k]

            # update d_log_alpha_t/d_eta
            self.d_log_alpha_d_eta[t][k] = np.zeros((self.K[0],self.K[0]))

            return

        # now deal with t != 0
        self.log_alphas[:,t] = logdotexp(self.log_alphas[:,t-1],self.Gamma[0]) + log_f

        # create matrix of P(X_{t-1} = i | X_{t} = j , Y_{1:t+1})
        xi = np.zeros((self.K[0],self.K[0]))
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                xi[i,j] =   self.log_alphas[i,t-1] + np.log(self.Gamma[0][i,j]) + log_f[j]
                xi[i,j] += -self.log_alphas[j,t]

        xi = np.exp(xi)

        # update d_log_alpha_t/d_theta
        for k in range(self.K[0]): # parameter k
            for feature in grad_log_f[0]:
                for param in grad_log_f[0][feature]:

                    # get d_log_fj/d_theta_j
                    d_log_fj_d_theta_k = np.zeros(self.K[0])
                    d_log_fj_d_theta_k[k] = grad_log_f[0][feature][param][k]

                    for j in range(self.K[0]): # alpha_j
                        self.d_log_alpha_d_theta[t][j][0][feature][param][k] = d_log_fj_d_theta_k[j]
                        for i in range(self.K[0]):
                            self.d_log_alpha_d_theta[t][j][0][feature][param][k] += \
                            xi[i,j]*self.d_log_alpha_d_theta[t-1][i][0][feature][param][k]

        # get d_log_Gamma/d_eta
        d_log_Gamma_d_eta = np.zeros((self.K[0],self.K[0],self.K[0],self.K[0]))
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                for k in range(self.K[0]):
                    if j == k:
                        d_log_Gamma_d_eta[i,j,i,k] = 1.0-self.Gamma[0][i,k]
                    else:
                        d_log_Gamma_d_eta[i,j,i,k] = -self.Gamma[0][i,k]


        # update d_log_alpha_t/d_eta
        for j in range(self.K[0]): # alpha_j
            self.d_log_alpha_d_eta[t][j] = np.zeros((self.K[0],self.K[0]))
            for i in range(self.K[0]):
                self.d_log_alpha_d_eta[t][j] += xi[i,j]*self.d_log_alpha_d_eta[t-1][i]
                self.d_log_alpha_d_eta[t][j] += xi[i,j]*d_log_Gamma_d_eta[i,j,:,:]

        return

    def bwd_step(self,t,grad=True):

        # deal with t = T-1
        if t == self.T-1:

            # update log_beta
            self.log_betas[:,t] = np.zeros(self.K[0])

            # update d_log_alpha_t/d_theta
            for k in range(self.K[0]):
                for feature in self.d_log_beta_d_theta[t][k][0]:
                    for param in self.d_log_beta_d_theta[t][k][0][feature]:
                        self.d_log_beta_d_theta[t][k][0][feature][param][k] = 0

            # update d_log_beta_t/d_eta
            self.d_log_beta_d_eta[t][k] = np.zeros((self.K[0],self.K[0]))

            return

        # get log_f, grad_log_f
        log_f,grad_log_f = self.log_f(t+1)

        # get log_beta
        self.log_betas[:,t] = logdotexp(self.log_betas[:,t+1] + log_f,
                                        np.transpose(self.Gamma[0]))

        # create matrix of P(X_{t+1} = j | X_{t} = i , Y_{t+1:T})
        xi = np.zeros((self.K[0],self.K[0]))
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                xi[i,j] =   np.log(self.Gamma[0][i,j]) + log_f[j] + self.log_betas[j,t+1]
                xi[i,j] += -self.log_betas[i,t]

        xi = np.exp(xi)

        # update d_log_beta_t/d_theta
        for k in range(self.K[0]): # parameter k
            for feature in grad_log_f[0]:
                for param in grad_log_f[0][feature]:
                    for i in range(self.K[0]): # beta_i

                        self.d_log_beta_d_theta[t][i][0][feature][param][k] = 0

                        # get d_log_fj/d_theta_k
                        d_log_fj_d_theta_k = np.zeros(self.K[0])
                        d_log_fj_d_theta_k[k] =  grad_log_f[0][feature][param][k]

                        for j in range(self.K[0]):

                            self.d_log_beta_d_theta[t][i][0][feature][param][k] += \
                            xi[i,j]*(self.d_log_beta_d_theta[t+1][j][0][feature][param][k] + \
                                     d_log_fj_d_theta_k[j])

        # get d_log_Gamma_ij/d_eta_kl
        d_log_Gamma_d_eta = np.zeros((self.K[0],self.K[0],self.K[0],self.K[0]))
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                for k in range(self.K[0]):
                    if j == k:
                        d_log_Gamma_d_eta[i,j,i,k] = 1.0-self.Gamma[0][i,k]
                    else:
                        d_log_Gamma_d_eta[i,j,i,k] = -self.Gamma[0][i,k]


        # update d_log_beta_t/d_eta
        for i in range(self.K[0]):
            self.d_log_beta_d_eta[t][i] = np.zeros((self.K[0],self.K[0]))
            for j in range(self.K[0]):
                self.d_log_beta_d_eta[t][i] += xi[i,j]*self.d_log_beta_d_eta[t+1][j]
                self.d_log_beta_d_eta[t][i] += xi[i,j]*d_log_Gamma_d_eta[i,j,:,:]

        return

    def fwd_pass(self):

        for t in range(self.T):
            self.fwd_step(t)

        return

    def bwd_pass(self):

        for t in reversed(range(self.T)):
            self.bwd_step(t)

        return

    def log_likelihood(self,t=None):

        if t is None:
            t = self.T - 1

        return np.copy(logsumexp(self.log_alphas[:,t] + self.log_betas[:,t]))

    def grad_log_likelihood(self,t=None):

        if t is None:
            t = self.T-1

        ll = logsumexp(self.log_alphas[:,t] + self.log_betas[:,t])
        p_X = np.exp(self.log_alphas[:,t] + self.log_betas[:,t] - ll)

        # get derivative with respect to theta
        for feature in self.d_log_alpha_d_theta[t][0][0]:
            for param in self.d_log_alpha_d_theta[t][0][0][feature]:
                for k in range(self.K[0]):
                    self.d_log_like_d_theta[t][0][feature][param][k] = \
                        sum([p_X[i]*(self.d_log_alpha_d_theta[t][i][0][feature][param][k] + \
                                      self.d_log_beta_d_theta[t][i][0][feature][param][k]) \
                             for i in range(self.K[0])])


        # update derivative with respect to eta
        self.d_log_like_d_eta[t] = np.zeros((self.K[0],self.K[0]))
        for i in range(self.K[0]):
            self.d_log_like_d_eta[t] += p_X[i]*(self.d_log_alpha_d_eta[t][i] + self.d_log_beta_d_eta[t][i])

        return

    def update_params(self,t,decay_ind=0):

        # update gradient of log_likelihood
        self.grad_log_likelihood(t=t)

        # find step size
        alpha = self.step_size / np.sqrt(max(self.step_num-decay_ind,1))

        # update eta
        #delta = alpha * np.copy(self.d_log_like_d_eta[t])
        #print("change in eta: ", delta)
        #self.eta[0] += delta

        # make sure |eta| < 15
        #self.eta[0] = np.clip(self.eta[0], -10, 10)
        #np.fill_diagonal(self.eta[0],0)

        # update Gamma
        #self.Gamma = eta_2_Gamma(self.eta)

        # update theta
        for feature in self.theta[0]:
            for param in ['mu']:#,'log_sig']:

                # update parameter
                delta = alpha * np.copy(self.d_log_like_d_theta[t][0][feature][param])
                #print("change in theta feature %s %s" % (feature,param), delta)
                self.theta[0][feature][param] += delta

                # make sure we are in realistic range
                self.theta[0][feature][param] = np.clip(self.theta[0][feature][param],
                                                        self.param_bounds[feature][param][0],
                                                        self.param_bounds[feature][param][1])

        return

    def train_HHMM(self,num_epochs,h=None,decay_ind=0):

        # set h
        if h is None:
            h = self.T

        # initialize important info
        epoch_num = 0
        direction = "forward"
        t = 0

        while epoch_num < num_epochs:
            for _ in range(h):

                # get prev_t for trace purposes
                prev_t = t

                #print("epoch num: ", epoch_num)
                #print("t: ", t)
                #print("direction: ", direction)
                #print("")

                # move forward one step
                if direction == "forward":

                    #print("pre alphas: ", self.log_alphas[:,t])
                    #print("pre betas: ", self.log_betas[:,t])
                    #print("pre log-likelihood: ", self.log_likelihood(t))
                    #print("")

                    self.fwd_step(t)

                    #print("post alphas: ", self.log_alphas[:,t])
                    #print("post betas: ", self.log_betas[:,t])
                    #print("post log-likelihood: ", self.log_likelihood(t))
                    #print("")

                    t += 1

                    if t == self.T:

                        # print results
                        print("finished epoch num: ", epoch_num)
                        print("last direction: ", direction)
                        print("log-likelihood: ", self.log_likelihood(prev_t))

                        # change direction to backward
                        direction = "backward"
                        epoch_num += 1
                        self.step_num += 1
                        t += -1

                # move backward one step
                elif direction == "backward":

                    #print("pre alphas: ", self.log_alphas[:,t])
                    #print("pre betas: ", self.log_betas[:,t])
                    #print("pre log-likelihood: ", self.log_likelihood(t))
                    #print("")

                    self.bwd_step(t)

                    #print("post alphas: ", self.log_alphas[:,t])
                    #print("post betas: ", self.log_betas[:,t])
                    #print("post log-likelihood: ", self.log_likelihood(t))
                    #print("")

                    t += -1

                    if t == -1:

                        # print results
                        print("finished epoch num: ", epoch_num)
                        print("last direction: ", direction)
                        print("log-likelihood: ", self.log_likelihood(prev_t))

                        # change direction to forward
                        direction = "forward"
                        epoch_num += 1
                        self.step_num += 1
                        t += 1

                        # finish off with forward step to complete likelihood
                        self.fwd_step(t)

            # update the paramters
            #print("updating parameters")
            self.update_params(t,decay_ind=decay_ind)

            if direction == "forward":
                self.log_like_trace.append(self.log_likelihood(prev_t))
            else:
                self.log_like_trace.append(self.log_likelihood(prev_t))

            self.theta_trace.append(deepcopy(self.theta))
            self.eta_trace.append(deepcopy(self.eta))

            self.grad_theta_trace.append(deepcopy(self.d_log_like_d_theta[prev_t]))
            self.grad_eta_trace.append(deepcopy(self.d_log_like_d_eta[prev_t]))

            # update the step

            # print log-likelihood
            #print("epoch num: ", epoch_num)
            #print("t: ", t)
            #print("direction: ", direction)
            #print(epoch_num)
            #print("")
            #print("t:")
            #print(t)
            #print("")
            #print("theta:")
            #print(self.theta)
            #print("")
            #print("eta:")
            #print(self.eta)
            #print("")
            #print("alphas: ", self.log_alphas[:,t])
            #print("betas: ", self.log_betas[:,t])
            #print("log-likelihood: ", self.log_likelihood(t))
            #print(self.log_likelihood(t))
            #print("")
            #print("grad log_liklihood:")
            #print(self.d_log_like_d_theta[t])
            #print(self.d_log_like_d_eta[t])
            #print("")
            #print("step size: ", self.step_size / np.sqrt(max(self.step_num-decay_ind,1)))

            #print("")

        return
