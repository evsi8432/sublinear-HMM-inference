import pandas as pd
import numpy as np
import numdifftools as ndt
import statsmodels.api as sm
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

class Uncertainty(HHMM):

    def __init__(self,data,features,K):

        '''
        constructor for optimizor class
        '''

        # get all of the stuff from HHMM
        super().__init__(data,features,K)

        # initialize inverse hessian and CIs
        self.M = None
        self.M_hess = None

        self.hess_inds = None

        self.working_hess = None
        self.working_hess_inv = None
        self.natural_hess = None
        self.natural_hess_inv = None

        # initialize CIs
        self.theta_low = deepcopy(self.theta)
        self.theta_high = deepcopy(self.theta)
        self.Gamma_low = deepcopy(self.eta)
        self.Gamma_high = deepcopy(self.eta)
        self.delta_low = deepcopy(self.eta0)
        self.delta_high = deepcopy(self.eta0)

        # monte carlo estimates of natural params
        self.theta_MC = []
        self.eta_MC = []
        self.eta0_MC = []

        # monte carlo estimates of probabilites
        self.p_Xt_MC = None

        self.p_fine_MC = None
        self.p_fine = None
        self.p_fine_high = None
        self.p_fine_low = None

        self.p_coarse_MC = None
        self.p_coarse = None
        self.p_coarse_high = None
        self.p_coarse_low = None

        return

    def get_working_hess(self):

        x0 = self.theta_2_x()

        # get log likelihood to find hessian indices
        def get_ll(x):

            print(x-x0)

            self.x_2_theta(x)

            for t in range(self.T):
                self.update_alpha(t)

            self.x_2_theta(x0)

            return logsumexp(self.log_alphas[-1])

        # get indices where hessian nonnegative
        hess_diag = ndt.Hessdiag(get_ll,step=0.001)(x0)

        # only keep negative entries, and remove duplicates of hess_diag
        # (these correspond to values of eta that violate identifiability)
        hess_inds = np.where(hess_diag < -1.0e-15)[0]

        hess_inds = []
        seen = set()
        for ind, hess in enumerate(hess_diag):
            if (hess not in seen) and (hess < -1e-15):
                hess_inds.append(ind)
            seen.add(hess)

        print(hess_inds)

        self.hess_inds = hess_inds

        # redefine ll func for hessian with relevant entries
        def get_ll_hess(x):

            x1 = np.copy(x0)
            for i,hess_ind in enumerate(hess_inds):
                x1[hess_ind] = x[i]

            print(x1-x0)

            self.x_2_theta(x1)

            for t in range(self.T):
                self.update_alpha(t)

            self.x_2_theta(x0)

            return logsumexp(self.log_alphas[-1])

        x0_hess = np.array([x0[ind] for ind in hess_inds])

        hess = ndt.Hessian(get_ll_hess,step=0.001)(x0_hess)

        self.working_hess = hess
        self.working_hess_inv = np.linalg.inv(hess)

        self.E_step()

        print(hess)

        return

    def get_natural_hess(self):

        # get number of params
        nparams = len(self.theta_2_x())

        # include diagonals and first elements of Gamma and delta
        # M[i,j] = d natural_i / d working_j
        M = np.zeros((nparams,nparams))

        # update parameters
        ind = 0

        # update theta
        for feature,settings in self.features.items():

            if settings['share_coarse'] and settings['share_fine']:
                for k0,k1 in product(range(self.K[0]),range(self.K[1])):
                    for param in self.theta[k0][feature]:
                        if param == "mu":
                            M[ind,ind] = 1.0
                        elif param == "log_sig":
                            M[ind,ind] = np.exp(self.theta[k0][feature][param][k1])
                        else:
                            print("unrecognized parameter %s" % param)
                ind += 1

            elif settings['share_fine']:
                for k0 in range(self.K[0]):
                    for k1 in range(self.K[1])):
                        for param in self.theta[k0][feature]:
                            if param == "mu":
                                M[ind,ind] = 1.0
                            elif param == "log_sig":
                                M[ind,ind] = np.exp(self.theta[k0][feature][param][k1])
                            else:
                                print("unrecognized parameter %s" % param)
                    ind += 1

            elif settings['share_coarse']:
                for k1 in range(self.K[1]):
                    for k0 in range(self.K[0])):
                        for param in self.theta[k0][feature]:
                            if param == "mu":
                                M[ind,ind] = 1.0
                            elif param == "log_sig":
                                M[ind,ind] = np.exp(self.theta[k0][feature][param][k1])
                            else:
                                print("unrecognized parameter %s" % param)
                    ind += 1

            else:
                for k0,k1 in product(range(self.K[0]),range(self.K[1])):
                    for param in self.theta[k0][feature]:
                        if param == "mu":
                            M[ind,ind] = 1.0
                        elif param == "log_sig":
                            M[ind,ind] = np.exp(self.theta[k0][feature][param][k1])
                        else:
                            print("unrecognized parameter %s" % param)
                    ind += 1

        # get Gamma and delta
        Gamma_coarse,Gammas_fine = eta_2_Gamma(self.eta)
        delta_coarse,deltas_fine = eta0_2_delta(self.eta0)

        # update eta coarse
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                for k in range(self.K[0]):
                    if k == j:
                        M[ind,ind+(k-j)] = Gamma_coarse[i,j]*(1.0-Gamma_coarse[i,j])
                    else:
                        M[ind,ind+(k-j)] = -Gamma_coarse[i,j]*Gamma_coarse[i,k]
                ind += 1

        # update eta fine
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                for j in range(self.K[1]):
                    for k in range(self.K[1]):
                        if k == j:
                            M[ind,ind+(k-j)] = Gammas_fine[k0][i,j]*(1.0-Gammas_fine[k0][i,j])
                        else:
                            M[ind,ind+(k-j)] = -Gammas_fine[k0][i,j]*Gammas_fine[k0][i,k]
                    ind += 1

        # update eta0 coarse
        for i in range(self.K[0]):
            for k in range(self.K[0]):
                if k == i:
                    M[ind,ind+(k-i)] = delta_coarse[i]*(1.0-delta_coarse[i])
                else:
                    M[ind,ind+(k-i)] = -delta_coarse[i]*delta_coarse[k]
            ind += 1

        # update eta0 fine
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                for k in range(self.K[1]):
                    if k == i:
                        M[ind,ind+(k-i)] = deltas_fine[k0][i]*(1.0-deltas_fine[k0][i])
                    else:
                        M[ind,ind+(k-i)] = -deltas_fine[k0][i]*deltas_fine[k0][k]
                ind += 1

        M_hess = M[self.hess_inds,:]

        self.M = M
        self.M_hess = M_hess
        self.natural_hess_inv = np.linalg.multi_dot([np.transpose(M_hess),
                                                     self.working_hess_inv,
                                                     M_hess])

        return

    def get_CIs(self):

        if self.working_hess_inv is None:
            print("Working Hessian is None. Calculating hessian:")
            self.get_working_hess()

        if self.natural_hess_inv is None:
            print("Natural Hessian is None. Calculating hessian:")
            self.get_natural_hess()

        SEs = np.diagonal(-self.natural_hess_inv)

        self.theta_low = deepcopy(self.theta)
        self.theta_high = deepcopy(self.theta)
        self.theta_SE = deepcopy(self.theta)

        self.Gamma_low = deepcopy(self.eta)
        self.Gamma = deepcopy(self.eta)
        self.Gamma_high = deepcopy(self.eta)
        self.Gamma_SE = deepcopy(self.eta)

        self.delta_low = deepcopy(self.eta0)
        self.delta = deepcopy(self.eta0)
        self.delta_high = deepcopy(self.eta0)
        self.delta_SE = deepcopy(self.eta0)

        # update parameters
        ind = 0

        # update theta
        for feature,settings in self.features.items():

            if settings['share_coarse'] and settings['share_fine']:
                for k0,k1 in product(range(self.K[0]),range(self.K[1])):
                    for param in self.theta[k0][feature]:
                        self.theta_low[k0][feature][param][k1] = self.theta[k0][feature][param][k1] - 1.96*SEs[ind]
                        self.theta_high[k0][feature][param][k1] = self.theta[k0][feature][param][k1] + 1.96*SEs[ind]
                        self.theta_SE[k0][feature][param][k1] = SEs[ind]
                ind += 1

            elif settings['share_fine']:
                for k0 in range(self.K[0]):
                    for k1 in range(self.K[1])):
                        for param in self.theta[k0][feature]:
                            self.theta_low[k0][feature][param][k1] = self.theta[k0][feature][param][k1] - 1.96*SEs[ind]
                            self.theta_high[k0][feature][param][k1] = self.theta[k0][feature][param][k1] + 1.96*SEs[ind]
                            self.theta_SE[k0][feature][param][k1] = SEs[ind]
                    ind += 1

            elif settings['share_coarse']:
                for k1 in range(self.K[1]):
                    for k0 in range(self.K[0])):
                        for param in self.theta[k0][feature]:
                            self.theta_low[k0][feature][param][k1] = self.theta[k0][feature][param][k1] - 1.96*SEs[ind]
                            self.theta_high[k0][feature][param][k1] = self.theta[k0][feature][param][k1] + 1.96*SEs[ind]
                            self.theta_SE[k0][feature][param][k1] = SEs[ind]
                    ind += 1

            else:
                for k0,k1 in product(range(self.K[0]),range(self.K[1])):
                    for param in self.theta[k0][feature]:
                        self.theta_low[k0][feature][param][k1] = self.theta[k0][feature][param][k1] - 1.96*SEs[ind]
                        self.theta_high[k0][feature][param][k1] = self.theta[k0][feature][param][k1] + 1.96*SEs[ind]
                        self.theta_SE[k0][feature][param][k1] = SEs[ind]
                    ind += 1

        # get Gamma and delta
        Gamma_coarse,Gammas_fine = eta_2_Gamma(self.eta)
        delta_coarse,deltas_fine = eta0_2_delta(self.eta0)

        # update eta coarse
        for i in range(self.K[0]):
            for j in range(self.K[0]):
                self.Gamma_low[0][i,j] = Gamma_coarse[i,j] - 1.96*SEs[ind]
                self.Gamma[0][i,j] = Gamma_coarse[i,j]
                self.Gamma_high[0][i,j] = Gamma_coarse[i,j] + 1.96*SEs[ind]
                self.Gamma_SE[0][i,j] = SEs[ind]
                ind += 1

        # update eta fine
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                for j in range(self.K[1]):
                    self.Gamma_low[1][k0][i,j] = Gammas_fine[k0][i,j] - 1.96*SEs[ind]
                    self.Gamma[1][k0][i,j] = Gammas_fine[k0][i,j]
                    self.Gamma_high[1][k0][i,j] = Gammas_fine[k0][i,j] + 1.96*SEs[ind]
                    self.Gamma_SE[1][k0][i,j] = SEs[ind]
                    ind += 1

        # update eta0 coarse
        for i in range(self.K[0]):
            self.delta_low[0][i] = delta_coarse[i] - 1.96*SEs[ind]
            self.delta[0][i] = delta_coarse[i]
            self.delta_high[0][i] = delta_coarse[i] + 1.96*SEs[ind]
            self.delta_SE[0][i] = SEs[ind]
            ind += 1

        # update eta0 fine
        for k0 in range(self.K[0]):
            for i in range(self.K[1]):
                self.delta_low[1][k0][i] = deltas_fine[k0][i] - 1.96*SEs[ind]
                self.delta[1][k0][i] = deltas_fine[k0][i]
                self.delta_high[1][k0][i] = deltas_fine[k0][i] + 1.96*SEs[ind]
                self.delta_SE[1][k0][i] = SEs[ind]
                ind += 1

        return

    def get_monte_carlo_theta(self,num_samples):

        # get mu and Sigma
        x = self.output['x']
        mu = x[self.hess_inds]
        Sig = self.working_hess_inv

        # initialize MC params
        self.theta_MC = []
        self.eta_MC = []
        self.eta0_MC = []

        for _ in range(num_samples):

            x_star = deepcopy(x)
            x_star_hess = np.random.multivariate_normal(mu,Sig)

            for ind0,ind in enumerate(self.hess_inds):
                x_star[ind] = x_star_hess[ind0]

            self.x_2_theta(x_star)

            self.theta_MC.append(deepcopy(self.theta))
            self.eta_MC.append(deepcopy(self.eta))
            self.eta0_MC.append(deepcopy(self.eta0))

        # return parameters to true MLE
        self.x_2_theta(x)

        return

    def get_fwd_bwd_prob_CIs(self,perc):

        num_samples = len(self.theta_MC)

        self.p_Xt_MC = np.zeros((self.T,self.K_total,num_samples))

        self.p_fine_MC = np.zeros((self.T,self.K[1],num_samples))
        self.p_fine = np.zeros((self.T,self.K[1]))
        self.p_fine_high = np.zeros((self.T,self.K[1]))
        self.p_fine_low = np.zeros((self.T,self.K[1]))

        self.p_coarse_MC = np.zeros((self.T,self.K[0],num_samples))
        self.p_coarse = np.zeros((self.T,self.K[0]))
        self.p_coarse_high = np.zeros((self.T,self.K[0]))
        self.p_coarse_low = np.zeros((self.T,self.K[0]))

        # hold on to old parameters
        theta_old = deepcopy(self.theta)
        eta_old = deepcopy(self.eta)
        eta0_old = deepcopy(self.eta0)

        sample_num = 0

        for theta,eta,eta0 in zip(self.theta_MC,self.eta_MC,self.eta0_MC):

            print(sample_num/num_samples)

            # set new parameters
            self.theta = deepcopy(theta)
            self.eta = deepcopy(eta)
            self.eta0 = deepcopy(eta0)

            self.get_log_Gamma(jump=True)
            self.get_log_Gamma(jump=False)
            self.get_log_delta()

            # update log_alphas
            for t in range(self.T):
                self.update_alpha(t)

            # update log_betas
            for t in reversed(range(self.T)):
                self.update_beta(t)

            # get p_Xt
            ll = logsumexp(self.log_alphas[-1])
            for t in range(self.T):
                self.p_Xt_MC[t,:,sample_num] = np.exp(self.log_alphas[t] + self.log_betas[t] - ll)

            sample_num += 1

        # return old parameters
        self.theta = theta_old
        self.eta = eta_old
        self.eta0 = eta0_old

        self.get_log_Gamma(jump=True)
        self.get_log_Gamma(jump=False)
        self.get_log_delta()

        self.E_step()

        # get p_coarse, p_fine, and CIs
        for k0 in range(self.K[0]):

            self.p_coarse[:,k0] = np.sum(self.p_Xt[:,(k0*self.K[1]):((k0+1)*self.K[1])],1)
            self.p_coarse_MC[:,k0,:] = np.sum(self.p_Xt_MC[:,(k0*self.K[1]):((k0+1)*self.K[1]),:],1)

            for t in range(self.T):
                self.p_coarse_low[t,k0] = np.quantile(self.p_coarse_MC[t,k0,:],0.5-(perc/2.0))
                self.p_coarse_high[t,k0] = np.quantile(self.p_coarse_MC[t,k0,:],0.5+(perc/2.0))

        for k1 in range(self.K[1]):

            self.p_fine[:,k1] = np.sum(self.p_Xt[:,k1::self.K[1]],1)
            self.p_fine_MC[:,k1,:] = np.sum(self.p_Xt_MC[:,k1::self.K[1],:],1)

            for t in range(self.T):
                self.p_fine_low[t,k1] = np.quantile(self.p_fine_MC[t,k1,:],0.5-(perc/2.0))
                self.p_fine_high[t,k1] = np.quantile(self.p_fine_MC[t,k1,:],0.5+(perc/2.0))

        return

    def get_emissions_dist_CIs(self,feature,K_total,perc):

        num_samples = len(self.theta_MC)

        # get bounds for the feature
        min_mu = np.infty
        max_mu = -np.infty
        max_sig = -np.infty

        for theta in self.theta_MC:
            min_mu0 = np.min([theta_k0[feature]['mu'] for theta_k0 in theta])
            max_mu0 = np.max([theta_k0[feature]['mu'] for theta_k0 in theta])
            max_sig0 = np.exp(np.max([theta_k0[feature]['log_sig'] for theta_k0 in theta]))

            min_mu = np.min([min_mu,min_mu0])
            max_mu = np.max([max_mu,max_mu0])
            max_sig = np.max([max_sig,max_sig0])

        # get param values
        x = np.linspace(min_mu-3*max_sig,max_mu+3*max_sig,1000)

        # see if we are using a truncnorm
        a = self.features[feature]["lower_bound"]
        b = self.features[feature]["upper_bound"]
        if not a is None:
            a = np.concatenate([self.features[feature]["lower_bound"] for _ in self.theta])
        if not b is None:
            b = np.concatenate([self.features[feature]["upper_bound"] for _ in self.theta])

        # now get the y values for each x
        y = np.zeros((1000,K_total,num_samples))
        y_low = np.zeros((1000,K_total))
        y_high = np.zeros((1000,K_total))
        y0 = np.zeros((1000,K_total))

        # true ys
        mu = np.concatenate([theta_k0[feature]['mu'] for theta_k0 in self.theta])
        sig = np.exp(np.concatenate([theta_k0[feature]['log_sig'] for theta_k0 in self.theta]))

        for state in range(K_total):
            if (not a is None) and (not b is None):
                y0[:,state] = truncnorm.pdf(x,a=(a[state]-mu[state])/sig[state],
                                              b=(b[state]-mu[state])/sig[state],
                                              loc=mu[state],scale=sig[state])
            else:
                y0[:,state] = norm.pdf(x,mu[state],sig[state])

        # Monte Carlo ys
        for sample_num,theta in enumerate(self.theta_MC):

            mu = np.concatenate([theta_k0[feature]['mu'] for theta_k0 in theta])
            sig = np.exp(np.concatenate([theta_k0[feature]['log_sig'] for theta_k0 in theta]))

            for state in range(K_total):
                if (not a is None) and (not b is None):
                    y[:,state,sample_num] = truncnorm.pdf(x,a=(a[state]-mu[state])/sig[state],
                                                            b=(b[state]-mu[state])/sig[state],
                                                            loc=mu[state],scale=sig[state])
                else:
                    y[:,state,sample_num] = norm.pdf(x,mu[state],sig[state])

        # find y_high and y_low
        for i in range(1000):
            for state in range(K_total):
                y_low[i,state] = np.quantile(y[i,state,:],0.5-(perc/2.0))
                y_high[i,state] = np.quantile(y[i,state,:],0.5+(perc/2.0))

        return x,y_low,y0,y_high

    def get_pseudoresids(self,feature):

        pseudoresids = []

        # define the feature
        dist = self.features[feature]['f']

        if dist == 'normal':

            # store log-likelihood
            mu = np.concatenate([theta_k0[feature]['mu'] for theta_k0 in self.theta])
            log_sig = np.concatenate([theta_k0[feature]['log_sig'] for theta_k0 in self.theta])
            sig = np.exp(log_sig)

            a = self.features[feature]['lower_bound']
            b = self.features[feature]['upper_bound']

        else:
            raise("only normal pseudoresiduals supported")

        for t in range(self.T):

            y = self.data[t]

            a = self.features[feature]['lower_bound']
            b = self.features[feature]['upper_bound']

            # get cdf for y
            if np.isnan(y[feature]):
                continue
            elif (a is None) or (b is None):
                log_f = norm.logcdf(y[feature],loc=mu,scale=sig)
            else:
                a = np.concatenate([a for _ in self.theta])
                b = np.concatenate([b for _ in self.theta])
                log_f = truncnorm.logcdf(y[feature],
                                         a=(a-mu)/sig,
                                         b=(b-mu)/sig,
                                         loc=mu,scale=sig)

            # get initial index
            seq_num = np.argmax(self.initial_ts > t)-1
            t0 = self.initial_ts[seq_num]

            # get numerator and denominator for pseudoresid
            if t == t0:
                num = logsumexp(self.log_delta + log_f + self.log_betas[t])
                denom = logsumexp(self.log_delta + self.log_betas[t])
            elif (t-t0) % self.jump_every == 0:
                num = logsumexp(logdotexp(self.log_alphas[t-1],self.log_Gamma_jump) + log_f + self.log_betas[t])
                denom = logsumexp(logdotexp(self.log_alphas[t-1],self.log_Gamma_jump) + self.log_betas[t])
            else:
                num = logsumexp(logdotexp(self.log_alphas[t-1],self.log_Gamma) + log_f + self.log_betas[t])
                denom = logsumexp(logdotexp(self.log_alphas[t-1],self.log_Gamma) + self.log_betas[t])

            unif_pseudoresid = np.exp(num-denom)
            norm_pseudoresid = norm.ppf(unif_pseudoresid)
            pseudoresids.append(norm_pseudoresid)

        return pseudoresids
