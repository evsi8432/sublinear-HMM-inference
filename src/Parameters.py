import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.optimize import LinearConstraint
from scipy.signal import convolve
from scipy.signal import stft
from scipy.special import expit

import divebomb

import time
import pickle


class Parameters:


    def __init__(self):

        # loading parameters
        self.cvc_file = '../Data/20190902-182840-CATs_OB_1.cvc'
        self.csv_file = '../Data/20190902-182840-CATs_OB_1_001.csv'

        self.cvc_cols = ['date','time',
                         'GPS lat','GPS long','GPS alt','GPS speed',
                         'quats0','quats1','quats2','quats3',
                         'acc x','acc y','acc z',
                         'gyro x','gyro y','gyro z',
                         'mag x','mag y','mag z',
                         'mag cal x','mag cal y','mag cal z',
                         'dyn acc x','dyn acc y','dyn acc z',
                         'press','pitot speed','GPS num sat',
                         'roll','pitch','yaw']

        self.renamed_cvc_cols = {'dyn acc x': 'Ax',
                                 'dyn acc y': 'Ay',
                                 'dyn acc z': 'Az',
                                 'mag cal x': 'Mx',
                                 'mag cal y': 'My',
                                 'mag cal z': 'Mz',
                                 'Depth (100bar) [m]' : 'depth',
                                 'Temperature (depth) [°C]': 'temp'}

        # preprocessing parameters
        self.freq = 50 # in Hz
        self.stime = '2019-09-02 13:20:00'
        self.etime = '2019-09-02 18:00:00'
        self.drop_times = [[1.6,1.8],[3.3,4.3]] # in hr after start
        self.smoother = [0.2]*5
        self.smooth_cols = ['depth','elevation']#,'Vz','heading','roll',]


        # parameters for HHMM features
        self.timescales = [(10,'min'),(2,'sec')]
        self.K = [2,3] # number of states per level
        self.share_fine_states = True # are fine states same for all crude states?

        self.features = [{'max_depth':{'corr':False,'f':'gamma'},
                          'bottom_variance':{'corr':False,'f':'gamma'},
                          'dive_duration':{'corr':False,'f':'gamma'}},

                         {'Ax':{'corr':True,'f':'normal'},
                          'Ay':{'corr':True,'f':'normal'},
                          'Az':{'corr':True,'f':'normal'},
                          'FoVeDBA_low':{'thresh':5,'corr':True,'f':'gamma'},
                          'FoVeDBA_high': {'thresh':5,'corr':True,'f':'gamma'},
                          'Vz': {'corr':True,'f':'normal'},
                          'peak_jerk': {'corr':False,'f':'normal'}, # in seconds
                          'roll_at_pj': {'corr':False,'f':'vonmises'},
                          'heading_var': {'corr':False,'f':'gamma'}}]

        return
