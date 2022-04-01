import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
from scipy.stats import circvar
from scipy.special import expit
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.signal import convolve
from scipy.interpolate import interp1d

import divebomb

import time
import pickle



class Preprocessor:


    def __init__(self,pars):

        self.pars = pars
        self.df = None
        self.dive_df = None

        return


    def load_data(self,cvc_file,csv_file,cvc_cols):

        '''
        load in the dataframe in pandas and do basic manipulations
        '''

        df = pd.read_csv(cvc_file,skiprows=[0],names=cvc_cols,encoding='cp1252')
        df = df.merge(pd.read_csv(csv_file,encoding='cp1252'),
                      right_on = ['Date (UTC)', 'Time (UTC)'],
                      left_on = ['date','time'])

        df['time'] = pd.to_datetime(df[' Date (local)']+' '+df[' Time (local)'],
                                    format = "%d.%m.%Y %H:%M:%S.%f")

        df = df.rename(columns = self.pars.renamed_cvc_cols)

        self.df = df

        return df


    def prune_cols(self,df):

        '''
        only include needed columns in the dataset
        '''

        cols_to_keep = set(['time','depth','Ax','Ay','Az'])
        acc_cols = ['A','Ahat_low','Ahat_high','FoVeDBA_low','FoVeDBA_high',
                    'peak_jerk','roll_at_pj']
        roll_cols = ['roll_at_pj']
        head_cols = ['heading_var']

        for level in self.pars.features:
            for feature in level:
                if feature in acc_cols:
                    df['VeDBA'] = np.sqrt(np.square(df['Ax']) + \
                                          np.square(df['Ay']) + \
                                          np.square(df['Az']))
                    cols_to_keep.add('VeDBA')

                if feature in roll_cols:
                    df['roll'] = (np.pi/180.0)*df['roll']
                    df['roll'] = (df['roll'] - df['roll'].mean())
                    df['roll'] = (df['roll']+np.pi)%(2*np.pi)-np.pi
                    cols_to_keep.add('roll')

                if feature in head_cols:
                    df['heading'] = np.arctan2(df['My'],df['Mx'])
                    cols_to_keep.add('heading')

        df = df[list(cols_to_keep)]

        return df


    def prune_times(self,df,stime,etime,drop_times):

        '''
        get rid of times in the dataset that are missing data
        '''

        # get times in terms of time from start
        df['sec_from_start'] = (df['time']-min(df['time']))/np.timedelta64(1,'s')
        df['hr_from_start'] = df['sec_from_start']/3600

        # get rid of begining buffer
        df = df[df['time'] >= stime]
        df = df[df['time'] <= etime]

        # get rid of times between drop_times:
        for drop_time in drop_times:
            df = df[(df['hr_from_start'] < drop_time[0])| \
                    (df['hr_from_start'] > drop_time[1])]

        # name each time series in the dataset
        def get_time_series(x):
            i = 0
            for drop_time in drop_times:
                if x < drop_time[0]:
                    return i
                else:
                    i += 1
            return i

        df['time_series'] = df['hr_from_start'].apply(get_time_series)

        return df


    def fix_pressure(self,df):

        '''
        correct for offset in dive dataset
        '''

        times = []
        depths = []
        surf_df = df[df['depth'] < np.quantile(df['depth'],0.05)]
        gbo = surf_df.groupby(surf_df.index.to_series().diff().ne(1).cumsum())
        for key in gbo.groups:
            group = gbo.get_group(key)
            if len(group) > 50:
                ind = group['depth'].argmin()
                times.append(group['sec_from_start'].iloc[ind])
                depths.append(group['depth'].iloc[ind])

        f = interp1d(times, depths,
                     bounds_error = False,
                     fill_value = (depths[0],depths[-1]))

        offset = f(df['sec_from_start'])
        df['depth'] = df['depth']-offset
        df['elevation'] = -df['depth']
        return df


    def find_Vz(self,df):

        '''
        find vertical velocity
        '''

        # calculate Vz
        df['Vz'] = (df['depth'].shift(1)-df['depth'].shift(-1))*self.pars.freq/2
        df['Vz'].iloc[0] = df['Vz'].iloc[1]
        df['Vz'].iloc[-1] = df['Vz'].iloc[-2]

        # smooth Vz
        smoother = norm.pdf(np.linspace(-3,3,self.pars.freq+1)) # smooth over 1s
        smoother = smoother/sum(smoother)

        new_col = []
        for time_series in range(int(df['time_series'].max() + 1)):
            overflow = int((self.pars.freq-1)/2)
            new_col0 = []
            new_col0.extend([0]*int(self.pars.freq/2))
            new_col0.extend(list(convolve(smoother,
                            df[df['time_series'] == time_series]['Vz']))\
                            [self.pars.freq:-self.pars.freq])
            new_col0.extend([0]*int(self.pars.freq/2))


            for i in range(overflow):
                new_col0[i] = new_col0[overflow]
                new_col0[-(i+1)] = new_col0[-overflow-1]

            new_col.extend(new_col0)

        df['Vz'] = new_col

        return df


    def smooth_columns(self,df,smoother,cols):

        '''
        smooth columns that need it
        '''

        overflow = len(smoother)-1

        for col in cols:
            new_col = []
            for time_series in range(df['time_series'].max() + 1):
                df_col = df[df['time_series'] == time_series][col]

                new_col.extend(list(df_col.iloc[list(range(int(overflow/2)))]))
                new_col.extend(list(convolve(smoother,df_col))[overflow:-overflow])
                new_col.extend(list(df_col.iloc[list(range(-int(overflow/2),0))]))

            df[col] = new_col

        return df


    def find_dives(self,df):

        # use divebomb to get dives
        raw_data = divebomb.profile_dives(df,
                                          minimal_time_between_dives=10,
                                          surface_threshold=0.5,
                                          ipython_display_mode=False)

        # merge dive info with raw data
        dives = raw_data[0]
        dives = pd.merge(dives,
                         raw_data[2][['time','time_series']]\
                            .rename(columns={'time':'bottom_start'})\
                            .drop_duplicates(),
                         on='bottom_start')

        # drop first and last incomplete dives
        dives = dives.drop(len(dives)-1)
        dives = dives.drop(0)

        # get rid of dives between time series
        dives = dives[dives['td_total_duration'] < 1000]
        dives = dives.reset_index().drop(columns=['index'])
        self.dive_df = dives

        # get dive numbers
        df_temp = raw_data[2]
        df_temp = df_temp.reset_index().drop(columns=['index'])
        dive_nums = pd.Series([-1]*len(df))
        for dive_num,(s,e) in enumerate(zip(dives['dive_start'],dives['dive_end'])):
            ind = (df_temp['time'] > s) & (df_temp['time'] < e)
            dive_nums[ind] = dive_num

        # move everything over to df_temp
        df_temp['dive_num'] = dive_nums
        df_temp['time'] = list(df['time'])
        df = df_temp.copy()
        del df_temp

        df = df[df['dive_num'] != -1]
        df = df[df['depth'] > 0.2]

        return df,dives


    def get_dive_features(self,dive_df,dive_num):

        features = {}

        if 'max_depth' in self.pars.features[0]:
            features['max_depth'] = dive_df['max_depth'][dive_num]
        if 'bottom_variance' in self.pars.features[0]:
            features['bottom_variance'] = dive_df['bottom_variance'][dive_num]
        if 'dive_duration' in self.pars.features[0]:
            features['dive_duration'] = dive_df['td_total_duration'][dive_num]

        return features


    def get_subdive_features(self,subdive_df):

        subdive_features = []
        nperseg = int(self.pars.timescales[1][0]*self.pars.freq)
        if self.pars.timescales[1][1] == 'min':
            nperseg *= 60
        inds = np.arange(0,len(subdive_df)-nperseg,nperseg)

        for ind_start in inds:
            dive_seg_features = {}
            dive_seg = subdive_df.iloc[ind_start:ind_start+nperseg]

            # find average Acceleration
            if 'A' in self.pars.features[1]:
                Ax = np.mean(dive_seg['Ax'])
                Ay = np.mean(dive_seg['Ay'])
                Az = np.mean(dive_seg['Az'])
                dive_seg_features['A'] = np.array([Ax,Ay,Az])

            # find average Acceleration
            if 'Ax' in self.pars.features[1]:
                dive_seg_features['Ax'] = np.mean(dive_seg['Ax'])
                dive_seg_features['Ax_all'] = dive_seg['Ax'].to_numpy()
            if 'Ay' in self.pars.features[1]:
                dive_seg_features['Ay'] = np.mean(dive_seg['Ay'])
                dive_seg_features['Ay_all'] = dive_seg['Ay'].to_numpy()
            if 'Az' in self.pars.features[1]:
                dive_seg_features['Az'] = np.mean(dive_seg['Az'])
                dive_seg_features['Az_all'] = dive_seg['Az'].to_numpy()

            # find depth
            dive_seg_features['depth_all'] = dive_seg['depth'].to_numpy()

            # find FoVeDBA
            if ('Ahat_low' in self.pars.features[1]) or ('Ahat_high' in self.pars.features[1]):

                freqs = np.fft.rfftfreq(nperseg, d=1/self.pars.freq)
                fftx = np.absolute(np.fft.rfft(dive_seg['Ax']))
                ffty = np.absolute(np.fft.rfft(dive_seg['Ay']))
                fftz = np.absolute(np.fft.rfft(dive_seg['Az']))

                if 'Ahat_low' in self.pars.features[1]:
                    thresh = self.pars.features[1]['Ahat_low']['thresh']
                    thresh_ind = max(np.where(freqs <= thresh)[0]) + 1
                    dive_seg_features['Ahat_low'] = np.sum(fftx[1:thresh_ind]**2)
                    dive_seg_features['Ahat_low'] += np.sum(ffty[1:thresh_ind]**2)
                    dive_seg_features['Ahat_low'] += np.sum(fftz[1:thresh_ind]**2)

                if 'Ahat_high' in self.pars.features[1]:
                    thresh = self.pars.features[1]['Ahat_high']['thresh']
                    thresh_ind = max(np.where(freqs <= thresh)[0]) + 1
                    dive_seg_features['Ahat_high'] = np.sum(fftx[thresh_ind:]**2)
                    dive_seg_features['Ahat_high'] += np.sum(ffty[thresh_ind:]**2)
                    dive_seg_features['Ahat_high'] += np.sum(fftz[thresh_ind:]**2)

            # find FoVeDBA
            if ('FoVeDBA_low' in self.pars.features[1]) or ('FoVeDBA_high' in self.pars.features[1]):

                freqs = np.fft.rfftfreq(nperseg, d=1/self.pars.freq)
                fft = np.absolute(np.fft.rfft(dive_seg['VeDBA']))

                if 'FoVeDBA_low' in self.pars.features[1]:
                    thresh = self.pars.features[1]['FoVeDBA_low']['thresh']
                    thresh_ind = max(np.where(freqs <= thresh)[0]) + 1
                    dive_seg_features['FoVeDBA_low'] = np.sum(fft[1:thresh_ind]**2)

                if 'FoVeDBA_high' in self.pars.features[1]:
                    thresh = self.pars.features[1]['FoVeDBA_high']['thresh']
                    thresh_ind = max(np.where(freqs <= thresh)[0]) + 1
                    dive_seg_features['FoVeDBA_high'] = np.sum(fft[thresh_ind:]**2)

            # find Vz
            if 'Vz' in self.pars.features[1]:
                dive_seg_features['Vz'] = dive_seg['depth'].iloc[0] - dive_seg['depth'].iloc[-1]#dive_seg['Vz'].mean()
                dive_seg_features['Vz'] = dive_seg_features['Vz']*self.pars.freq/nperseg

            # find peak_jerk and roll at peak_jerk
            if ('peak_jerk' in self.pars.features[1]) or ('roll_at_pj' in self.pars.features[1]):
                pj = dive_seg['VeDBA'].diff().max()

                # find peak_jerk
                if 'peak_jerk' in self.pars.features[1]:
                    dive_seg_features['peak_jerk'] = pj*self.pars.freq

                # find roll at peak peak_jerk
                if 'roll_at_pj' in self.pars.features[1]:
                    roll = dive_seg[dive_seg['VeDBA'].diff() == pj]['roll'].iloc[0]
                    dive_seg_features['roll_at_pj'] = roll

            # find heading variance
            if 'heading_var' in self.pars.features[1]:
                dive_seg_features['heading_var'] = circvar(dive_seg['heading'])

            # find start and end time of subdive
            dive_seg_features['start_time'] = dive_seg['time'].min()
            dive_seg_features['end_time'] = dive_seg['time'].max()

            subdive_features.append(dive_seg_features)

        return subdive_features


    def get_all_features(self,df,dive_df):

        features = []
        gbo = df.groupby('dive_num')

        for dive_num in gbo.groups:
            subdive_df = gbo.get_group(dive_num)
            dive_features = self.get_dive_features(dive_df,dive_num)
            dive_features['subdive_features'] = self.get_subdive_features(subdive_df)
            dive_features['start_dive'] = dive_features['subdive_features'][0]['start_time']
            dive_features['end_dive'] = dive_features['subdive_features'][-1]['end_time']
            features.append(dive_features)

        return features
