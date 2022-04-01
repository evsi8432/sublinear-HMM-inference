import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import vonmises
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
from scipy.stats import circstd
from scipy.special import iv
from scipy.special import expit
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.signal import convolve
from scipy.interpolate import interp1d

import divebomb

import time
import pickle


class Visualisor:


    def __init__(self,pars,data,df,hhmm=None):

        self.pars = pars
        self.data = data
        self.df = df
        self.hhmm = hhmm

        return


    def lagplot(self,lims={},file=None):

        ncols = 2
        nrows = int(np.ceil(len(self.pars.features[0]) + len(self.pars.features[1])/2))
        fig,ax = plt.subplots(nrows,ncols,figsize=(2.5*ncols,2.5*nrows))
        fig.delaxes(ax[2,1])
        fig_num = 1

        features = list(self.pars.features[0].keys()) + list(self.pars.features[1].keys())
        xlabs = ['$Y_t$',
                 r'$\left(\tilde A^*_{t,\tilde t^*}\right)_x$',
                 r'$\left(\tilde A^*_{t,\tilde t^*}\right)_y$',
                 r'$\left(\tilde A^*_{t,\tilde t^*}\right)_z$',
                 r'$\tilde W^*_{t,\tilde t^*}$']
        ylabs = ['$Y_{t+1}$',
                 r'$\left(\tilde A^*_{t,\tilde t^*+1}\right)_x$',
                 r'$\left(\tilde A^*_{t,\tilde t^*+1}\right)_y$',
                 r'$\left(\tilde A^*_{t,\tilde t^*+1}\right)_z$',
                 r'$\tilde W^*_{t,\tilde t^*+1}$']

        # lag plots of dive-level data
        for i,feature in enumerate(features):

            # get data
            if feature in self.pars.features[0]:
                x = [x0[feature] for x0 in self.data[:-1]]
                y = [y0[feature] for y0 in self.data[1:]]
            else:
                x = []
                y = []
                for dive in self.data:
                    x.extend([x0[feature] for x0 in dive['subdive_features'][:-1]])
                    y.extend([y0[feature] for y0 in dive['subdive_features'][1:]])

            if feature in lims:
                xlim = lims[feature]
                ylim = lims[feature]
            else:
                xlim = [min(x),max(x)]
                ylim = [min(y),max(y)]

            # print autocorrelation
            print(feature)
            print(np.corrcoef(x,y))

            # KDE of lag plot
            plt.subplot(nrows,ncols,fig_num)
            kernel = gaussian_kde([x,y])
            Xtemp, Ytemp = np.mgrid[xlim[0]:xlim[1]:100j,ylim[0]:ylim[1]:100j]
            positions = np.vstack([Xtemp.ravel(), Ytemp.ravel()])
            Ztemp = np.reshape(kernel.pdf(positions).T, Xtemp.shape)

            #np.set_printoptions(suppress=True,precision=4)
            im = plt.imshow(np.rot90(Ztemp),extent = xlim + ylim)
            plt.xlabel(xlabs[i],fontsize=10)
            plt.ylabel(ylabs[i],fontsize=10)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.gca().ticklabel_format(style='sci',scilimits = (-10,10),useMathText=True)
            plt.gca().yaxis.get_offset_text().set_fontsize(10)
            plt.gca().xaxis.get_offset_text().set_fontsize(10)
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", size="5%", pad=0.05)
            c = fig.colorbar(im,cax=cax)#im, orientation='horizontal',pad=0.125)
            c.ax.tick_params(labelsize=8)
            c.ax.ticklabel_format(style='sci',scilimits = (-10,10))
            c.ax.yaxis.get_offset_text().set_fontsize(10)
            if i%2 == 1 or i==4:
                c.set_label('Density',fontsize = 8)

            fig_num += 1

        plt.subplots_adjust(wspace=0.9, hspace=0.9)
        #fig.text(0.5, 1.0, 'Lag Plots', ha='center', fontsize=50)

        if file is None:
            plt.show()
        else:
            #plt.tight_layout()
            plt.savefig(file,dpi=500)

        return


    def plot_emission_probs(self,level,file=None,features=None):

        if self.hhmm is None:
            print('No trained model')
            return

        if level == 0:
            features = list(self.hhmm.theta[0].keys())
            nrows = 1
            ncols = len(self.hhmm.theta[0])
        else:
            if features is None:
                features = ['Ax','Ay','Az','Ahat_low']#list(self.hhmm.theta[1][0].keys())
            nrows = 2 #self.pars.K[0]
            ncols = int((len(features)+1)/2)

        if level == 0:
            fig, ax = plt.subplots(nrows,ncols,figsize=(max(5,2.5*ncols),2.5*nrows))
            bbox = Bbox([[0,0],[max(5,2.5*ncols),2.5*nrows]])
        else:
            fig, ax = plt.subplots(nrows,ncols,figsize=(max(5,2.5*ncols),max(5,2.5*nrows)))
            bbox = Bbox([[0,0],[max(5,2.5*ncols),2.5*nrows + 0.5]])
        ax = np.reshape(ax,(nrows,ncols))

        for feature_num,feature in enumerate(features):

            row_num = int(feature_num/2)
            col_num = feature_num%2

            if level == 0:
                mu = self.hhmm.theta[0][feature]['mu']
                sig = self.hhmm.theta[0][feature]['sig']

                dist = self.pars.features[0][feature]['f']
                K = self.pars.K[0]
                colors = [cm.get_cmap('tab10')(i) for i in range(K)]
                legend = ['Dive type %d'%(x+1) for x in range(K)]
            else:
                mu = self.hhmm.theta[1][0][feature]['mu']
                sig = self.hhmm.theta[1][0][feature]['sig']

                dist = self.pars.features[1][feature]['f']
                K = self.pars.K[1]
                colors = [cm.get_cmap('viridis')(i/(K-1.0)) for i in range(K)]
                legend = ['Subdive Behavior %d'%(x+1) for x in range(K)]

            for state in range(K):
                if dist == 'gamma':
                    shape = np.square(mu)/np.square(sig)
                    scale = np.square(sig)/np.array(mu)
                    x = np.linspace(0.01,max(mu)+5*max(sig),100000)
                    y = gamma.pdf(x,shape[state],0,scale[state])
                elif dist == 'normal':
                    x = np.linspace(min(mu)-3*max(sig),max(mu)+3*max(sig),100000)
                    y = norm.pdf(x,mu[state],sig[state])
                elif dist == 'vonmises':
                    x = np.linspace(-np.pi,np.pi,100000)
                    y = vonmises.pdf(x,sig[state],loc=mu[state])
                else:
                    raise('distribution %s not recognized' % dist)
                ax[row_num,col_num].plot(x,y,color=colors[state])
                #if col_num == 0:
                ax[row_num,col_num].set_ylabel('Density',fontsize=12)
                if level == 0:
                    title = 'Emission Distributions, Dive Duration'
                    ax[row_num,col_num].set_xlabel('$Y_t$ $(s)$')
                    #ax[row_num,col_num].set_title(title,fontsize=12)
                    plt.legend(['Dive type 1','Dive type 2'],fontsize=10)
                else:
                    titles = [r'$\left(\tilde A^*_{t,\tilde t^*}\right)_x$ $(m/s^2)$',
                              r'$\left(\tilde A^*_{t,\tilde t^*}\right)_y$ $(m/s^2)$',
                              r'$\left(\tilde A^*_{t,\tilde t^*}\right)_z$ $(m/s^2)$',
                              r'$\tilde W^*_{t,\tilde t^*}$']
                    title = titles[feature_num]
                    ax[row_num,col_num].set_xlabel(title,fontsize=12)
                    if feature == 'Ahat_low':
                        ax[row_num,col_num].set_xscale('log')
                        ax[row_num,col_num].set_yscale('log')
                        ax[row_num,col_num].set_ylim([10e-8,10e-1])
                        ax[row_num,col_num].set_xlim([10e-1,10e4])

                    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Subdive state 1',
                                           markerfacecolor=colors[0], markersize=8),
                                       Line2D([0], [0], marker='o', color='w', label='Subdive state 2',
                                           markerfacecolor=colors[1], markersize=8),
                                       Line2D([0], [0], marker='o', color='w', label='Subdive state 3',
                                           markerfacecolor=colors[2], markersize=8)]

                    fig.legend(handles=legend_elements,
                               prop={'size': 10}, ncol=3,
                               mode='expand',
                               bbox_to_anchor=(0.0, 0.99, 1., .1),
                               loc='lower left')

                    #fig.text(0.5, 1.0, 'Fine Scale Emission Distributions', ha='center', fontsize=50)

        if file is None:
            plt.show()
        elif file == 'noshow':
            return (fig, ax)
        else:
            plt.tight_layout()
            plt.savefig(file,bbox_inches=bbox,dpi=500)

        return


    def print_ptms(self):

        if self.hhmm is None:
            print('No trained model')
            return

        print('Probability transistion matrix for Dive types:')

        ptm = np.exp(self.hhmm.eta[0])
        ptm = (ptm.T/np.sum(ptm,1)).T
        print(ptm)
        print('')

        print('Stationary distribution for dive types:')
        delta = np.ones((1,self.pars.K[0]))/self.pars.K[0]
        for _ in range(100):
            delta = delta.dot(ptm)
        print(delta)
        print('')
        print('')
        print('')
        print('')

        for dive_type in range(self.pars.K[0]):

            print('Probability transistion matrix for subdive behaviors, '
                  'dive type %s:'%(dive_type+1))
            ptm = np.exp(self.hhmm.eta[1][dive_type])
            ptm = (ptm.T/np.sum(ptm,1)).T
            print(ptm)
            print('')

            print('Stationary Distribution for subdive behaviors, '
                  'dive type %s:'%(dive_type+1))
            delta = np.ones((1,self.pars.K[1]))/self.pars.K[1]
            for _ in range(100):
                delta = delta.dot(ptm)
            print(delta)
            print('')
            print('')

        return


    def plot_dive_features(self,sdive,edive,df_cols,data_cols,file=None,ylabs=None,nrows=None):

        df = self.df
        data = self.data

        if nrows is None:
            nrows = 2*len(df_cols)
            for col in data_cols:
                if col in self.pars.features[0]:
                    nrows += 1
                else:
                    nrows += 2

        plt.subplots(nrows,1,figsize=(6,1.5*nrows))
        fignum = 1

        # get df state-by-state
        subdive = df[(df['dive_num'] >= sdive) & (df['dive_num'] <= edive)].copy()
        subdive['sec_from_start'] -= min(subdive['sec_from_start'])
        subdives = []
        for state in range(self.pars.K[1]):
            subdives.append(subdive[subdive['ML_subdive'] == state])

        dive = df[(df['dive_num'] >= sdive) & (df['dive_num'] <= edive)].copy()
        dive['sec_from_start'] -= min(dive['sec_from_start'])
        dives = []
        for state in range(self.pars.K[0]):
            dives.append(dive[dive['ML_dive'] == state])

        if ylabs is None:
            ylabs = [r'$Pr\left(X_t = 1\right)$',
                     r'$Pr\left(X_t = 2\right)$',
                     r'$Pr\left(\tilde X^*_{t,\tilde t^*} = 1\right)$',
                     r'$Pr\left(\tilde X^*_{t,\tilde t^*} = 2\right)$',
                     r'$Pr\left(\tilde X^*_{t,\tilde t^*} = 3\right)$']
            #ylabs = [r'$\left(Z^{*(1)}\right)_x$ $(m/s^2)$',
            #         r'Depth $(m)$']

        #title = 'Hidden State Probability Estimates'
        title = 'Decoded Dive Profile / Accelerometer Data'
        for i,col in enumerate(df_cols):

            # dive-level coloring
            if 'subdive_state_' not in col:

                plt.subplot(nrows,1,fignum)
                colors = [cm.get_cmap('tab10')(i) for i in range(self.pars.K[0])]
                legend = ['Dive type %d' % (i+1) for i in range(self.pars.K[0])]
                for state,dive_df in enumerate(dives):
                    plt.plot(dive_df['sec_from_start']/60,dive_df[col],
                             '.',color=colors[state],markersize=4)
                if 'prob' in col:
                    plt.plot(dive[dive[col] > -0.01]['sec_from_start']/60,dive[dive[col] > -0.01][col],
                             'k-',alpha=0.5,linewidth=0.5)
                    plt.axhline(0.5,color='k',linestyle=':',alpha=0.5,linewidth=0.5)
                    plt.ylim([-0.05,1.05])
                    plt.yticks([0,0.5,1.0],fontsize=30)
                else:
                    plt.plot(dive['sec_from_start']/60,dive[col],'k--',alpha=0.5,linewidth=0.5)
                plt.yticks(fontsize=10)
                plt.ylabel(ylabs[i],fontsize=10)
                plt.xticks([])
                if col == df_cols[0]:
                    #plt.title(title,fontsize=36)
                    pass
                if col == 'depth':
                    plt.gca().invert_yaxis()

                # plot vlines
                for dive_num in range(sdive+1,edive+1):
                    vline = min(dive[dive['dive_num'] == dive_num]['sec_from_start']/60)
                    plt.axvline(vline,color='k',linewidth=2)
                #if col == 'Ax':
                #plt.ylim([-2,2])
                fignum += 1#2

            # subdive-level coloring
            if 'dive_state_' not in col or 'subdive_state' in col:
                plt.subplot(nrows,1,fignum)
                colors = [cm.get_cmap('viridis')(i/(self.pars.K[1]-1.0)) for i in range(self.pars.K[1])]
                #legend = ['Subdive Behavior %d' % (i+1) for i in range(self.pars.K[1])]
                for state,subdive_df in enumerate(subdives):
                    plt.plot(subdive_df['sec_from_start']/60,subdive_df[col],
                             '.',color=colors[state],markersize=4)
                if 'prob' in col:
                    plt.plot(dive[dive[col] > -0.01]['sec_from_start']/60,
                             dive[dive[col] > -0.01][col],'k-',alpha=0.5,linewidth=0.5)
                    plt.axhline(0.5,color='k',linestyle=':',alpha=0.5,linewidth=0.5)
                    plt.ylim([-0.05,1.05])
                    plt.yticks([0,0.5,1.0],fontsize=10)
                else:
                    plt.plot(dive['sec_from_start']/60,dive[col],'k--',alpha=0.5,linewidth=0.5)
                plt.yticks(fontsize=10)
                plt.ylabel(ylabs[i],fontsize=10)
                if col == df_cols[-1]:
                    plt.xlabel('Time $(min)$',fontsize=10)
                    plt.xticks(fontsize=10)
                else:
                    plt.xticks([])
                #plt.legend(legend,prop={'size': 20})
                #plt.title(col + ', dives %d-%d'%(sdive,edive),fontsize=24)
                if col == 'depth':
                    plt.gca().invert_yaxis()

                # plot vlines
                for dive_num in range(sdive+1,edive+1):
                    vline = min(dive[dive['dive_num'] == dive_num]['sec_from_start']/60)
                    plt.axvline(vline,color='k',linewidth=2)
                # if col == 'Ax':
                #     plt.ylim([-2,2])
                fignum += 1#-1

        t_start = dive['time'].min()
        def time2sec(t):
            return (t-t_start)/pd.Timedelta(1,'s')

        for col in data_cols:

            if col in self.pars.features[0]:

                # dive-level columns - color by dive type only
                plt.subplot(nrows,1,fignum)
                colors = [cm.get_cmap('tab10')(i) for i in range(self.pars.K[0])]
                legend = ['Dive type %d' % (i+1) for i in range(self.pars.K[0])]

                times = [[]] * self.pars.K[0]
                features = [[]] * self.pars.K[0]

                time = []
                feature = []

                for dive in data[sdive:edive+1]:
                    if 'dive_state_probs' in dive:
                        ML_state = np.argmax(dive['dive_state_probs'])
                        avg_time = 0.5*(time2sec(dive['start_dive']) + time2sec(dive['end_dive']))
                        times[ML_state].append(avg_time)
                        time.append(avg_time)
                        features[ML_state].append(dive[col])
                        feature.append(dive[col])
                    #plt.axvline(max(time),color='k',linewidth=2)
                for state in range(self.pars.K[0]):
                    plt.plot(times[state],features[state],
                             '.',color=colors[state],markersize=4)
                plt.plot(time,feature,'k--',alpha=0.5,linewidth=1)
                plt.ylabel(col,fontsize = 10)
                plt.xlabel('Time (s)',fontsize=10)
                plt.legend(legend,prop={'size': 10})

                fignum += 1

            else:

                # subdive-level columns - color by dive type
                plt.subplot(nrows,1,fignum)
                colors = [cm.get_cmap('tab10')(i) for i in range(self.pars.K[0])]
                legend = ['Dive type %d' % (i+1) for i in range(self.pars.K[0])]

                times = [ [] for _ in range(self.pars.K[0]) ]
                features = [ [] for _ in range(self.pars.K[0]) ]

                time = []
                feature = []

                vlines = []

                for dive in data[sdive:edive+1]:
                    if 'dive_state_probs' in dive:
                        ML_state = np.argmax(dive['dive_state_probs'])
                        for seg in dive['subdive_features']:
                            times[ML_state].append(time2sec(seg['start_time']))
                            time.append(time2sec(seg['start_time']))
                            features[ML_state].append(seg[col])
                            feature.append(seg[col])
                    vlines.append(max(time))
                for state in range(self.pars.K[0]):
                    plt.plot(times[state],features[state],
                             '.',color=colors[state],markersize=4)
                plt.plot(time,feature,'k--',alpha=0.5,linewidth=1)
                plt.ylabel(col,fontsize = 14)
                plt.xlabel('Time (s)',fontsize=10)
                plt.legend(legend,prop={'size': 10})
                #for vline in vlines:
                    #plt.axvline(vline,color='k',linewidth=2)
                fignum += 1


                # subdive-level columns - color by subdive type
                plt.subplot(nrows,1,fignum)
                colors = [cm.get_cmap('viridis')(i/self.pars.K[1]-1.0) for i in range(self.pars.K[1])]
                legend = ['Subdive Behavior %d' % (i+1) for i in range(self.pars.K[1])]

                times = [ [] for _ in range(self.pars.K[1]) ]
                features = [ [] for _ in range(self.pars.K[1]) ]

                time = []
                feature = []

                for dive in data[sdive:edive+1]:
                    for seg in dive['subdive_features']:
                        if 'subdive_state_probs' in seg:
                            ML_state = np.argmax(seg['subdive_state_probs'])
                            times[ML_state].append(time2sec(seg['start_time']))
                            time.append(time2sec(seg['start_time']))
                            features[ML_state].append(seg[col])
                            feature.append(seg[col])
                for state in range(self.pars.K[1]):
                    plt.plot([t for t in times[state]],features[state],
                             '.',color=colors[state],markersize=4)
                plt.plot([t for t in time],feature,'k--',alpha=0.5,linewidth=0.5)
                plt.yticks(fontsize=10)
                plt.ylabel(col,fontsize=10)
                plt.xticks(fontsize=10)
                plt.xlabel('Time (secs)',fontsize=10)
                plt.legend(legend,prop={'size': 10})
                plt.title('Depth Data',fontsize=10)
                #for vline in vlines:
                    #plt.axvline(vline,color='k',linewidth=2)
                fignum += 1

        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Subdive state 1',
                                  markerfacecolor=cm.get_cmap('viridis')(0.0), markersize=8),
                           Line2D([0], [0], marker='o', color='w', label='Dive type 1',
                                  markerfacecolor=cm.get_cmap('tab10')(0), markersize=8),
                           Line2D([0], [0], marker='o', color='w', label='Subdive state 2',
                                  markerfacecolor=cm.get_cmap('viridis')(0.5), markersize=8),
                           Line2D([0], [0], marker='o', color='w', label='Dive type 2',
                                  markerfacecolor=cm.get_cmap('tab10')(1), markersize=8),
                           Line2D([0], [0], marker='o', color='w', label='Subdive state 3',
                                  markerfacecolor=cm.get_cmap('viridis')(1.0), markersize=8)]
        plt.subplot(nrows,1,1)
        plt.gca().legend(handles=legend_elements,prop={'size': 8}, ncol=3, mode="expand", borderaxespad=0.,
                         bbox_to_anchor=(0., 1.1, 1.0, .102), loc='lower left')
        plt.subplots_adjust(wspace=0, hspace=0.1)

        if file:
            #plt.tight_layout()
            plt.savefig(file,dpi=500)
        else:
            plt.show()

        return
