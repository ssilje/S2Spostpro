import xarray as xr
import pandas as pd
import numpy as np
import xskillscore as xs

import json
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from S2S.local_configuration import config
from S2S.graphics import latex, graphics

from matplotlib.colors import BoundaryNorm

from S2S.data_handler import ERA5, BarentsWatch
from S2S.process import Hindcast, Observations, Grid2Point
from S2S import models
import S2S.scoring as sc
import S2S.location_cluster as lc

from . import norkyst_restack

nk = norkyst_restack.run()



def plus_minus_15_days(t_start,t_end):
    if t_start[1]==1:
        t_start = (t_start[0]-1,12,15)
    else:
        t_start = (t_start[0],t_start[1]-1,15)

    if t_end[1]==12:
        t_end = (t_end[0]+1,1,15)
    else:
        t_end = (t_end[0],t_end[1]+1,15)

    return t_start,t_end

# bounds = (0,28,55,75)
bounds = (0,28,55,75)
var      = 'sst'

clim_t_start  = (2000,1,1)
clim_t_end    = (2021,1,4)

steps    = pd.to_timedelta([9,16,23,30,37],'D')

high_res = True

mparser = {
            '1':'JAN','2':'FEB','3':'MAR',
            '4':'APR','5':'MAY','6':'JUN',
            '7':'JUL','8':'AUG','9':'SEP',
            '10':'OCT','11':'NOV','12':'DEC'
        }
months  = ['01','02','03','04','05','06','07','08','09','10','11','12']

mae_fc, mae_clim, mae_pers = [], [], []
################################################################################
# bw = BarentsWatch().load('all',no=0).sortby('time')[var]
print(len(nk.location.values))
# nk = lc.cluster(nk,'Hisdalen',3,1.5) # change filenames!
print(len(nk.location.values))
t_start  = (2020,8,1) #can start with 8
t_end    = (2021,9,1)
model    = ''
extent   = [4.5,7.1,59.3,61]

nk = nk.where(nk.lon <= extent[1], drop=True)
nk = nk.where(nk.lon >= extent[0], drop=True)

nk = nk.where(nk.lat <= extent[3], drop=True)
nk = nk.where(nk.lat >= extent[2], drop=True)

# nk = []
# for loc in bw.location.values:
#
#     fname =                 config['NORKYST'] +\
#                                 'NorKyst800_' +\
#                                      str(loc) +\
#                                         '.nc'
#
#     nk.append(xr.open_dataset(fname)[var].drop('radius'))
# nk = xr.concat(nk,'location')

hindcast = Hindcast(
                    var,
                    t_start,
                    t_end,
                    bounds,
                    high_res   = high_res,
                    steps      = steps,
                    process    = False,
                    download   = False,
                    split_work = False,
                    period     = [nk.time.min(),nk.time.max()]
                )

observations = Observations(
                            name='Hardanger_NorKyst-800_hisdalenC',
                            observations=nk,
                            forecast=hindcast,
                            process=False
                            )

for month in np.arange(1,13,1):
    print(np.isfinite(observations.data.where(observations.data.time.dt==month).values).sum())

del nk

hindcast = Grid2Point(observations,hindcast)\
                        .correlation(step_dependent=True)

mae_fc = xs.mae(
        hindcast.data_a.mean('member'),
        observations.data_a,
        dim=[]
    )

crps_fc = sc.crps_ensemble(observations.data_a,hindcast.data_a)


del hindcast

crps_clim = xs.crps_gaussian(
                                observations.data_a,
                                xr.zeros_like(observations.data_a),
                                xr.ones_like(observations.data_a),
                                dim=[]
                            )

mae_clim = xs.mae(
        xr.zeros_like(observations.data_a),
        observations.data_a,
        dim=[]
    )

pers    = models.persistence(
                init_value   = observations.init_a,
                observations = observations.data_a
                )

mae_pers = xs.mae(
        pers,
        observations.data_a,
        dim=[]
    )

del observations

for season in ['DJF','MAM','JJA','SON']:

    season = str(season)

    mae_fc_mon = mae_fc.where(mae_fc.time.dt.season==season,drop=True)

    crps_fc_mon = crps_fc.where(crps_fc.time.dt.season==season,drop=True)

    crps_clim_mon = crps_clim.where(crps_clim.time.dt.season==season,drop=True)

    mae_pers_mon = mae_pers.where(mae_pers.time.dt.season==season,drop=True)

    mae_clim_mon = mae_clim.where(mae_clim.time.dt.season==season,drop=True)


    for ref_fc,lab in zip([mae_clim_mon,mae_pers_mon],['CLIM','PERS']):

        subplots = (3,5)
        ###########################
        #### Initialize figure ####
        ###########################
        latex.set_style(style='white')
        fig,axes = plt.subplots(subplots[0],subplots[1],
                        figsize=latex.set_size(width='thesis',
                            subplots=(subplots[0],subplots[1]))
                        )
        ###########################

        for m,month in enumerate(np.unique(ref_fc.time.dt.month.values)):
            for s,step in enumerate(steps):

                fc  = mae_fc_mon.where(mae_fc_mon.time.dt.month==month)\
                                                        .mean('time',skipna=True)
                ref = ref_fc.where(ref_fc.time.dt.month==month)\
                                                        .mean('time',skipna=True)

                ax = axes[m,s]

                ss = ( 1 - fc.sel(step=step)/ref.sel(step=step) )

                N, bins, patches = ax.hist(
                    ss,
                    density=False,
                    edgecolor='k',
                    linewidth=0.1,
                    alpha=0.75,
                    bins=[-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.4,0.5]
                )
                for i in range(0,5):
                    patches[i].set_facecolor('r')
                for i in range(5,6):
                    patches[i].set_facecolor('yellow')
                for i in range(6, 11):
                    patches[i].set_facecolor('blue')

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)

                if s==0:
                    ax.set_ylabel(mparser[str(month)])
                    ax.get_yaxis().set_ticks([0,50,100,150,200])
                else:
                    ax.get_yaxis().set_ticks([])
                if m==2:
                    ax.set_xlabel('DAYS: '+str(step.days-4)+'-'+str(step.days+3))
                else:
                    ax.get_xaxis().set_ticks([])

                ax.set_ylim([0,215])
                ax.set_xlim([-0.5,0.5])

        fig.suptitle('Hardanger og omegn, MAESS: EC mot '+lab)

        graphics.save_fig(fig,
                        'histogram/82020-82021_hardanger_mae_hist_NorKyst'+season+lab
                        )

    subplots = (3,5)
    ###########################
    #### Initialize figure ####
    ###########################
    latex.set_style(style='white')
    fig,axes = plt.subplots(subplots[0],subplots[1],
                    figsize=latex.set_size(width='thesis',
                        subplots=(subplots[0],subplots[1]))
                    )
    ###########################

    for m,month in enumerate(np.unique(crps_clim_mon.time.dt.month.values)):
        for s,step in enumerate(steps):

            fc  = crps_fc_mon.where(crps_fc_mon.time.dt.month==month)\
                                                    .mean('time',skipna=True)
            ref = crps_clim_mon.where(crps_clim_mon.time.dt.month==month)\
                                                    .mean('time',skipna=True)

            ax = axes[m,s]

            ss = ( 1 - fc.sel(step=step)/ref.sel(step=step) )

            N, bins, patches = ax.hist(
                ss.values,
                density=False,
                edgecolor='k',
                linewidth=0.1,
                # edgecolor='white',
                alpha=0.75,
                bins=[-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.4,0.5]
            )
            for i in range(0,5):
                patches[i].set_facecolor('r')
            for i in range(5,6):
                patches[i].set_facecolor('yellow')
            for i in range(6, 11):
                patches[i].set_facecolor('blue')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            if s==0:
                ax.set_ylabel(mparser[str(month)])
                ax.get_yaxis().set_ticks([0,50,100,150,200])
            else:
                ax.get_yaxis().set_ticks([])
            if m==2:
                ax.set_xlabel('DAYS: '+str(step.days-4)+'-'+str(step.days+3))
            else:
                ax.get_xaxis().set_ticks([])

            ax.set_ylim([0,215])
            ax.set_xlim([-0.5,0.5])

    fig.suptitle('Hardanger og omegn, CRPSS: EC mot CLIM')

    graphics.save_fig(fig,
            'histogram/82020-82021_hardanger_crps_hist_NorKyst'+season
            )
