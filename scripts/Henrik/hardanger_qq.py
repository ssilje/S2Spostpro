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

def run():

    nk = norkyst_restack.run()

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

    t_start  = (2020,8,1) #can start with 8
    t_end    = (2021,9,1)
    model    = ''
    extent   = [4.5,7.1,59.3,61]

    nk = nk.where(nk.lon <= extent[1], drop=True)
    nk = nk.where(nk.lon >= extent[0], drop=True)

    nk = nk.where(nk.lat <= extent[3], drop=True)
    nk = nk.where(nk.lat >= extent[2], drop=True)

    print(len(nk.location.values))

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

    del nk

    hindcast = Grid2Point(observations,hindcast)\
                            .correlation(step_dependent=True)

    pers    = models.persistence(
                    init_value   = observations.init_a,
                    observations = observations.data_a
                    )


    for season in ['DJF','MAM','JJA','SON']:

        season = str(season)

        fc_a = hindcast.data_a.where(hindcast.data_a.time.dt.season==season,drop=True).mean('member')
        o_a  = observations.data_a.where(observations.data_a.time.dt.season==season,drop=True)
        p_a  = pers.where(pers.time.dt.season==season,drop=True)


        # fc_a,o_a,p_a = xr.align(fc_a,o_a,p_a,join='outer')
        # fc_a = fc_a.broadcast_like(o_a)
        # p_a  = p_a.broadcast_like(o_a)
        # o_a  = o_a.broadcast_like(fc_a)

        # print(fc_a)
        # print(p_a)
        # print(o_a)

        for ref_fc,lab in zip([fc_a,p_a],['EC','PERS']):
        # for ref_fc,lab in zip([fc_a],['EC']):

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

                    ax = axes[m,s]

                    o = o_a.where(o_a.time.dt.month==month).sel(step=step)
                    f = ref_fc.where(ref_fc.time.dt.month==month).sel(step=step)

                    f,o = xr.align(f,o)

                    ax.scatter(
                            o.isel(location=0).values.flatten(),
                            f.isel(location=0).values.flatten(),
                            label=lab,
                            s=0.5,
                            color={'EC':'red','PERS':'blue'}[lab],
                            alpha=0.4
                            )

                    ax.plot([-5,5],[-5,5],'--',color='k',linewidth=0.5)

                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)

                    if s==0:
                        ax.set_ylabel('Month')
                    else:
                        ax.get_yaxis().set_ticks([])
                    if m==2:
                        ax.set_xlabel('LT')
                    else:
                        ax.get_xaxis().set_ticks([])

                    ax.set_xlim([-5,5])
                    ax.set_ylim([-5,5])

                    if s==0:
                        ax.set_ylabel(mparser[str(month)])
                    else:
                        ax.get_yaxis().set_ticks([])
                    if m==2:
                        ax.set_xlabel('DAYS: '+str(step.days-4)+'-'+str(step.days+3))
                    else:
                        ax.get_xaxis().set_ticks([])

        fig.suptitle('Hardanger og omegn, QQ-plot')

        graphics.save_fig(fig,
                        'qq/82020-82021_hardanger_qq_NorKyst'+season+lab
                        )
