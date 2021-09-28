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
nk = lc.cluster(nk,'Hisdalen',5,5)
print(len(nk.location.values))
t_start  = (2020,8,1) #can start with 8
t_end    = (2021,9,1)
model    = ''
extent   = [4.5,7.1,59.3,61]

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
                            name='Hardanger_NorKyst-800',
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

for month in np.arange(1,13,1):

    month = str(month)

    mae_fc_mon = mae_fc.where(mae_fc.time.dt.month==int(month),drop=True)\
                                    .mean('time',skipna=True)

    crps_fc_mon = crps_fc.where(crps_fc.time.dt.month==int(month),drop=True)\
                                    .mean('time',skipna=True)

    crps_clim_mon = crps_clim.where(crps_clim.time.dt.month==int(month),drop=True)\
                                    .mean('time',skipna=True)

    mae_pers_mon = mae_pers.where(mae_pers.time.dt.month==int(month),drop=True)\
                                    .mean('time',skipna=True)

    mae_clim_mon = mae_clim.where(mae_clim.time.dt.month==int(month),drop=True)\
                                    .mean('time',skipna=True)
    print('FC',np.isfinite(mae_fc_mon.values).sum())
    print('Clim',np.isfinite(mae_clim_mon.values).sum())
    for step in steps:

        for ref_fc,lab in zip([mae_clim_mon,mae_pers_mon],['CLIM','PERS']):

            latex.set_style(style='white')
            fig,ax = plt.subplots(1,1,\
                figsize=latex.set_size(width=345,subplots=(1,1),fraction=0.95),\
                subplot_kw=dict(projection=ccrs.NorthPolarStereo()))

            ss = ( 1 - mae_fc_mon/ref_fc ).sel(step=step)

            cmap   = latex.skill_cmap().reversed()
            levels = levels = [-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.4,0.5] #np.arange(-0.5,0.6,0.1)
            norm   = BoundaryNorm(levels,cmap.N)

            cs = ax.scatter(
                        ss.lon.values,
                        ss.lat.values,
                        c=ss.values,
                        s=3.1,
                        cmap=cmap,
                        norm=norm,
                        alpha=0.95,
                        transform=ccrs.PlateCarree(),
                        zorder=30,
                        edgecolors='k',
                        linewidth=0.2
                    )
            ax.set_extent(extent, crs=ccrs.PlateCarree())

            resol = '10m'  # use data at this scale
            land = cfeature.NaturalEarthFeature('physical', 'land', \
                scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
            river = cfeature.RIVERS(scale=resol)
            # ocean = cfeature.NaturalEarthFeature('physical', 'ocean', \
            #     scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
            ax.add_feature(land, zorder=1, facecolor='beige', linewidth=0.2)
            ax.add_feature(river, zorder=2)
            # ax.add_feature(ocean, linewidth=0.2, zorder=0 )

            # ax.coastlines(resolution='10m', color='grey',\
            #                         linewidth=0.2)
            ax.set_title(mparser[month] + ' MAESS, DAYS: '\
                                        +str(step.days-4)+'-'+str(step.days+3))
            cbar=fig.colorbar(cs,ax=ax)
            cbar.set_ticks(levels)
            cbar.set_ticklabels(levels)
            cbar.ax.set_title('EC')
            cbar.ax.set_xlabel(lab)
            graphics.save_fig(fig,
                            '82020-82021_hardanger_mae_skill_map_NorKyst_month'+month+lab+str(step.days)
                            )

        latex.set_style(style='white')
        fig,ax = plt.subplots(1,1,\
            figsize=latex.set_size(width=345,subplots=(1,1),fraction=0.95),\
            subplot_kw=dict(projection=ccrs.NorthPolarStereo()))

        ss = ( 1 - crps_fc_mon/crps_clim_mon ).sel(step=step)

        cmap   = latex.skill_cmap().reversed()
        levels = [-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.4,0.5] #np.arange(-0.5,0.6,0.1)
        norm   = BoundaryNorm(levels,cmap.N)

        cs = ax.scatter(
                    ss.lon.values,
                    ss.lat.values,
                    c=ss.values,
                    s=3.1,
                    cmap=cmap,
                    norm=norm,
                    alpha=0.95,
                    transform=ccrs.PlateCarree(),
                    zorder=30,
                    edgecolors='k',
                    linewidth=0.2
                )
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        resol = '10m'  # use data at this scale
        land = cfeature.NaturalEarthFeature('physical', 'land', \
            scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
        # ocean = cfeature.NaturalEarthFeature('physical', 'ocean', \
            # scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
        ax.add_feature(land, zorder=1, facecolor='beige', linewidth=0.2)
        # ax.add_feature(ocean, linewidth=0.2, zorder=0 )

        # ax.coastlines(resolution='10m', color='grey',\
        #                         linewidth=0.2)

        ax.set_title(mparser[month] + ' CRPSS, DAYS: '\
                                        +str(step.days-4)+'-'+str(step.days+3))
        cbar = fig.colorbar(cs,ax=ax)
        cbar.set_ticks(levels)
        cbar.set_ticklabels(levels)
        cbar.ax.set_title('EC')
        cbar.ax.set_xlabel('CLIM')
        graphics.save_fig(fig,
                        '82020-82021_hardanger_crps_skill_map_NorKyst_month'+month+'CLIM'+str(step.days)
                        )
