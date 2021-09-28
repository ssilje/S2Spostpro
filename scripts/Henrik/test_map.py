import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from S2S.data_handler import ERA5, BarentsWatch
from S2S.process import Hindcast, Observations, Grid2Point

from S2S.graphics import mae,crps,graphics,latex
from S2S import models, location_cluster

import cartopy.crs as ccrs
import cartopy.feature as cfeature


def loc(name):
    return str(location_cluster.loc_from_name(name))

def go():

    mparser = {
                '1':'JAN','2':'FEB','3':'MAR',
                '4':'APR','5':'MAY','6':'JUN',
                '7':'JUL','8':'AUG','9':'SEP',
                '10':'OCT','11':'NOV','12':'DEC'
            }
    winter_months  = ['10','11','12','01','02','03']
    summer_months  = ['04','05','06','07','08','09']

    bounds = (0,28,55,75)
    var      = 'sst'

    t_start  = (2020,7,2)
    t_end    = (2021,7,2)

    high_res = True
    steps    = pd.to_timedelta([9,16,23,30,37],'D')

    clim_t_start  = (2000,6,15)
    clim_t_end    = (2020,8,25)

    era = ERA5(high_res=True).load(
        var         = var,
        start_time  = clim_t_start,
        end_time    = clim_t_end,
        bounds      = bounds
    )

    era = era.resample(time='D').mean()
    era = era.sst.rolling(time=7,center=True).mean()

    resol  = '10m'
    extent = [0,28,55,75]

    for mlabel,months in zip(['winter','summer'],[winter_months,summer_months]):

        latex.set_style(style='white')
        fig,axes = plt.subplots(6,5,\
            figsize=latex.set_size(width=345,subplots=(6,2),fraction=0.95),\
            subplot_kw=dict(projection=ccrs.NorthPolarStereo())
        )

        for ax_month,month,row in zip(axes,months,range(len(months))):
            for ax,step,col in zip(ax_month,steps,range(len(steps))):


                data = era.where(era.time.dt.month==int(month))\
                    .var('time',skipna=True).transpose('lat','lon')

                ax.contourf(
                    data.lon,
                    data.lat,
                    data,
                    transform=ccrs.PlateCarree(),
                    zorder=30
                )

                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.coastlines(resolution=resol,linewidth=0.1)
                # land = cfeature.NaturalEarthFeature('physical', 'land', \
                #     scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land']
                # )
                # ax.add_feature(land, zorder=1, facecolor='beige', linewidth=0.2)

                if row==len(months)-1:
                    ax.text(0.5, -0.2, str(step.days),
                        size = 'xx-large',
                        va='bottom',
                        ha='center',
                        rotation='horizontal',
                        rotation_mode='anchor',
                        transform=ax.transAxes
                    )
                    print(str(step.days))
                if col==0:
                    ax.text(-0.07, 0.55, mparser[str(int(month))],
                        size = 'xx-large',
                        va='bottom',
                        ha='center',
                        rotation='vertical',
                        rotation_mode='anchor',
                        transform=ax.transAxes
                    )
                    print(mparser[str(int(month))])


        graphics.save_fig(fig,'paper/test_map_'+mlabel)

        # ax.set_extent(extent, crs=ccrs.PlateCarree())
        #
        # resol = '10m'  # use data at this scale
        # land = cfeature.NaturalEarthFeature('physical', 'land', \
        #     scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
        # # ocean = cfeature.NaturalEarthFeature('physical', 'ocean', \
        # #     scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
        # ax.add_feature(land, zorder=1, facecolor='beige', linewidth=0.2)
        # # ax.add_feature(ocean, linewidth=0.2, zorder=0 )
        #
        # # ax.coastlines(resolution='10m', color='grey',\
        # #                         linewidth=0.2)
        # ax.set_title(mparser[month] + ' MAEss EC vs. '+lab+', NorKyst, lt: '\
        #                             +str(step.days-4)+'-'+str(step.days+3))
        # cbar=fig.colorbar(cs,ax=ax)
        # cbar.set_ticks(levels)
        # cbar.set_ticklabels(levels)
        # graphics.save_fig(fig,
        #                 year+'hardanger_mae_skill_map_NorKyst_month'+month+lab+str(step.days)
        #                 )
