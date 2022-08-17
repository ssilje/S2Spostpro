import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from S2S.data_handler import ERA5, BarentsWatch
from S2S.process import Hindcast, Observations, Grid2Point

from S2S.graphics import mae,crps,graphics,latex
from S2S import models, location_cluster, wilks

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from S2S.local_configuration import config

import xskillscore as xs
import S2S.scoring as sc

from matplotlib.colors import BoundaryNorm, ListedColormap
import seaborn as sns
from S2S.xarray_helpers import o_climatology
import cmocean

tmp_path = '/projects/NS9853K/DATA/norkyst800/processed/'

def calc():

    path = '/projects/NS9853K/DATA/norkyst800/processed/hardanger/'

    mparser = {
                '1':'JAN','2':'FEB','3':'MAR',
                '4':'APR','5':'MAY','6':'JUN',
                '7':'JUL','8':'AUG','9':'SEP',
                '10':'OCT','11':'NOV','12':'DEC'
            }

    winter_months  = ['10','11','12','01','02','03']
    summer_months  = ['04','05','06','07','08','09']

    bounds = (4.25-0.5,6.75+0.5,59.3-0.5,61.+0.5)
    var      = 'sst'

    t_start  = (2020,7,1)
    t_end    = (2021,7,3)

    clim_t_start  = (2000,6,15)
    clim_t_end    = (2020,8,25)

    high_res = True
    steps    = pd.to_timedelta([9,16,23,30,37],'D')

    print('Get hindcast')
    hindcast = Hindcast(
        var,
        t_start,
        t_end,
        bounds,
        high_res=high_res,
        steps=steps,
        process=False,
        download=False,
        split_work=True,
        cross_val=True,
    )

    print('Get era')
    era = ERA5(high_res=True).load(
        var         = var,
        start_time  = clim_t_start,
        end_time    = clim_t_end,
        bounds      = bounds
    )

    print('ERA: resample and roll')
    era = era.resample(time='D').mean()
    era = era.sst.rolling(time=7,center=True).mean()

    print('ERA: process')
    observations = Observations(
        name='ERA_hardanger_ext',
        observations=era,
        forecast=hindcast,
        process=False
    )

    #del hindcast
    del era

    #hindcast = xr.open_dataset(path+'hindcast_hardanger')

    print(hindcast.data)

    if False:

        print('Fit combo model')
        # combo = models.combo(
        #                        init_value      = observations.init_a,
        #                        model           = hindcast.data_a,
        #                        observations    = observations.data_a
        #                    )
        # combo.to_netcdf(tmp_path+'tmp_combo_fig7.nc')

        combo_a = models.combo(
                                init_value      = observations.init_a,
                                model           = hindcast.data_a,
                                observations    = observations.data_a,
                                adj_amplitude   = True
                            )

        combo_a.to_netcdf(tmp_path+'tmp_comboa_fig7.nc')

        # combo = hindcast.data_a - hindcast.data_a.mean('member') + combo

        # adjust spread
        # combo = models.bias_adjustment_torralba(
        #                             forecast        = combo,
        #                             observations    = observations.data_a,
        #                             spread_only     = True
        #                             )

        print('Calculate CRPS')
        #crps_co  = sc.crps_ensemble(observations.data_a,combo,fair=True).rename('crps')
        #crps_fc  = sc.crps_ensemble(observations.data_a,hindcast.data_a,fair=True).rename('crps')
        crps_ref = xs.crps_gaussian(observations.data_a,mu=0,sig=1,dim=[]).rename('crps')

        #crps_co.to_netcdf(tmp_path+'tmp_crps_co_fig7.nc')
        #crps_fc.to_netcdf(tmp_path+'tmp_crps_fc_fig7.nc')
        crps_ref.to_netcdf(tmp_path+'tmp_crps_clim_fig7.nc')

        pers = models.persistence(observations.init_a,observations.data_a,window=30)
        #pers = pers.expand_dims('member')
        #mae_pe  = sc.crps_ensemble(observations.data_a,pers,fair=True).rename('crps')
        pers.to_netcdf(tmp_path+'tmp_pers_fig7.nc')

    ec = hindcast.data_a - hindcast.data_a.mean('member')
    ec = ec /hindcast.data_a.std('member')
    ec = ec + hindcast.data_a.mean('member')

    combo   = xr.open_dataarray(tmp_path+'tmp_comboa_fig7.nc')
    combo_u = hindcast.data_a - hindcast.data_a.mean('member')
    combo_u = combo_u / combo_u.std('member')
    combo = combo_u + combo

    pers      = xr.open_dataarray(tmp_path+'tmp_pers_fig7.nc')
    mse_pers  = xs.mae(observations.data_a,pers,dim=[]).rename('mse').mean('time')

    return sc.crps_ensemble(observations.data_a,combo,fair=True).rename('crps'),sc.crps_ensemble(observations.data_a,ec,fair=True).rename('crps'),mse_pers

def main():

    steps    = pd.to_timedelta([9,16,23,30,37],'D')

    path = '/projects/NS9853K/DATA/norkyst800/processed/hardanger/'

    crps_co,crps_fc,mse_pers  = calc()
    crps_co = crps_co.mean('time')
    crps_fc = crps_fc.mean('time')
    crps_ref = xr.open_dataset(tmp_path+'tmp_crps_clim_fig7.nc').crps.mean('time')

    crps_co,crps_fc,crps_ref = xr.align(crps_co,crps_fc,crps_ref)

    resol  = '10m'
    extent = [4.25,6.75,59.3,61.]

    latex.set_style(style='white')
    fig,axes = plt.subplots(5,5,\
        figsize=latex.set_size(width=345,subplots=(4,2),fraction=0.95),\
        subplot_kw=dict(projection=ccrs.NorthPolarStereo())
    )

    cmap   = cmocean.cm.speed_r
    levels = np.arange(0.0,1.1,0.1)
    norm   = BoundaryNorm(levels,cmap.N,extend='both')

    coast = cfeature.NaturalEarthFeature(category='physical', scale='10m',
                                facecolor='none', name='coastline')

    for _ax_,crps,label,c in zip(axes,[crps_ref,crps_fc,mse_pers,crps_co],['CLIM','EC','PERS','COMBO'],['darkblue','darkgreen','darkred','orange']):

        _ax_[0].text(-0.07, 0.55, label,
            color=c,
            size = 'xx-large',
            va='bottom',
            ha='center',
            rotation='vertical',
            rotation_mode='anchor',
            transform=_ax_[0].transAxes
        )

        for ax,step,col in zip(_ax_,steps,range(len(steps))):

            _ss_ = crps.sel(step=step).transpose('lat','lon')


            cs = ax.pcolor(
                _ss_.lon.values,
                _ss_.lat.values,
                _ss_.values,
                #s=2.6,
                cmap=cmap,
                norm=norm,
                #alpha=0.95,
                transform=ccrs.PlateCarree(),
                zorder=20,
                #linewidth=0.3,
                #edgecolors='k'
            )

            ax.set_extent(extent, crs=ccrs.PlateCarree())
            # land = cfeature.NaturalEarthFeature('physical', 'land', \
            #     scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
            # ax.add_feature(land, zorder=1, facecolor='beige', linewidth=0.2)
            feature = ax.add_feature(coast, edgecolor='k', linewidth=0.5, zorder=40)
    cbar=fig.colorbar(
        cs,
        ax=axes[:-1].ravel().tolist(),
        fraction=0.046,
        pad=0.06
    )

    data = xr.concat(
        [crps_ref,crps_fc,mse_pers,crps_co],
        dim=pd.Index([1,3,5,7],name='model')
    )
    data = data.idxmin(dim='model')

    axes[-1][0].text(-0.07, 0.55, 'Best Model',
        color='k',
        size = 'xx-large',
        va='bottom',
        ha='center',
        rotation='vertical',
        rotation_mode='anchor',
        transform=axes[-1][0].transAxes
    )

    cmap   = ListedColormap(['darkblue','darkgreen','darkred','orange'])
    levels = [0,2,4,6,8]
    norm   = BoundaryNorm(levels,cmap.N)


    for ax,step,col in zip(axes[-1],steps,range(len(steps))):

        _ss_ = data.sel(step=step).transpose('lat','lon')

        cs = ax.pcolor(
            _ss_.lon.values,
            _ss_.lat.values,
            _ss_.values,
            #s=2.6,
            #alpha=0.60,
            transform=ccrs.PlateCarree(),
            zorder=20,
            cmap = cmap,
            norm = norm
            #linewidth=0.3,
            #edgecolors='k'
        )

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        # land = cfeature.NaturalEarthFeature('physical', 'land', \
        #     scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
        # ax.add_feature(land, zorder=1, facecolor='beige', linewidth=0.2)
        feature = ax.add_feature(coast, edgecolor='k', linewidth=0.5, zorder=40)
        ax.text(0.5, -0.2, str(step.days),
            size = 'xx-large',
            va='bottom',
            ha='center',
            rotation='horizontal',
            rotation_mode='anchor',
            transform=ax.transAxes
        )

    fig.suptitle('')
    latex.save_figure(fig,'/nird/home/heau/figures_paper/Figure7')
