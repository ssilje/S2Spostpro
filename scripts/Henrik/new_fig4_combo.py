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

from matplotlib.colors import BoundaryNorm
import seaborn as sns

from S2S.xarray_helpers import o_climatology

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

    bounds = (0,28,55,75)
    var      = 'sst'

    t_start  = (2020,7,1)
    t_end    = (2021,7,3)

    high_res = True
    steps    = pd.to_timedelta([9,16,23,30,37],'D')

    print('Get norkyst')
    norkyst = xr.open_dataset(
        path+'norkyst800_sst_daily_mean_hardanger_atBW.nc'
    )
    print('Norkyst: roll')
    norkyst = norkyst.sst.rolling(time=7,center=True).mean()

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
        period=[norkyst.time.values.min(),norkyst.time.values.max()]
    )

    print('Norkyst: process')
    observations = Observations(
        name='norkyst_in_hardangerfjorden_all',
        observations=norkyst,
        forecast=hindcast,
        process=False
    )

    del norkyst
    del hindcast

    hindcast = xr.open_dataset(path+'hindcast_hardanger')

    if False:

        print('Fit combo model')
        combo = models.combo(
                                init_value      = observations.init_a,
                                model           = hindcast.data_a,
                                observations    = observations.data_a
                            )

        combo_a = models.combo(
                                init_value      = observations.init_a,
                                model           = hindcast.data_a,
                                observations    = observations.data_a,
                                adj_amplitude   = True
                            )

        combo.to_netcdf(tmp_path+'tmp_combo_fig4.nc')
        combo_a.to_netcdf(tmp_path+'tmp_comboa_fig4.nc')

        combo = hindcast.data_a - hindcast.data_a.mean('member') + combo

        # adjust spread
        # combo = models.bias_adjustment_torralba(
        #                             forecast        = combo,
        #                             observations    = observations.data_a,
        #                             spread_only     = True
        #                             )

        print('Calculate CRPS')
        crps_co  = sc.crps_ensemble(observations.data_a,combo,fair=True).rename('crps')
        crps_fc  = sc.crps_ensemble(observations.data_a,hindcast.data_a,fair=True).rename('crps')
        crps_ref = xs.crps_gaussian(observations.data_a,mu=0,sig=1,dim=[]).rename('crps')

        crps_co.to_netcdf(tmp_path+'tmp_crps_co_fig4.nc')
        crps_fc.to_netcdf(tmp_path+'tmp_crps_fc_fig4.nc')
        crps_ref.to_netcdf(tmp_path+'tmp_crps_clim_fig4.nc')

        pers = models.persistence(observations.init_a,observations.data_a,window=30)
        #pers = pers.expand_dims('member')
        #mae_pe  = sc.crps_ensemble(observations.data_a,pers,fair=True).rename('crps')
        pers.to_netcdf(tmp_path+'tmp_pers_fig4.nc')

    combo   = xr.open_dataarray(tmp_path+'tmp_combo_fig4.nc')
    combo_u = hindcast.data_a - hindcast.data_a.mean('member')
    combo_u = combo_u / combo_u.std('member')
    combo_d = combo_u + combo

    combo   = xr.open_dataarray(tmp_path+'tmp_comboa_fig4.nc')
    combo_u = hindcast.data_a - hindcast.data_a.mean('member')
    combo_u = combo_u / combo_u.std('member')
    combo_u = combo_u + combo

    crps_d = sc.crps_ensemble(observations.data_a,combo_d,fair=True).rename('crps').mean('time')
    crps_u = sc.crps_ensemble(observations.data_a,combo_u,fair=True).rename('crps').mean('time')

    return crps_d,crps_u


def main():

    combo_d,combo_u = calc()

    steps    = pd.to_timedelta([9,16,23,30,37],'D')

    path = '/projects/NS9853K/DATA/norkyst800/processed/hardanger/'

    combo_d,combo_u = xr.align(combo_d,combo_u)

    resol  = '10m'
    extent = [4.25,6.75,59.3,61.]

    latex.set_style(style='white')
    fig,axes = plt.subplots(3,5,
        figsize=(4.535076795350768, 2.8028316011177257),
        subplot_kw=dict(projection=ccrs.NorthPolarStereo())
    )

    cmap   = plt.get_cmap('coolwarm')
    levels = np.arange(0.3,0.825,0.025)
    norm   = BoundaryNorm(levels,cmap.N,extend='both')

    for _ax_,crps,label,c in zip(axes,[combo_d,combo_u],['COMBO','COMBO ADJ'],['darkgreen','orange']):

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

            _ss_ = crps.sel(step=step)

            cs = ax.scatter(
                _ss_.lon.values,
                _ss_.lat.values,
                c=_ss_.values,
                s=2.6,
                cmap=cmap,
                norm=norm,
                alpha=0.95,
                transform=ccrs.PlateCarree(),
                zorder=20,
                linewidth=0.3,
                edgecolors='k'
            )

            ax.set_extent(extent, crs=ccrs.PlateCarree())
            land = cfeature.NaturalEarthFeature('physical', 'land', \
                scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
            ax.add_feature(land, zorder=1, facecolor='beige', linewidth=0.2)

    cbar=fig.colorbar(
        cs,
        ax=axes.ravel().tolist(),
        fraction=0.046,
        pad=0.06
    )

    data = xr.concat(
        [combo_d,combo_u],
        dim=pd.Index(['darkgreen','orange'],name='model')
    )
    data = data.idxmin(dim='model')
    for ax,step,col in zip(axes[-1],steps,range(len(steps))):

        _ss_ = data.sel(step=step)

        cs = ax.scatter(
            _ss_.lon.values,
            _ss_.lat.values,
            c=_ss_.values,
            s=2.6,
            alpha=0.95,
            transform=ccrs.PlateCarree(),
            zorder=20,
            linewidth=0.3,
            edgecolors='k'
        )

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        land = cfeature.NaturalEarthFeature('physical', 'land', \
            scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
        ax.add_feature(land, zorder=1, facecolor='beige', linewidth=0.2)
        ax.text(0.5, -0.2, str(step.days),
            size = 'xx-large',
            va='bottom',
            ha='center',
            rotation='horizontal',
            rotation_mode='anchor',
            transform=ax.transAxes
        )

    fig.suptitle('')
    latex.save_figure(fig,'/nird/home/heau/figures_paper/new062022_fig4_combo_spread_comparison')
