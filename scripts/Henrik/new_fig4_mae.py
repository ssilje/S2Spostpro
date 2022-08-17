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
import cmocean
from scripts.Henrik.distance_from_coast import get_shore

from sklearn.neighbors import BallTree

tmp_path = '/projects/NS9853K/DATA/norkyst800/processed/'

class HC:
    def __init__(self):

        path = '/projects/NS9853K/DATA/norkyst800/processed/hardanger/'

        with xr.open_dataset(path+'norkyst800_sst_daily_mean_hardanger_atBW.nc') as d:
            norkyst = d

        with xr.open_dataarray('/projects/NS9853K/DATA/norkyst800/processed/hardanger/hc_coast_nk_time.nc') as data:

            hc = data
            hc = hc.stack(point=['lat','lon'])
            hc = hc.where(np.isfinite(hc),drop=True)

            hc_loc = np.deg2rad(np.stack([hc.lat,hc.lon],axis=-1))
            nk_loc = np.deg2rad(np.stack([norkyst.lat,norkyst.lon],axis=-1))

            tree = BallTree(hc_loc)
            ind = tree.query(nk_loc,return_distance=False)

            hc  = hc.isel( point=ind.flatten() )
            lat = norkyst.lat.values
            lon = norkyst.lon.values
            hc  = hc.assign_coords(point=norkyst.location.values).rename(point='location')
            hc  = hc.assign_coords(lon=('location',lon))
            hc  = hc.assign_coords(lat=('location',lat))



            self.data   = hc
            self.data_a = hc

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

    hindcast = HC()

    print('Norkyst: process')
    observations = Observations(
        name='norkyst_in_hardangerfjorden_all',
        observations=norkyst,
        forecast=hindcast,
        process=False
    )


    if False:

        print('Fit combo model')
        combo = models.combo(
                                init_value      = observations.init_a,
                                model           = hindcast.data_a,
                                observations    = observations.data_a
                            )
        combo.to_netcdf(tmp_path+'tmp_combo_fig4.nc')

    if False:
        combo_a = models.combo(
                                init_value      = observations.init_a,
                                model           = hindcast.data_a,
                                observations    = observations.data_a,
                                adj_amplitude   = True
                            )

        combo_a.to_netcdf(tmp_path+'tmp_comboa_fig4.nc')

    if False:
        # adjust spread
        # combo = models.bias_adjustment_torralba(
        #                             forecast        = combo,
        #                             observations    = observations.data_a,
        #                             spread_only     = True
        #                             )

        print('Calculate CRPS')
        # crps_co  = sc.crps_ensemble(observations.data_a,combo,fair=True).rename('crps')
        # crps_fc  = sc.crps_ensemble(observations.data_a,hindcast.data_a,fair=True).rename('crps')


        # crps_co.to_netcdf(tmp_path+'tmp_crps_co_fig4.nc')
        # crps_fc.to_netcdf(tmp_path+'tmp_crps_fc_fig4.nc')


        pers = models.persistence(observations.init_a,observations.data_a,window=30)
        #pers = pers.expand_dims('member')
        #mae_pe  = sc.crps_ensemble(observations.data_a,pers,fair=True).rename('crps')
        pers.to_netcdf(tmp_path+'tmp_pers_fig4.nc')

    crps_ref = xs.mae(observations.data_a,xr.full_like(observations.data_a,0),dim=[]).rename('crps')
    crps_ref.to_netcdf(tmp_path+'tmp_mae_clim_fig4.nc')

    ec = hindcast.data_a.mean('member')

    combo   = xr.open_dataarray(tmp_path+'tmp_combo_fig4.nc')

    pers      = xr.open_dataarray(tmp_path+'tmp_pers_fig4.nc')
    mse_pers  = xs.mae(observations.data_a,pers,dim=[]).rename('mse').mean('time')

    return xs.mae(observations.data_a,combo,dim=[]).rename('crps'),xs.mae(observations.data_a,ec,dim=[]).rename('crps'),mse_pers

def main():

    steps    = pd.to_timedelta([9,16,23,30,37],'D')

    path = '/projects/NS9853K/DATA/norkyst800/processed/hardanger/'

    crps_co,crps_fc,mse_pers  = calc()
    crps_co = crps_co.mean('time')
    crps_fc = crps_fc.mean('time')
    crps_ref = xr.open_dataset(tmp_path+'tmp_mae_clim_fig4.nc').crps.mean('time')

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

        out = {'distance_from_coast_metric':[]}
        for ax,step,col in zip(_ax_,steps,range(len(steps))):

            _ss_ = crps.sel(step=step)
            print(_ss_)
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

            # MAKE TABLE, SET TO FALSE TO MAKE FIGURE

            out = {**out,label+'_'+str(step.days):[]}

            shore = get_shore()
            shore = np.floor( shore.shore * 10 ) / 10

            shore,_ss_ = xr.align(shore,_ss_)
            _ss_ = _ss_.assign_coords(shore=('location',shore.values))
            _ss_ = _ss_.groupby('shore').mean()

            for s in np.unique(shore.values):

                out['distance_from_coast_metric'] = np.unique(shore.values)
                out[label+'_'+str(step.days)].append(_ss_.sel(shore=s).values)

        pd.DataFrame.from_dict(out).to_csv('/nird/home/heau/figures_paper/crps_table_'+label+'.csv')

    cbar=fig.colorbar(
        cs,
        ax=axes[:-1].ravel().tolist(),
        fraction=0.046,
        pad=0.06
    )

    data = xr.concat(
        [crps_ref,crps_fc,mse_pers,crps_co],
        dim=pd.Index(['darkblue','darkgreen','darkred','orange'],name='model')
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
    latex.save_figure(fig,'/nird/home/heau/figures_paper/Figure5_mae_original_adjustment')
