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

import cmocean

def lin_weight(data,dist,r):
    """
    Linearly weighted mean where the closest points are given more weight.
    """
    weights = 1 - dist/r
    return ( data * weights ).sum() / weights.sum()

def distance(lat1, lon1, lat2, lon2):
    """
    Returns the great circle distance between lat1,lon1 and lat2,lon2.
    Function is positive definite.
    """
    p = np.pi/180
    a = 0.5 - np.cos((lat2-lat1)*p)/2 +\
        np.cos(lat1*p) * np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p))/2
    return 12742 * np.arcsin(np.sqrt(a))

def interpolate_hincast_to_observations():

    path = '/projects/NS9853K/DATA/norkyst800/processed/hardanger/'
    filename = path + 'norkyst800_sst_daily_mean_hardanger_atBW.nc'

    locations = xr.open_dataset(filename).location#.sel(location=['12022','15196'])
    time = xr.open_dataset(filename).time

    # get hindcast data
    bounds = (0,28,55,75)
    bounds2 = (4.25,6.75,59.3,61.)
    var      = 'sst'

    t_start  = (2020,7,1)
    t_end    = (2021,7,3)

    high_res = True
    steps    = pd.to_timedelta([9,16,23,30,37],'D')

    print('Get hindcast')
    hindcast = Hindcast(
        var,
        t_start,
        t_end,
        bounds,
        bounds2 = bounds2,
        high_res=high_res,
        steps=steps,
        process=False,
        download=False,
        split_work=True,
        cross_val=True,
        #period=[time.values.min(),time.values.max()]
    )

    hc = hindcast.data_a

    extent = [4.,7.1,58.,62.]

    hc = hc.where(extent[0] < hc.lon, drop=True)
    hc = hc.where(extent[2] < hc.lat, drop=True)

    hc = hc.where(extent[1] > hc.lon, drop=True)
    hc = hc.where(extent[3] > hc.lat, drop=True)

    hc = hc.stack(points=['lon','lat'])

    # points = tuple(map(np.array,zip(*hc.points.values)))
    out = []

    for loc in locations:

        print(loc.values.item())

        nlon = loc.lon.values.item()
        nlat = loc.lat.values.item()

        t_hc = hc.assign_coords(d=distance(nlat, nlon, hc.lat, hc.lon))

        t_hc = t_hc.sortby('d')

        for p in t_hc.points:
            if 0 < np.isfinite(
                xr.ufuncs.isfinite(
                    t_hc.sel(points=p)
                ).sum(['member','step','time'],skipna=False).values
            ).sum():
                out.append(t_hc.sel(points=p).drop('points').drop('d'))
                break

    xr.concat(out,locations).to_netcdf(path+'hindcast_hardanger.nc')

def calc_crps():

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

    print('Fit combo model')
    # combo = models.combo(
    #                         init_value      = observations.init_a,
    #                         model           = hindcast.data_a,
    #                         observations    = observations.data_a
    #                     )
    tmp_path = '/projects/NS9853K/DATA/norkyst800/processed/'
    combo  = xr.open_dataarray(tmp_path+'tmp_comboa_fig4.nc')
    combo_ = hindcast.data_a - hindcast.data_a.mean('member')
    combo_ = combo_/combo_.std('member')
    combo  = combo_ + combo

    ec = hindcast.data_a - hindcast.data_a.mean('member')
    ec = ec /hindcast.data_a.std('member')
    ec = ec + hindcast.data_a.mean('member')

    # adjust spread
    # combo = models.bias_adjustment_torralba(
    #                             forecast        = combo,
    #                             observations    = observations.data_a,
    #                             spread_only     = True
    #                             )

    print('Calculate CRPS')
    crps_co  = sc.crps_ensemble(observations.data_a,combo,fair=True).rename('crps')
    crps_fc  = sc.crps_ensemble(observations.data_a,ec,fair=True).rename('crps')
    crps_ref = xs.crps_gaussian(observations.data_a,mu=0,sig=1,dim=[]).rename('crps')

    crps_co.to_netcdf(path+'crps_co_unit_var.nc')
    crps_fc.to_netcdf(path+'crps_fc_unit_var.nc')
    crps_ref.to_netcdf(path+'crps_ref_unit_var.nc')

    resol  = '10m'
    t_alt  = 'two-sided'
    extent = [0,28,55,75]

def main():

    steps    = pd.to_timedelta([9,16,23,30,37],'D')

    path = '/projects/NS9853K/DATA/norkyst800/processed/hardanger/'

    mparser = {
                '1':'JAN','2':'FEB','3':'MAR',
                '4':'APR','5':'MAY','6':'JUN',
                '7':'JUL','8':'AUG','9':'SEP',
                '10':'OCT','11':'NOV','12':'DEC'
            }

    winter_months  = ['10','11','12','01','02','03']
    summer_months  = ['04','05','06','07','08','09']
    all_months = ['01','02','03','04','05','06','07','08','09','10','11','12']

    crps_co = xr.open_dataset(path+'crps_co_unit_var.nc').crps
    crps_fc = xr.open_dataset(path+'crps_fc_unit_var.nc').crps
    crps_ref = xr.open_dataset(path+'crps_ref_unit_var.nc').crps

    crps_co,crps_fc,crps_ref = xr.align(crps_co,crps_fc,crps_ref)

    resol  = '10m'
    t_alt  = 'two-sided'
    extent = [4.25,6.75,59.3,61.]

    figsize = latex.set_size(width=345,subplots=(1,1),fraction=0.95)
    figsize = (figsize[0]*1.61,figsize[0])

    latex.set_style(style='white')
    fig,axes = plt.subplots(6,10,\
        figsize=figsize,\
        subplot_kw=dict(projection=ccrs.NorthPolarStereo())
    )

    # cmap   = sns.color_palette("Spectral", as_cmap=True)
    cmap   = cmocean.cm.curl_r
    #levels = [-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.4,0.5]
    levels = np.sort(np.hstack([-np.arange(0.2,1.2,0.2),np.arange(0.2,1.2,0.2),np.array([-0.05,0.05])]))
    levels = [round(lvl,2) for lvl in levels]
    norm   = BoundaryNorm(levels,cmap.N)

    print('Combo against Norkyst')
    for _axes_,months in zip([axes[:,:5],axes[:,5:]],[all_months[:6],all_months[6:]]):

        for ax_month,month,row in zip(_axes_,months,range(len(months))):

            score_fc  = crps_co.where(crps_co.time.dt.month==int(month),drop=True)
            score_ref = crps_ref.where(crps_ref.time.dt.month==int(month),drop=True)

            # Two sided t-test for annual means of crps_fc and crps_ref
            if False:
                dist,pval = sc.ttest_upaired(
                    score_fc.groupby('time.year').mean(skipna=True),
                    score_ref.groupby('time.year').mean(skipna=True),
                    dim=['year'],
                    alternative=t_alt,
                    welch=True
                )

                pmap = wilks.xsignificans_map(pval,alpha_FDR=0.2)

            ss = 1 - (score_fc.groupby('time.year').mean(skipna=True) /\
                score_ref.groupby('time.year').mean(skipna=True)
            ).mean('year')

            for ax,step,col in zip(ax_month,steps,range(len(steps))):

                _ss_ = ss.sel(step=step)

                #pcoords = pmap.sel(step=step)
                #pcoords = pcoords.where(xr.ufuncs.isfinite(pcoords),drop=True)

                #_ss_p = _ss_.where(_ss_.location.isin(pcoords.location),drop=True)


                # ---
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

                #ax.scatter(
                #            _ss_p.lon.values,
                #            _ss_p.lat.values,
                #            c=_ss_p.values,
                #            s=2.6,
                #            cmap=cmap,
                #            norm=norm,
                #            alpha=0.95,
                #            transform=ccrs.PlateCarree(),
                #            zorder=30,
                #            linewidth=0.3,
                #            edgecolors='k'
                #        )

                ax.set_extent(extent, crs=ccrs.PlateCarree())

                resol = '10m'  # use data at this scale
                land = cfeature.NaturalEarthFeature('physical', 'land', \
                    scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
                ax.add_feature(land, zorder=1, facecolor='beige', linewidth=0.2)

                if row==len(months)-1:
                    ax.text(0.5, -0.2, str(step.days),
                        size = 'xx-large',
                        va='bottom',
                        ha='center',
                        rotation='horizontal',
                        rotation_mode='anchor',
                        transform=ax.transAxes
                    )
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

    cbar=fig.colorbar(
        cs,
        ax=axes.ravel().tolist(),
        fraction=0.046,
        pad=0.06
    )
    cbar.set_ticks(levels)
    cbar.set_ticklabels(levels)
    cbar.ax.set_title('COMBO')
    cbar.ax.set_xlabel('CLIM')
    #fig.suptitle('CRPSS    SST: COMBO against Norkyst')
    latex.save_figure(fig,'/nird/home/heau/figures_paper/Figure6')


    figsize = latex.set_size(width=345,subplots=(1,1),fraction=0.95)
    figsize = (figsize[0]*1.61,figsize[0])

    latex.set_style(style='white')
    fig,axes = plt.subplots(6,10,\
        figsize=figsize,\
        subplot_kw=dict(projection=ccrs.NorthPolarStereo())
    )

    print('Combo against EC')
    for _axes_,months in zip([axes[:,:5],axes[:,5:]],[all_months[:6],all_months[6:]]):

        for ax_month,month,row in zip(_axes_,months,range(len(months))):

            score_fc  = crps_co.where(crps_co.time.dt.month==int(month),drop=True)
            score_ref = crps_fc.where(crps_fc.time.dt.month==int(month),drop=True)

            if False:
                # Two sided t-test for annual means of crps_fc and crps_ref
                dist,pval = sc.ttest_upaired(
                    score_fc.groupby('time.year').mean(skipna=True),
                    score_ref.groupby('time.year').mean(skipna=True),
                    dim=['year'],
                    alternative=t_alt,
                    welch=True
                )

                pmap = wilks.xsignificans_map(pval,alpha_FDR=0.2)

            ss = 1 - (score_fc.groupby('time.year').mean(skipna=True) /\
                score_ref.groupby('time.year').mean(skipna=True)
            ).mean('year')

            for ax,step,col in zip(ax_month,steps,range(len(steps))):

                _ss_ = ss.sel(step=step)

                #pcoords = pmap.sel(step=step)
                #pcoords = pcoords.where(xr.ufuncs.isfinite(pcoords),drop=True)

                #_ss_p = _ss_.where(_ss_.location.isin(pcoords.location),drop=True)


                # ---
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

                #ax.scatter(
                #            _ss_p.lon.values,
                #            _ss_p.lat.values,
                #            c=_ss_p.values,
                #            s=2.6,
                #            cmap=cmap,
                #            norm=norm,
                #            alpha=0.95,
                #            transform=ccrs.PlateCarree(),
                #            zorder=30,
                #            linewidth=0.3,
                #            edgecolors='k'
                #        )

                ax.set_extent(extent, crs=ccrs.PlateCarree())

                resol = '10m'  # use data at this scale
                land = cfeature.NaturalEarthFeature('physical', 'land', \
                    scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
                ax.add_feature(land, zorder=1, facecolor='beige', linewidth=0.2)

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

    cbar=fig.colorbar(
        cs,
        ax=axes.ravel().tolist(),
        fraction=0.046,
        pad=0.06
    )
    cbar.set_ticks(levels)
    cbar.set_ticklabels(levels)
    cbar.ax.set_title('COMBO')
    cbar.ax.set_xlabel('EC')
    #fig.suptitle('CRPSS    SST: COMBO against EC; Obs: Norkyst')
    latex.save_figure(fig,'/nird/home/heau/figures_paper/FigureS3')

    exit()
    print('EC against NK')
    for mlabel,months in zip(['winter','summer'],[winter_months,summer_months]):

        latex.set_style(style='white')
        fig,axes = plt.subplots(6,5,\
            figsize=latex.set_size(width=345,subplots=(6,2),fraction=0.95),\
            subplot_kw=dict(projection=ccrs.NorthPolarStereo())
        )

        # cmap   = sns.color_palette("Spectral", as_cmap=True)
        cmap   = latex.skill_cmap().reversed()
        levels = [-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.3,0.4,0.5]
        norm   = BoundaryNorm(levels,cmap.N)

        for ax_month,month,row in zip(axes,months,range(len(months))):

            score_fc  = crps_fc.where(crps_fc.time.dt.month==int(month),drop=True)
            score_ref = crps_ref.where(crps_ref.time.dt.month==int(month),drop=True)

            # Two sided t-test for annual means of crps_fc and crps_ref
            dist,pval = sc.ttest_upaired(
                score_fc.groupby('time.year').mean(skipna=True),
                score_ref.groupby('time.year').mean(skipna=True),
                dim=['year'],
                alternative=t_alt,
                welch=True
            )

            pmap = wilks.xsignificans_map(pval,alpha_FDR=0.2)

            ss = 1 - (score_fc.groupby('time.year').mean(skipna=True) /\
                score_ref.groupby('time.year').mean(skipna=True)
            ).mean('year')

            for ax,step,col in zip(ax_month,steps,range(len(steps))):

                _ss_ = ss.sel(step=step)

                pcoords = pmap.sel(step=step)
                pcoords = pcoords.where(xr.ufuncs.isfinite(pcoords),drop=True)

                _ss_p = _ss_.where(_ss_.location.isin(pcoords.location),drop=True)


                # ---
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

                ax.scatter(
                            _ss_p.lon.values,
                            _ss_p.lat.values,
                            c=_ss_p.values,
                            s=2.6,
                            cmap=cmap,
                            norm=norm,
                            alpha=0.95,
                            transform=ccrs.PlateCarree(),
                            zorder=30,
                            linewidth=0.3,
                            edgecolors='k'
                        )

                ax.set_extent(extent, crs=ccrs.PlateCarree())

                resol = '10m'  # use data at this scale
                land = cfeature.NaturalEarthFeature('physical', 'land', \
                    scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
                # ocean = cfeature.NaturalEarthFeature('physical', 'ocean', \
                #     scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
                ax.add_feature(land, zorder=1, facecolor='beige', linewidth=0.2)
                # ax.add_feature(ocean, linewidth=0.2, zorder=0 )

                # ax.coastlines(resolution='10m', color='grey',\
                #                         linewidth=0.2)
                ##########################

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

        cbar=fig.colorbar(
            cs,
            ax=axes.ravel().tolist(),
            fraction=0.046,
            pad=0.06
        )
        cbar.set_ticks(levels)
        cbar.set_ticklabels(levels)
        cbar.ax.set_title('EC')
        cbar.ax.set_xlabel('CLIM')
        fig.suptitle('CRPSS    SST: EC against Norkyst')
        latex.save_figure(fig,'/nird/home/heau/figures_paper/062022'+mlabel+t_alt+'_EC_NK')
