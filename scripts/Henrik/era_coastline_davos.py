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

import xskillscore as xs
import S2S.scoring as sc

from matplotlib.colors import BoundaryNorm
import seaborn as sns

tpath = '/projects/NS9853K/DATA/norkyst800/processed/'

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
        split_work=True
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
        name='ERA_the_whole_coast',
        observations=era,
        forecast=hindcast,
        process=False
    )

    del era

    print('Calculate CRPS')
    crps_fc  = sc.crps_ensemble(observations.data_a,hindcast.data_a,fair=True)
    crps_ref = xs.crps_gaussian(observations.data_a,mu=0,sig=1,dim=[])

    resol  = '10m'
    t_alt  = 'less'
    extent = [0,28,55,75]

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
            ).mean('year').transpose('lat','lon','step')

            for ax,step,col in zip(ax_month,steps,range(len(steps))):

                pcoords = pmap.sel(step=step).stack(point=('lon','lat'))
                pcoords = pcoords.where(xr.ufuncs.isfinite(pcoords),drop=True)

                cs = ax.contourf(
                    ss.lon,
                    ss.lat,
                    ss.sel(step=step),
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                    extend='both'
                )

                ax.scatter(
                    pcoords.lon,
                    pcoords.lat,
                    c='k',
                    marker='+',
                    s=0.1,
                    alpha=0.95,
                    transform=ccrs.PlateCarree(),
                    zorder=30,
                    linewidth=0.2
                )

                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.coastlines(resolution=resol,linewidth=0.1,zorder=40)
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

        cbar=fig.colorbar(
            cs,
            ax=axes.ravel().tolist(),
            fraction=0.046,
            pad=0.04
        )
        cbar.set_ticks(levels)
        cbar.set_ticklabels(levels)
        cbar.ax.set_title('EC')
        cbar.ax.set_xlabel('Clim')
        fig.suptitle('CRPSS    SST: EC against ERA5')
        graphics.save_fig(fig,'paper/map_'+mlabel+t_alt)

def go_combo_data():

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

    clim_t_start  = (2012,6,27)
    clim_t_end    = (2019,2,26)

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
        split_work=True
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
        name='ERA_the_whole_coast',
        observations=era,
        forecast=hindcast,
        process=True
    )

    del era

    crps_ref = xs.crps_gaussian(observations.data_a,mu=0,sig=1,dim=[])
    crps_ref.to_netcdf(tpath+'go_combo_crps_ref_unit_var.nc')
    del crps_ref

    print('Fit combo model')
    combo = models.combo(
                            init_value      = observations.init_a,
                            model           = hindcast.data_a,
                            observations    = observations.data_a
                        )

    combo = hindcast.data_a - hindcast.data_a.mean('member') + combo

    # adjust spread
    # combo = models.bias_adjustment_torralba(
    #                            forecast        = combo,
    #                             observations    = observations.data_a,
    #                             spread_only     = True
    #                             )

    print('Calculate CRPS')
    crps_co  = sc.crps_ensemble(observations.data_a,combo,fair=True)
    crps_fc  = sc.crps_ensemble(observations.data_a,hindcast.data_a,fair=True)

    crps_co.to_netcdf(tpath+'go_combo_crps_co_unit_var.nc')
    crps_fc.to_netcdf(tpath+'go_combo_crps_fc_unit_var.nc')


def go_combo():

    extent = [4.25,6.75,59.3,61.]

    steps    = pd.to_timedelta([9,16,23,30,37],'D')

    mparser = {
                '1':'JAN','2':'FEB','3':'MAR',
                '4':'APR','5':'MAY','6':'JUN',
                '7':'JUL','8':'AUG','9':'SEP',
                '10':'OCT','11':'NOV','12':'DEC'
            }

    winter_months  = ['10','11','12','01','02','03']
    summer_months  = ['04','05','06','07','08','09']

    crps_co = xr.open_dataset(tpath+'go_combo_crps_co_unit_var.nc').rename(dict(__xarray_dataarray_variable__='crps')).crps
    crps_fc = xr.open_dataset(tpath+'go_combo_crps_fc_unit_var.nc').sst.rename('crps')
    crps_ref = xr.open_dataset(tpath+'go_combo_crps_ref_unit_var.nc').rename(dict(__xarray_dataarray_variable__='crps')).crps

    crps_co,crps_fc,crps_ref = xr.align(crps_co,crps_fc,crps_ref)

    resol  = '10m'
    t_alt  = 'two-sided'
    extent = [4.25,6.75,59.3,61.]


    print('Combo against ERA')
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

            score_fc  = crps_co.where(crps_co.time.dt.month==int(month),drop=True)
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
            ).mean('year').transpose('lat','lon','step')

            for ax,step,col in zip(ax_month,steps,range(len(steps))):

                pcoords = pmap.sel(step=step).stack(point=('lon','lat'))
                pcoords = pcoords.where(xr.ufuncs.isfinite(pcoords),drop=True)
                print(month,step,np.isfinite(ss.sel(step=step)).values.sum() )
                cs = ax.contourf(
                    ss.lon,
                    ss.lat,
                    ss.sel(step=step),
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                    extend='both'
                )

                ax.scatter(
                    pcoords.lon,
                    pcoords.lat,
                    c='k',
                    marker='+',
                    s=0.1,
                    alpha=0.95,
                    transform=ccrs.PlateCarree(),
                    zorder=30,
                    linewidth=0.2
                )

                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.coastlines(resolution=resol,linewidth=0.1,zorder=40)
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

        cbar=fig.colorbar(
            cs,
            ax=axes.ravel().tolist(),
            fraction=0.046,
            pad=0.04
        )
        cbar.set_ticks(levels)
        cbar.set_ticklabels(levels)
        cbar.ax.set_title('COMBO')
        cbar.ax.set_xlabel('Clim')
        fig.suptitle('CRPSS    SST: COMBO against ERA5')
        latex.save_figure(fig,'/nird/home/heau/figures_paper/fig7_'+mlabel)

    print('EC against ERA')
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
            ).mean('year').transpose('lat','lon','step')

            for ax,step,col in zip(ax_month,steps,range(len(steps))):

                pcoords = pmap.sel(step=step).stack(point=('lon','lat'))
                pcoords = pcoords.where(xr.ufuncs.isfinite(pcoords),drop=True)

                cs = ax.contourf(
                    ss.lon,
                    ss.lat,
                    ss.sel(step=step),
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                    extend='both'
                )

                ax.scatter(
                    pcoords.lon,
                    pcoords.lat,
                    c='k',
                    marker='+',
                    s=0.1,
                    alpha=0.95,
                    transform=ccrs.PlateCarree(),
                    zorder=30,
                    linewidth=0.2
                )

                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.coastlines(resolution=resol,linewidth=0.1,zorder=40)
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

        cbar=fig.colorbar(
            cs,
            ax=axes.ravel().tolist(),
            fraction=0.046,
            pad=0.04
        )
        cbar.set_ticks(levels)
        cbar.set_ticklabels(levels)
        cbar.ax.set_title('EC')
        cbar.ax.set_xlabel('Clim')
        fig.suptitle('CRPSS    SST: EC against ERA5')
        latex.save_figure(fig,'/nird/home/heau/figures_paper/fig6_'+mlabel)


    print('Combo against EC')
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

            score_fc  = crps_co.where(crps_co.time.dt.month==int(month),drop=True)
            score_ref = crps_fc.where(crps_fc.time.dt.month==int(month),drop=True)

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
            ).mean('year').transpose('lat','lon','step')

            for ax,step,col in zip(ax_month,steps,range(len(steps))):

                pcoords = pmap.sel(step=step).stack(point=('lon','lat'))
                pcoords = pcoords.where(xr.ufuncs.isfinite(pcoords),drop=True)

                cs = ax.contourf(
                    ss.lon,
                    ss.lat,
                    ss.sel(step=step),
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                    extend='both'
                )

                ax.scatter(
                    pcoords.lon,
                    pcoords.lat,
                    c='k',
                    marker='+',
                    s=0.1,
                    alpha=0.95,
                    transform=ccrs.PlateCarree(),
                    zorder=30,
                    linewidth=0.2
                )

                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.coastlines(resolution=resol,linewidth=0.1,zorder=40)
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

        cbar=fig.colorbar(
            cs,
            ax=axes.ravel().tolist(),
            fraction=0.046,
            pad=0.04
        )
        cbar.set_ticks(levels)
        cbar.set_ticklabels(levels)
        cbar.ax.set_title('COMBO')
        cbar.ax.set_xlabel('EC')
        fig.suptitle('CRPSS    SST: COMBO against EC predicting ERA5')
        graphics.save_fig(fig,'u_var/map_'+mlabel+t_alt+'combo_ec_unit_var')



def combo():

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

    hindcast = Hindcast(
        var,
        t_start,
        t_end,
        bounds,
        high_res=high_res,
        steps=steps,
        process=False,
        download=False,
        split_work=True
    )

    print(
        hindcast.data.time.min(),
        hindcast.data.time.max()+hindcast.data.step.max()
        )

    era = ERA5(high_res=True).load(
        var         = var,
        start_time  = clim_t_start,
        end_time    = clim_t_end,
        bounds      = bounds
    )

    print(np.array_equal(hindcast.data.lat.values,era.lat.values))
    print(np.array_equal(hindcast.data.lon.values,era.lon.values))

    era = era.resample(time='D').mean()
    era = era.sst.rolling(time=7,center=True).mean()

    observations = Observations(
        name='ERA_the_whole_coast',
        observations=era,
        forecast=hindcast,
        process=False
    )

    del era

    print('Fit combo model')
    combo = models.combo(
                            init_value      = observations.init_a,
                            model           = hindcast.data_a,
                            observations    = observations.data_a
                        )

    combo = hindcast.data_a - hindcast.data_a.mean('member') + combo

    # adjust spread
    combo = models.bias_adjustment_torralba(
                                forecast        = combo,
                                observations    = observations.data_a,
                                spread_only     = True
                                )

    crps_co  = sc.crps_ensemble(observations.data_a,combo,fair=True)
    crps_fc  = sc.crps_ensemble(observations.data_a,hindcast.data_a,fair=True)
    crps_ref = xs.crps_gaussian(observations.data_a,mu=0,sig=1,dim=[])

    resol  = '10m'
    extent = [0,28,55,75]

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

            # not finished from here
            ss = 1 - ( crps_co.where(crps_co.time.dt.month==int(month))\
                .mean('time',skipna=True) /\
                    crps_ref.where(crps_ref.time.dt.month==int(month))\
                        .mean('time',skipna=True)
                    ).transpose('lat','lon','step')

            # SS = sc.SSCORE(
            #     observations=crps_ref.where(crps_ref.time.dt.month==int(month)),
            #     forecast    =crps_fc.where(crps_fc.time.dt.month==int(month))
            # ).bootstrap( N = 10000, min_period = 2 )
            #
            # ss   = SS.est.transpose('lat','lon','step')
            # ss_lq = SS.low_q.transpose('lat','lon','step')
            # ss_hq = SS.low_q.transpose('lat','lon','step')

            for ax,step,col in zip(ax_month,steps,range(len(steps))):

                cs = ax.contourf(
                    ss.lon,
                    ss.lat,
                    ss.sel(step=step),
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                    extend='both'
                )

                # p = ss.sel(step=step)/(ss_hq.sel(step=step)-ss_lq.sel(step=step))

                # print(p)
                # plon,plat = wilks.significans_map(
                #     p.lon.values,
                #     p.lat.values,
                #     p.values,
                #     0.05
                # )
                # print(plon,plat)

                # cs = ax.scatter(
                #     plon,
                #     plat,
                #     c='k',
                #     marker='+',
                #     alpha=0.95,
                #     transform=ccrs.PlateCarree(),
                #     zorder=30,
                #     edgecolors='k',
                #     linewidth=0.2
                # )

                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.coastlines(resolution=resol,linewidth=0.1,zorder=40)
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

        cbar=fig.colorbar(
            cs,
            ax=axes.ravel().tolist(),
            fraction=0.046,
            pad=0.04
        )
        cbar.set_ticks(levels)
        cbar.set_ticklabels(levels)
        cbar.ax.set_title('COMBO')
        cbar.ax.set_xlabel('Clim')
        fig.suptitle('CRPSS    var: SST obs: ERA5')
        graphics.save_fig(fig,'paper/era_combo_clim_'+mlabel)

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

            ss = 1 - ( crps_co.where(crps_fc.time.dt.month==int(month))\
                .mean('time',skipna=True) /\
                    crps_ref.where(crps_ref.time.dt.month==int(month))\
                        .mean('time',skipna=True)
                    ).transpose('lat','lon','step')

            # SS = sc.SSCORE(
            #     observations=crps_ref.where(crps_ref.time.dt.month==int(month)),
            #     forecast    =crps_fc.where(crps_fc.time.dt.month==int(month))
            # ).bootstrap( N = 10000, min_period = 2 )
            #
            # ss   = SS.est.transpose('lat','lon','step')
            # ss_lq = SS.low_q.transpose('lat','lon','step')
            # ss_hq = SS.low_q.transpose('lat','lon','step')

            for ax,step,col in zip(ax_month,steps,range(len(steps))):

                cs = ax.contourf(
                    ss.lon,
                    ss.lat,
                    ss.sel(step=step),
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                    extend='both'
                )

                # p = ss.sel(step=step)/(ss_hq.sel(step=step)-ss_lq.sel(step=step))

                # print(p)
                # plon,plat = wilks.significans_map(
                #     p.lon.values,
                #     p.lat.values,
                #     p.values,
                #     0.05
                # )
                # print(plon,plat)

                # cs = ax.scatter(
                #     plon,
                #     plat,
                #     c='k',
                #     marker='+',
                #     alpha=0.95,
                #     transform=ccrs.PlateCarree(),
                #     zorder=30,
                #     edgecolors='k',
                #     linewidth=0.2
                # )

                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.coastlines(resolution=resol,linewidth=0.1,zorder=40)
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

        cbar=fig.colorbar(
            cs,
            ax=axes.ravel().tolist(),
            fraction=0.046,
            pad=0.04
        )
        cbar.set_ticks(levels)
        cbar.set_ticklabels(levels)
        cbar.ax.set_title('COMBO')
        cbar.ax.set_xlabel('EC')
        fig.suptitle('CRPSS    var: SST obs: ERA5')
        graphics.save_fig(fig,'paper/era_combo_ec_'+mlabel)

def var():

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
    window   = [1,7,14,30,90]

    # clim_t_start  = (2000,6,15)
    # clim_t_end    = (2020,8,25)

    clim_t_start  = (1979,6,15)
    clim_t_end    = (1999,8,25)

    era = ERA5(high_res=True).load(
        var         = var,
        start_time  = clim_t_start,
        end_time    = clim_t_end,
        bounds      = bounds
    )

    era = era.resample(time='D').mean()

    resol  = '10m'
    extent = [0,28,55,75]

    for mlabel,months in zip(['winter','summer'],[winter_months,summer_months]):

        latex.set_style(style='white')
        fig,axes = plt.subplots(6,5,\
            figsize=latex.set_size(width=345,subplots=(6,2),fraction=0.95),\
            subplot_kw=dict(projection=ccrs.NorthPolarStereo())
        )

        # cmap   = sns.color_palette("Spectral", as_cmap=True)
        cmap   = sns.color_palette("Blues", as_cmap=True)
        levels = np.arange(0,3.25,0.25)
        norm   = BoundaryNorm(levels,cmap.N)

        for ax_month,month,row in zip(axes,months,range(len(months))):

            for ax,wdw,col in zip(ax_month,window,range(len(window))):

                rolled = era.sst.rolling(time=wdw,center=True).mean()
                rolled = rolled.where(rolled.time.dt.month==int(month))\
                    .var('time',skipna=True).transpose('lat','lon')

                cs = ax.contourf(
                    rolled.lon,
                    rolled.lat,
                    rolled,
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                    extend='both'
                )

                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.coastlines(resolution=resol,linewidth=0.1,zorder=40)
                # land = cfeature.NaturalEarthFeature('physical', 'land', \
                #     scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land']
                # )
                # ax.add_feature(land, zorder=1, facecolor='beige', linewidth=0.2)

                if row==len(months)-1:
                    ax.text(0.5, -0.2, str(wdw)+' days',
                        size = 'xx-large',
                        va='bottom',
                        ha='center',
                        rotation='horizontal',
                        rotation_mode='anchor',
                        transform=ax.transAxes
                    )
                    print(wdw)
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
            pad=0.04
        )
        cbar.set_ticks(levels)
        cbar.set_ticklabels(levels)
        # cbar.ax.set_title('[degC^2]')
        cbar.ax.set_ylabel('[degC^2]')
        # cbar.ax.set_xlabel('Clim')
        fig.suptitle('Varians of SST: ERA5 {}-{}'.format(
                clim_t_start[0],clim_t_end[0]
            )
        )
        graphics.save_fig(fig,'paper/var_era_'+mlabel+'_{}-{}'.format(
                clim_t_start[0],clim_t_end[0]
            )
        )

def dvar():

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
    window   = [1,7,14,30,90]

    clim_t_start2  = (2000,6,15)
    clim_t_end2    = (2020,8,25)

    clim_t_start1  = (1979,6,15)
    clim_t_end1    = (1999,8,25)

    era1 = ERA5(high_res=True).load(
        var         = var,
        start_time  = clim_t_start1,
        end_time    = clim_t_end1,
        bounds      = bounds
    )

    era2 = ERA5(high_res=True).load(
        var         = var,
        start_time  = clim_t_start2,
        end_time    = clim_t_end2,
        bounds      = bounds
    )

    era1 = era1.resample(time='D').mean()
    era2 = era2.resample(time='D').mean()

    resol  = '10m'
    extent = [0,28,55,75]

    for mlabel,months in zip(['winter','summer'],[winter_months,summer_months]):

        latex.set_style(style='white')
        fig,axes = plt.subplots(6,5,\
            figsize=latex.set_size(width=345,subplots=(6,2),fraction=0.95),\
            subplot_kw=dict(projection=ccrs.NorthPolarStereo())
        )

        # cmap   = sns.color_palette("Spectral", as_cmap=True)
        cmap   = latex.skill_cmap()
        levels = np.arange(-3,3.25,0.25)
        norm   = BoundaryNorm(levels,cmap.N)

        for ax_month,month,row in zip(axes,months,range(len(months))):

            for ax,wdw,col in zip(ax_month,window,range(len(window))):

                rolled1 = era1.sst.rolling(time=wdw,center=True).mean()
                rolled1 = rolled1.where(rolled1.time.dt.month==int(month))\
                    .var('time',skipna=True).transpose('lat','lon')

                rolled2 = era2.sst.rolling(time=wdw,center=True).mean()
                rolled2 = rolled2.where(rolled2.time.dt.month==int(month))\
                    .var('time',skipna=True).transpose('lat','lon')

                rolled = rolled2 - rolled1

                cs = ax.contourf(
                    rolled.lon,
                    rolled.lat,
                    rolled,
                    levels=levels,
                    cmap=cmap,
                    norm=norm,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                    extend='both'
                )

                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.coastlines(resolution=resol,linewidth=0.1,zorder=40)
                # land = cfeature.NaturalEarthFeature('physical', 'land', \
                #     scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land']
                # )
                # ax.add_feature(land, zorder=1, facecolor='beige', linewidth=0.2)

                if row==len(months)-1:
                    ax.text(0.5, -0.2, str(wdw)+' days',
                        size = 'xx-large',
                        va='bottom',
                        ha='center',
                        rotation='horizontal',
                        rotation_mode='anchor',
                        transform=ax.transAxes
                    )
                    print(wdw)
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
            pad=0.04
        )
        cbar.set_ticks(levels)
        cbar.set_ticklabels(levels)
        # cbar.ax.set_title('[degC^2]')
        cbar.ax.set_ylabel('[degC^2]')
        # cbar.ax.set_xlabel('Clim')
        fig.suptitle('Difference in varians of SST: ERA5 ({}-{}) - ({}-{})'.format(
                clim_t_start2[0],clim_t_end2[0],clim_t_start1[0],clim_t_end1[0]
            )
        )
        graphics.save_fig(fig,'paper/delta_var_era_'+mlabel+'_{}-{}_{}-{}'.format(
                clim_t_start2[0],clim_t_end2[0],clim_t_start1[0],clim_t_end1[0]
            )
        )

# point_observations = Observations(
#                             name='BarentsWatch',
#                             observations=point_observations,
#                             forecast=grid_hindcast,
#                             process=True
#                             )
#
# point_hindcast     = Grid2Point(point_observations,grid_hindcast)\
#                             .correlation(step_dependent=True)
#
# pers  = models.persistence(
#                 init_value   = point_observations.init_a,
#                 observations = point_observations.data_a
#                 )
#
# combo = models.combo(
#                         init_value      = point_observations.init_a,
#                         model           = point_hindcast.data_a,
#                         observations    = point_observations.data_a
#                     )
#
# combo = point_hindcast.data_a - point_hindcast.data_a.mean('member') + combo
#
# # adjust spread
# combo = models.bias_adjustment_torralba(
#                             forecast        = combo,
#                             observations    = point_observations.data_a,
#                             spread_only     = True
#                             )
