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
from S2S.organize_barentzwatch import name_from_loc
import xskillscore as xs
import S2S.scoring as sc

from matplotlib.colors import BoundaryNorm
import seaborn as sns

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
        combo.to_netcdf(tmp_path+'tmp_combo_fig4.nc')

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

    ec = hindcast.data_a - hindcast.data_a.mean('member')
    ec = ec /hindcast.data_a.std('member')
    ec = ec + hindcast.data_a.mean('member')

    combo   = xr.open_dataarray(tmp_path+'tmp_comboa_fig4.nc')
    combo_u = hindcast.data_a - hindcast.data_a.mean('member')
    combo_u = combo_u / combo_u.std('member')
    combo = combo_u + combo

    return combo,observations.data_a

def main():

    steps    = pd.to_timedelta([9,16,23,30,37],'D')
    lt = 1
    path = '/projects/NS9853K/DATA/norkyst800/processed/hardanger/'

    combo,obs  = calc()

    latex.set_style(style='white')

    for loc in combo.location.values:

        fc = combo.sel(location=loc,step=steps[lt])
        ob = obs.sel(location=loc,step=steps[lt])

        fig,ax = plt.subplots(1,1,
            figsize=(4.535076795350768, 2.8028316011177257)
        )

        ax.fill_between(fc.time,fc.min('member'),fc.max('member'),color='orange',alpha=0.2)
        ax.fill_between(fc.time,fc.quantile(0.25,dim='member'),fc.quantile(0.75,dim='member'),color='orange',alpha=0.5)
        ax.plot(fc.time,fc.mean('member'),color='orange',linewidth=0.8,label='COMBO')
        ax.plot(ob.time,ob,'o',color='k',markersize=0.3,label='NorKyst800')

        ax.set_title('Forecast at '+str(name_from_loc(loc))+'for lead time: '+str(steps[lt].days)+' days')
        ax.set_xlabel('Forecast initialization time')
        ax.set_ylabel('Anomalies from climatology')

        ax.legend(loc='upper left',fontsize=6)

        latex.save_figure(fig,'/nird/home/heau/figures_paper/timeseries/new062022_fig4_timeseries_adj_spread_'+str(name_from_loc(loc)))
        exit()
