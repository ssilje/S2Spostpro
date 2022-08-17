import pandas as pd
import xarray as xr
import numpy as np

from S2S.data_handler import ERA5, BarentsWatch
from S2S.process import Hindcast, Observations, Grid2Point

from S2S.graphics import mae,crps,graphics as mae,crps,graphics
from S2S import models

def plot():

    bounds = (0,28,55,75)
    var      = 'sst'

    t_start  = (2020,1,23)
    t_end    = (2021,1,4)

    clim_t_start  = (2000,1,1)
    clim_t_end    = (2021,1,4)


    high_res = True
    steps    = pd.to_timedelta([9,16,23,30,37],'D')

    # observations must be weekly mean values with a time dimension
    barentswatch = BarentsWatch().load(
                                        ['Ljonesbjørgene','Aga Ø']
                                    )[var]

    print('Get norkyst')
    norkyst = xr.open_dataset(
        '/projects/NS9853K/DATA/norkyst800/processed/hardanger/norkyst800_sst_daily_mean_hardanger_atBW.nc'
    )

    norkyst = norkyst.sel(location=barentswatch.location)

    print('Norkyst: roll')
    norkyst = norkyst.sst.rolling(time=7,center=True).mean()

    hindcast = xr.open_dataarray(
        '/projects/NS9853K/DATA/norkyst800/processed/hardanger/hindcast_hardanger.nc'
    )

    print('Norkyst: process')
    observations = Observations(
        name='norkyst_in_hardangerfjorden',
        observations=norkyst,
        forecast=hindcast,
        process=False
    )

    print('Barentswatch: process')
    observations_bw = Observations(
        name='barentswatch_in_hardangerfjorden',
        observations=barentswatch,
        forecast=hindcast,
        process=False
    )

    clim_fc = models.clim_fc(observations.mean,observations.std)

    ec_sb = ( hindcast * observations.std ) + observations.mean

    pers = models.persistence(observations.init_a,observations.data_a,window=30)
    pers = ( pers * observations.std ) + observations.mean

    print('Fit combo model')
    combo = models.combo(
        init_value      = observations.init_a,
        model           = hindcast,
        observations    = observations.data_a
    )

    combo = hindcast - hindcast.mean('member') + combo
    combo = ( combo * observations.std ) + observations.mean

    # In absolute values
    graphics.timeseries(
                            observations    = observations.data,
                            cast            = [clim_fc,ec_sb,pers,combo],
                            lead_time       = [9,16,23],
                            clabs           = ['CLIM','EC','PERS','COMBO'],
                            filename        = '/nird/home/heau/figures_paper/fig3',
                            title           = ''
                        )

    # In absolute values
    graphics.timeseries(
                            observations    = observations.data,
                            cast            = [clim_fc,ec_sb,pers,combo],
                            lead_time       = [30,37],
                            clabs           = ['CLIM','EC','PERS','COMBO'],
                            filename        = '/nird/home/heau/figures_paper/fig3-5',
                            title           = ''
                        )
