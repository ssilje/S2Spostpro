import os
import numpy  as np
import pandas as pd
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
import cartopy.crs       as ccrs
# from sklearn.neighbors import BallTree
#
from S2S         import models, xarray_helpers
from S2S.process import Hindcast, Observations

obs_path = '/projects/NS9853K/DATA/norkyst800/station_3m/temperature_insitu_3m_norkyst800_barentswatch_closest_20060101-20220920.nc'
tmp_path = "/projects/NS9853K/DATA/tmp/"

def main():

    obs_time1 = datetime(year=2005,month=11,day=1) # noted 22/10/22 for new NK data
    obs_time2 = datetime(year=2021,month=7,day=19)

    hindcast = Hindcast(
        var='sst',
        t_start=(2020,7,16),
        t_end=(2021,7,19),
        bounds=(0,28,55,75),
        high_res=True,
        process=False,
        steps=pd.to_timedelta([4,11,18,25,32,39],'D'),
        download=False,
        split_work=True,
        cross_val=True,
        period=[obs_time1,obs_time2]
    )

    with xr.open_dataset(obs_path) as obs:
        obs = obs.rolling(time=7,center=True).mean()
        obs = obs.temperature

    observations = Observations(
        name='temp3m_norkyst',
        observations=obs,
        forecast=hindcast,
        process=False
    )

    # PERSISTENCE
    pe_path = tmp_path + "temp3m_pers.nc"
    if not os.path.exists(pe_path):

        pers = models.persistence(observations.init_a,observations.data_a,window=30)
        pers.to_netcdf(pe_path)

    pe_path = tmp_path + "temp3m_pers_scaled.nc"
    if not os.path.exists(pe_path):

        pers = models.persistence_scaled(observations.init_a,observations.data_a,window=30)
        pers.to_netcdf(pe_path)

    print('Finished')
