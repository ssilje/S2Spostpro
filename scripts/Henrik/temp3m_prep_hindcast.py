from S2S import models
from datetime import datetime
import numpy  as np
import xarray as xr
import os
from glob import glob
import pandas as pd
from sklearn.neighbors import BallTree
from S2S.process       import Hindcast, Observations

def main():

    obs_path = '/projects/NS9853K/DATA/norkyst800/station_3m/temperature_insitu_3m_norkyst800_barentswatch_closest_20060101-20220920.nc'
    tmp_path = '/projects/NS9853K/DATA/tmp/'
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

    hc = hindcast
    hindcast = hindcast.data_a
    hindcast = hindcast.stack(point=['lat','lon']).dropna(dim='point',how='all')

    with xr.open_dataarray(obs_path) as obs:
        obs = obs
        location = obs.location

    ycoords = np.deg2rad(
        np.stack([hindcast.lat,hindcast.lon],axis=-1)
    )
    xcoords = np.deg2rad(
        np.stack([location.lat,location.lon],axis=-1)
    )

    tree    = BallTree(ycoords,metric="haversine")
    k_indices = tree.query(xcoords,return_distance=False).squeeze()

    hindcast = hindcast.isel(point=k_indices)
    hindcast = hindcast.assign_coords(point=location.values)
    hindcast = hindcast.rename({'point':'location'})

    hindcast.to_netcdf(tmp_path+'temp3_hindcast_at_point_location.nc')
    print(hindcast)

    observations = Observations(
        name='temp3m_norkyst',
        observations=obs,
        forecast=hc,
        process=False
    )

    co_path = tmp_path + "temp3_combo_at_point_location.nc"

    if not os.path.exists(co_path):

        combo = models.combo(
            init_value      = observations.init_a,
            model           = hindcast,
            observations    = observations.data_a
        )
        combo.to_netcdf(co_path)

    co_path = tmp_path + "temp3_combo_scaled_at_point_location.nc"

    if not os.path.exists(co_path):

        combo = models.combo_scaled(
            init_value      = observations.init_a,
            model           = hindcast,
            observations    = observations.data_a
        )
        combo.to_netcdf(co_path)
