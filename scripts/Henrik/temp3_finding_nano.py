from S2S import models, xarray_helpers
from datetime import datetime
import numpy  as np
import xarray as xr
import os
from glob import glob
import pandas as pd
from sklearn.neighbors import BallTree
from S2S.process       import Hindcast, Observations
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cmocean

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

    print(data_a)
    exit()
