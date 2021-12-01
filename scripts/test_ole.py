import pandas as pd
import xarray as xr
import xskillscore as xs
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

from S2S.data_handler import ERA5
#from S2S.process import Hindcast, Observations, Forecast, Observations_hcfc
from S2S.process import Hindcast, Observations, Forecast

from S2S.graphics import mae,crps,graphics as mae,crps,graphics
from S2S import models
import S2S.xarray_helpers    as xh
from S2S.scoring import uncentered_acc, centered_acc
from S2S.local_configuration import config


bounds          = (0,28,55,75)
var             = 'tp'


t_start         = (2018,3,1)
t_end           = (2018,4,1)


clim_t_start    = (1999,1,1)
clim_t_end      = (2019,12,31)


high_res        = False

steps           = pd.to_timedelta([7, 14, 21, 28, 35, 42],'D')


print('\tProcess Hindcast')

grid_hindcast = Hindcast(
                        var,
                        t_start,
                        t_end,
                        bounds,
                        high_res=high_res,
                        steps=steps,  
                        process=False,
                        download=False,
                        split_work=False
                    )

grid_forecast = Forecast(
                        var,
                        t_start,
                        t_end,
                        bounds,
                        high_res=high_res,
                        steps=steps,
                        process=True,
                        download=False,
                        split_work=False
                    )
## Hindcast/Forecast is mm/6h
## ERA is mm/24h
era = ERA5(high_res=high_res)\
                            .load(var,clim_t_start,clim_t_end,bounds)[var]

## need to have grid_observations_fc first. if grid_observations_hc is first, the date in grid_observations_fc is the same as grid_observations_hc.
# still something strange with hc fc obs. 

grid_observations_fc = Observations(
                                   name='Era',
                                   observations=era,
                                   forecast=grid_forecast,
                                   process=True
)


del(era)
era = ERA5(high_res=high_res)\
                            .load(var,clim_t_start,clim_t_end,bounds)[var]


grid_observations_hc = Observations(
  name='Era',
  observations=era,
  forecast=grid_hindcast,
  process=True
)


obs = []
obs.append(grid_observations_hc.data)
obs.append(grid_observations_fc.data)

obs_full = xr.concat(obs,dim='time') ## stacking the data along month dimension
obs_full = obs_full.rename(var)


reanalysis         = xh.assign_validation_time(obs_full)


hindcast           = xh.assign_validation_time(grid_hindcast.data)


forecast           = xh.assign_validation_time(grid_forecast.data)
