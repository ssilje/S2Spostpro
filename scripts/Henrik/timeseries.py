import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt

from S2S.local_configuration import config
from S2S.data_handler        import BarentsWatch
from S2S.graphics            import latex,graphics
from S2S.process             import Hindcast, Observations, Grid2Point

def customizer():

    ticks = {
        'labelsize' : 'x-large'
        # 'major.size': 5,
        # 'minor.size': 3
    }

    axes = {
        'labelsize' : 'x-large',
        'titlesize' : 'x-large'
    }

    legend = {
        'markerscale' : 3

    }

    plt.rc('legend', **legend)
    plt.rc('xtick', **ticks)      # fontsize of the tick labels
    plt.rc('ytick', **ticks)      # fontsize of the tick labels


def observations(name):

    customizer()

    BW           = BarentsWatch()
    barentswatch = BW.load([name]).sortby('time')
    loc_no       = BW.loc_from_name(name)
    norkyst_path = config['NORKYST'][:-10]+'NorKyst800_'+str(loc_no)+'.nc'
    norkyst      = xr.open_dataset(norkyst_path)
    norkyst      = norkyst.assign_coords(
                            time=pd.Timestamp('1970-01-01 00:00:00')\
                            +pd.to_timedelta(norkyst.time,unit='S')\
                            -pd.Timedelta(12,unit='H')
                            ).rename({'__xarray_dataarray_variable__':'sst'})

    bw = barentswatch.rolling(time=7,center=True).mean().squeeze()
    nk = norkyst.rolling(time=7,center=True).mean().squeeze()

    # Plot inital time series
    latex.set_style(style='white')
    subplots = (1,1)

    fig,ax = plt.subplots(
                            subplots[0],subplots[1],
                            figsize=latex.set_size(width='thesis',
                            subplots=(subplots[0],subplots[1]))
                            )

    n_bw = np.isfinite(bw.sst.values).sum()
    n_nk = np.isfinite(nk.sst.values).sum()

    ax.plot(
            bw.time,
            bw.sst,
            'o-',
            ms=1,
            linewidth=0.3,
            alpha=0.95,
            label='BarentsWatch',
            zorder=30
            )
    ax.plot(
            nk.time,
            nk.sst,
            'o-',
            ms=1,
            linewidth=0.3,
            alpha=0.6,
            label='NorKyst-800',
            color='k',
            zorder=0
            )

    ax.legend(prop={'size': 8})
    ax.set_title('Havtemperatur ved '+ name +\
                    '    Ant. obs. BW:{}  NK:{}'.format(n_bw,n_nk))

    ax.set_ylabel('Havtemperatur [degC]')
    ax.set_xlabel('Tid')

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    graphics.save_fig(fig,'presentation/observations_first_look')

def observations_closeup(name):

    customizer()

    BW           = BarentsWatch()
    barentswatch = BW.load([name]).sortby('time')
    loc_no       = BW.loc_from_name(name)
    norkyst_path = config['NORKYST'][:-10]+'NorKyst800_'+str(loc_no)+'.nc'
    norkyst      = xr.open_dataset(norkyst_path)
    norkyst      = norkyst.assign_coords(
                            time=pd.Timestamp('1970-01-01 00:00:00')\
                            +pd.to_timedelta(norkyst.time,unit='S')\
                            -pd.Timedelta(12,unit='H')
                            ).rename({'__xarray_dataarray_variable__':'sst'})

    bw = barentswatch.rolling(time=7,center=True).mean().squeeze()
    nk = norkyst.rolling(time=7,center=True).mean().squeeze()

    # bw = bw.sel(time=slice('2016-09-01','2017-03-01'))
    # nk = nk.sel(time=slice('2016-09-01','2017-03-01'))

    min_t = bw.time.min()
    max_t = bw.time.max()

    grid_hindcast = Hindcast(
                                var        = 'sst',
                                t_start    = (2020,1,23),
                                t_end      = (2021,1,4),
                                bounds     = (0,28,55,75),
                                high_res   = True,
                                steps      = pd.to_timedelta([9,16,23,30,37],'D'),
                                process    = True,
                                download   = False,
                                split_work = True,
                                period     = [min_t,max_t]
                            )

    point_nk = Observations(
                                name='NorKyst_closeup',
                                observations=norkyst,
                                forecast=grid_hindcast,
                                process=False
                                )

    point_bw = Observations(
                                name='BW_closeup',
                                observations=bw,
                                forecast=grid_hindcast,
                                process=False
                                )

    point_hindcast     = Grid2Point(point_observations,grid_hindcast)\
                                .correlation(step_dependent=True)
    exit()
    # Plot inital time series
    latex.set_style(style='white')
    subplots = (1,1)

    fig,ax = plt.subplots(
                            subplots[0],subplots[1],
                            figsize=latex.set_size(width='thesis',
                            subplots=(subplots[0],subplots[1]))
                            )

    n_bw = np.isfinite(bw.sst.values).sum()
    n_nk = np.isfinite(nk.sst.values).sum()

    ax.plot(
            bw.time,
            bw.sst,
            'o-',
            ms=1,
            linewidth=0.3,
            alpha=0.95,
            label='BarentsWatch',
            zorder=30
            )
    ax.plot(
            nk.time,
            nk.sst,
            'o-',
            ms=1,
            linewidth=0.3,
            alpha=0.6,
            label='NorKyst-800',
            color='k',
            zorder=0
            )

    ax.legend(prop={'size': 8})
    ax.set_title('Havtemperatur ved '+ name)

    ax.set_ylabel('Havtemperatur [degC]')
    ax.set_xlabel('Tid')

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    graphics.save_fig(fig,'presentation/prediction1')
