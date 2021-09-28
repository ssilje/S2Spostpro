import numpy as np
import xarray as xr
import pandas as pd

import matplotlib.pyplot as plt

from S2S.local_configuration import config
from S2S.data_handler        import BarentsWatch
from S2S.graphics            import latex,graphics
from S2S.process             import Hindcast, Observations, Grid2Point
from S2S                     import xarray_helpers as xh

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

def to_location(data,lat,lon,landmask,reflat,reflon):

    appr_data = []
    rs        = []

    landmask  = landmask.astype('int')

    for rlon,rlat in zip(reflon,reflat):

        dist = distance(rlat,rlon,lat,lon)

        over_land     = 0
        target_radius = 0.
        while not over_land:

            target_radius += 1.

            idx  = dist < target_radius

            lm   = landmask[idx]

            over_land = lm.sum()

        lm = lm.astype('bool')

        # wr: within radius
        data_wr = data[idx][lm]
        dist_wr = dist[idx][lm]

        appr_data.append( lin_weight( data_wr, dist_wr, target_radius ) )
        rs.append( target_radius )

    return np.array(appr_data),np.array(rs)

def to_location_1D(data,lat,lon,landmask,rlat,rlon):

    landmask  = landmask.astype('int')

    dist = distance(rlat,rlon,lat,lon)

    over_land     = 0
    target_radius = 0.
    while not over_land:

        target_radius += 1.

        idx  = dist < target_radius

        lm   = landmask[idx]

        over_land = lm.sum()

    lm = lm.astype('bool')

    # wr: within radius
    data_wr = data[idx][lm]
    dist_wr = dist[idx][lm]

    return lin_weight( data_wr, dist_wr, target_radius ),target_radius

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

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),prop={'size': 8})

def observations_closeup(names):

    customizer()

    BW            = BarentsWatch()
    barentswatch  = BW.load(names).sortby('time')
    loc_nos       = [BW.loc_from_name(name) for name in names]
    norkyst_paths = [config['NORKYST'][:-10]+'NorKyst800_'+str(loc_no)+'.nc'\
                        for loc_no in loc_nos]

    chunks = []
    for norkyst_path in norkyst_paths:
        data = xr.open_dataset(norkyst_path)
        data = data.assign_coords(
                time=pd.Timestamp('1970-01-01 00:00:00')\
                +pd.to_timedelta(data.time,unit='S')\
                -pd.Timedelta(12,unit='H')
                ).rename({'__xarray_dataarray_variable__':'sst'})\
                .drop(('depth','radius'))
        chunks.append(data)
    norkyst = xr.concat(chunks,'location')
    del chunks,data

    bw = barentswatch.rolling(time=7,center=True).mean().squeeze()
    nk = norkyst.rolling(time=7,center=True).mean().squeeze()

    min_t = bw.time.min()
    max_t = bw.time.max()

    grid_hindcast = Hindcast(
                            var        = 'sst',
                            t_start    = (2020,1,23),
                            t_end      = (2021,1,4),
                            bounds     = (0,28,55,75),
                            high_res   = True,
                            steps      = pd.to_timedelta([9,16,23,30,37],'D'),
                            process    = False,
                            download   = False,
                            split_work = True,
                            period     = [min_t,max_t]
                            )

    point_nk = Observations(
                                name         = 'NorKyst_closeup',
                                observations = norkyst.sst,
                                forecast     = grid_hindcast,
                                process      = False
                                )

    point_bw = Observations(
                                name         = 'BW_closeup',
                                observations = bw.sst,
                                forecast     = grid_hindcast,
                                process      = False
                                )

    hc_nk    = Grid2Point(point_nk,grid_hindcast)\
                                .correlation(step_dependent=False)

    hc_bw    = Grid2Point(point_bw,grid_hindcast)\
                                .correlation(step_dependent=False)
    exit()
    # landmask = xr.ufuncs.isfinite(grid_hindcast.data).sum(['member','time','step'])
    # landmask = landmask == landmask.max()
    #
    # hc,r = xr.apply_ufunc(
    #         to_location,
    #         grid_hindcast.data_a,
    #         grid_hindcast.data_a.lat,
    #         grid_hindcast.data_a.lon,
    #         landmask,
    #         point_nk.data_a.lat,
    #         point_nk.data_a.lon,
    #         input_core_dims=[
    #                             ['lon','lat'],['lon','lat'],['lon','lat'],['lon','lat'],
    #                             ['location'],['location'],
    #                         ],
    #         output_core_dims=[['location'],['location']],
    #         vectorize=True
    #     )
    # print(hc)
    # print(r)
    # exit()
    # bw = point_bw.data_a
    nk = point_nk.data_a
    hc = grid_hindcast.data_a
    cm = point_nk.mean
    cs = point_nk.std

    hc = hc.interp(coords={'lon':nk.lon,'lat':nk.lat},method='nearest')
    # cm = cm.interp(coords={'lon':nk.lon,'lat':nk.lat},method='nearest')
    # cs = cs.interp(coords={'lon':nk.lon,'lat':nk.lat},method='nearest')

    # hc_nk    = Grid2Point(point_nk,grid_hindcast)\
    #                             .correlation(step_dependent=False)
    #
    # hc_bw    = Grid2Point(point_bw,grid_hindcast)\
    #                             .correlation(step_dependent=False)

    # Plot inital time series
    latex.set_style(style='white')
    subplots = (1,1)

    fig,ax = plt.subplots(
                            subplots[0],subplots[1],
                            figsize=latex.set_size(width='thesis',
                            subplots=(subplots[0],subplots[1]))
                            )

    init_date = '2018-8-10'
    ll        = str(loc_nos[1])
    # variable to plot
    # bw = bw.sel(time=init_date,location=str(loc_nos[0]))
    nk = nk.sel(time=init_date,location=ll)
    hc = hc.sel(time=init_date,location=ll)
    cm = cm.sel(time=init_date,location=ll)
    cs = cs.sel(time=init_date,location=ll)

    # for member in hc.member()

    # assign validation time
    hc = xh.assign_validation_time(hc)
    nk = xh.assign_validation_time(nk)
    cm = xh.assign_validation_time(cm)
    cs = xh.assign_validation_time(cs)

    hc = hc.sortby('step')
    nk = nk.sortby('step')
    cm = cm.sortby('step')
    cs = cs.sortby('step')
    # bw = xh.assign_validation_time(bw)



    ax.plot(
            cm.validation_time,
            cm,
            'o-',
            ms=1,
            linewidth=1,
            alpha=0.95,
            label='Klimatologisk gjennomsnitt',
            zorder=1
            )

    ax.plot(
            nk.validation_time,
            nk + cm,
            'o-',
            ms=1,
            linewidth=1,
            alpha=0.95,
            label='NorKyst-800',
            color='k',
            zorder=2
            )

    for member in hc.member:

        ax.plot(
                hc.validation_time,
                hc.sel(member=member)*cs + cm,
                'o-',
                ms=1,
                linewidth=0.6,
                alpha=0.6,
                label='Varsel EC',
                color='orange',
                zorder=0
                )

    ax.plot(
            hc.validation_time,
            hc.mean('member')*cs + cm,
            'o--',
            ms=1.2,
            linewidth=1.7,
            alpha=1,
            label='Varsel EC ens. mean',
            color='orange'
            )

    legend_without_duplicate_labels(ax)
    ax.set_title('Havtemperatur ved '+ names[0])

    ax.set_ylabel('Havtemperatur [degC]')
    ax.set_xlabel('Tid')

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    ax.set_xticks(hc.validation_time.values)

    graphics.save_fig(fig,'presentation/closeup_prediction1')
