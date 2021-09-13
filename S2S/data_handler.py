# environment dependencies
import numpy  as np
import xarray as xr
import pandas as pd
import os
import json
import glob
import dask

# local dependencies
from S2S.local_configuration import config
import S2S.handle_datetime as dt
import scripts.Henrik.organize_barentzwatch as organize_barentzwatch
from . import xarray_helpers as xh
from . import date_to_model as d2m

class Archive:
    """
    Returns local paths and filename functions
    """

    def __init__(self):

        self.source = ['ERA5','S2SH','S2SF','BW']
        self.path   = dict((source,config[source]) for source in self.source)
        self.ftype  = {
                        'ERA5':'reanalysis',
                        'S2SH':'hindcast',
                        'S2SF':'forecast'
                        }

        self.dimension_parser = {
                                    'number'   :'member',
                                    'longitude':'lon',
                                    'latitude' :'lat'
                                    }

    def ERA5_in_filename(self,**kwargs):
        """
        Returns list of filenames to load from the ERA database
        """
        var    = self.given('var',kwargs)
        start  = self.given('start_time',kwargs)
        end    = self.given('end_time',kwargs)
        path   = self.given('path',kwargs)

        var = {
                'sst':'sea_surface_temperature',
                'u10':'10m_u_component_of_wind',
                'v10':'10m_v_component_of_wind',
                't2m':'2m_temperature'
            }[var]

        # add variable sub-directory to path
        path += var + '/'

        filenames = self.get_filenames(path)

        return filenames.sel(time=slice(start,end))


    def S2S_in_filename(self,**kwargs):
        """
        Returns list of filenames to load from the S2S database
        """
        var         = self.given('var',kwargs)
        start       = self.given('start_time',kwargs)
        end         = self.given('end_time',kwargs)
        run         = self.given('run',kwargs)
        path        = self.given('path',kwargs)

        # add variable sub-directory to path
        path += var + '/'

        filenames = self.get_filenames(path,run=run)

        filenames = filenames.sel(time=slice(start,end))

        return filenames.values

    def get_filenames(self,path,**kwargs):
        """
        Returns a xarray.DataArray with all filenames in folder
        with coordinates:
            time
            run (cf,pf) (if run is given)

        If run is not given: The point is to make a 2D array of filename such
        that the open_mfdataset can concat member and timedimension correctly.
        """

        # get all filenames in folder

        # if run is specified
        if self.given('run',kwargs) is not None:

            all_files = np.array(glob.glob(path+'*'+kwargs['run']+'.grb'))
            coords    = {'time':[self.find_time(file) for file in all_files]}
            dims      = list(coords)

        # if both runs should be included
        else:

            # complete lists of cf and pf in folder
            cf = np.array(glob.glob(path+'*cf.grb'))
            pf = np.array(glob.glob(path+'*pf.grb'))

            # get times associated with each file
            cf_time = np.array([self.find_time(file) for file in cf])
            pf_time = np.array([self.find_time(file) for file in pf])

            # sort times and filenames after time
            cf_idx = np.argsort(cf_time)
            pf_idx = np.argsort(pf_time)

            cf = cf[cf_idx]
            pf = pf[pf_idx]

            cf_time = cf_time[cf_idx]
            pf_time = pf_time[pf_idx]

            # lenghts of filelists
            lcf,lpf = len(cf_time),len(pf_time)

            # pop the non-matching times NB THIS HAS NOT BEEN TESTED YET
            n,popped = 0,[]
            while lcf != lpf:

                cf_is_longest = lcf>lpf

                shortest_time = pf_time if cf_is_longest else cf_time
                shortest_name = pf      if cf_is_longest else cf

                longest_time  = cf_time if cf_is_longest else pf_time
                longest_name  = cf      if cf_is_longest else pf

                while np.isin(shortest_time,longest_time[n],unique=True):
                    n += 1

                popped.append(longest_name[n])
                longest_time = np.delete(longest_time,n)
                longest_name = np.delete(longest_name,n)

                lcf,lpf = len(cf_time),len(pf_time)

            if n>0:

                pf_time = shortest_time if cf_is_longest else longest_time
                pf      = shortest_name if cf_is_longest else longest_name

                cf_time = longest_time  if cf_is_longest else shortest_time
                cf      = longest_name  if cf_is_longest else shortest_name

                print('Files not matching between runs:')
                for file in popped:
                    print(file)

            if ( cf_time == pf_time ).all():

                all_files = np.array([cf,pf]).T
                coords    = {
                                'time': pf_time,
                                'run' : np.array(['cf','pf'])
                            }
                dims      = list(coords)

            else:
                raise NameError('cf and pf files does not match on time')
                exit(1)

        # make time object from files
        return xr.DataArray(
                                data   = all_files,
                                dims   = dims,
                                coords = coords
                            ).sortby('time')

    @staticmethod
    def BW_in_filename(**kwargs):

        location = kwargs['location']

        return '_'.join(['barentswatch',str(location)])+'.nc'

    @staticmethod
    def out_filename(var,start,end,bounds,label):
        return '%s_%s-%s_%s_%s%s'%(
                            var,
                            dt.to_datetime(start).strftime('%Y-%m-%d'),
                            dt.to_datetime(end).strftime('%Y-%m-%d'),
                            label,
                            '%s_%s-%s_%s'%(bounds),
                            '.nc'
                        )

    @staticmethod
    def make_dir(path):
        """
        Creates directory if it does not exist.

        args:
            path: str
        """
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def given(key,dictionary):
        try:
            return dictionary[key]
        except KeyError:
            return None

    @staticmethod
    def find_time(filename,*args):
        """
        Returns pd.Timestamp of date found in filename of format *_yyyy-mm-dd_*
        """
        for instance in filename.split('_'):
            try:
                if len(instance)==10:
                    return pd.Timestamp(instance)
            except ValueError:
                pass
        return None

    # @staticmethod
    # def get_filename_info(filename,info):
    #     """
    #     DEPRICATED
    #
    #     Returns the either model_cycle or run (specified by info) from filename
    #     """
    #     return filename.split('_')[{'model_cycle':1,'run':-1}[info]]

class LoadLocal:
    """
    Base class for loading local data
    """

    def __init__(self):

        self.prnt            = True

        self.label           = ''
        self.var             = ''
        self.download        = False
        self.loading_options = {
                                'label':            None,
                                'load_time':        None,
                                'concat_dimension': None,
                                'resample':         None,
                                'sort_by':          None,
                                'control_run':      None,
                                'engine':           None,
                                'high_res':         None
                            }

        self.out_path     = ''
        self.in_path      = ''

        self.out_filename = ''

        self.start_time   = None
        self.end_time     = None

        self.run          = None

        self.bounds       = ()

        self.dimension_parser = Archive().dimension_parser

    def rename_dimensions(self,ds):
        """
        rename dimensions given in self.dimension_parser of dataArray

        args:
            da: xarray.DataArray

        returns:
            da: xarray.DataArray
        """
        for dim in ds.dims:
            try:
                ds = ds.rename({dim:self.dimension_parser[dim]})
            except KeyError:
                pass
        return ds

    def adjust_in_data(self,da):
        """
        rename dimensions given in self.dimension_parser of dataArray,
        assign member dim if not already given and select gridbox

        args:
            da: xarray.DataArray

        returns:
            da: xarray.DataArray
        """

        # rename dimensions
        da = self.rename_dimensions(da)

        # assign member dimension to cf runs
        try:
            da['member']
        except KeyError:
            da = da.expand_dims('member').assign_coords(member=[0])

        # select the gridbox specified by self.bounds
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            da = da.sortby('lat','lon').sel(
                lat=slice(self.bounds[2],self.bounds[3]),
                lon=slice(self.bounds[0],self.bounds[1])
            )
        # da = da.where( ( da.lon <= self.bounds[1] | self.bounds[0] <= da.lon ),
        #     drop=True
        # )
        # da = da.where( ( da.lat <= self.bounds[3] | self.bounds[2] <= da.lat ),
        #     drop=True
        # )
        return da

    def filename(self,key='in'):
        """
        Returns functions returning filenames dependent on self.label and 'key'
        argument
        """

        label = self.label

        if label=='ERA5' and key=='in':
            return Archive().ERA5_in_filename

        if label=='S2SH' and key=='in':
            return Archive().S2S_in_filename

        if label=='S2SF' and key=='in':
            return Archive().S2S_in_filename

        if label=='BW' and key=='in':
            return Archive().BW_in_filename

        if key=='out':
            return Archive().out_filename

    def execute_loading_sequence(self,chunks=None):
        """
        Gets filenames and loads the specified dataset. Uses Dask if argument
        chunks is given.

        args:
            chunks: list
        returns:
            xarray.Dataset
        """
        sort_by     = self.loading_options['sort_by']
        resample    = self.loading_options['resample']
        engine      = self.loading_options['engine']
        dimension   = self.loading_options['concat_dimension']
        control_run = self.loading_options['control_run']
        high_res    = self.loading_options['high_res']
        ftype       = Archive().ftype[self.label]

        filename_func = self.filename(key='in')

        # get flexible filename (filename with at least one * in it)
        filenames = filename_func(
                                var         = self.var,
                                start_time  = self.start_time,
                                end_time    = self.end_time,
                                path        = self.in_path,
                                run         = self.run,
                                high_res    = high_res
                            )

        # get dims to concatenate on
        c_dims = ['time'] if self.run is not None else ['time','member']

        print('\n')
        for filename in filenames.flatten():
            print(filename)
        print('\n...files loading...')

        # load_mfdataset needs nested lists
        filenames = [row.tolist() for row in filenames]

        # Load dataset using mf. What does mf actually stand for?
        data = xr.open_mfdataset(
                                filenames,
                                concat_dim     = c_dims,
                                chunks         = chunks,
                                combine        = 'nested',
                                parallel       = True,
                                engine         = engine,
                                backend_kwargs = {'indexpath':''},
                                preprocess     = self.adjust_in_data
                            )

        # make alternative load by series function ?

        return data

    def load(
                self,
                var,
                bounds,
                start_time  = None,
                end_time    = None,
                model_cycle = None,
                download    = False,
                prnt        = True,
                chunks      = None,
                run         = None
            ):
        """
        Load dataset.

        args:
            var:        string
            bounds:     tuple
            start_time: string of yyyy-mm-dd or tuple of (yyyy,mm,dd)
            end_time:   string of yyyy-mm-dd or tuple of (yyyy,mm,dd)
            download:   bool, if True force download even if temp file exists
            prnt:       bool, if True print steps
            chunks:     dict, uses dask if given
            run:        string, options = ['cf','pf']. If not given, loads both.
        """
        if isinstance(start_time,tuple) and isinstance(end_time,tuple):
            start_time = dt.to_datetime(start_time)
            end_time   = dt.to_datetime(end_time)

        elif isinstance(model_cycle,str):
            start_time, end_time = d2m.mv_2_init(model_cycle)
            start_time = dt.to_datetime(start_time)
            end_time   = dt.to_datetime(end_time)

        else:
            raise ValueError('Must supply either start/end time or model_cycle.')
            exit(1)


        archive = Archive()

        self.prnt         = prnt

        self.var          = var
        self.start_time   = start_time
        self.end_time     = end_time
        self.bounds       = bounds
        self.run          = run
        self.download     = download
        self.out_filename = archive.out_filename(
                                            var    = var,
                                            start  = start_time,
                                            end    = end_time,
                                            bounds = bounds,
                                            label  = archive.ftype[self.label]
                                            )

        while self.download or not os.path.exists(self.out_path
                                                        +self.out_filename):

            archive.make_dir(self.out_path)

            data = self.execute_loading_sequence(chunks=chunks)
            data.transpose(
                        'member','step','time','lon','lat',missing_dims='ignore'
                        ).to_netcdf(self.out_path + self.out_filename)

            self.download = False

        return xr.open_dataset(self.out_path+self.out_filename,chunks=chunks)

class ERA5(LoadLocal):

    def __init__(self,high_res=False):

        super().__init__()

        self.label           = 'ERA5'

        self.loading_options['load_time']        = 'daily'
        self.loading_options['concat_dimension'] = 'time'
        self.loading_options['resample']         = 'D'
        self.loading_options['sort_by']          = 'lat'
        self.loading_options['control_run']      = False
        self.loading_options['engine']           = 'netcdf4'

        if high_res:
            self.loading_options['high_res']     = True
            self.in_path                         = config[self.label+'_HR']
        else:
            self.in_path                         = config[self.label]

        self.out_path                            = config['VALID_DB']

class ECMWF_S2SH(LoadLocal):

    def __init__(self,high_res=False):

        super().__init__()

        self.label           = 'S2SH'

        self.loading_options['load_time']        = 'daily'
        self.loading_options['concat_dimension'] = 'time'
        self.loading_options['resample']         = False
        self.loading_options['sort_by']          = 'lat'
        self.loading_options['control_run']      = True
        self.loading_options['engine']           = 'cfgrib'

        if high_res:
            self.loading_options['high_res']     = True
            self.in_path                         = config[self.label+'_HR']
        else:
            self.in_path                         = config[self.label]

        self.out_path                            = config['VALID_DB']

class ECMWF_S2SF(LoadLocal):

    def __init__(self,high_res=False):

        super().__init__()

        self.label           = 'S2SF'

        self.loading_options['load_time']        = 'daily'#'weekly_forecast_cycle'
        self.loading_options['concat_dimension'] = 'time'
        self.loading_options['resample']         = False
        self.loading_options['sort_by']          = 'lat'
        self.loading_options['control_run']      = True
        self.loading_options['engine']           = 'cfgrib'

        if high_res:
            self.loading_options['high_res']     = True
            self.in_path                         = config[self.label+'_HR']
        else:
            self.in_path                         = config[self.label]

        self.out_path                            = config['VALID_DB']

class BarentsWatch:

    def __init__(self,prnt=True):

        self.labels = [
                        'DATA',
                        ]

        self.ftype = {
                        'DATA':'point_measurement',
                        }

        self.loaded = dict.fromkeys(self.labels)
        self.path   = dict.fromkeys(self.labels)
        self.time   = dict.fromkeys(self.labels)

        self.path['DATA'] = config['BW']

    def load(self,location,no=400,data_label='DATA'):

        if location=='all':
            location = self.all_locs(number_of_observations=no)

        archive = Archive()

        if not hasattr(location,'__iter__'):
            if isinstance(location,str) or isinstance(location,int):
                location = [location]
            else:
                raise ValueError

        location_2 = []
        if isinstance(location[0],str):
            for loc in location:
                location_2.append(archive.get_domain(loc)['localityNo'])
            location = location_2

        if not os.path.exists(self.path['DATA']+\
                                archive.BW_in_filename(location=location[0])):
            organize_barentzwatch.organize_files()

        chunk = []
        for loc in location:

            chunk.append(self.load_barentswatch(self.path[data_label],loc))

        return xr.concat(chunk,'location')

    @staticmethod
    def all_locs(number_of_observations=400):
        with open(config['BW']+'temp_BW.json', 'r') as file:
            data = json.load(file)
            out  = []
            i    = 0
            for loc,dat in data.items():
                values = np.array(dat['value'],dtype='float32')
                if np.count_nonzero(~np.isnan(values))>number_of_observations:
                    i += 1
                    print(dat['name'])
                    out.append(int(dat['localityNo']))

            print(
                    '\n...',
                    i,
                    'locations, with',
                    number_of_observations,
                    'observations or more, included\n'
                )
        return out

    @staticmethod
    def load_barentswatch(path,location):

        filename  = '_'.join(['barentswatch',str(location)])+'.nc'

        open_data = xr.open_dataset(path+filename)

        return open_data

class IMR:

    def __init__(self):

        self.path = config['IMR']

    def load(self,location):

        chunk = []
        for loc in location:

            filename = loc+'_organized.nc'

            data = xr.open_dataset(self.path+filename)
            data = data.sortby('time').resample(time='D',skipna=True).mean()
            chunk.append(data)

        return xr.concat(chunk,'location',join='outer')
