# environment dependencies
import numpy  as np
import xarray as xr
import pandas as pd
import os
import json
import glob

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

    @staticmethod
    def ERA5_in_filename(**kwargs):
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


    @staticmethod
    def S2S_in_filename(**kwargs):
        """
        Returns list of filenames to load from the S2S database
        """
        var         = self.given('var',kwargs)
        start       = self.given('start_time',kwargs)
        end         = self.given('end_time',kwargs)
        model_cycle = self.given('model_cycle',kwargs)
        run         = self.given('run',kwargs)
        path        = self.given('path',kwargs)

        # add variable sub-directory to path
        path += var + '/'

        filenames = self.get_filenames(path,model_cycle=model_cycle,run=run)

        filenames = filenames.sel(time=slice(start,end))

        if run is not None:
            filenames = filenames.sel(run=run)

        if model_cycle is not None:
            filenames = filenames.sel(model_cycle=model_cycle)

        return filenames


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
    def find_time(filename):
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

    @staticmethod
    def get_filename_info(filename,info):
        """
        Returns the either model_cycle or run (specified by info) from filename
        """
        return filename.split('_')[{'model_cycle':1,'run':-1}[info]]

    @staticmethod
    def get_filenames(path,**kwargs):
        """
        Returns a xarray.DataArray with all filenames in folder
        with coordinates:
            time,
            model_cycle (if model_cycle is given),
            run (cf,pf) (if run is given)

        """
        # parse argument to its retrieval function
        p_pars = {
                    'time'        : self.find_time,
                    'model_cycle' : self.get_filename_info,
                    'run'         : self.get_filename_info
                }

        # define coords based on kwargs
        coords = {'time':[]}
        if self.given('model_cycle',kwargs) is not None:
            coords['model_cycle'] = []
        if self.given('run',kwargs) is not None:
            coords['run'] = []

        # get all filenames in folder
        all_files = glob.glob(path+'*')

        # get coords
        for file in all_files:
            for key in coords:
                coords[key].append(p_pars[key](file))

        # make time object from files
        return xr.DataArray(data=all_files,coords=coords).sort('time')

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

        self.run          = ''

        self.bounds       = ()

        self.dimension_parser = Archive().dimension_parser

    def rename_dimensions(self,ds):

        for dim in ds.dims:
            try:
                ds = ds.rename({dim:self.dimension_parser[dim]})
            except KeyError:
                pass
        return ds

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

    def load_frequency(self):
        """
        Returns a list like item of pandas.Datetime
        from self.start_time to self.end_time. Frequency dependent on input.
        """

        # Only the option 'daily' is used so far.
        # self.excecute_loading_sequence checks if file exists.
        # Consider deleting the remaining options, in case this function
        # is deprocated and dt.days_from(self.start_time,self.end_time) could be
        # called directly.

        # Second note, this function is depricated after update to
        # load_mfdataset

        option = self.loading_options['load_time']

        if option=='daily':
            return dt.days_from(self.start_time,self.end_time)

        elif option=='weekly_forecast_cycle':
            return dt.weekly_forecast_cycle(self.start_time,self.end_time)

        elif option=='forecast_cycle':
            return dt.forecast_cycle(self.start_time,self.end_time)
        else:
            raise KeyError
            exit()

    def execute_loading_sequence(self,x_landmask=False,chunks=None):

        # Consider removing x_landmask option unless filling dataset holes
        # become something we want to do in the future

        chunk = [] # depricated using mf

        # CHANGE FILENAME TO LOAD USING MF
        # THEN BACK TO WASHING MACHINE
        # AND PROPAGATE chunks to surface

        sort_by     = self.loading_options['sort_by']
        resample    = self.loading_options['resample']
        engine      = self.loading_options['engine']
        dimension   = self.loading_options['concat_dimension']
        control_run = self.loading_options['control_run']
        high_res    = self.loading_options['high_res']
        ftype       = Archive().ftype[self.label]

        filename_func = self.filename(key='in')

        # get flexible filename (filename with at least one * in it)
        filename = filename_func(
                                var         = self.var,
                                start_time  = self.start_time,
                                end_time    = self.end_time,
                                model_cycle = self.model_cycle,
                                path        = self.in_path,
                                run         = self.run
                                high_res    = high_res
                            )

        # Load dataset using mf. What does mf actually stand for?
        data = xarray.open_mfdataset(
                                self.in_path + filename,
                                chunks     = chunks,
                                parallel   = True # good idea?
                            )

        print(data)

        # for time in self.load_frequency():
        #
        #     runs   = ['pf','cf'] if control_run else [None]
        #
        #     OK = True
        #     for n,run in enumerate(runs):
        #
        #         filename = filename_func(
        #                                 var      = self.var,
        #                                 date     = time,
        #                                 run      = run,
        #                                 ftype    = ftype,
        #                                 high_res = high_res
        #                             )
        #         if not os.path.exists(self.in_path+filename):
        #             OK = False
        #
        #     if OK:
        #         members = []
        #         for n,run in enumerate(runs):
        #
        #             filename = filename_func(
        #                                     var      = self.var,
        #                                     date     = time,
        #                                     run      = run,
        #                                     ftype    = ftype,
        #                                     high_res = high_res
        #                                 )
        #
        #             if n>0:
        #                 members.append(open_data)
        #
        #
        #             # to suppress generation of the index file when
        #             # using cfgrib engine
        #             if engine=='cfgrib':
        #                 with xr.open_dataset(
        #                                     self.in_path + \
        #                                     filename,engine = engine,
        #                                     backend_kwargs  = {'indexpath':''}
        #                                     ) as temp_data:
        #                     open_data = temp_data
        #
        #             else:
        #                 with xr.open_dataset(self.in_path+\
        #                                             filename,engine=engine
        #                                             ) as temp_data:
        #                     open_data = temp_data
        #
        #             open_data = self.rename_dimensions(open_data)
        #
        #             if sort_by:
        #                 open_data = open_data.sortby(sort_by,ascending=True)
        #
        #             open_data = open_data.sel(
        #                             lat=slice(self.bounds[2],self.bounds[3]),
        #                             lon=slice(self.bounds[0],self.bounds[1])
        #                             )
        #
        #             if resample:
        #                 open_data = open_data.resample(time=resample).mean()
        #
        #             if x_landmask:
        #                 open_data = xh.extrapolate_land_mask(open_data)
        #
        #             if run=='cf':
        #                 open_data = open_data.expand_dims('member')\
        #                                 .assign_coords(member=pd.Index([0]))
        #
        #             if self.prnt:
        #                 print(filename)
        #
        #         if n>0:
        #             members.append(open_data)
        #             open_data = xr.concat(members,'member')
        #
        #         chunk.append(open_data)
        #
        # return xr.concat(chunk,dimension)

    def load(
                self,
                var,
                bounds,
                start_time  = None, # times are dominant
                end_time    = None, #
                model_cycle = None,
                download    = False,
                prnt        = True,
                chunks      = None,
                run         = None,
                x_landmask  = False # consider moving this option
            ):

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
        self.model_cycle  = model_cycle
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

            data = self.execute_loading_sequence(
                                                    x_landmask = x_landmask,
                                                    chunks     = chunks
                                                )

            # This is the advertised washingmachine, maybe wrap this as three
            # functions. The washingmachine should convert everything to a
            # common format
            if self.label=='ERA5':
                data.transpose('time','lon','lat').to_netcdf(
                                                              self.out_path
                                                            + self.out_filename
                                                            )
            elif self.label=='S2SH':

                data.transpose(
                            'member','step','time','lon','lat'
                            ).to_netcdf(self.out_path + self.out_filename)

            elif self.label=='S2SF':

                data.transpose(
                            'member','step','time', 'lon','lat'
                            ).to_netcdf(self.out_path + self.out_filename)

            else:
                data.to_netcdf(self.out_path+self.out_filename)

            self.download = False

        return xr.open_dataset(self.out_path+self.out_filename)

        # OLD VERSION
        # def execute_loading_sequence(self,x_landmask=False):
        #
        #     # Consider removing x_landmask option unless filling dataset holes
        #     # become something we want to do in the future
        #
        #     chunk = []
        #
        #     sort_by     = self.loading_options['sort_by']
        #     resample    = self.loading_options['resample']
        #     engine      = self.loading_options['engine']
        #     dimension   = self.loading_options['concat_dimension']
        #     control_run = self.loading_options['control_run']
        #     high_res    = self.loading_options['high_res']
        #     ftype       = Archive().ftype[self.label]
        #
        #     filename_func = self.filename(key='in')
        #
        #     for time in self.load_frequency():
        #
        #         runs   = ['pf','cf'] if control_run else [None]
        #
        #         OK = True
        #         for n,run in enumerate(runs):
        #
        #             filename = filename_func(
        #                                     var      = self.var,
        #                                     date     = time,
        #                                     run      = run,
        #                                     ftype    = ftype,
        #                                     high_res = high_res
        #                                 )
        #             if not os.path.exists(self.in_path+filename):
        #                 OK = False
        #
        #         if OK:
        #             members = []
        #             for n,run in enumerate(runs):
        #
        #                 filename = filename_func(
        #                                         var      = self.var,
        #                                         date     = time,
        #                                         run      = run,
        #                                         ftype    = ftype,
        #                                         high_res = high_res
        #                                     )
        #
        #                 if n>0:
        #                     members.append(open_data)
        #
        #
        #                 # to suppress generation of the index file when
        #                 # using cfgrib engine
        #                 if engine=='cfgrib':
        #                     with xr.open_dataset(
        #                                         self.in_path + \
        #                                         filename,engine = engine,
        #                                         backend_kwargs  = {'indexpath':''}
        #                                         ) as temp_data:
        #                         open_data = temp_data
        #
        #                 else:
        #                     with xr.open_dataset(self.in_path+\
        #                                                 filename,engine=engine
        #                                                 ) as temp_data:
        #                         open_data = temp_data
        #
        #                 open_data = self.rename_dimensions(open_data)
        #
        #                 if sort_by:
        #                     open_data = open_data.sortby(sort_by,ascending=True)
        #
        #                 open_data = open_data.sel(
        #                                 lat=slice(self.bounds[2],self.bounds[3]),
        #                                 lon=slice(self.bounds[0],self.bounds[1])
        #                                 )
        #
        #                 if resample:
        #                     open_data = open_data.resample(time=resample).mean()
        #
        #                 if x_landmask:
        #                     open_data = xh.extrapolate_land_mask(open_data)
        #
        #                 if run=='cf':
        #                     open_data = open_data.expand_dims('member')\
        #                                     .assign_coords(member=pd.Index([0]))
        #
        #                 if self.prnt:
        #                     print(filename)
        #
        #             if n>0:
        #                 members.append(open_data)
        #                 open_data = xr.concat(members,'member')
        #
        #             chunk.append(open_data)
        #
        #     return xr.concat(chunk,dimension)
        #
        # def load(
        #             self,
        #             var,
        #             start_time,
        #             end_time,
        #             bounds,
        #             download=False,
        #             prnt=True,
        #             x_landmask=False
        #         ):
        #
        #     archive = Archive()
        #
        #     self.prnt         = prnt
        #
        #     self.var          = var
        #     self.start_time   = start_time
        #     self.end_time     = end_time
        #     self.bounds       = bounds
        #     self.download     = download
        #     self.out_filename = archive.out_filename(
        #                                         var    = var,
        #                                         start  = start_time,
        #                                         end    = end_time,
        #                                         bounds = bounds,
        #                                         label  = archive.ftype[self.label]
        #                                         )
        #
        #     while self.download or not os.path.exists(self.out_path
        #                                                     +self.out_filename):
        #
        #         archive.make_dir(self.out_path)
        #
        #         data = self.execute_loading_sequence(x_landmask=x_landmask)
        #
        #         if self.label=='ERA5':
        #             data.transpose('time','lon','lat').to_netcdf(
        #                                                           self.out_path
        #                                                         + self.out_filename
        #                                                         )
        #         elif self.label=='S2SH':
        #
        #             data.transpose(
        #                         'member','step','time','lon','lat'
        #                         ).to_netcdf(self.out_path + self.out_filename)
        #
        #         elif self.label=='S2SF':
        #
        #             data.transpose(
        #                         'member','step','time', 'lon','lat'
        #                         ).to_netcdf(self.out_path + self.out_filename)
        #
        #         else:
        #             data.to_netcdf(self.out_path+self.out_filename)
        #
        #         self.download = False
        #
        #     return xr.open_dataset(self.out_path+self.out_filename)

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
