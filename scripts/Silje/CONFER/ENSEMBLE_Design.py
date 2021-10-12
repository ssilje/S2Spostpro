import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import xskillscore as xs
import time as time_lib
import properscoring as ps
from datetime import date, datetime, timedelta, timezone



import csv


def read_csv(data_dir,
             csv_name):
    file = open(data_dir + '/' + csv_name)
    csvreader = csv.reader(file)
    header = next(csvreader)
    print(header)
    csv_file = []
    for row in csvreader:
            csv_file.append(row)

    file.close()

    return csv_file


IODfcs = read_csv(data_dir = '/nird/projects/NS9853K/DATA/ENS_DESIGN',
             csv_name = 'IOD_fcs.csv')


IODdt = read_csv(data_dir = '/nird/projects/NS9853K/DATA/ENS_DESIGN',
             csv_name = 'IOD_dt.csv')

OBSdt = read_csv(data_dir = '/nird/projects/NS9853K/DATA/ENS_DESIGN',
             csv_name = 'obs_dt.csv')
precdt = read_csv(data_dir = '/nird/projects/NS9853K/DATA/ENS_DESIGN',
             csv_name = 'prec_dt.csv')

##data_dir = '/nird/projects/NS9853K/DATA/ENS_DESIGN'
#csv_name = 'IOD_fcs.csv'



