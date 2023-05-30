import sys

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from utils import *

#Read in candidate_objects.dat, exposures.dat
dcdf = pd.read_table(sys.argv[1], names=['field',
'candidate id',
'object id',
'object ra', 
'object dec', 
'object MJD', 
'object filter', 
'object magnitude', 
'object magnitude error', 
'object real/bogus score', 
'exposure id'], comment='#', delim_whitespace=True)

#Assert all object IDs are unique
assert dcdf.shape[0] == dcdf['object id'].unique().shape[0]

expdf = pd.read_table(sys.argv[2], names=[
'field',
'exposure calendar date',
'filename base', 
'exposure id', 
'modified julian date', 
'filter', 
'mean limiting magnitude',
'mean seeing',
'mean sky background',
'total number of objects',
'total number of objects with R/B>0.6', 
'airmass',
], usecols=['exposure id', 'airmass', 'mean seeing'], comment='#', delim_whitespace=True)

#Merge dataframes
dcdfnew = pd.merge(dcdf, expdf, on=['exposure id'], how='left')

#Add datetime column
obj_dt = []

for i,time in enumerate(dcdfnew['object MJD'].values):
    t = Time(time, format='mjd')
    obj_dt.append(t.datetime)

dcdfnew['object datetime'] = obj_dt

#Add parallactic angles
pa_arr = np.zeros_like(dcdfnew['object id'], dtype=float)

for i,id in enumerate(dcdfnew['object id'].unique()):
    obj_ra = dcdfnew[dcdfnew['object id'] == id]['object ra'].values
    obj_dec = dcdfnew[dcdfnew['object id'] == id]['object dec'].values
    obj_mjd = dcdfnew[dcdfnew['object id'] == id]['object MJD'].values

    obj_pa = celest_to_pa(ra = obj_ra, dec = obj_dec, time = Time(obj_mjd, format='mjd'), loc = EarthLocation.of_site('Cerro Tololo'))
    pa_arr[i] = obj_pa

dcdfnew['parallactic angle'] = pa_arr

#Write file
dcdfnew.to_csv('{}/ddf_flares.csv'.format(os.path.abspath(os.getcwd())))