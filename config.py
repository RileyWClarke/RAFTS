import numpy as np
from astropy.coordinates import EarthLocation
WAVELENGTH = np.arange(0,12001,1)
WMIN = 3825
WMAX = 9200
MDSPEC = 'm5.active.ha.na.k.fits'
AMS = np.linspace(1.05,1.2,num=6,dtype='float')

deg2rad = np.pi / 180
ha2deg = 15.0

def get_img_string(key):
    codes={'flare1ep1':'20200206287824000819zg13o2',
           'flare1ep3':'20200206302627000819zg13o2',
           'flare2ep1':'20190418224780000864zg08o3',
           'flare3ep1':'20180402425405000864zg10o2'
    }
    return codes[key]

def get_flr_coord(key):
    coords = {'flare1':[186.09002468556912, 65.31234685437119],
              'flare2':[239.9337200957272, 75.04305873103452],
              'flare3':[255.0794984723742, 78.31401820936068]}
    return coords[key]