import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.wcs import WCS

def getsciimg(filefracday,paddedfield,filtercode,paddedccdid,imgtypecode,qid, flip=True):

    year = filefracday[0:4]
    month = filefracday[4:6]
    day = filefracday[6:8]
    fracday = filefracday[8:]

    url = 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/'+year+'/'+month+day+'/'+fracday+'/ztf_'+filefracday+'_'+paddedfield+'_'+filtercode+'_c'+paddedccdid+'_'+imgtypecode+'_q'+qid+'_'+'sciimg.fits'

    hdu = fits.open(url)

    hdu.writeto('srcext/'+filefracday+'_'+paddedfield+'_sciimg'+'.fits', overwrite=True)

    if flip:
        with fits.open('srcext/'+filefracday+'_'+paddedfield+'_sciimg'+'.fits', mode='update') as hdu:
            flipdata = np.fliplr(np.copy(hdu[0].data))
            hdu[0].data = flipdata


def getrefimg(paddedfield,filtercode,paddedccdid,qid):

    fieldprefix = paddedfield[0:3]

    url = 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/ref/'+fieldprefix+'/field'+paddedfield+'/'+filtercode+'/ccd'+paddedccdid+'/q'+qid+'/ztf_'+paddedfield+'_'+filtercode+'_c'+paddedccdid+'_q'+qid+'_refimg.fits'

    hdu = fits.open(url)

    hdu.writeto('srcext/'+str(paddedfield)+'_refimg'+'.fits', overwrite=True)


def srcext(file):

    os.chdir('srcext')
    print(file)
    print(os.path.isfile(file))
    print(os.getcwd())
    os.system('sex ' + file + ' -c default.sex')

    cata_df = pd.read_table('out.cat', names=['NUMBER',
    'X_IMAGE',
    'Y_IMAGE',
    'XWIN_IMAGE',
    'YWIN_IMAGE',
    'XMODEL_IMAGE',
    'YMODEL_IMAGE',
    'FLUX_AUTO',
    'FLUX_MODEL',
    'MAG_AUTO',
    'MAG_MODEL',
    'FLUX_RADIUS',
    'FLAGS',
    'NITER_MODEL',
    'ALPHA_SKY',
    'DELTA_SKY',
    'THETA_WORLD',
    'ELLIPTICITY'], index_col=0, comment='#', delim_whitespace=True)

    os.chdir('..')

    return cata_df

def xmatch(cat1, cat2):

    c1 = SkyCoord(ra=cat1["ALPHA_SKY"]*u.degree, dec=cat1["DELTA_SKY"]*u.degree)
    c2 = SkyCoord(ra=cat2["ALPHA_SKY"]*u.degree, dec=cat2["DELTA_SKY"]*u.degree)
    idx, d2d, d3d = c1.match_to_catalog_sky(c2)

    return idx, d2d, d3d