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
    print("Querying: "+url)
    hdu = fits.open(url)
    hdu.writeto('srcext/'+filefracday+'_'+paddedfield+'_sciimg'+'.fits', overwrite=True)



def getrefimg(paddedfield,filtercode,paddedccdid,qid):

    fieldprefix = paddedfield[0:3]

    url = 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/ref/'+fieldprefix+'/field'+paddedfield+'/'+filtercode+'/ccd'+paddedccdid+'/q'+qid+'/ztf_'+paddedfield+'_'+filtercode+'_c'+paddedccdid+'_q'+qid+'_refimg.fits'
    print("Querying: "+url)
    hdu = fits.open(url)
    hdu.writeto('srcext/'+str(paddedfield)+'_refimg'+'.fits', overwrite=True)


def srcext(file, det_thresh, ana_thresh):
    print(os.getcwd())
    os.chdir('srcext')
    print(file)
    #print(os.path.isfile(file))
    print(os.getcwd())
    os.system('sex ' + file + ' -c default.sex' + ' -DETECT_THRESH ' + str(det_thresh) + ' -ANALYSIS_THRESH ' + str(ana_thresh))

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

def calc_zenith(date):

    mtn = EarthLocation.of_site('Palomar')
    mjd = Time(date, format='mjd')

    zenith = SkyCoord(AltAz(alt=90 * u.degree, az=0 * u.degree, obstime = mjd, location=mtn)).transform_to(ICRS())

    return zenith

def plot_shifts(ref_ra, ref_dec, sci_ra, sci_dec, zen_ra, zen_dec, flr_ind, mjd, am, centered=False):

    #Calc delta coords
    d_ra = (ref_ra - sci_ra) * 3600
    d_dec = (ref_dec - sci_dec) * 3600

    #Calc zenith delta coord
    d_zra = (ref_ra.mean() - zen_ra) * 3600
    d_zdec = (ref_dec.mean() - zen_dec) * 3600

    #Calc centroid
    centroid = (sum(d_ra) / len(d_ra), sum(d_dec) / len(d_dec))

    #Plotting
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(d_ra,d_dec, alpha=0.5)
    ax.scatter(d_ra[flr_ind], d_dec[flr_ind], color='r', label='Flare star')

    if not centered:
        ax.scatter(centroid[0], centroid[1], color='k', marker='x', label="Centroid")

    ax.plot([0,d_zra],[0,d_zdec], c='gray',ls='--', label="to zenith")

    # Move left y-axis and bottim x-axis to centre, passing through (0,0)

    if centered:
        ax.spines['left'].set_position(('data', centroid[0]))
        ax.spines['bottom'].set_position(('data', centroid[1]))
    else:
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xlabel(r"$\Delta$ RA (arcsec)", labelpad=150)
    ax.set_ylabel(r"$\Delta$ Dec (arcsec)", labelpad=150)
    ax.set_xticks([-0.4,-0.2,0.2,0.4])
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    ax.set_title("MJD: {0}, Airmass = {1}".format(mjd, am))
    ax.legend()
    ax.grid(False)
    plt.gca().set_aspect('equal')
