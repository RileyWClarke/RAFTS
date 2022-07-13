import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
from astropy.io import fits

from astroquery.gaia import Gaia

from scipy.interpolate import interp1d

import rubin_sim.photUtils.Bandpass as Bandpass
import rubin_sim.photUtils.Sed as Sed

from mdwarf_interp import *
from config import *

import astropy.constants as const
import astropy.units as u

def make_bb(wavelengths, temp, normed = 1.0):

    """
    Creates blackbody spectrum

    Parameters
    -----------
    temp: float
        Effective temperature of blackbody
    wavelenths: ndarray
        wavelength array of blackbody
    normed : float
        blackbodies are normalized to 1.0 at this value

    Returns
    -----------
    ndarray:
        flux array of blackbody
    """
    h = (const.h).cgs
    c = (const.c).cgs
    k = (const.k_B).cgs
    
    l = u.Quantity(wavelengths, unit=u.angstrom)

    T = u.Quantity(temp, unit=u.K)

    F_lambda = (((2*h*c**2)/l**5) * (1/(np.exp((h*c)/(l*k*T)) - 1)))
    
    return F_lambda.value * normed

def sed_integ(w, f):
    return np.nansum(f) / np.nanmean(np.diff(w))

import globals
globals.initialize()

def compspec(temp, mdname, ff, compplot=False):
    """
    Creates composite blackbody + m dwarf spectrum

    Parameters
    -----------
    temp: int
        blackbody temperature in Kelvin

    mdname: str
        mdwarf spectrum .fits filename

    ff: float
        blackbody fill factor

    Returns
    -----------
    ndarray:
        composite spectrum
    """

    bb = make_bb(WAVELENGTH, temp) * globals.BBnorm
    mdf = mdwarf_interp(mdname)
    md = mdf(WAVELENGTH)

    ff = ff / globals.FF #change to "units" of 0.05

    if compplot:
        plt.figure()
        plt.plot(WAVELENGTH, md, label="dM5e Template Only")
        plt.plot(WAVELENGTH, md + (bb * ff), label="Composite Spectrum")
        plt.xlabel(r'Wavelength $(\AA)$')
        plt.ylabel(r'$F_\lambda$ (arb. units)')
        plt.title(r'Composite Spectrum w/ {0}K BB & $ff$ = {1} * 0.05'.format(temp,ff))
        plt.legend()
  
    return md + (bb * ff)

def filt_interp(band,plotit=False):

    """
    Imports and interpolates LSST filter

    Parameters
    -----------
    band: string
        which LSST band to use (u,g,r,i,z,y)

    Returns
    -----------
    interp1d(filtx,filty, bound_error = False, fill_value = 0.0) : function
        function that interpolates filtx,filty
    """

    lsst = {}
    lsst[band] = Bandpass()
    lsst[band].readThroughput('baseline/total_' + band + '.dat')

    sb, w = lsst[band].sb, lsst[band].wavelen*10 #scale flux, conv nm to A

    if plotit:
        plt.plot(w,sb)

    return interp1d(w, sb, bounds_error=False, fill_value=0.0)

def lamb_eff_md(band, temp, ff=globals.FF, WAVELENGTH=WAVELENGTH, mdonly=False, compplot=False):

    """
    Calculates the effective wavelength in arcsec for md + BB sed

    Parameters
    -----------
    band: string
        which LSST band to use

    temp: float
        BB effective temperature

    mdname: string
       filename of mdwarf spectra .fits file

    ff: float
        filling factor for mdwarf + (BB*ff)

    Returns
    -----------
    float
        effective wavelength in Angstroms
    """

    #Create composite spectrum
    wave = WAVELENGTH
    mdbb = compspec(temp, mdname=MDSPEC, ff=ff, compplot=compplot)

    if mdonly:
        mdbb = compspec(temp, mdname=MDSPEC, ff=0.0)
    
    #Import filter
    f = filt_interp(band=band)
    interpolated_filt = f(wave)

    #Create left slice ind
    for i,s in enumerate(interpolated_filt):
        if s == 0:
            pass
        else:
            s_left = wave[i]
            break
        
    #Create right slice ind
    for i,s in reversed(list(enumerate(interpolated_filt))):
        if s == 0:
            pass
        else:
            s_right = wave[i]
            break

    #take slice where band is non-zero
    BBleft = np.where(np.abs(wave - s_left) == np.abs(wave - s_left).min())[0][0]
    BBright = np.where(np.abs(wave - s_right) == np.abs(wave - s_right).min())[0][0]
    
    #Slice spectrum
    mdbb = mdbb[BBleft:BBright]
    wave = wave[BBleft:BBright]

    #if verbose:
        #print("Calculating BB at T = {} K".format(temp))
        
    #Calc effective lambda
    w_eff = np.exp(np.sum(mdbb * interpolated_filt[BBleft:BBright] * np.log(wave)) / 
                   np.sum(mdbb * interpolated_filt[BBleft:BBright]))
   
    return w_eff

def lamb_eff_BB(band, temp, verbose=False):

    """
    Calculates the effective wavelength in arcsec for BB sed

    Parameters
    -----------
    band: string
        which LSST band to use

    temp: float
        BB effective temperature

    Returns
    -----------
    float
        effective wavelength in Angstroms
    """

    #Create BB
    BBwave = WAVELENGTH
    BBflux = make_bb(BBwave, temp) / globals.BBnorm

    #Import filter
    f = filt_interp(band=band)
    interpolated_filt = f(BBwave)

    #Create left slice ind
    for i,s in enumerate(interpolated_filt):
        if s == 0:
            pass
        else:
            s_left = BBwave[i]
            break
        
    #Create right slice ind
    for i,s in reversed(list(enumerate(interpolated_filt))):
        if s == 0:
            pass
        else:
            s_right = BBwave[i]
            break

    #take slice where band is non-zero
    BBleft = np.where(np.abs(BBwave - s_left) == np.abs(BBwave - s_left).min())[0][0]
    BBright = np.where(np.abs(BBwave - s_right) == np.abs(BBwave - s_right).min())[0][0]
    
    #Slice BB
    BBfluxc = BBflux[BBleft:BBright]
    BBwavec = BBwave[BBleft:BBright]

    #if verbose:
        #print("Calculating w_eff")
    
    #Calc effective lambda
    w_eff = np.exp(np.sum( BBfluxc * interpolated_filt[BBleft:BBright] * np.log(BBwavec)) / 
                   np.sum( BBfluxc * interpolated_filt[BBleft:BBright]))
   
    return w_eff

def dcr_offset(w_eff, airmass):

    """
    Calculates the DCR offset in arcsec

    Parameters
    -----------
    w_eff: float
        effective wavelength

    airmass: float
        airmass value

    Returns
    -----------
    float
        DCR offset in arcsec
    """

    w_effn = np.copy(w_eff) / 1e4 #Convert angstrom to micron

    #Calc index of refr
    n = (10**-6 * (64.328 + (29498.1 / (146-(1/w_effn**2))) + (255.4 / (41 - (1/w_effn**2))))) + 1

    #Calc R_0
    R_0 = (n**2 - 1) / (2 * n**2)

    #Calc Z
    Z = np.arccos(1/airmass)

    R = R_0*np.tan(Z)

    return np.rad2deg(R) * 3600

def getsciimg(filefracday,paddedfield,filtercode,paddedccdid,imgtypecode,qid):

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

def getgaiacat(sra,sdec,srad,verb=2):
    
    Gaia.ROW_LIMIT = 200 
    coord = SkyCoord(ra=sra, dec=sdec, unit=(u.degree, u.degree), frame='icrs')
    radius = u.Quantity(srad, u.deg)
    j = Gaia.cone_search_async(coord, radius)
    r = j.get_results()
   
    gaia_df = pd.DataFrame({'ra':r['ra'], 'dec':r['dec'], 'gmag':r['phot_g_mean_mag'], 'rmag':r['phot_rp_mean_mag']})

    return gaia_df

def srcext(file, det_thresh, ana_thresh, catname):
    #print(os.getcwd())
    os.chdir('srcext')
    print('Making SExtractor catalog of '+file+'...')
    #print(os.path.isfile(file))
    #print(os.getcwd())

    if os.path.isfile(catname) == True:
        print('This catalogue already exists, moving on...')
    else:
        os.system('sex ' + file + ' -c default.sex' + ' -DETECT_THRESH ' + str(det_thresh) + ' -ANALYSIS_THRESH ' + str(ana_thresh) + ' -CATALOG_NAME ' + str(catname))

    cata_df = pd.read_table(catname, names=['NUMBER',
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

    print("Matching catalogs...")
    c1 = SkyCoord(ra=cat1["ALPHA_SKY"]*u.degree, dec=cat1["DELTA_SKY"]*u.degree)
    c2 = SkyCoord(ra=cat2["ALPHA_SKY"]*u.degree, dec=cat2["DELTA_SKY"]*u.degree)

    if len(c1) < len(c2):
        idx, d2d, d3d = c1.match_to_catalog_sky(c2)
    
    else:
        idx, d2d, d3d = c2.match_to_catalog_sky(c1)

    return idx, d2d, d3d

def xmatch_gaia(cat1, cat2):

    c1 = SkyCoord(ra=cat1["ALPHA_SKY"]*u.degree, dec=cat1["DELTA_SKY"]*u.degree)
    c2 = SkyCoord(ra=cat2["ra"].values*u.degree, dec=cat2["dec"].values*u.degree)
    idx, d2d, d3d = c1.match_to_catalog_sky(c2)

    return idx, d2d, d3d

def circle_cut(imgcen_ra, imgcen_dec, cat, radius):
    '''
    Returns indices from cat within radius of imgcenter
    '''
    
    c1 = SkyCoord(ra=imgcen_ra*u.degree, dec=imgcen_dec*u.degree)
    c2 = SkyCoord(ra=cat["ra"].values*u.degree, dec=cat["dec"].values*u.degree)
    ind = c1.separation(c2) < radius

    return ind

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
    ax.scatter(d_ra, d_dec, alpha=0.75)
    ax.scatter(d_ra[flr_ind], d_dec[flr_ind], color='red', s=100, marker='*', label='Flare star')

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

def plot_shifts_gaia(ref_ra, ref_dec, sci_ra, sci_dec, zen_ra, zen_dec, flr_ind, mjd, am, circ_inds, colors, centered=False):

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
    inners = ax.scatter(d_ra,d_dec, c = colors, cmap='RdBu_r', alpha=0.75)
    #outers = ax.scatter(d_ra[~circ_inds],d_dec[~circ_inds], facecolors='none', edgecolors='black', label='peripheral (>0.5 deg)', alpha=0.5)
    ax.scatter(d_ra[flr_ind], d_dec[flr_ind], color='chartreuse', s=100, marker='*', label='Flare star')

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

    cbar = plt.colorbar(inners)
    cbar.set_label('Gaia g-rp (mag)', rotation=270, labelpad=20)

    ax.set_xlabel(r"$\Delta$ RA (arcsec)", labelpad=150)
    ax.set_ylabel(r"$\Delta$ Dec (arcsec)", labelpad=150)
    ax.set_xticks([-0.4,-0.2,0.2,0.4])
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    ax.set_title("MJD: {0}, Airmass = {1}".format(mjd, am))
    ax.legend()
    ax.grid(False)
    
    plt.gca().set_aspect('equal')