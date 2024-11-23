import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS, FK5
from astropy.io import fits

import scipy
from scipy.interpolate import interp1d
from scipy.optimize import minimize

import rubin_sim.photUtils.Bandpass as Bandpass
import rubin_sim.photUtils.Sed as Sed

from mdwarf_interp import *
from config import *

import astropy.constants as const
import astropy.units as u

import warnings
#suppress warnings
warnings.filterwarnings('ignore')

SQ2 = np.sqrt(2)

def gaussian(x, A=1.0, mu=0.0, sigma=1.0):
    """
    Calculate the Gaussian function.

    Parameters:
    x (array-like): Input values where the Gaussian function will be evaluated.
    A (float): Amplitude of the Gaussian (default is 1.0).
    mu (float): Mean of the Gaussian (default is 0.0).
    sigma (float): Standard deviation of the Gaussian (default is 1.0).

    Returns:
    array: Values of the Gaussian function evaluated at x.
    """
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def lorentzian(x, A, x0, gamma):
    """
    Generate a Lorentzian function.

    Parameters:
    - x : array-like
        The input values (independent variable).
    - A : float
        The amplitude (height) of the peak.
    - x0 : float
        The center of the peak.
    - gamma : float
        The full-width at half-maximum (FWHM) of the peak.

    Returns:
    - f : array-like
        The Lorentzian function values.
    """
    return A / (1 + ((x - x0) / (gamma / 2)) ** 2)

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

def fitbb_to_m5(a, T, m5spec):
    bb = make_bb(WAVELENGTH, 3000) * 1e27 * a
    relevant_w = np.argmin(np.abs(WAVELENGTH - WMAX))
    indices = range(relevant_w-50, relevant_w)
    x = np.abs(((bb[indices] - m5spec[indices])).sum())
    return x

def gen_mdspec(mdname, filename, extended=True):

    mdf = mdwarf_interp(mdname)
    md = mdf(WAVELENGTH)

    if extended:
        amplitude = 1
        res = scipy.optimize.minimize(fitbb_to_m5, [amplitude], args=(3000, md))
        md[WAVELENGTH >= WMAX] = (make_bb(WAVELENGTH, 3000) * 1e27 * res.x)[WAVELENGTH >= WMAX]

    np.save(filename, md)

def compspec(temp, md, ff, balmer_ratio = 1, lines = None, lorentz_lines=False, linefrac=0.0, band='g', compplot=False):
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

    bandf = filt_interp(band=band)(WAVELENGTH)
    maxpt = np.where(bandf == bandf.max())[0][0]
    band_edges = (WAVELENGTH[bandf == bandf[(bandf > 0) * (WAVELENGTH < maxpt)].min()][0], 
                        WAVELENGTH[bandf == bandf[(bandf > 0) * (WAVELENGTH > maxpt)].min()][0])


    bb = make_bb(WAVELENGTH, temp) * globals.BBnorm
    sed_plain = np.copy(bb + md)
    ff = ff / globals.FF #change to "units" of 0.05

    balmer_step = np.ones_like(WAVELENGTH, dtype=float)
    balmer_step[WAVELENGTH < 3700] = balmer_ratio
    
    if lines is not None:
        
        line_s = 3
        sed_eng = sed_plain[band_edges[0]:band_edges[1]].sum()
        #print(sed_eng)
        lf = linefrac[0]
        nl = 2
        for i, line in enumerate(lines):
            
            if i == 2:
                lf = linefrac[1]
                nl = 3
            
            amp = np.sqrt(1 / (2 * np.pi * line_s**2)) * sed_eng * lf / nl
            #print(gaussian(WAVELENGTH, A=amp, mu=line, sigma=line_s).sum())
            bb += gaussian(WAVELENGTH, A=amp, mu=line, sigma=line_s)

    if lorentz_lines:

        sed_sum = sed_plain[band_edges[0]:band_edges[1]].sum()
        #print('Sum under blackbody = {}'.format(sed_sum))

        lf = linefrac[0]
        l = lorentzian(WAVELENGTH, *linedict[linenames[0]]) +  lorentzian(WAVELENGTH, *linedict[linenames[1]])
        lnew = l * (sed_sum / l.sum()) * lf 
        bb += lnew
        #print('Sum under Ca lines = {0} ({1}% of blackbody)'.format(lnew.sum(), (lnew.sum() / sed_sum)*100))

        lf = linefrac[1]
        l = lorentzian(WAVELENGTH, *linedict[linenames[2]]) +  lorentzian(WAVELENGTH, *linedict[linenames[3]]) + lorentzian(WAVELENGTH, *linedict[linenames[4]])
        lnew = l * (sed_sum / l.sum()) * lf 
        bb += lnew
        #print('Sum under H lines = {0} ({1}% of blackbody)'.format(lnew.sum(), (lnew.sum() / sed_sum)*100))

    return md + (bb * ff * balmer_step)

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
    lsst[band].readThroughput('/Users/riley/Desktop/RAFTS/baseline/total_' + band + '.dat')

    sb, w = lsst[band].sb, lsst[band].wavelen*10 #scale flux, conv nm to A

    if plotit:
        plt.plot(w,sb)

    return interp1d(w, sb, bounds_error=False, fill_value=0.0)

def lamb_eff_md(band, temp, mdpath = '/Users/riley/Desktop/RAFTS/sdsstemplates/m5.active.ha.na.k_ext.npy', ff=globals.FF, balmer_ratio = 1.0, 
                lines=None, lorentz_lines=False, linefrac=0.0, WAVELENGTH=WAVELENGTH, compplot=False, ax=None, ax2=None):

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
    mdspec = np.load(mdpath)
    mdbb = compspec(temp, md=mdspec, ff=ff, balmer_ratio=balmer_ratio, lines=lines, lorentz_lines=False, linefrac=linefrac, compplot=compplot)
    mdbb_lines = compspec(temp, md=mdspec, ff=ff, balmer_ratio=balmer_ratio, lines=lines, lorentz_lines=lorentz_lines, linefrac=linefrac, compplot=False)
    mdq = compspec(temp=0, md=mdspec, ff=ff, balmer_ratio=balmer_ratio, lines=lines, lorentz_lines=False, linefrac=linefrac, compplot=False)

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
    mdbb_band = mdbb[BBleft:BBright]
    mdbb_lines_band = mdbb_lines[BBleft:BBright]
    mdq_band = mdq[BBleft:BBright]
    wave_band = wave[BBleft:BBright]

    #if verbose:
        #print("Calculating BB at T = {} K".format(temp))
        
    #Calc effective lambda
    w_eff = np.exp(np.sum(mdbb_band * interpolated_filt[BBleft:BBright] * np.log(wave_band)) / 
                   np.sum(mdbb_band * interpolated_filt[BBleft:BBright]))
    
    w_eff_lines = np.exp(np.sum(mdbb_lines_band * interpolated_filt[BBleft:BBright] * np.log(wave_band)) / 
                   np.sum(mdbb_lines_band * interpolated_filt[BBleft:BBright]))
    
    w_effq = np.exp(np.sum(mdq_band * interpolated_filt[BBleft:BBright] * np.log(wave_band)) / 
                    np.sum(mdq_band * interpolated_filt[BBleft:BBright]))
    
    if compplot:

        ax.plot(WAVELENGTH, mdq, label="dM only")
        ax.plot(WAVELENGTH, mdbb, label="dM + blackbody")
        ax.plot(WAVELENGTH, mdbb_lines, label="dM + blackbody + lines")
        ax.set_xlabel(r'Wavelength $(\AA)$', fontsize=16)
        ax.set_ylabel(r'$F_\lambda$ (arb. units)', fontsize=16)
        ax.set_title(r'$T_{BB}$ = ' + '{0}K'.format(temp) + ', Ca line energy = {0:.1f}%, H line energy = {1:.1f}%'.format(linefrac[0] * 100, linefrac[1] * 100), fontsize=16)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)

        ax2.set_ylabel('Filter Throughput', fontsize=16)
        ax2.tick_params(axis ='y')
        ax2.yaxis.set_tick_params(labelsize=12)
        ax2.vlines(w_effq, 0, interpolated_filt[np.where(abs(WAVELENGTH - w_effq) == abs(WAVELENGTH - w_effq).min())[0][0]], 
                   color='C0', ls='--', label=r'$\lambda_{eff, quiescent}$')
        ax2.vlines(w_eff, 0, interpolated_filt[np.where(abs(WAVELENGTH - w_eff) == abs(WAVELENGTH - w_eff).min())[0][0]], 
                   color='C1', ls='--', label=r'$\lambda_{eff, flare}$')
        ax2.vlines(w_eff_lines, 0, interpolated_filt[np.where(abs(WAVELENGTH - w_eff_lines) == abs(WAVELENGTH - w_eff_lines).min())[0][0]], 
                   color='C2', ls='--', label=r'$\lambda_{eff, flare}$ (with lines)')

        ax2.plot(WAVELENGTH, interpolated_filt, color='k', alpha=0.4)

        ax.set_xlim(BBleft, BBright)
        ax.set_ylim(None, np.nanmax(mdbb_lines_band))
        #ax.legend()

    if lorentz_lines:
        return w_eff_lines
    else:
        return w_eff

def lamb_eff_BB(band, temp, verbose=False):

    """
    Calculates the effective wavelength in angstroms for BB sed

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

def R0(w_eff):
    #Docstring
    w_effn = np.copy(w_eff) / 1e4 #Convert angstrom to micron

    #Calc index of refr
    n = (10**-6 * (64.328 + (29498.1 / (146-(1/w_effn**2))) + (255.4 / (41 - (1/w_effn**2))))) + 1

    #Calc R_0
    return (n**2 - 1) / (2 * n**2)

def dcr_offset(w_eff, airmass, coord = None, header = None, chrDistCorr=False):

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

    #w_effn = np.copy(w_eff) / 1e4 #Convert angstrom to micron

    #Calc index of refr
    #n = (10**-6 * (64.328 + (29498.1 / (146-(1/w_effn**2))) + (255.4 / (41 - (1/w_effn**2))))) + 1

    #Calc R_0
    #R_0 = (n**2 - 1) / (2 * n**2)

    R_0 = R0(w_eff)

    Z = np.arccos(1/airmass)

    R = R_0*np.tan(Z)

    if chrDistCorr:

        corr = chrDistCorr(w_eff, coord, header)

        return np.rad2deg(R) * 3600 * corr

    return np.rad2deg(R) * 3600 

def dcr_offset_inverse(w_eff_1, w_eff_0, dcr):

    q = np.deg2rad(dcr / 3600)

    R0_1 = R0(w_eff_1)
    R0_0 = R0(w_eff_0)

    z_crit = np.arctan(q / (R0_1 - R0_0))

    return 1 / np.cos(z_crit)


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    #plt.register_cmap(cmap=newcmap)

    return newcmap

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

def srcext(file, det_thresh, ana_thresh, catname, ext_number=0):
    #print(os.getcwd())
    os.chdir('srcext')
    print('Making SExtractor catalog of '+file+'...')
    #print(os.path.isfile(file))
    #print(os.getcwd())

    if os.path.isfile(catname) == True:
        print('This catalogue already exists, moving on...')
    else:
        os.system('sex ' + file + ' -c default.sex' + ' -DETECT_THRESH ' + str(det_thresh) + ' -ANALYSIS_THRESH ' 
                  + str(ana_thresh) + ' -CATALOG_NAME ' + str(catname))

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

def calc_zenith(date, site):

    mtn = EarthLocation.of_site(site)
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

def pa(h, phi, d):

    '''
    PA equation from Astronomical Algorithms

    Parameters
    -------------
    h: float
        hour angle in hours
    phi: float
        Geographic latitude of observatory in degrees
    d: float
        Declination in degrees

    Returns
    -------------
    float
        Parallactic angle in degrees
    '''

    q = np.arctan2(np.sin(h * ha2deg * deg2rad), \
        np.cos(d * deg2rad) * np.tan(phi * deg2rad) - \
        np.sin(d * deg2rad) * np.cos(h * ha2deg * deg2rad))

    return q / deg2rad

def celest_to_pa(ra, dec, time, loc, delra=None, deldec=None, round_lmst = False, verbose = False):

    '''
    Convert celestial coordinates to a parallactic angle given
    a observation time and observatory location

    Parameters
    -------------
    ra: float
        Right Ascension in degrees
    dec: float
        Declination in degrees
    time: float
        astropy.time.Time object
    location: astropy.coordinates.EarthLocation object
        EarthLocation object of observing site

    Returns
    -------------
    astropy.Quantity object
        Parallactic angle quantity
    '''

    t = time
    lat = loc.lat.deg
    lon = loc.lon.deg
    scoord = SkyCoord(ra=ra * u.deg, dec = dec * u.deg)
    lst = t.sidereal_time('mean', longitude=lon)

    if round_lmst:
        lst = (lst * 60).round() / 60

    ha = lst.hour - scoord.ra.hour

    if verbose:
        print('Location = Lon:{0:.3f}, Lat:{1:.3f}'.format(loc.lon, loc.lat))
        print('RA = {0}, Dec = {1}'.format(scoord.ra.hms, scoord.dec.dms))
        print('time = {}'.format(t))
        print('LMST = {}'.format(lst.hms))
        print('ha = {}'.format(ha))

    if delra is not None and deldec is not None:

        delh = ha2deg * delra
        dpa = pa_error(ha, dec, loc.lat.deg, delh, deldec)

        return pa(ha, lat, dec), dpa
    
    else:

        return pa(ha, lat, dec) 

def celest_to_ha(ra, dec, time, loc, round_lmst = False, verbose = False):

    '''
    Convert celestial coordinates to a parallactic angle given
    a observation time and observatory location

    Parameters
    -------------
    ra: float
        Right Ascension in degrees
    dec: float
        Declination in degrees
    time: float
        astropy.time.Time object
    location: astropy.coordinates.EarthLocation object
        EarthLocation object of observing site

    Returns
    -------------
    astropy.Quantity object
        Parallactic angle quantity
    '''

    t = time
    lat = loc.lat.deg
    lon = loc.lon.deg
    scoord = SkyCoord(ra=ra * u.deg, dec = dec * u.deg)
    lst = t.sidereal_time('mean', longitude=lon)

    if round_lmst:
        lst = (lst * 60).round() / 60

    ha = lst.hour - scoord.ra.hour

    if verbose:
        print('Location = Lon:{0:.3f}, Lat:{1:.3f}'.format(loc.lon, loc.lat))
        print('RA = {0}, Dec = {1}'.format(scoord.ra.hms, scoord.dec.dms))
        print('time = {}'.format(t))
        print('LMST = {}'.format(lst.hms))
        print('ha = {}'.format(ha))
    return ha

def dpar(dra, ddec, pa2, delra = None, deldec = None, delpa2 = None):

    '''
    Compute component of positional offset parallel to zenith direction

    Parameters
    -------------
    dra: float
        Change in right Ascension in degrees
    ddec: float
        Change in declination in degrees
    pa2: float
        Parallactic angle of second position in degreees

    Returns
    -------------
    float 
        zenith-parallel component
    '''

    dparallel = np.sqrt(dra**2 + ddec**2) * np.cos((np.pi/2) - np.deg2rad(pa2) - np.arctan2(ddec, dra))

    if delra is not None and deldec is not None and delpa2 is not None:

        dparallel_err, ddra, dddec, ddpar2 = dpar_error(dra, ddec, pa2, delra, deldec, delpa2)

        return dparallel, dparallel_err, ddra, dddec, ddpar2
    
    else:

        return dparallel

def dtan(dra, ddec, pa2):

    '''
    Compute component of positional offset perpendicular to zenith direction

    Parameters
    -------------
    dra: float
        Change in right Ascension in degrees
    ddec: float
        Change in declination in degrees
    pa2: float
        Parallactic angle of second position in degreees

    Returns
    -------------
    float 
        zenith-parallel component
    '''

    return np.sqrt(dra**2 + ddec**2) * np.sin((np.pi/2) - np.deg2rad(pa2) - np.arctan(ddec/dra))

def gcd(lat1, lat2, lon1, lon2, haversine=False):
    dlat = np.abs(lat2 - lat1)
    dlon = np.abs(lon2 - lon1)
    dsig = np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon))
    
    if haversine:
        dsig = 2 * np.arcsin(np.sqrt(np.sin(dlat / 2))**2 + (1 - np.sin(dlat/2)**2 - np.sin((lat1 + lat2) / 2)**2) * np.sin(dlon / 2)**2)
        
    return dsig

def pa_error(h, dec, phi, dh, ddec):

    h = h * ha2deg * deg2rad
    dec = dec * deg2rad
    phi = phi * deg2rad
    dh = dh * deg2rad
    ddec = ddec * deg2rad

    dPdh = (-(np.cos(dec) * np.cos(h) * np.tan(phi)) + (np.sin(dec) * np.sin(h)**2) + (np.sin(dec) * np.cos(h)**2)) / ((-2 * np.sin(dec) * np.cos(dec) * np.cos(h) * np.tan(phi)) + (np.sin(dec)**2 * np.cos(h)**2) + (np.cos(dec)**2 * np.tan(phi)**2) + np.sin(h)**2)
    
    dPdd = (np.sin(h) * (np.cos(h) * np.cos(dec) + np.sin(dec) * np.tan(phi))) / ((np.cos(h) * np.sin(dec) - np.cos(dec) * np.tan(phi))**2 + np.sin(h)**2)

    err = np.sqrt( (dPdh * np.rad2deg(dh))**2 + (dPdd * np.rad2deg(ddec))**2 )

    return err

def dpar_error(dra, ddec, pa2, delra, deldec, delpa2):

    ddpar_ddra = (dra * np.sin(np.arctan(ddec/dra) + np.deg2rad(pa2)) - ddec * np.cos(np.arctan(ddec/dra) + np.deg2rad(pa2))) / np.sqrt(dra**2 + ddec**2)
    ddpar_dddec = (ddec * np.sin(np.arctan(ddec/dra) + np.deg2rad(pa2)) + dra * np.cos(np.arctan(ddec/dra) + np.deg2rad(pa2))) / np.sqrt(dra**2 + ddec**2)
    ddpar_dpa2 = np.sqrt(dra**2 + ddec**2) * np.cos(np.arctan(ddec/dra) + np.deg2rad(pa2))

    err = np.sqrt( (ddpar_ddra * delra)**2 + (ddpar_dddec * deldec)**2 + (ddpar_dpa2 * delpa2)**2 )

    return err, ddpar_ddra, ddpar_dddec, ddpar_dpa2

def obj(T, weff_0=4841.425781369825):

    manual_linefrac = [0.50, 0.50]

    weff = lamb_eff_md(band = 'g', temp = T, mdpath = '/Users/riley/Desktop/RAFTS/sdsstemplates/m7.active.ha.na.k_ext.npy', lorentz_lines=True, linefrac=manual_linefrac)
    
    return abs(weff_0 - weff)

Nfeval = 1

def callbackF(Xi):
    global Nfeval
    print(Nfeval, obj(Xi), Xi)
    Nfeval += 1

def inverse_Teff(delta_dcr, quiescent_dcr, airmass, callback=False, return_weff = False):

    dcr_f = delta_dcr + quiescent_dcr

    R_0 = dcr_f / np.rad2deg(np.tan(np.arccos(1 / airmass)))

    n = 1 / np.sqrt(1 - (2 * R_0))
    ir_factor = ((1.4965 * n - 1.496907944477) * 1e3)
    weff = np.sqrt((5 * np.sqrt(7) * 
                    np.sqrt(3.9375 * n**2 - 7.8776997855 * n + 3.940200259046195407)) / ir_factor
                    #np.sqrt(3_937_500_000_000_000_000 * n**2 - 7_877_699_785_500_000_000 * n + 3_940_200_259_046_195_407)) / 
                   #(1_496_500_000_000 * n - 1_496_907_944_477) 
                   + (46.75 * n) / ir_factor - 46.760445709 / ir_factor) / SQ2
    weff *= 1e4

    init_guess = 2800.0
    if callback:
        result = minimize(obj, init_guess, args=weff, callback=callbackF, method='Nelder-Mead', options = {'disp':True, 'gtol':1e-2})
    else:
        result = minimize(obj, init_guess, args=weff, method='Nelder-Mead', options = {'gtol':1e-2})

    if return_weff:
        return result.x, weff
    else:
        return result.x 

###DMTN-037 refraction calculations

def R(l, Z):

    chi = CHI
    beta = BETA
    n = n_0(l)

    return (chi * (n - 1) * (1 - beta) * np.tan(np.deg2rad(Z)) - chi * (1 - n) * (beta - ((n - 1) / 2)) * np.tan(np.deg2rad(Z))**3) * 3600

def n_0(l):

    sigma = 1e4 / l
    dn_s = (2371.34 + (683939.7 / (130 - sigma**2)) + (4547.3 / (38.9 - sigma**2))) * D_S * 1e-8
    dn_w = (6487.31 + 58.058 * sigma**2 - 0.7115 * sigma**4 + 0.08851 * sigma**6) * D_W * 1e-8

    return 1 + dn_s + dn_w

###

def chrDistCorr(wavelength, coord, header):

    source = np.array([coord.ra.value, coord.dec.value])
    zenith = np.zeros_like(source)
    center = np.zeros_like(source)

    zenith[0] = SkyCoord(AltAz(alt=90 * u.degree, az=0 * u.degree, obstime = coord.obstime, location=EarthLocation.of_site('Cerro Tololo'))).transform_to(ICRS()).ra.value
    zenith[1] = SkyCoord(AltAz(alt=90 * u.degree, az=0 * u.degree, obstime = coord.obstime, location=EarthLocation.of_site('Cerro Tololo'))).transform_to(ICRS()).dec.value
    center[0] = header['CENTRA']
    center[1] = header['CENTDEC']

    a = gcd(np.deg2rad(center[1]), np.deg2rad(zenith[1]), np.deg2rad(center[0]), np.deg2rad(zenith[0]))
    b = gcd(np.deg2rad(source[1]), np.deg2rad(center[1]), np.deg2rad(source[0]), np.deg2rad(center[0]))
    c = gcd(np.deg2rad(zenith[1]), np.deg2rad(source[1]), np.deg2rad(zenith[0]), np.deg2rad(source[0]))

    A = np.arccos( (np.cos(a) - np.cos(b) * np.cos(c))  / (np.sin(b) * np.sin(c)) ) 

    theta = np.pi - A

    pixpermm = 153 / 2.3
    arcsecperpix = 0.2637

    new_w = np.arange(batoid_trace[0][0],batoid_trace[0][-1],1)
    f = interp1d(x = batoid_trace[0], y=batoid_trace[1] * 1e-3 * pixpermm * arcsecperpix, kind='quadratic')

    dist_mag = f(new_w)[np.where(abs(new_w - wavelength) == abs(new_w - wavelength).min())[0]]

    return dist_mag * np.cos(theta)

    