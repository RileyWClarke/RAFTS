from matplotlib.pyplot import fill
import numpy as np
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

def lamb_eff_md(band, temp, mdname, ff = 0.0, verbose=False):

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

    #Create BB
    BBwave = np.arange(1,12000,1)
    BBflux = make_bb(BBwave,temp,normed=BBnorm)

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

    #Import mdwarf spectrum
    f = mdwarf_interp(fname=mdname)
    interpolated_md = f(BBwave)
    
    #Slice BB
    BBfluxc = BBflux[BBleft:BBright]
    BBwavec = BBwave[BBleft:BBright]

    if verbose:
        print("Calculating BB at T = {} K".format(temp))
        print(BBfluxc)
    
    #Calc effective lambda
    w_eff = np.exp(np.sum( (interpolated_md[BBleft:BBright] + (ff * BBfluxc) ) * interpolated_filt[BBleft:BBright] * np.log(BBwavec)) / 
                   np.sum( (interpolated_md[BBleft:BBright] + (ff * BBfluxc) ) * interpolated_filt[BBleft:BBright]))
   
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
    BBwave = np.arange(1,12000,1)
    BBflux = make_bb(BBwave,temp,normed=BBnorm)

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

    if verbose:
        print("Calculating BB $T_{eff} = {}$".format(temp))
    
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

