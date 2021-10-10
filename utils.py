import numpy as np
from scipy.interpolate import interp1d

import rubin_sim.photUtils.Bandpass as Bandpass
import rubin_sim.photUtils.Sed as Sed

from mdwarf_interp import *

import astropy.constants as const
import astropy.units as u

def make_bb(wavelengths, temp, normed = -1.0):

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

def filt_interp(band, x_new):

    """
    Imports and interpolates LSST filter

    Parameters
    -----------
    band: string
        which LSST band to use (u,g,r,i,z,y)

    x_new: ndarray
        the array to interpolate on

    Returns
    -----------
    filt_array(wavelens) : function
        function that evaluates the interpolated filter
        over the specified wavelengths
    """

def filt_array(wavelens):

    """
    Imports LSST filter

    Parameters
    -----------
    wavelens : ndarray
        wavelengths to generate the filter over

    Returns
    -----------
    sb : ndarray
        filter throughput
    w : ndarray
        filter wavelengths
    s_left : float
        wavelength of the left side of band
    s_right : float
        wavelength of the right side of band
    """

def md_lamb_eff(band, temp, mdname, ff = 0.0, normT = -1.0):

    """
    Calculates the DCR offset in arcsec for md + BB sed

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

    normT: float
        normalization temperature. Blackbody peak flux normalized to 1.0 at this temp

    Returns
    -----------
    float
        effective wavelength
    """

def lamb_eff_BB(band, temp):

    """
    Calculates the DCR offset in arcsec for md + BB sed

    Parameters
    -----------
    band: string
        which LSST band to use

    temp: float
        BB effective temperature

    normT: float
        normalization temperature. Blackbody peak flux normalized to 1.0 at this temp

    Returns
    -----------
    float
        effective wavelength
    """

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