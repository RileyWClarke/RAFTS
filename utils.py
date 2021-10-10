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

def filt_interp(band):

    """
    Imports and interpolates LSST filter

    Parameters
    -----------
    band: string
        which LSST band to use (u,g,r,i,z,y)

    Returns
    -----------
    interp1d(filtx,filty,fill_value = 0.0) : function
        function that interpolates filtx,filty
    """


def md_lamb_eff(band, temp, mdname, ff = 0.0):

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
        effective wavelength
    """

def lamb_eff_BB(band, temp):

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