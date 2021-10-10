import os
import matplotlib.pyplot as plt
import numpy as np

from make_bb import *
from filt_interp import *
from mdwarf_interp import *

def lamb_eff(band, temp, mdname, ff = 0.0, normT = -1.0):

    """
    Calculates the DCR offset in arcsec

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

    
    #Create BB
    BBwave = np.arange(1,12000,1)
    BBflux = make_bb(BBwave,temp)

    #Import filter
    sb, w, s_left, s_right = filt_interp(band=band)

    #Import mdwarf spectrum
    m_spec, m_wave = mdwarf_interp(mdname, x_new=BBwave,
                                s_left = s_left, s_right = s_right)
    
    #take slice where band is non-zero
    cleft = np.where(np.abs(BBwave - s_left) == np.abs(BBwave - s_left).min())[0][0]
    cright = np.where(np.abs(BBwave - s_right) == np.abs(BBwave - s_right).min())[0][0]
    
    #Slice BB
    BBfluxc = BBflux[cleft:cright]
    BBwavec = BBwave[cleft:cright]
    
    #Calc effective lambda
    w_eff = np.exp(np.sum( (m_spec + (ff * BBfluxc)) * sb * np.log(BBwavec)) / np.sum((m_spec + (ff * BBfluxc)) * sb))
   
    return w_eff

