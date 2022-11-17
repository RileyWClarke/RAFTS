import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from scipy.interpolate.interpolate import interp1d

import os

def mdwarf_interp(fname, plotit=False):

    """
    Imports and interpolates mdwarf spectrum

    Parameters
    -----------
    fname: string
        filename of .fits file

    plotit : bool
        plotting toggle

    Returns
    -----------
    interp1d(mdx,mdy,fill_value = 0.0) : function
        function that interpolates mdx, mdy
    """

    with fits.open(fname, mode="readonly") as hdulist:
        data = hdulist[0].data[0]
        wstart = hdulist[0].header[3]
        wstep = hdulist[0].header[5]
        wave = np.arange(wstart,wstart + len(data) * wstep,wstep)

    if plotit:
        plt.plot(wave, data, label='MD Spectrum')
        plt.xlabel(r'Wavelength $(\AA)$')
        plt.ylabel(r'$F_\lambda$ ($10^{-17}\;erg\;s^{-1}\;cm^{-2}\;\AA^{-1}$)')
        plt.savefig('Figures/m_spectra.png', dpi=300, bbox_inches='tight')

    return interp1d(wave, data, bounds_error=False, fill_value=0.0)