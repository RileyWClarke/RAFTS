import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from scipy.interpolate.interpolate import interp1d

def mdwarf_interp(fname, x_new, s_left, s_right):

    """
    Imports and interpolates mdwarf spectrum

    Parameters
    -----------
    fname: string
        filename of .fits file
    x_new: ndarray
        array to interpolate on
    s_left: float
        left side of filter band
    s_right: float
        right side of filter band

    Returns
    -----------
    ndarray:
        interpolated mdwarf spectrum flux
    """

    with fits.open(fname, mode="readonly") as hdulist:
        data = hdulist[0].data[0]
        wave = np.arange(3825,9200,0.1)
    
    #take slice where band is non-zero
    wleft = np.where(np.abs(wave - s_left) == np.abs(wave - s_left).min())[0][0]
    wright = np.where(np.abs(wave - s_right) == np.abs(wave - s_right).min())[0][0]

    xleft = np.where(np.abs(x_new - s_left) == np.abs(x_new - s_left).min())[0][0]
    xright = np.where(np.abs(x_new - s_right) == np.abs(x_new - s_right).min())[0][0]

    swave, sdata = wave[wleft:wright],data[wleft:wright]
    sx = x_new[xleft:xright]

    f = interp1d(swave,sdata)
    sdata = f(sx)

    #plt.plot(wave, data)
    #plt.xlabel(r'$\lambda \;(\AA)$')
    #plt.ylabel('Flux')
    #plt.savefig('Figures/m_spectra.png', dpi=200, bbox_inches='tight')

    return sdata,swave