import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import rubin_sim.photUtils.Bandpass as Bandpass
import rubin_sim.photUtils.Sed as Sed

def filt_interp(band, x_new):

    """
    Imports LSST filter

    Parameters
    -----------
    band: string
        which LSST band to use (u,g,r,i,z,y)

    x_new: ndarray
        the array to interpolate on

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

    # Read the LSST throughput curves.
    filterlist = ['u', 'g', 'r', 'i', 'z', 'y']
    filtercolors = {'u':'b', 'g':'c', 'r':'g', 'i':'orange', 'z':'r', 'y':'m'}
    # Get the throughputs directory using the 'throughputs' package env variables.
    #throughputsDir = os.getenv('LSST_THROUGHPUTS_BASELINE')
    lsst = {}
  
    lsst[band] = Bandpass()
    # Use os.path.join to join directory and filenames - it's safer.
    #throughputsFile = os.path.join(throughputsDir, 'total_' + f + '.dat')
    lsst[band].readThroughput('baseline/total_' + band + '.dat')

    sb, w = lsst[band].sb, lsst[band].wavelen*10 #scale flux, conv nm to A

    #Create left slice ind
    for i,s in enumerate(sb):
        if s == 0:
            pass
        else:
            s_left = w[i]
            break
        
    #Create right slice ind
    for i,s in reversed(list(enumerate(sb))):
        if s == 0:
            pass
        else:
            s_right = w[i]
            break

    #take slice where band is non-zero
    wleft = np.where(np.abs(w - s_left) == np.abs(w - s_left).min())[0][0]
    wright = np.where(np.abs(w - s_right) == np.abs(w - s_right).min())[0][0]

    xleft = np.where(np.abs(x_new - s_left) == np.abs(x_new - s_left).min())[0][0]
    xright = np.where(np.abs(x_new - s_right) == np.abs(x_new - s_right).min())[0][0]
    sx = x_new[xleft:xright]

    f = interp1d(w, sb)
    sb_new = f(sx)

    return sb_new[wleft:wright], w[wleft:wright], s_left, s_right