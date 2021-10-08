import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as const
import astropy.units as u

h = (const.h).cgs
c = (const.c).cgs
k = (const.k_B).cgs

def make_bb(wavelengths, temp):

    """
    Creates blackbody spectrum

    Parameters
    -----------
    temp: float
        Effective temperature of blackbody
    wavelenths: ndarray
        wavelength array of blackbody

    Returns
    -----------
    ndarray:
        flux array of blackbody
    """

    l = u.Quantity(wavelengths, unit=u.angstrom)

    T = u.Quantity(temp, unit=u.K)
    #nT = u.Quantity(norm_temp, unit=u.K)

    F_lambda = ((2*h*c**2)/l.to(u.nm)**5) * (1/(np.exp((h*c)/(l.to(u.nm)*k*T)) - 1))
   
    return F_lambda.value

