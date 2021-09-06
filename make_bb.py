import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as const
import astropy.units as u

h = (const.h).cgs
c = (const.c).cgs
k = (const.k_B).cgs

def make_bb(wavelength, temp, makefile=False):

    l = u.Quantity(wavelength, unit=u.nm)

    T = u.Quantity(temp, unit=u.K)

    E_lambda = ((8*np.pi*h*c)/l**5) * (1/(np.exp((h*c)/(l*k*T)) - 1))

    return E_lambda.value

