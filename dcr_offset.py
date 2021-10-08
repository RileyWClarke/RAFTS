import numpy as np

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

    #Calc index of refr
    n = (10**-6 * (64.328 + (29498.1 / (146-(1/w_eff**2))) + (255.4 / (41 - (1/w_eff**2))))) + 1

    #Calc R_0
    R_0 = (n**2 - 1) / (2 * n**2)

    #Calc Z
    Z = np.arccos(1/airmass)

    return R_0*np.tan(Z)