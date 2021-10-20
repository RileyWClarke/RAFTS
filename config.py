import numpy as np
BBWAVE = np.arange(0,12001,1)
BBnorm = 5.833e-24
airmass = 1.4
FFACTOR = 0.05
WMIN = 3825
WMAX = 9200

def sed_integ(w, f):
    return np.nansum(f) / np.nanmean(np.diff(w))