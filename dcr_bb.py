import os
import matplotlib.pyplot as plt
import numpy as np

import rubin_sim.photUtils.Bandpass as Bandpass
import rubin_sim.photUtils.Sed as Sed
from make_bb import *


def dcr_bb(band, temp, airmass):
    # Read the LSST throughput curves.
    filterlist = ['u', 'g', 'r', 'i', 'z', 'y']
    filtercolors = {'u':'b', 'g':'c', 'r':'g', 'i':'orange', 'z':'r', 'y':'m'}
    # Get the throughputs directory using the 'throughputs' package env variables.
    #throughputsDir = os.getenv('LSST_THROUGHPUTS_BASELINE')
    lsst = {}
    for f in filterlist:
        lsst[f] = Bandpass()
        # Use os.path.join to join directory and filenames - it's safer.
        #throughputsFile = os.path.join(throughputsDir, 'total_' + f + '.dat')
        lsst[f].readThroughput('baseline/total_' + f + '.dat')
 
    bb_sed = Sed()
    bb_sed.setSED(wavelen = lsst[band].wavelen, flambda = make_bb(lsst[band].wavelen, temp))
    fluxNorm = bb_sed.calcFluxNorm(24.80, lsst['r'])
    bb_sed.multiplyFluxNorm(fluxNorm)
    
    #Calc effective wavelen
    num = np.sum(bb_sed.wavelen * bb_sed.flambda * lsst[band].sb * np.log(lsst[band].wavelen))
    denom = np.sum(bb_sed.wavelen * bb_sed.flambda * lsst[band].sb)
    w_eff = np.exp(num/denom)
   

    #Calc index of refr
    n = 10**-6 * (64.328 + (29498.1 / (146-w_eff**-2)) + (255.4 / (41 - w_eff**-2))) + 1

    #Calc R_0
    R_0 = (n**2 - 1) / (2 * n**2)

    return R_0, R_0*np.tan(airmass)

temps = np.arange(5000,35000,5000)

dcr_R_AM1 = []
dcr_R_AM2 = []
dcr_R_AM3 = []

for temp in temps:
    _, r1 = dcr_bb(band = 'u', temp=temp, airmass=1.1)
    dcr_R_AM1.append(r1)
    _, r2 = dcr_bb(band = 'u', temp=temp, airmass=1.25)
    dcr_R_AM2.append(r2)
    _, r3 = dcr_bb(band = 'u', temp=temp, airmass=1.4)
    dcr_R_AM3.append(r3)

plt.scatter(temps,dcr_R_AM1, label = 'airmass = 1.1')
plt.scatter(temps,dcr_R_AM2, label = 'airmass = 1.25')
plt.scatter(temps,dcr_R_AM3, label = 'airmass = 1.4')
plt.ylim(-0.0005,0.0005)
plt.xlabel("Temperature (K)")
plt.ylabel("R (arcsec)")
plt.legend()
plt.savefig('Figures/rplot.png', dpi=300, bbox_inches='tight')