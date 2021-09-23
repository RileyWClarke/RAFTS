import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

import rubin_sim.photUtils.Bandpass as Bandpass
import rubin_sim.photUtils.Sed as Sed
from make_bb import *

compw, compf = np.loadtxt('vandenberk_qsrcompspc.txt', usecols=(0,1), unpack=True)

def dcr_qsr(band, z, airmass, s_left, s_right, compw=compw, compf=compf):
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
    '''
    pl_sed = Sed()
    pl_sed.setSED(wavelen = lsst[band].wavelen, flambda = lsst[band].wavelen**(-0.5))
    fluxNormpl = pl_sed.calcFluxNorm(24.80, lsst['r'])
    pl_sed.multiplyFluxNorm(fluxNormpl)
  
    comp_sed = Sed()
    comp_sed.setSED(wavelen = compw, flambda = compf)
    fluxNormcp = comp_sed.calcFluxNorm(24.80, lsst['r'])
    comp_sed.multiplyFluxNorm(fluxNormcp)
    '''
    sb, w = lsst[band].sb*35, lsst[band].wavelen*10 #scale flux, conv nm to A
   
    compw_c = np.copy(compw)
    compw_c *= (1+z)

    #take slice where band is non-zero
    cleft = np.where(np.abs(compw_c - s_left) == np.abs(compw_c - s_left).min())[0][0]
    cright = np.where(np.abs(compw_c - s_right) == np.abs(compw_c - s_right).min())[0][0]
    wleft = np.where(np.abs(w - s_left) == np.abs(w - s_left).min())[0][0]
    wright = np.where(np.abs(w - s_right) == np.abs(w - s_right).min())[0][0]

    f = interpolate.interp1d(w[wleft:wright],sb[wleft:wright], fill_value='extrapolate')
    new_sb = f(compw_c[cleft:cright])
    
    compf_clip = compf[cleft:cright]
    compw_c_clip = compw_c[cleft:cright]

    #Calc weff
    w_eff = np.exp(np.sum(compf_clip * new_sb * np.log(compw_c_clip))/ np.sum(compf_clip * new_sb))
    
    #Calc index of refr
    n = (10**-6 * (64.328 + (29498.1 / (146-(1/w_eff**2))) + (255.4 / (41 - (1/w_eff**2))))) + 1

    #Calc R_0
    R_0 = (n**2 - 1) / (2 * n**2)

    #Calc Z
    Z = np.arccos(1/airmass)
   
    return R_0, R_0*np.tan(Z), w_eff, compw_c, sb, w

Rlist = []
wefflist = []
zs = np.arange(0,3.2,0.2)
zs = [1.1,1.4,1.7,2.0]
fig, ax = plt.subplots()

for z in zs:
    R_0, R, w_eff, compw_c, sb, w = dcr_qsr(band='g', z=z, airmass = 1.414, 
                                            s_left=3500, s_right=6000)
    Rlist.append(R_0)
    wefflist.append(w_eff)

    ax.plot(compw_c, compf, label='Z = {}'.format(z))
    
ax.plot(w, sb, color='k', label='LSST g-band')
ax.set_xlim(0,15000)
ax.set_xlabel(r'Wavelength $\AA$')
ax.legend()
fig.savefig('Figures/sed_comp.png',dpi=300,bbox_inches='tight')

fig1, ax1 = plt.subplots()
ax1.plot(zs, Rlist)
ax1.set_xlabel('Z (redshift)')
ax1.set_ylabel('R (arcsec)')
fig1.savefig('Figures/zplot.png', dpi=300, bbox_inches='tight')

fig2, ax2 = plt.subplots()
ax2.plot(zs, wefflist)
ax2.set_xlabel('Z (redshift)')
ax2.set_ylabel(r'$\lambda_{eff}$ ($\AA$)')
fig2.savefig('Figures/weffplot.png', dpi=300, bbox_inches='tight')