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
    
    sb, w = lsst[band].sb*35, lsst[band].wavelen*10 #scale flux, conv nm to A
    
    ##Composite SED calculation
    compw_c = np.copy(compw)
    compw_c *= (1+z)

    #take slice where band is non-zero
    cleft = np.where(np.abs(compw_c - s_left) == np.abs(compw_c - s_left).min())[0][0]
    cright = np.where(np.abs(compw_c - s_right) == np.abs(compw_c - s_right).min())[0][0]
    wleft = np.where(np.abs(w - s_left) == np.abs(w - s_left).min())[0][0]
    wright = np.where(np.abs(w - s_right) == np.abs(w - s_right).min())[0][0]
    
    #Interp SED 
    f = interpolate.interp1d(w[wleft:wright],sb[wleft:wright], bounds_error=False, fill_value=0.0)
    new_sb = f(compw_c[cleft:cright])

    compf_clip = compf[cleft:cright]
    compw_c_clip = compw_c[cleft:cright]

    #Calc weff
    w_eff = np.exp(np.sum(compf_clip * new_sb * np.log(compw_c_clip))/ np.sum(compf_clip * new_sb))
    w_effm = w_eff / 1e4

    #Calc index of refr
    n = (10**-6 * (64.328 + (29498.1 / (146-(1/w_effm**2))) + (255.4 / (41 - (1/w_effm**2))))) + 1

    #Calc R_0
    R_0_sed = (n**2 - 1) / (2 * n**2)

    ##Calculate power-law stuff
    pl_w = np.copy(compw_c)
    pl_f = pl_w**(-0.5)

    #take slice where band is non-zero
    cleft = np.where(np.abs(pl_w - s_left) == np.abs(pl_w - s_left).min())[0][0]
    cright = np.where(np.abs(pl_w - s_right) == np.abs(pl_w - s_right).min())[0][0]
    '''
    #Interp SED 
    f = interpolate.interp1d(w[wleft:wright],sb[wleft:wright], bounds_error=False, fill_value=0.0)
    new_sb = f(pl_w[cleft:cright])
    '''
    pl_f_clip = pl_f[cleft:cright]
    pl_w_clip = pl_w[cleft:cright]

    #Calc weff
    w_eff_pl = np.exp(np.sum(pl_f_clip * new_sb * np.log(pl_w_clip))/ np.sum(pl_f_clip * new_sb))
    w_eff_plm  = w_eff_pl / 1e4

    #Calc index of refr
    n_pl = (10**-6 * (64.328 + (29498.1 / (146-(1/w_eff_plm**2))) + (255.4 / (41 - (1/w_eff_plm**2))))) + 1

    #Calc R_0
    R_0_pl = (n_pl**2 - 1) / (2 * n_pl**2)

    ##Calc Z
    Z = np.arccos(1/airmass)

    R = (R_0_sed - R_0_pl) * np.tan(Z)
 
    return R, w_eff, compw_c, sb, w

#Plotting----------------
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots(figsize=(15,5))
fig2, ax2 = plt.subplots(figsize=(15,5))
ams = [1.1,1.25,1.4]
for am in ams:
    Rlist = []
    wefflist = []
    zs = np.arange(0,5,0.02)
    print('Starting airmass = {}'.format(am))
    for z in zs:
        R, w_eff, compw_c, sb, w = dcr_qsr(band='i', z=z, airmass = am, 
                                            s_left=6500, s_right=8500)
        Rlist.append(R)
        wefflist.append(w_eff)
    
        ax.plot(compw_c, compf, label='Z = {}'.format(z))
        print('Calculating z={}'.format(z))
    ax.plot(w, sb, color='k', label='LSST g-band')
    ax.set_xlim(0,15000)
    ax.set_xlabel(r'Wavelength $\AA$')
    
    ax1.plot(zs, Rlist,label='airmass = {}'.format(am))
    ax1.axhlines(0)
    ax1.set_xlabel('Z (redshift)')
    ax1.set_ylabel('R (arcsec)')
    ax1.legend()

    ax2.plot(zs, wefflist)
    ax2.set_xlabel('Z (redshift)')
    ax2.set_ylabel(r'$\lambda_{eff}$ ($\AA$)')

    np.save('rlist_am'+str(am), Rlist)
    np.save('wefflist_am'+str(am), wefflist)

fig.savefig('Figures/sed_comp.png',dpi=300,bbox_inches='tight')
fig1.savefig('Figures/zplot.png', dpi=300, bbox_inches='tight')
fig2.savefig('Figures/weffplot.png', dpi=300, bbox_inches='tight')