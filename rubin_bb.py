import os
import matplotlib.pyplot as plt
import numpy as np

import rubin_sim.photUtils.Bandpass as Bandpass
import rubin_sim.photUtils.Sed as Sed
from make_bb import *


def bb_mags(bb_temp, makeplot=False):
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

    wave = np.arange(10,1200,0.5)

    if makeplot==True:    
        fig = plt.figure()
        ax1 = plt.subplot(1,1,1)
        ax2 = ax1.twinx()
        lines = [":","--","-","-."]

        for i,f in enumerate(filterlist):
            ax1.plot(lsst[f].wavelen, lsst[f].sb, color=filtercolors[f], label='LSST '+filterlist[i])
           
        plt.xlim(300, 1100)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Transmission (0-1)')
        ax1.set_ylim(0, 1)
        ax2.set_ylabel('$\lambda F_\lambda$ (ergs/cm$^2$/s)')
        ax1.legend()

    for j,temp in enumerate(bb_temp):
        bb = Sed()
        bb.setSED(wave,flambda=make_bb(wave,temp))

        mags = {}
        diffs = {}
        print('{}K Magnitudes'.format(temp))
        for f in filterlist:
            mags[f] = bb.calcMag(lsst[f])
            print('%s  %.2f' % (f, mags[f]))
        maglist = list(mags.values())
        for i in range(5):
            print(filterlist[i]+'-'+filterlist[i+1]+': {0:.2f}'.format(np.abs(maglist[i] - maglist[i+1])))
            diffs[str(filterlist[i]+'-'+filterlist[i+1])] = np.abs(maglist[i] - maglist[i+1])
        
        if makeplot==True:    
           
            ax2.plot(bb.wavelen, bb.wavelen*bb.flambda, color='k', 
                    linestyle=lines[j], label='{}K'.format(temp))
            ax2.legend(loc=9)

    plt.savefig('Figures/bb_passbands.png', dpi=300, bbox_inches='tight')

    return diffs

diffs = bb_mags([5000,10000,15000],makeplot=True)