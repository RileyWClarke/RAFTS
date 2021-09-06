import os
import matplotlib.pyplot as plt
import numpy as np

import rubin_sim.photUtils.Bandpass as Bandpass
import rubin_sim.photUtils.Sed as Sed
import make_bb

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

bb10k = Sed()
bb10k.setSED(wave,flambda=make_bb.make_bb(wave,10000))

bb3k = Sed()
bb3k.setSED(wave,flambda=make_bb.make_bb(wave,5000))

mags = {}

print('10K Magnitudes')
for f in filterlist:
    mags[f] = bb10k.calcMag(lsst[f])
    print('%s  %.2f' % (f, mags[f]))

print(" ")
print('3K Magnitudes')
for f in filterlist:
    mags[f] = bb3k.calcMag(lsst[f])
    print('%s  %.2f' % (f, mags[f]))

    
fig = plt.figure()
ax1 = plt.subplot(1,1,1)
ax2 = ax1.twinx()
for f in filterlist:
    ax1.plot(lsst[f].wavelen, lsst[f].sb, color=filtercolors[f])
ax2.plot(bb10k.wavelen, bb10k.wavelen*bb10k.flambda, color='k', label='10000K')
ax2.plot(bb3k.wavelen, bb3k.wavelen*bb3k.flambda, color='k',linestyle='--',
         label='5000K')
plt.xlim(300, 1100)
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Transmission (0-1)')
ax1.set_ylim(0, 1)
ax2.set_ylabel('$\lambda F_\lambda$ (ergs/cm$^2$/s)')
plt.legend()
plt.savefig('Figures/bb_passbands.png', bbox_inches='tight')

