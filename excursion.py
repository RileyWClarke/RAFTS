import sys
from utils import make_bb, filt_interp
from config import filters, filtercolors
import numpy as np
import matplotlib.pyplot as plt
from config import WMIN, WMAX
import scipy
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.modeling.models import BlackBody
from astropy.visualization import quantity_support
from astropy.io import fits

from sys import argv

band = str(argv[1])
spt = str(argv[2])

wav = np.arange(100, 12000) * u.AA
plx = 10.0
dist = 1000 / plx * u.pc
Rstar = 0.1 * u.Rsun
t = 5000 
a = 0.0025
balmer_ratio = 1

def excursion(band, spt, wav=wav, dist=dist, Rstar=Rstar, t=t, a=a, balmer_ratio=balmer_ratio):

    with fits.open('/Users/riley/Desktop/RAFTS/sdsstemplates/{}.active.ha.na.k.fits'.format(spt), mode="readonly") as hdulist:
        data = hdulist[0].data[0]
        header = hdulist[0].header
        wstart = hdulist[0].header[3]
        wstep = hdulist[0].header[5]
        wave = np.arange(wstart,wstart + len(data) * wstep,wstep)

    f = filt_interp(band=band, survey='LSST')
    g = interp1d(wave, data, bounds_error=False, fill_value=0.0)
    mds = g(wav)

    sdss_inds = [np.where(wav == WMIN * u.AA)[0][0], np.where(wav == WMAX * u.AA)[0][0].sum()]

    ##Augment spectra
    def fitbb_to_m5(a, T, m5spec):
        bb = make_bb(wav, 3000) * 1e27 * a
        relevant_w = np.argmin(np.abs(wav - WMAX*u.AA))
        indices = range(relevant_w-50, relevant_w)
        x = np.abs(((bb[indices] - m5spec[indices])).sum())
        return x

    amplitude = 1
    res = scipy.optimize.minimize(fitbb_to_m5, [amplitude], args=(3000, mds))
    mds[wav >= WMAX*u.AA] = (make_bb(wav, 3000) * 1e27 * res.x)[wav >= WMAX*u.AA]

    balmer_step = np.ones_like(wav.value)
    balmer_step[wav.value < 3700] = balmer_ratio

    bb = BlackBody(temperature = t * u.K, scale = 1*u.erg/u.s/u.cm/u.cm/u.AA/u.sr) #Blackbody
    C = bb(wav)[sdss_inds[0]:sdss_inds[1]].sum().value * (a**2) / mds[sdss_inds[0]:sdss_inds[1]].sum() #Blackbody scaling factor
    bbdm = (mds * C) * (1*u.erg/u.s/u.cm/u.cm/u.AA/u.sr) #Scaled quiescent spectrum
    combined_bb = (bb(wav) * a**2 * balmer_step) + bbdm #Combined flare spectrum

    assert bb(wav)[sdss_inds[0]:sdss_inds[1]].sum() * a**2 == bbdm[sdss_inds[0]:sdss_inds[1]].sum()

    flux = (np.sum(f(wav) * bbdm) / np.sum(f(wav))) * (np.pi*u.sr) * (Rstar.to('cm') ** 2) / ((dist.to('cm')) ** 2)
    flux_flare = (np.sum(f(wav) * combined_bb) / np.sum(f(wav))) * (np.pi*u.sr) * (Rstar.to('cm') ** 2) / ((dist.to('cm')) ** 2)
    flux_bb = (np.sum(f(wav) * (bb(wav) * a**2)) / np.sum(f(wav))) * (np.pi*u.sr) * (Rstar.to('cm') ** 2) / ((dist.to('cm')) ** 2)
    mag = 22.5 - 2.5 * np.log10(flux.value)
    mag_flare = 22.5 - 2.5 * np.log10(flux_flare.value)
    mag_bb = 22.5 - 2.5 * np.log10(flux_bb.value)
    delta = mag_flare - mag
    abs_mag = mag - 5 * np.log10(dist.value) + 5
    abs_mag_flare = mag_flare - 5 * np.log10(dist.value) + 5
    abs_delta = abs_mag_flare - abs_mag

    #print('flare absolute mag = {}, band: {}, SpT: {}'.format(abs_mag_flare, band, spt))
    print('delta mag = {}, band: {}, SpT: {}'.format(delta, band, spt))
    print('max rise rate = {} mag/min'.format((delta / 1))) 
    print('min rise rate = {} mag/min'.format((delta / 152))) 
    print('max fade rate = {} mag/min'.format((delta / 4.9))) 
    print('min fade rate = {} mag/min'.format((delta / 216.8))) 


    with quantity_support():
        fig, ax = plt.subplots()
        ax.plot(wav, bb(wav)*a**2, ls='--', c='k')
        ax.plot(wav, bbdm, c='C3')
        ax.plot(wav, combined_bb, c='C0')
        ax.set_ylim(None, combined_bb.max() * 1.2)
        ax.set_xlabel(r'Wavelength ($\AA$)', fontsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)
        ax.set_title('T = {} K, SpT = {}, band = {}'.format(t, spt, band))
        ax.yaxis.set_tick_params(labelsize=12)
        ax.xaxis.set_tick_params(labelsize=12)

        ax2 = ax.twinx()
        ax2.set_ylabel('Filter Throughput', fontsize=16)
        ax2.tick_params(axis ='y')
        ax2.yaxis.set_tick_params(labelsize=12)

        for f in filters:
            func = filt_interp(band=f, survey='LSST')
            if f == band:
                ax2.plot(wav, func(wav), c=filtercolors[f])
            else:
                ax2.plot(wav, func(wav), c='grey', alpha=0.3)

        plt.show()

excursion(band, spt)