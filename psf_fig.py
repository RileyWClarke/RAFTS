from icecream import ic
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from astropy.io import fits
from photutils.centroids import centroid_2dg

from utils import (
    filt_interp,
    make_bb,
    dcr_offset,
    create_2d_gaussian,
    shift_gaussian,
    dilate_gaussian
)
from config import WAVELENGTH
import globals

globals.initialize()

### Make quiescent and flaring PSFs for an input temp, airmass, etc
#PSFNORM = 9.89881469e-04  # so that the mag scale goes up tp ~2
PSFNORM = 1.080779116444788e-3 
# set constants
size = 201

center0 = (int(size / 2), int(size / 2))  # array cell coordunates
arcsec2lsstpixels = 0.2  # arcsec/pixel
lsstpixels2img = 0.1  # lsstpixel/cell 
arcsec2img = arcsec2lsstpixels * lsstpixels2img   # arcsec/image cell
fwhm = 0.7 / arcsec2img   # 0.7" seeing in image scale
stdev = fwhm / 2.355  # imgscale

temp = 10000
am_i = 1.0 / np.cos(np.deg2rad(35))  # initial airmass: median survey airmass 1.2
am_f = 1.0 / np.cos(np.deg2rad(55))  # final airmass
band = "g"
flux_factor = 1.5  # brightness increase
f = filt_interp(band)  # filter throughput

hdul = fits.open("ukg2v.fits")  # sun spectrum
g2vwavelength = np.arange(1150, 25005, 5)
g2vSED_old = hdul[0].data

g2vf = interp1d(g2vwavelength, g2vSED_old, fill_value="extrapolate")
g2vSED = g2vf(WAVELENGTH)

# import spectra
iSED = np.load("mdspec_full.npy")  # dM spectrum
fSED = (
    np.load("mdspec_full.npy") + make_bb(WAVELENGTH, temp) * globals.BBnorm
)  # dM spectrum plus flare
flatSED = np.full_like(iSED, 0.1)

hdul = fits.open("ukg2v.fits")  # sun spectrum
g2vwavelength = np.arange(1150, 25005, 5)
g2vSED_old = hdul[0].data

g2vf = interp1d(g2vwavelength, g2vSED_old, fill_value="extrapolate")
g2vSED = g2vf(WAVELENGTH)


def calc_weff(p, f):
    return np.exp(np.sum(p * np.log(WAVELENGTH[f(WAVELENGTH) > 0])) / np.sum(p))


# DCR First moment
def Rbar(SED, am, plot=False):
    p = SED[f(WAVELENGTH) > 0] * f(WAVELENGTH)[f(WAVELENGTH) > 0]
    if plot:
        plt.plot(WAVELENGTH[f(WAVELENGTH) > 0], p)

    weff = calc_weff(p, f)
    R = dcr_offset(weff, airmass=am)  # acrsec

    return np.sum(p * R) / np.sum(p), weff  # arcsec


# DCR Second moment
def V(SED, am):
    p = SED[f(WAVELENGTH) > 0] * f(WAVELENGTH)[f(WAVELENGTH) > 0]  # filtered SED
    R = dcr_offset(WAVELENGTH[f(WAVELENGTH) > 0], airmass=am)  # arcsec #dcr offset
    Rbar = np.sum(p * R) / np.sum(p)  # arcsec the DCR offset integrated over the SED

    return np.sum(p * (R - Rbar) ** 2) / np.sum(p)  # arcsec^2


Rbar_x0_dm, weff_dm = Rbar(iSED, am_i)  # arcsec
Rbar_x0_flare, weff_flare = Rbar(fSED, am_i)  # arcsec
Rbar_x1_dm, weff_dm = Rbar(iSED, am_f)  # arcsec
Rbar_x1_flare, weff_flare = Rbar(fSED, am_f)  # arcsec
Rbar_x0_g, weff_g = Rbar(g2vSED, am_i)  # arcsec
Rbar_x1_g, weff_g = Rbar(g2vSED, am_f)  # arcsec

dRbar_x0_sed = Rbar_x0_flare - Rbar_x0_dm  # arcsec
dRbar_x_dm = Rbar_x1_dm - Rbar_x0_dm  # arcsec
dRbar_x_sed = Rbar_x1_flare - Rbar_x0_dm  # arcsec
dRbar_x_g = Rbar_x1_g - Rbar_x0_g  # arcsec

V_x0_dm = V(iSED, am_i)  # arcsec^2
V_x0_flare = V(fSED, am_i)  # arcsec^2
V_x1_dm = V(iSED, am_f)  # arcsec^2
V_x1_flare = V(fSED, am_f)  # arcsec^2
V_x0_g = V(g2vSED, am_i)  # arcsec^2
V_x1_g = V(g2vSED, am_f)  # arcsec^2

dstd_x0_sed = V_x0_flare - V_x0_dm  # stdev
dstd_x_sed = V_x1_flare - V_x0_dm
dstd_x_dm = V_x1_dm - V_x0_dm
dstd_x_g = V_x1_g - V_x0_g  # arcsec


weffs = {
    "dm": weff_dm,
    "flare": weff_flare,
    "g2v": weff_g,
}

print(
    f"DCR offset - same airmass {dRbar_x0_sed:.2f} arcsec\n"
    f"DCR offset - true {dRbar_x_sed:.2f} arcsec\n"
    f"dM at different airmass {dRbar_x_dm:.2f} arcsec\n"
    f"g-star different airmass {dRbar_x_g:.2f} arcsec\n"
)

print(
    f"DCR PSF dispersion, same airmass {dstd_x0_sed:.5f} arcsec\n"
    f"DCR PSF dispersion, true {dstd_x_sed:.5f} arcsec\n"
    f"DCR PSF dispersion, different airmass {dstd_x_dm:.5f} arcsec\n"
    f"DCR PSF dispersion, different airmass, g assumption {dstd_x_g:.5f} arcsec\n"
)

print(r"lambda effective")
p = flatSED[f(WAVELENGTH) > 0] * f(WAVELENGTH)[f(WAVELENGTH) > 0]
print(
    "flat  {:.2f} AA".format(
        np.exp(np.sum(p * np.log(WAVELENGTH[f(WAVELENGTH) > 0])) / np.sum(p))
    )
)
p = g2vSED[f(WAVELENGTH) > 0] * f(WAVELENGTH)[f(WAVELENGTH) > 0]
print(
    "sum   {:.2f} AA".format(
        np.exp(np.sum(p * np.log(WAVELENGTH[f(WAVELENGTH) > 0])) / np.sum(p))
    )
)
p = iSED[f(WAVELENGTH) > 0] * f(WAVELENGTH)[f(WAVELENGTH) > 0]
print(
    "dM    {:.2f} AA".format(
        np.exp(np.sum(p * np.log(WAVELENGTH[f(WAVELENGTH) > 0])) / np.sum(p))
    )
)
p = fSED[f(WAVELENGTH) > 0] * f(WAVELENGTH)[f(WAVELENGTH) > 0]
print(
    "flare {:.2f} AA".format(
        np.exp(np.sum(p * np.log(WAVELENGTH[f(WAVELENGTH) > 0])) / np.sum(p))
    )
)


def make_PSF(weffs, dRbs, Vs, am1, am2):

    center = center0  # Center of the Gaussian
    sigma_x_original = stdev  # Original standard deviation along the x-axis
    sigma_y_original = stdev  # Original standard deviation along the y-axis
    yshift = dRbs[0]
    yshiftsub = dRbs[1]

    ic(yshift, "arcsec")
    ic(yshift / arcsec2img * lsstpixels2img, "LSST pix" )
    ic(yshift / arcsec2img, "sim pix" )

    #scale_factor_x = np.sqrt(stdev**2 * (weffs[1] / weffs[0]) ** (-2 / 5)) / stdev
    #scale_factor_x *= np.sqrt(stdev**2 * (am2 / am1)**0.6) / stdev
    #scale_factor_xsub = np.sqrt(stdev**2 * (weffs[2] / weffs[0]) ** (-2 / 5)) / stdev 
    #scale_factor_xsub *= np.sqrt(stdev**2 * (am2 / am1)**0.6) / stdev

    #scale_factor_y = scale_factor_x  
    #scale_factor_ysub = scale_factor_xsub 
    #scale_factor_y2 = np.sqrt(stdev**2 + (Vs[1] - Vs[0]) / arcsec2img**2) / stdev
    #scale_factor_y2sub = np.sqrt(stdev**2 + (Vs[3] - Vs[2]) / arcsec2img**2) / stdev

    ic(weffs)
    ic(dRbs[0], dRbs[1])
    ic(Vs)
    #ic(scale_factor_x, scale_factor_y)
    #ic(scale_factor_y2, scale_factor_y2sub)
    
    # Create the original Gaussian and apply corrections
    g0 = create_2d_gaussian((size, size), center, sigma_x_original, sigma_y_original, flux_factor=1, PSFNORM=PSFNORM)
    #gflare = create_2d_gaussian((size, size), center, sigma_x_original, sigma_y_original, flux_factor=flux_factor, PSFNORM=PSFNORM)  
    #g1 = dilate_gaussian(gflare, scale_factor_x, scale_factor_y) #isotropic correction
    #g2 = dilate_gaussian(g1, 1, scale_factor_y2) #dispersion in zenith direction
    #g3 = shift_gaussian(g2, shift_vector=(-(yshift / arcsec2img), 0)) #centroid shift
    
    #Create model of the assumed correction
    #gsub0 = create_2d_gaussian((size, size), center, sigma_x_original, sigma_y_original)
    #gsub1 = dilate_gaussian(gsub0, scale_factor_xsub, scale_factor_ysub) #isotropic correction
    #gsub2 = dilate_gaussian(gsub1, 1, scale_factor_y2sub) #dispersion in zenith direction
    #gsub3 = shift_gaussian(gsub2, shift_vector=(-(yshiftsub / arcsec2img), 0)) #centroid shift

    sigma_x = np.sqrt(sigma_x_original**2 * (weffs[1] / weffs[0]) ** (-2 / 5))
    sigma_x *= (am2 / am1)**0.6

    sigma_y = np.sqrt(sigma_y_original**2 * (weffs[1] / weffs[0]) ** (-2 / 5))
    sigma_y *= (am2 / am1)**0.6
    sigma_y2 = np.sqrt(sigma_y**2 +  (Vs[1] - Vs[0]) / arcsec2img**2)

    center_x = center[0]
    center_y = center[1] - (yshift / arcsec2img)


    g3 = create_2d_gaussian((size, size), (center_x, center_y), 
                            sigma_x, sigma_y2, flux_factor=flux_factor, PSFNORM=PSFNORM)

    sigma_x_sub = np.sqrt(sigma_x_original**2 * (weffs[2] / weffs[0]) ** (-2 / 5))
    sigma_x_sub *= (am2 / am1)**0.6

    sigma_y_sub = np.sqrt(sigma_y_original**2 * (weffs[2] / weffs[0]) ** (-2 / 5))
    ic((weffs[2] / weffs[0]))
    sigma_y_sub *= (am2 / am1)**0.6
    sigma_y2_sub = np.sqrt(sigma_y_sub**2 +  (Vs[3] - Vs[2]) / arcsec2img**2)

    center_x_sub = center[0]
    center_y_sub = center[1] - (yshiftsub / arcsec2img)

    gsub3 = create_2d_gaussian((size, size), (center_x, center_y_sub), 
                            sigma_x_sub, sigma_y2_sub, flux_factor=1, PSFNORM=PSFNORM)

    return g0, g3, gsub3

def plotrow(
    psf_i,
    psf_f,
    psf_sub,
    am1,
    am2,
    ax,
    cmap,
    zenith=False,
    SEDlabel="",
):
    # Plotting
    vrange = (-0.15, 2)

    ic(psf_i.sum(), psf_i.max())
    ic(psf_f.sum(), psf_f.max(), psf_f.sum() / psf_i.sum())
    ic(psf_sub.max(), psf_sub.sum(), psf_sub.sum() / psf_i.sum())
    ic((psf_f - psf_sub).sum())

    a = ax[0].imshow(psf_i, vmin=vrange[0], vmax=vrange[1], cmap=cmap)
    b = ax[1].imshow(psf_f, vmin=vrange[0], vmax=vrange[1], cmap=cmap)
    c = ax[2].imshow(psf_f - psf_sub, vmin=vrange[0], vmax=vrange[1], cmap=cmap)
 
    print("--------------")
    levels = np.arange(0.1, 0.9, 0.1)

    # contours
    cc = ax[2].contour(psf_f - psf_sub, levels, alpha=0.8, cmap="Reds_r")
    labels = ax[2].clabel(cc, levels, inline=False, fmt="%1.1f", fontsize=0)

    x = WAVELENGTH[f(WAVELENGTH) > 0.0]
    yg = f(WAVELENGTH)[f(WAVELENGTH) > 0.0]
    yi = yg * iSED[f(WAVELENGTH) > 0.0]
    yf = yg * fSED[f(WAVELENGTH) > 0.0]

    ax[0].set_title(
        "Quiescent, X = {0:.2f}".format(am1)
    )  # , fontsize = plt.rcParams["font.size"] - 1)
    ax[1].set_title(
        "Flaring, X = {0:.2f}".format(am2)
    )  # , fontsize = plt.rcParams["font.size"] - 1)
    ax[2].set_title(
        " ".join([SEDlabel, "Difference"])
    )  # , fontsize = plt.rcParams["font.size"] - 1)

    if zenith:
        ax[0].annotate(
            "zenith",
            xy=(size * 1 / 7, size * 2 / 5),
            xytext=(size * 1 / 7, size * 4 / 5),
            horizontalalignment="center",
            color="white",
            alpha=0.7,
            arrowprops=dict(facecolor="#aaaaaa", color="#aaaaaa", width=2, headwidth=6),
        )
    return a, cc, labels


# PLOTTING
fig, ax = plt.subplots(3, 3, figsize=(7.2, 8))

########ROW 1
psf_i, psf_f, psf_sub = make_PSF(
    (weffs["dm"], weffs["flare"], weffs["g2v"]), 
    (dRbar_x0_sed, dRbar_x0_sed), 
    (V_x0_dm, V_x0_flare, V_x0_dm, V_x0_dm), 
    am_i, am_i
)

xcen, ycen = centroid_2dg(psf_f - psf_i)
#ax[0][2].scatter(xcen, ycen, marker='+', color='purple')

print("Max pixel: {}".format(np.where(psf_f - psf_i == (psf_f - psf_i).max())))
print("Difference Centroid: [x,y]=[{0},{1}]".format(xcen, ycen))
print("Derived DCR = {} arcsec".format((center0[1] - ycen) * arcsec2img))
np.save("Outdata/dm_am1.npy", psf_i)
np.save("Outdata/flare_am1.npy", psf_f)

a, cc, labels = plotrow(psf_i, psf_f, psf_i, am_i, am_i, ax=ax[0], cmap='Greys_r', zenith=True)
cbar_ax = fig.add_axes([0.31, 0.05, 0.4, 0.03])
fig.colorbar(a, cax=cbar_ax, label="Relative flux", orientation="horizontal")

########ROW 2

psf_i, psf_f, psf_sub = make_PSF(
    (weffs["dm"], weffs["flare"], weffs["dm"]),
    (dRbar_x_sed - dRbar_x_dm, dRbar_x_dm - dRbar_x_dm),
    (V_x0_dm, V_x1_flare, V_x0_dm, V_x1_dm),
    am_i,
    am_f,
)

xcen, ycen = centroid_2dg(psf_f - psf_sub)
#ax[1][2].scatter(xcen, ycen, marker='+', color='purple')

print("Max pixel: {}".format(np.where(psf_f - psf_sub == (psf_f - psf_sub).max())))
print("Difference Centroid: [x,y]=[{0},{1}]".format(xcen, ycen))
print("Derived DCR = {} arcsec".format((center0[1] - ycen) * arcsec2img))
np.save("Outdata/dm_am2.npy", psf_sub)
np.save("Outdata/flare_am2.npy", psf_f)

a, cc, labels = plotrow(psf_i, psf_f, psf_sub, am_i, am_f, ax=ax[1], cmap='Greys_r', zenith=False, SEDlabel=r"$dM$")

########ROW 3
psf_i, psf_f, psf_sub = make_PSF(
    (weffs["dm"], weffs["flare"], weffs["dm"]),
    (dRbar_x_sed - dRbar_x_dm, dRbar_x_g - dRbar_x_dm),
    (V_x0_dm, V_x1_flare, V_x0_g, V_x1_g),
    am_i,
    am_f,
)

xcen, ycen = centroid_2dg(psf_f - psf_sub)
#ax[2][2].scatter(xcen, ycen, marker='+', color='purple')

print("Max pixel: {}".format(np.where(psf_f - psf_sub == (psf_f - psf_sub).max())))
print("Difference Centroid: [x,y]=[{0:.3f},{1:.3f}]".format(xcen, ycen))
print("Derived DCR = {0:.3f} arcsec".format((center0[1] - ycen) * arcsec2img))

np.save("Outdata/g2v_am2.npy", psf_sub)

a, cc, labels = plotrow(psf_i, psf_f, psf_sub, am_i, am_f, ax=ax[2], cmap='Greys_r', zenith=False, SEDlabel=r"$G2V$")

for gridx in range(3):
    for gridy in range(3):
        ax[gridx, gridy].plot(
            [int(size / 2), int(size / 2)], [0, size], "w-", lw=0.3
        )
        ax[gridx, gridy].plot(
            [0, size], [int(size / 2), int(size / 2)], "w-", lw=0.3
        )

letters = ["(a)", "(b)", "(c)", "", "(d)", "(e)", "", "", "(f)"]
# Grid
for axis in ax.flatten():
    
    axis.set_xticks(np.arange(0, size, 1/lsstpixels2img))  # size*0.1 is the pixel scale of the camera
    axis.set_yticks(np.arange(0, size, 1/lsstpixels2img))
    
    axis.set_xlim(0, size)
    axis.set_ylim(size, 0)

    for tick in axis.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    for tick in axis.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    axis.grid(True, alpha=0.4)

for axis, letter in zip(ax.flatten(), letters):
    axis.annotate("{}".format(letter), xy=(size / 20, size / 10), color="white")

###############
fig.savefig("Figures/psfshift9panels.png", dpi=300, bbox_inches="tight")
