import importlib
import os
import sys
import numpy as np
import pandas as pd
from Utils.ztf_shift_utils import *

scistring = sys.argv[1]
refstring = sys.argv[2]
date = float(sys.argv[3])
flr_ra = float(sys.argv[4])
flr_dec = float(sys.argv[5])
imcen_ra = float(sys.argv[6])
imcen_dec = float(sys.argv[7])
search_rad = float(sys.argv[8])

##Import images files from IRSA
print(" ")
print("--------------------------")
print("Importing ZTF science and GAIA reference files...")

#Download science image from ZTF API
getsciimg(filefracday = scistring[0:14], paddedfield = scistring[14:20],filtercode = scistring[20:22],
          paddedccdid = scistring[22:24],imgtypecode = scistring[24], qid = scistring[25])

#Create reference catalog from Gaia positions
ref_cata = pd.read_csv(refstring)
#print(imcen_ra, imcen_dec, search_rad)
#ref_cata = getgaiacat(imcen_ra, imcen_dec, search_rad)
grcolors = ref_cata['gmag'].values - ref_cata['rmag'].values
gicolors = ref_cata['gmag'].values - ref_cata['imag'].values

##Generate srcext catalogues
print(" ")
print("--------------------------")

if os.path.isfile('srcext/'+scistring[0:14]+'_'+scistring[14:20]+'_sciimg.fits') == True:
    print('SExtracted Catalogue Exists, moving on...')
    sci_cata = pd.read_table('srcext/out.cat', names=['NUMBER',
    'X_IMAGE',
    'Y_IMAGE',
    'XWIN_IMAGE',
    'YWIN_IMAGE',
    'XMODEL_IMAGE',
    'YMODEL_IMAGE',
    'FLUX_AUTO',
    'FLUX_MODEL',
    'MAG_AUTO',
    'MAG_MODEL',
    'FLUX_RADIUS',
    'FLAGS',
    'NITER_MODEL',
    'ALPHA_SKY',
    'DELTA_SKY',
    'THETA_WORLD',
    'ELLIPTICITY'], index_col=0, comment='#', delim_whitespace=True)

    with fits.open('srcext/'+scistring[0:14]+'_'+scistring[14:20]+'_sciimg.fits') as hdu:
        sci_data = hdu[0].data
        sci_header = hdu[0].header

else:
    print("Creating source extractor catalogues")
    sci_cata = srcext(scistring[0:14]+'_'+scistring[14:20]+'_sciimg.fits', det_thresh=15, ana_thresh=15)

    with fits.open('srcext/'+scistring[0:14]+'_'+scistring[14:20]+'_sciimg.fits') as hdu:
        sci_data = hdu[0].data
        sci_header = hdu[0].header

##Crossmatch catalogues
print(" ")
print("--------------------------")
print("Running Astropy x-match...")

circ_ind = circle_cut(185.68352128, 65.33824434, ref_cata, radius=0.5*u.degree)

ind, d2d, d3d = xmatch_gaia(sci_cata,ref_cata)

seps = d2d.to(u.arcsec).value

ref_coord = SkyCoord(ref_cata["ra"][ind].values * u.degree, 
                ref_cata["dec"][ind].values * u.degree, frame='icrs')
sci_coord = SkyCoord(sci_cata["ALPHA_SKY"].values * u.degree, 
                sci_cata["DELTA_SKY"].values * u.degree, frame='icrs')

#Calculate zenith direction
zenith = calc_zenith(date)

theta_1 = ref_coord.position_angle(zenith).arcsecond
theta_2 = ref_coord.position_angle(sci_coord).arcsecond

good_sep = seps < 1

proj = seps[good_sep] * np.cos(theta_1[good_sep] - theta_2[good_sep])

am  = sci_header[97]

##Find flare star

f_ind = np.where(np.abs(sci_cata[["ALPHA_SKY","DELTA_SKY"]][good_sep].values - [flr_ra, flr_dec]) == np.abs(sci_cata[["ALPHA_SKY","DELTA_SKY"]][good_sep].values - [flr_ra, flr_dec]).min())[0][0]

print(" ")
print("--------------------------")
print("Plotting...")
plot_shifts_gaia(ref_cata["ra"].values[ind][good_sep],ref_cata["dec"].values[ind][good_sep],
            sci_cata["ALPHA_SKY"].values[good_sep],sci_cata["DELTA_SKY"].values[good_sep], 
            zenith.ra.value, zenith.dec.value, f_ind, date, am, circ_ind[ind][good_sep], grcolors[ind][good_sep], centered=False)
plt.savefig('Figures/flare_dshift_plots/'+str(date)+'GAIA.png',dpi=300,bbox_inches='tight')