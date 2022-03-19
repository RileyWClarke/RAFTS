import importlib
import os
import sys
import numpy as np
from ztf_shift_utils import *

scistring = sys.argv[1]
refstring = sys.argv[2]
date = float(sys.argv[3])
flr_ra = float(sys.argv[4])
flr_dec = float(sys.argv[5])

##Import images files from IRSA
print(" ")
print("--------------------------")
print("Importing image files...")

getsciimg(filefracday = scistring[0:14], paddedfield = scistring[14:20],filtercode = scistring[20:22],
          paddedccdid = scistring[22:24],imgtypecode = scistring[24], qid = scistring[25], flip=False)
getrefimg(paddedfield = refstring[0:6],filtercode = refstring[6:8], paddedccdid = refstring[8:10], qid = refstring[10])

##Generate srcext catalogues
print(" ")
print("--------------------------")
print("Creating source extractor catalogues")
sci_cata = srcext(scistring[0:14]+'_'+scistring[14:20]+'_sciimg.fits', det_thresh=15, ana_thresh=15)
ref_cata = srcext(refstring[0:6]+'_refimg.fits', det_thresh=15, ana_thresh=15)

with fits.open('srcext/'+scistring[0:14]+'_'+scistring[14:20]+'_sciimg.fits') as hdu:
    sci_data = hdu[0].data
    sci_header = hdu[0].header

with fits.open('srcext/'+refstring[0:6]+'_refimg.fits') as hdu:
    ref_data = hdu[0].data
    ref_header = hdu[0].header

##Crossmatch catalogues
print(" ")
print("--------------------------")
print("Running Astropy x-match...")
ind, d2d, d3d = xmatch(sci_cata,ref_cata)

seps = d2d.to(u.arcsec).value

ref_coord = SkyCoord(ref_cata["ALPHA_SKY"][ind].values * u.degree, 
                ref_cata["DELTA_SKY"][ind].values * u.degree, frame='icrs')
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
plot_shifts(ref_cata["ALPHA_SKY"].values[ind][good_sep],ref_cata["DELTA_SKY"].values[ind][good_sep],
            sci_cata["ALPHA_SKY"].values[good_sep],sci_cata["DELTA_SKY"].values[good_sep], zenith.ra.value, zenith.dec.value, f_ind, date, am, centered=True)
plt.savefig('Figures/flare_dshift_plots/'+refstring+'_'+str(date)+'.png',dpi=300,bbox_inches='tight')