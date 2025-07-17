import sys
def f(x):
    return x*x


from photutils.psf import IntegratedGaussianPRF, PSFPhotometry, ImagePSF, MoffatPSF, CircularGaussianPRF, CircularGaussianPSF, GaussianPSF, GaussianPRF, CircularGaussianSigmaPRF
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from photutils.background import MMMBackground, LocalBackground
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import iqr
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
from astropy.time import Time
from astropy.timeseries import LombScargle
from astropy.wcs import WCS
from astropy import units as u
from astropy.io import fits
from astropy import constants
from astropy.modeling.models import BlackBody
from astropy.visualization import quantity_support

from utils import filt_interp, dpar, dtan, lamb_eff_md, dcr_offset, celest_to_pa, inverse_Teff, inverseTeff, inverseWeff, lorentzian, find_min_max_adjacent, variance_weighted_mean, chrDistAng
import globals


threshold = 1#np.linspace(0.1,3,10)
fwhm = 2#0.1, 0.2, 0.3]#np.linspace(0.5,2,3)
roundlo = -0.5#np.linspace(-4,0,3)
roundhi = 0.5#np.linspace(0,4,5)
peakmax = 10_000#np.linspace(1000,100_000,10)
minsep = 5#np.linspace(2,20,10)
imgsize = 500
step=1

from photutils.detection import find_peaks
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData
from photutils.psf import extract_stars
from astropy.visualization import simple_norm
from photutils.psf import EPSFBuilder

def makefitsfromext(file, ext, outdir):
    hdu_list = fits.open(file)
    data = hdu_list[ext].data
    header = hdu_list[ext].header

    hdr = header
    hdr['DATE-OBS'] = hdu_list[0].header['DATE-OBS']

    empty_primary = fits.PrimaryHDU(header=hdr)

    image_hdu = fits.ImageHDU(data)

    hdul = fits.HDUList([empty_primary, image_hdu])

    hdul.writeto(outdir, overwrite=True)

def makefitpsf(data, header, fit_shape, oversampling=4, positions=None, plot=False):
    if positions is None:
        mmm_bkg = MMMBackground()
        localbkg_estimator = LocalBackground(5, 10, mmm_bkg)

        finder = DAOStarFinder(threshold=1*mmm_bkg(data), fwhm=2*header['FWHM'], 
                       roundlo=-0.3, roundhi=0.3, peakmax=10_000, exclude_border=True)

        psfphot = PSFPhotometry(CircularGaussianPRF(fwhm=header['FWHM']), fit_shape, finder=finder, aperture_radius=10,
                                                    localbkg_estimator=localbkg_estimator)
        phot = psfphot(data)
        phot = phot[phot['flux_fit'] / phot['flux_err'] > 20]

        #removing stars at the edge
        size = 25
        hsize = (size - 1) / 2
        x = phot['x_fit']
        y = phot['y_fit']
        mask = ((x > hsize) & (x < (data.shape[1] -1 - hsize)) &
            (y > hsize) & (y < (data.shape[0] -1 - hsize)))

        

        positions = np.transpose(((x[mask], y[mask])))
    stars_tbl = Table()
    stars_tbl['x'] = positions[:,0]
    stars_tbl['y'] = positions[:,1]
    xycoords = positions
    
    if plot:
        #plot star finder positions
        apertures = CircularAperture(positions, r=20.)
        plt.figure(figsize=(20,10))
        plt.imshow(data, clim=(0,100), cmap='Greys_r')
        apertures.plot(color='red')
        for i,p in enumerate(positions):
            plt.annotate('{0}'.format(i), xy=p, xytext=p+30, color='r')
        plt.title('Source detection')
        plt.show()
        
    #remove background
    mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.0)
    data_clean = data.copy() - median_val
    #get cutouts
    nddata = NDData(data=data_clean)
    stars = extract_stars(nddata, stars_tbl, size=25)

    if plot:
        #plot cutout around the target star
        plt.figure()
        nrows = len(stars) // 5 +1
        ncols = 5
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
                       squeeze=True)
        ax = ax.ravel()
        i=0
        while i <len(stars):
            norm = simple_norm(stars[i], 'log', percent=99.0)
            ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
            i+=1
        plt.show()
    #build empirical function
    epsf_builder = EPSFBuilder(oversampling=oversampling, maxiters=20,
                           progress_bar=False, center_accuracy=0.1)
    epsf, fitted_stars = epsf_builder(stars)
    
    #plot the psf fitted stars 
    if True:#plot:
        norm = simple_norm(epsf.data, 'log', percent=99.0)
        plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
        plt.colorbar()
        plt.show()
    return epsf

def readinfitsinfo(img):
    #read in image data
    data = fits.getdata(img)
    header = fits.getheader(img)
    wcs = WCS(header)
    time = Time(header['DATE-OBS'])   
    return data, header, wcs, time

def photometry(psf_model_init, positions, fit_shape, finder, localbkg_estimator,
                        data, aperture_radius=10, header=None, plot=False):
    ############### PSF photometry 
    if psf_model_init is None:
        print("build PSF model")
        assert header, "need to pass the header if you want to do epsf"
                                
        psfmodel = makefitpsf(data, header, fit_shape, oversampling=4, positions=positions, plot=plot)
        norm = simple_norm(psfmodel.data, 'log', percent=99.0)
                                
        plt.imshow(psfmodel.data, norm=norm, origin='lower', cmap='viridis')
        plt.colorbar()
        plt.show()
            
    else:
        psfmodel = psf_model_init 
            
    psfphot = PSFPhotometry(psfmodel, fit_shape, finder=finder, aperture_radius=10,
                                                    localbkg_estimator=localbkg_estimator)
    phot = psfphot(data)
    return phot.to_pandas()

      
def extract(start_to_end, psf_model_init, imnames_timeordered,  xycoords, 
            threshold, fwhm, roundlo, roundhi, peakmax, minsep,
            flare_ref_pos, fit_shape, match_dist, flr_id, plot=False):
    start, end = start_to_end
    matchescounts = []
    main_df_local = pd.DataFrame()
    Nimages = len(imnames_timeordered[start:end])
    for i, imname in enumerate(imnames_timeordered[start:end:1]):
        #loop over images
        
        print('{0}/{1}'.format(i+1, Nimages))
        print(imname)
        data, header, wcs, time =  readinfitsinfo('dwfflare/dwfflareS18/' + imname)    
        
        ################### find source in image
        mmm_bkg = MMMBackground()
        #print('Background: {}'.format(mmm_bkg.calc_background(data)))
        #print('Seeing: {} arcsec'.format(header['FWHM'] * 0.2637))
        localbkg_estimator = LocalBackground(5, 10, mmm_bkg)
        finder = DAOStarFinder(threshold=threshold*mmm_bkg.calc_background(data), 
                                                   fwhm=fwhm*header['FWHM'], 
                                                   roundlo=roundlo, roundhi=roundhi, 
                                                   peakmax=peakmax, exclude_border=True, 
                                                   xycoords=xycoords, min_separation=minsep)
        sources = finder(data) 
        fwhm2 = fwhm
        while not sources:
            fwhm2 = fwhm2 + 1
            print(f"No sources found, something is wrong, increasing FWHM to {fwhm2}")
            finder = DAOStarFinder(threshold=threshold*mmm_bkg.calc_background(data), 
                                                   fwhm=fwhm2*header['FWHM'], 
                                                   roundlo=roundlo, roundhi=roundhi, 
                                                   peakmax=peakmax, exclude_border=True, 
                                                   xycoords=xycoords, min_separation=minsep)
            sources = finder(data)          
        
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

        #check if the flare star was found
        flrmatche = (positions[:,0] - flare_ref_pos[0])**2 + (positions[:,1] - flare_ref_pos[1])**2
        
        #if flare star not found find stars again with relaxed PSF width
        if not (flrmatche<PIXDISTSQ).sum():
            print(f"WARNING: no object found near flare star - redoing DAOPHOT with {fwhm2+1}xFWHM")
            finder = DAOStarFinder(threshold=1.0*mmm_bkg.calc_background(data), fwhm=(fwhm2+1)*header['FWHM'], 
                               roundlo=-1, roundhi=1, peakmax=10_000, exclude_border=True, 
                               xycoords=xycoords, min_separation=10)
            sources = finder(data)
        assert sources, "No sources found, something is wrong"

        print(f"nsources detected {len(sources)}")
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

        _ = photometry(psf_model_init, positions, fit_shape, finder, localbkg_estimator,
                        data, aperture_radius=10, header=header, plot=plot)    

        flrmatche = (_['x_fit'] - flare_ref_pos[0])**2 + (_['y_fit'] - flare_ref_pos[1])**2
        if not (flrmatche<PIXDISTSQ).sum():
            print("WARNING: redoing psphot with Circular Gaussian PRF model")
            _ = photometry(CircularGaussianPRF(fwhm=header['FWHM']), positions, fit_shape, finder, localbkg_estimator,
                        data, aperture_radius=10, header=header, plot=plot)    
        
        if not (flrmatche < PIXDISTSQ).sum():
            print("WARNING!!!!!!! we were not able to get the photometry for the flare star")
      
        ################# Build astrometry and photoemtry dataframe
        if i == 0:
            #first image is a reference dataframe
            ref_df = _
            ref_df['RA'] = wcs.all_pix2world(ref_df[['x_fit', 'y_fit']].values, 1)[:,0]
            ref_df['DEC'] = wcs.all_pix2world(ref_df[['x_fit', 'y_fit']].values, 1)[:,1]
            ref_df['RA_ERR'] = np.abs( wcs.all_pix2world(ref_df[['x_fit', 'y_fit']].values + ref_df[['x_err', 'y_err']].values, 1)[:,0] - ref_df['RA'].values )
            ref_df['DEC_ERR'] = np.abs( wcs.all_pix2world(ref_df[['x_fit', 'y_fit']].values + ref_df[['x_err', 'y_err']].values, 1)[:,1] - ref_df['DEC'].values )
            ref_df['time'] = time.mjd
            main_df_local = pd.concat([main_df_local, ref_df])
    
        else:
            #all epoch past first one
            img_df = _
            img_df['RA'] = wcs.all_pix2world(img_df[['x_fit', 'y_fit']].values, 1)[:,0]
            img_df['DEC'] = wcs.all_pix2world(img_df[['x_fit', 'y_fit']].values, 1)[:,1]
            img_df['RA_ERR'] = np.abs(wcs.all_pix2world(img_df[['x_fit', 'y_fit']].values + 
                                    img_df[['x_err', 'y_err']].values, 1)[:,0] - img_df['RA'].values )
            img_df['DEC_ERR'] = np.abs(wcs.all_pix2world(img_df[['x_fit', 'y_fit']].values + 
                                    img_df[['x_err', 'y_err']].values, 1)[:,1] - img_df['DEC'].values )
            img_df['time'] = time.mjd
            #match rows in img_df to ref_df locations
            matchescount, flarestar_matchescount = 0, 0

            for index, row in img_df.iterrows(): #for every row in the img_df file
                #match if distance in arcsec is within 0.5 
                matches = (np.sqrt(( (row['RA'] - ref_df['RA']) * 3600 )**2 + ( (row['DEC'] - ref_df['DEC']) * 3600 )**2) < match_dist ).values
                #print(matches, ref_df['id'].loc[matches])
                if matches.sum() == 1: #matches.any():
                    if ref_df['id'].loc[matches].values[0] == flr_id:
                        print("did we macth flare star? YES")
                        flarestar_matchescount += 1
                    #print(index, matches.sum())
                    matchescount += 1
                    
                    img_df['id'].iloc[index] = ref_df['id'].loc[matches]
                else:
                    #print("here", ref_df['id'].loc[matches])
                    #if flr_id in ref_df['id'].loc[matches]:
                        #if matches.sum() == 0:                    
                        #    print("did we macth flare star? No, no matches")
                        #else:
                            #print("did we macth flare star? No, ambiguous matches")                    
                    img_df.iloc[index] = np.full(img_df.shape[1], np.nan)
            print(f"matchescount {matchescount}, {flarestar_matchescount}")
            matchescounts.append(matchescount)

            main_df_local = pd.concat([main_df_local, img_df]) #main_df.append(img_df)
            
            #################### plot finding around flare postage stamp
            imgsize = 1_000
            if True:#plot:
                plt.figure(figsize=(4,4))
                plt.imshow(data[:imgsize,:imgsize], clim=(0,100), cmap='Greys_r')
                
                plt.scatter(flare_ref_pos[0], flare_ref_pos[1], marker='s', s=200, edgecolors='white', facecolors='None')
                #plt.scatter(ref_df['x_fit'], ref_df['y_fit'], facecolor='None', edgecolor='red', s=50)
                plt.scatter(img_df['x_fit'], img_df['y_fit'], edgecolors='blue', s=50, facecolors='None')
                for i, row in img_df.iterrows():
                    plt.annotate('{0}'.format(row['id']), xy=row[['x_fit', 'y_fit']], xytext=row[['x_fit', 'y_fit']]+50, color='blue')
                #for i, row in ref_df.iterrows():
                #    plt.annotate('{0}'.format(row['id']), xy=row[['x_fit', 'y_fit']], xytext=row[['x_fit', 'y_fit']]-50, color='red')
                plt.scatter(positions[:,0], positions[:,1], edgecolors='r', s=70, facecolors='None', alpha=0.5)
                
                plt.xlim(0,imgsize)
                plt.ylim(0,imgsize)
                plt.title('{}'.format(time))
                plt.show()
                
    return main_df_local, matchescounts

PIXDISTSQ = 25

def psffittests(psf_model_init, threshold, fwhm, roundlo, roundhi, peakmax, minsep, images, xycoords, flare_ref_pos, match_dist=1,
                fit_shape = (25, 25), sharplo=0.2, shaprhi=1, plot=False, report=False):
    fwhms = []
    nsources = []
    matchescounts = []
    matched_flare_phot = []
    matched_flare = []
    bgs = []
    main_df_local = pd.DataFrame({})
    for i, imnameS18 in enumerate(images):
        print(i)
        #loop over images
        #print('{0}/{1}'.format(i+1, len(imnames_timeordered[start:end])))
        data, header, wcs, time =  readinfitsinfo('dwfflare/dwfflareS18/' + imnameS18)    
        
        #############STAR FINDING BEFORE PSF PHOT
        mmm_bkg = MMMBackground()
        bgs.append(mmm_bkg.calc_background(data))
        print('Background: {}'.format(mmm_bkg.calc_background(data)))
        print('Seeing: {} arcsec'.format(header['FWHM'] * 0.2637))
        fwhms.append(header['FWHM'] * 0.2637)
        localbkg_estimator = LocalBackground(5, 10, mmm_bkg)
        finder = DAOStarFinder(threshold=threshold*mmm_bkg.calc_background(data), 
                               fwhm=fwhm*header['FWHM'], 
                               roundlo=roundlo, roundhi=roundhi, 
                               peakmax=peakmax, exclude_border=True, 
                               xycoords=xycoords, min_separation=minsep)
        sources = finder(data)
        
        if not sources:
            print(f"bad combo THR {threshold}, FWHM {fwhm}, LO {roundlo}, HI {roundhi}, PEAK {peakmax}, MINSEP {minsep}")
            nsources.append(0)
            matched_flare.append(0)
            matchescounts.append(0)
            matched_flare_phot.append(0)
            continue
            
        nsources.append(len(sources))
        print(f"nsources detected {len(sources)}")
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        
        matched_flare.append(0)
        
        flrmatche = (positions[:,0] - flare_ref_pos[0])**2 + (positions[:,1] - flare_ref_pos[1])**2
        if not (flrmatche<PIXDISTSQ).sum():
            print("WARNING: no object found near flare star - redoing DAOPHOT with 3xFWHM")
            finder = DAOStarFinder(threshold=threshold*mmm_bkg.calc_background(data), 
                               fwhm=3*header['FWHM'], 
                               roundlo=roundlo, roundhi=roundhi, 
                               peakmax=peakmax, exclude_border=True, 
                               xycoords=xycoords, min_separation=minsep)
            sources = finder(data)
            if not sources:
                print(f"bad combo THR {threshold}, FWHM {fwhm}, LO {roundlo}, HI {roundhi}, PEAK {peakmax}, MINSEP {minsep}")
                nsources.append(0)
                matched_flare.append(0)
                matchescounts.append(0)
                matched_flare_phot.append(0)
            #continue
        print(f"good combo THR {threshold}, FWHM {fwhm}, LO {roundlo}, HI {roundhi}, PEAK {peakmax}, MINSEP {minsep}")
            
        nsources.append(len(sources))
        print(f"nsources detected {len(sources)}")
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        matched_flare.append(0)
        for p in positions:
            if (p[0] - flare_ref_pos[0])**2 + (p[1] - flare_ref_pos[1])**2 < PIXDISTSQ:
                matched_flare[-1] = 1
        
        ############# FITTING PHOTOMETRY 
        _ = photometry(psf_model_init, positions, fit_shape, finder, localbkg_estimator,
                        data, aperture_radius=10, header=header, plot=plot)    

        flrmatche = (_['x_fit'] - flare_ref_pos[0])**2 + (_['y_fit'] - flare_ref_pos[1])**2
        if not (flrmatche<PIXDISTSQ).sum():
            print("WARNING: redoing psphot with Circular Gaussian PRF model")
            _ = photometry(CircularGaussianPRF(fwhm=header['FWHM']), positions, fit_shape, finder, localbkg_estimator,
                        data, aperture_radius=10, header=header, plot=plot)    
        
        if not (flrmatche < PIXDISTSQ).sum():
            print("WARNING!!!!!!! we were not able to get the photometry for the flare start")
            
        if i == 0:
            #first image is a reference dataframe
            ref_df = _
            ref_df['RA'] = wcs.all_pix2world(ref_df[['x_fit', 'y_fit']].values, 1)[:,0]
            ref_df['DEC'] = wcs.all_pix2world(ref_df[['x_fit', 'y_fit']].values, 1)[:,1]
            assert ((ref_df['x_fit'] - flare_ref_pos[0])**2 + (
                        ref_df['y_fit'] - flare_ref_pos[1])**2).min() < 10, "WARNING!!!!!!! NO GOOD MATCHES FOR THE FLARE STAR in the reference df, butting out"
                    
            main_df_local = pd.concat([main_df_local, ref_df])
            matched_flare_phot.append(0)
            matchescounts.append(0)
            
        else:
            #all epoch past first one
            img_df = _
            img_df['RA'] = wcs.all_pix2world(img_df[['x_fit', 'y_fit']].values, 1)[:,0]
            img_df['DEC'] = wcs.all_pix2world(img_df[['x_fit', 'y_fit']].values, 1)[:,1]
            
            #match rows in img_df to ref_df locations
            print(f"photometry sources {len(img_df)}")
            print(f"THR {threshold}, FWHM {fwhm}, LO {roundlo}, HI {roundhi}, PEAK {peakmax}, MINSEP {minsep}")
            matchescount = 0
            matched_flare_phot.append(0)
            for index, row in img_df.iterrows(): #for every row in the img_df file
                #match if distance in arcsec is within match_dist pixels 
                matches = (np.sqrt(( (row['RA'] - ref_df['RA']) * 3600 )**2 + ( 
                    (row['DEC'] - ref_df['DEC']) * 3600 )**2) < match_dist ).values
                
                if matches.sum() == 1: #if no or more than one matches, reject
                    matchescount += 1
                    
                    img_df['id'].iloc[index] = ref_df['id'].loc[matches]
                    sep_from_target_star_sq = (img_df['x_fit'].iloc[index] - flare_ref_pos[0])**2 + (
                        img_df['y_fit'].iloc[index] - flare_ref_pos[1])**2 
                    #check if the separation fron the frlare star is < PIXDIST 
                    if sep_from_target_star_sq < PIXDISTSQ:
                        matched_flare_phot[-1] = 1 
                        print(f"{index} is a possible match with the target star")
                        if report:
                            print("\t", flare_ref_pos[0], flare_ref_pos[1])
                            print("\t", img_df['x_fit'].iloc[index], img_df['y_fit'].iloc[index])
                else:               
                    img_df.iloc[index] = np.full(img_df.shape[1], np.nan)
            img_df = img_df[(img_df['x_fit'] < imgsize) * (img_df['y_fit'] < imgsize)]
            print(f"matchescount {matchescount}, {matched_flare_phot}")
            matchescounts.append(matchescount)
            '''
            for index, row in ref_df.iterrows():
    
                matches = (np.sqrt(( (row['RA'] - img_df['RA']) * 3600 )**2 + ( (row['DEC'] - img_df['DEC']) * 3600 )**2) < match_dist ).values
    
                if matches.any():
    
                    img_df['id'].iloc[matches] = ref_df['id'].loc[index]
    
                else:
                    
                    img_df.loc[index + 0.5] = np.full(img_df.shape[1], np.nan)
                    img_df['id'].loc[index + 0.5] = ref_df['id'].loc[index]
                    img_df = img_df.sort_index().reset_index(drop=True)
            '''
                    
            main_df_local = pd.concat([main_df_local, img_df]) #main_df.append(img_df)
            if plot:
                print("celest finder chart, black photometry output, blue matched output")
                fig, axs = plt.subplots(1,3,figsize=(12,4))
                axs[0].imshow(data[:imgsize,:imgsize], clim=(0,100), cmap='Greys_r')
                axs[0].set_title(f"THR {threshold}, FWHM {fwhm}, LO {roundlo}, HI {roundhi}, PEAK {peakmax}, MINSEP {minsep}")
                axs[0].scatter(flare_ref_pos[0], flare_ref_pos[1], marker='s', s=200, edgecolors='white', facecolors='None')
                for i, row in positions:
                    axs[0].scatter(xycoords[:,0], xycoords[:,1], edgecolors='#447799', s=100, facecolors='None')
                axs[0].set_xlim(0,imgsize)
                axs[0].set_ylim(0,imgsize)
                axs[1].imshow(data[:imgsize,:imgsize], clim=(0,100), cmap='Greys_r')
                axs[1].scatter(flare_ref_pos[0], flare_ref_pos[1], marker='s', s=200, edgecolors='white', facecolors='None')
                for i, row in positions:
                    axs[1].scatter(positions[:,0], positions[:,1], edgecolors='r', s=50, facecolors='None')
                axs[1].set_xlim(0,imgsize)
                axs[1].set_ylim(0,imgsize)
                axs[2].imshow(data[:imgsize,:imgsize], clim=(0,100), cmap='Greys_r')
                axs[2].scatter(flare_ref_pos[0], flare_ref_pos[1], marker='s', s=200, edgecolors='white', facecolors='None')
                axs[2].scatter(img_df['x_fit'], img_df['y_fit'], edgecolors='blue', s=50, facecolors='None')
                axs[0].set_xlim(0,imgsize)
                axs[0].set_ylim(0,imgsize)
                axs[1].set_xlim(0,imgsize)
                axs[1].set_ylim(0,imgsize)
                axs[2].set_xlim(0,imgsize)
                axs[2].set_ylim(0,imgsize)
                plt.show()
    return fwhms, nsources,  matchescounts ,matched_flare_phot, matched_flare, bgs

if __name__ == '__main__':
    f(sys.argv[1:])
           