# Notebook workflow:

- `dwfflarePSF_prepimagefits.ipynb` => saves fits files pertaining to the relevant chip (16) as separate fits for reading (original images have to be in `dwfflare`, chip fits are saved into `dwfflareS18` 
- `dwfflarePSFastrometry_revisions.ipynb` => produces the astrometric and photometric time series for all stars used in the analysis. Data frame with photometry is saved into `Outdata/main_df_GaussPSF.csv`
- `dwfflarePSFastrometry_processing_NEWPSF.ipynb` => produces the time series of d_parallel which is used for the DCR analysis and saves it into `Outdata/flare_df.csv`. Saved intermediate files: 

      'Outdata/dpar_arr.npy' => dpar for all stars
      'Outdata/dpar_raw.npy' => dpar flare star raw
      'Outdata/dpar_subtracted.npy' => dpar flare star removed median of all other stars
      'Outdata/dparerr.npy' => dpar flare star raw errors
      'Outdata/dpar_smoothed.npy' => dpar flare star removed median and smoothed with rolling aggregate (median)
      'Outdata/dparerr_smoothed.npy' => dpar flare star removed median and smoothed with rolling aggregate (median) errors
      'Outdata/delta_g_mag.npy' =>  flare star magnitude in g band
      'Outdata/delta_g_mag_err.npy' =>  flare star magnitude errors in g band
      'Outdata/times.npy' => time stamps
      `Outdata/dt_sec.npy` => time in seconds
      `Outdata/ras.npy' => flare star RA arrays
      `Outdata/dec.npy' => flare star Dec arrays
      `Outdata/raserr.npy' => flare star RA errors arrays
      `Outdata/decerrs.npy' => flare star Dec errors arrays
      `Outdata/ra_change.npy` => relative RA (relative to initial position)
      `Outdata/dec_chane.npy` => relative Dec (relative to initial position)

- `dwfflare_dparrStatisticalAnalysis.ipynb` => finds the point of change in the d_parallel time series and measures the statistical significance

- `dwfflareTeffModel.ipynb` => Ingests data from `Outdata/`, creates lookup tables for delta magnitude and effective wavelength as functions of blackbody temperature ($T_{BB}$) and filling factor ($X_{BB}$), for both the emission line SED model and the blackbody-only SED model. The dpar inputs are converted to effective wavelengths (`flr_weffs.csv`) and corrected for instrumental distortion. Errors on derived effective wavelengths are calculated via bootstrap. For each epoch in flr_weffs and delta_g_mag, the lookup tables are searched for $T_{BB}$ and $X_{BB}$ that minizmize the reduced $\chi^2$ of the inputs. Resulting $T_{BB}$ and $X_{BB}$ curves are output and plotted.

  ...

=========================

Utility functions are in files which must be stored in the previous directory including:
- defs.py
- utils.py

  
=========================
Additional files not part of the pipeline, but used for tests throughout our study
- `dwfflarePSFastrometry_tests.ipynb` => tests different approaches to astrometry
- 
	
