# Notebook workflow:

- `dwfflarePSF_prepimagefits.ipynb` => saves fits files pertaining to the relevant chip (16) as separate fits for reading (original images have to be in `dwfflare`, chip fits are saved into `dwfflareS18` 
- `dwfflarePSFastrometry_revisions.ipynb` => produces the astrometric and photometric time series for all stars used in the analysis. Data frame with photometry is saved into `Outdata/main_df_GaussPSF.csv`
- `dwfflarePSFastrometry_processing_NEWPSF.ipynb` => produces the time series of d_parallel which is used for the DCR analysis. Saved intermediate files: 

      'Outdata/dpar_arr.npy' => dpar for all stars

      'Outdata/dpar_raw.npy' => dpar flare star raw

      'Outdata/dpar_subtracted.npy' => dpar flare star removed median of all other stars

      'Outdata/dparerr.npy' => dpar flare star raw errors

      'Outdata/delta_g_mag.npy' => dpar flare star magnitude in g band

      'Outdata/delta_g_mag_err.npy' => dpar flare star magnitude errors in g band

      'Outdata/times.npy' => time stamps
