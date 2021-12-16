import glob
import copy
import numpy as np
import os
import astropy.table as tbl
from astropy import time, coordinates as coord, units as u
from astropy.stats import LombScargle
from astropy.io import fits
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from scipy.stats import sigmaclip
import scipy.signal
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from matplotlib.patches import ConnectionPatch

#i, TIC_ID, RA, Dec, Year, Mon, Day, HR, Min, g_mags, TESS_mags, distance, SpT, mass, Prot, Ro, Evry_Erg, e_Evry_Erg, TESS_erg, e_TESS_Erg, evr_peakFF, tess_peakFF, n_peaks, tot_BB_data, e_tot_BB_data, tot_BB_data_trap, e_tot_BB_data_trap, E_tot_BB_data_trap, tot_BB_sampl, e_tot_BB_sampl, E_tot_BB_sampl, FWHM_BB_data, e_FWHM_BB_data, FWHM_BB_sampl, e_FWHM_BB_sampl, E_FWHM_BB_sampl, FWHM, impulse

i=[] #0
TIC_ID=[] #1
RA=[] #2
Dec=[] #3
Year=[] #4
Mon=[] #5
Day=[] #6
HR=[] #7
Min=[] #8
g_mags=[] #9
TESS_mags=[] #10
distance=[] #11
SpT=[] #12
mass=[] #13
Prot=[] #14
Ro=[] #15
Evry_Erg=[] #16
e_Evry_Erg=[] #17
TESS_erg=[] #18
e_TESS_Erg=[] #19
evr_peakFF=[] #20
tess_peakFF=[] #21
n_peaks=[] #22
tot_BB_data=[] #23
e_tot_BB_data=[] #24
tot_BB_data_trap=[] #25
e_tot_BB_data_trap=[] #26
E_tot_BB_data_trap=[] #27
tot_BB_sampl=[] #28
e_tot_BB_sampl=[] #29
E_tot_BB_sampl=[] #30
FWHM_BB_data=[] #31
e_FWHM_BB_data=[] #32
FWHM_BB_sampl=[] #33
e_FWHM_BB_sampl=[] #34
E_FWHM_BB_sampl=[] #35
FWHM=[] #36
impulse=[] #37

with open("evryflare_III_table_I.csv","r") as INFILE:
    next(INFILE)
    for lines in INFILE:
        i.append(int(lines.split(",")[0])) #0
        TIC_ID.append(int(lines.split(",")[1])) #1
        RA.append(float(lines.split(",")[2])) #2
        Dec.append(float(lines.split(",")[3])) #3
        Year.append(int(lines.split(",")[4])) #4
        Mon.append(int(lines.split(",")[5])) #5
        Day.append(int(lines.split(",")[6])) #6
        HR.append(int(lines.split(",")[7])) #7
        Min.append(int(lines.split(",")[8])) #8
        g_mags.append(float(lines.split(",")[9])) #9
        TESS_mags.append(float(lines.split(",")[10])) #10
        distance.append(float(lines.split(",")[11])) #11
        SpT.append(str(lines.split(",")[12])) #12
        mass.append(float(lines.split(",")[13])) #13
        Prot.append(float(lines.split(",")[14])) #14
        Ro.append(float(lines.split(",")[15])) #15
        Evry_Erg.append(float(lines.split(",")[16])) #16
        e_Evry_Erg.append(float(lines.split(",")[17])) #17
        TESS_erg.append(float(lines.split(",")[18])) #18
        e_TESS_Erg.append(float(lines.split(",")[19])) #19
        evr_peakFF.append(float(lines.split(",")[20])) #20
        tess_peakFF.append(float(lines.split(",")[21])) #21
        n_peaks.append(int(lines.split(",")[22])) #22
        tot_BB_data.append(float(lines.split(",")[23])) #23
        e_tot_BB_data.append(float(lines.split(",")[24])) #24
        tot_BB_data_trap.append(float(lines.split(",")[25])) #25
        e_tot_BB_data_trap.append(float(lines.split(",")[26])) #26
        E_tot_BB_data_trap.append(float(lines.split(",")[27])) #27
        tot_BB_sampl.append(float(lines.split(",")[28])) #28
        e_tot_BB_sampl.append(float(lines.split(",")[29])) #29
        E_tot_BB_sampl.append(float(lines.split(",")[30])) #30
        FWHM_BB_data.append(float(lines.split(",")[31])) #31
        e_FWHM_BB_data.append(float(lines.split(",")[32])) #32
        FWHM_BB_sampl.append(float(lines.split(",")[33])) #33
        e_FWHM_BB_sampl.append(float(lines.split(",")[34])) #34
        E_FWHM_BB_sampl.append(float(lines.split(",")[35])) #35
        FWHM.append(float(lines.split(",")[36])) #36
        impulse.append(float(lines.split(",")[37])) #37

i = np.array(i)  
TIC_ID=np.array(TIC_ID)
mass = np.array(mass)
FWHM = np.array(FWHM)
g_mags= np.array(g_mags)
distance= np.array(distance)
Evry_Erg = np.array(Evry_Erg)
TESS_Erg = np.array(TESS_erg)
e_Evry_Erg = np.array(e_Evry_Erg)
Evry_Erg_bol = np.log10((10.0**Evry_Erg)/0.19)
SpT = np.array(SpT)
FWHM_BB_data = np.array(FWHM_BB_data)
tot_BB_data = np.array(tot_BB_data)
e_FWHM_BB_data = np.array(e_FWHM_BB_data)
e_tot_BB_data = np.array(e_tot_BB_data)
FWHM_BB_sampl = np.array(FWHM_BB_sampl)
tot_BB_sampl = np.array(tot_BB_sampl)
tot_BB_data_trap=np.array(tot_BB_data_trap)
e_tot_BB_data_trap=np.array(e_tot_BB_data_trap)
E_tot_BB_data_trap=np.array(E_tot_BB_data_trap)
impulse = np.array(impulse)
evr_peakFF = np.array(evr_peakFF)
tess_peakFF = np.array(tess_peakFF)
color = evr_peakFF - tess_peakFF
n_peaks = np.array(n_peaks).astype(float)
Ro = np.array(Ro)
Prot = np.array(Prot)

e_FWHM_BB_data[e_FWHM_BB_data<300.0]=300.0
e_tot_BB_data[e_tot_BB_data<300.0]=300.0

#e_color = np.absolute(color)*np.sqrt((evry_pk_err/evr_peakFF)**2.0 + (tess_pk_err/tess_peakFF)**2.0)
#e_impulse = np.absolute(impulse)*np.sqrt((evry_pk_err/evr_peakFF)**2.0 + (1.0/FWHM)**2.0)

def compute_1sigma_CI(input_array):

    sorted_input_array = np.sort(input_array)

    low_ind = int(0.16*len(input_array))
    high_ind = int(0.84*len(input_array))

    bot_val = (sorted_input_array[:low_ind])[-1]
    top_val = (sorted_input_array[high_ind:])[0]

    bot_arr_err = abs(np.nanmedian(input_array) - bot_val)
    top_arr_err = abs(np.nanmedian(input_array) - top_val)

    return (np.nanmedian(input_array), bot_arr_err, top_arr_err)

def get_temp_data(i):

    data_times=[]
    data_temps=[]
    data_lowerr=[]
    data_upperr=[]
    data_formal_lowerr=[]
    data_formal_upperr=[]
    with open(str(i)+"_flaretemp_data_lc.csv","r") as INFILE:
        for lines in INFILE:
            data_times.append(float(lines.split(",")[0]))
            data_temps.append(float(lines.split(",")[1]))
            data_lowerr.append(float(lines.split(",")[2]))
            data_upperr.append(float(lines.split(",")[3]))
            data_formal_lowerr.append(float(lines.split(",")[4]))
            data_formal_upperr.append(float(lines.split(",")[5]))
    data_times=np.array(data_times)
    data_temps=np.array(data_temps)
    data_lowerr=np.array(data_lowerr)
    data_upperr=np.array(data_upperr)
    data_formal_lowerr=np.array(data_formal_lowerr)
    data_formal_upperr=np.array(data_formal_upperr)

    return (data_times, data_temps, data_lowerr, data_upperr, data_formal_lowerr, data_formal_upperr) 

def get_temp_model(i):

    model_times=[]
    model_temps=[]
    model_lowerr=[]
    model_upperr=[]
    with open(str(i)+"_flaretemp_model_lc.csv","r") as INFILE:
        for lines in INFILE:
            model_times.append(float(lines.split(",")[0]))
            model_temps.append(float(lines.split(",")[1]))
            model_lowerr.append(float(lines.split(",")[2]))
            model_upperr.append(float(lines.split(",")[3]))
    model_times=np.array(model_times)
    model_temps=np.array(model_temps)
    model_lowerr=np.array(model_lowerr)
    model_upperr=np.array(model_upperr)
    
    return (model_times, model_temps, model_lowerr, model_upperr)

def get_flare_fits(i):
    
    fit_times=[]
    fit_evry_fracflux=[]
    fit_tess_fracflux=[]
    with open(str(i)+"_flare_fits_lc.csv","r") as INFILE:
        for lines in INFILE:
            fit_times.append(float(lines.split(",")[0]))
            fit_evry_fracflux.append(float(lines.split(",")[1]))
            fit_tess_fracflux.append(float(lines.split(",")[2]))
    fit_times=np.array(fit_times)
    fit_evry_fracflux=np.array(fit_evry_fracflux)
    fit_tess_fracflux=np.array(fit_tess_fracflux)
            
    return (fit_times, fit_evry_fracflux, fit_tess_fracflux)

def get_fracflux(i):

    x_tess_and_evry=[]
    y_tess_and_evry=[]
    y_err_tess_and_evry=[]
    flag=[]
    with open(str(i)+"_flare_fluxes_lc.csv","r") as INFILE:
        for lines in INFILE:
            x_tess_and_evry.append(float(lines.split(",")[0]))
            y_tess_and_evry.append(float(lines.split(",")[1]))
            y_err_tess_and_evry.append(float(lines.split(",")[2]))
            flag.append(int(lines.split(",")[3]))
    x_tess_and_evry=np.array(x_tess_and_evry)
    y_tess_and_evry=np.array(y_tess_and_evry)
    y_err_tess_and_evry=np.array(y_err_tess_and_evry)
    flag=np.array(flag)

    x_tess = x_tess_and_evry[flag==0]
    y_tess = y_tess_and_evry[flag==0]
    y_tess_err = y_err_tess_and_evry[flag==0]
    x_evry = x_tess_and_evry[flag==1]
    y_evry = y_tess_and_evry[flag==1]
    y_evry_err = y_err_tess_and_evry[flag==1]
    
    return (x_tess, y_tess, y_tess_err, x_evry, y_evry, y_evry_err)

###################################################################


fig, ax = plt.subplots(figsize=(12,3))
plt.axis('off')

##### Flare 1 ########

x_tess, y_tess, y_tess_err, x_evry, y_evry, y_evry_err = get_fracflux(0)

fit_times, fit_evry_fracflux, fit_tess_fracflux = get_flare_fits(0)

y_tess = y_tess[x_tess<0.12]
y_tess_err = y_tess_err[x_tess<0.12] 
x_tess = x_tess[x_tess<0.12]
fit_evry_fracflux = fit_evry_fracflux[fit_times<0.12]
fit_tess_fracflux = fit_tess_fracflux[fit_times<0.12]
fit_times = fit_times[fit_times<0.12]

y_tess_mag = -2.5*np.log10(1.0 + y_tess)
y_evry_mag = -2.5*np.log10(1.0 + y_evry)
fit_evry_mag = -2.5*np.log10(1.0 + fit_evry_fracflux)
fit_tess_mag = -2.5*np.log10(1.0 + fit_tess_fracflux)

ax1a = fig.add_subplot(1,2,1)
plt.gca().invert_yaxis()

plt.xlabel("Time [min]",color="black",fontsize=13)
plt.ylabel("$\Delta g^{\prime}$-mag",color="royalblue",fontsize=13)
plt.yticks(color="royalblue",fontsize=13)
plt.xticks(color="black",fontsize=13)

ax1b = ax1a.twinx()
#ax1a.set_ylim([0.05, -0.45]) #Evryscope
ax1b.set_ylim([-0.9, 0.035]) #TESS [-0.685, 0.035]

ax1a.plot((fit_times-0.0573)*24.0*60.0, fit_evry_mag, color="lightgrey")
ax1a.plot((x_evry-0.0573)*24.0*60.0, y_evry_mag, marker="o",ls="none",color="cornflowerblue")

ax1b.plot((fit_times-0.0573)*24.0*60.0, fit_tess_mag, color="grey",alpha=0.66)
ax1b.plot((x_tess-0.0573)*24.0*60.0, y_tess_mag, marker="o",ls="none",color="firebrick")


plt.gca().invert_yaxis()

plt.yticks(color="firebrick",fontsize=13)
plt.xticks(color="black",fontsize=13)
plt.ylabel("$\Delta$TESS mag",color="firebrick",fontsize=13)

plt.text(38,-0.69,"TIC-294750180,\n2018-10-20 5:36 UT", fontsize=12)
#plt.text(-18, -0.75, "A", fontsize=16, color="black")

xyA=(45, -0.05)
xyB= (63,-0.25)

con = ConnectionPatch(xyA, xyB, coordsA="data", coordsB="data", axesA=ax1b, axesB=ax1b, color="grey")
ax1b.add_artist(con)

plt.text(65,-0.24, "TESS data",color="firebrick",fontsize=12)

xyA=(22, -0.21)
xyB= (39,-0.46)

con = ConnectionPatch(xyA, xyB, coordsA="data", coordsB="data", axesA=ax1b, axesB=ax1b, color="grey")
ax1b.add_artist(con)

plt.text(41.2,-0.444, "Evryscope data",color="royalblue",fontsize=12)

##### Flare 2 ########

data_times, data_temps, data_lowerr, data_upperr, data_formal_lowerr, data_formal_upperr = get_temp_data(0)

data_temps = data_temps[data_times<0.12]
data_lowerr = data_lowerr[data_times<0.12]
data_upperr = data_upperr[data_times<0.12]
data_formal_lowerr = data_formal_lowerr[data_times<0.12]
data_formal_upperr = data_formal_upperr[data_times<0.12]
data_times = data_times[data_times<0.12]

ax2 = fig.add_subplot(1,2,2)

ax2.fill_between((data_times-0.0573)*24.0*60.0, data_temps-data_lowerr,data_temps+data_upperr,color="lightgrey",zorder=0.01)
ax2.errorbar((data_times-0.0573)*24.0*60.0, data_temps, yerr=[data_formal_lowerr, data_formal_upperr], marker="o", color="black", ls="none",zorder=0.05)
ax2.plot((data_times[data_temps<900.0]-0.0573)*24.0*60.0, data_temps[data_temps<900.0], marker="o", ms=2, color="white", ls="none",zorder=1.0)

plt.xlabel("Time [min]",color="black", fontsize=13)
plt.ylabel("Temperature [K]", fontsize=13)
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)

#ax2.set_ylim([0.05, -0.45]) #Evryscope

#ax2.plot((data_times-0.0573)*24.0*60.0, y_evry_mag, marker="o",ls="none",color="cornflowerblue")

#plt.text(47,39000,"TIC-294750180,\n2018-10-20 5:36 UT", fontsize=12, bbox=dict(facecolor='white', alpha=0.999))
#plt.text(-19, 43500, "A", fontsize=16, color="black")

plt.tight_layout()
plt.savefig("simulflare_51Pegb.png")
plt.show()



