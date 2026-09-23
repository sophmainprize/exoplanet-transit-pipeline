import os
import matplotlib.pyplot as plt
import numpy as np
import lightkurve as lk
from scipy.optimize import curve_fit

#set plot visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

#download & clean light curve data
search_result = lk.search_lightcurve("Kepler-10", mission="Kepler", quarter=2)
lc_raw = search_result.download()

lc_cleaned = (
    lc_raw
    .remove_nans()
    .normalize()
    .flatten(window_length=101)
    .remove_outliers(sigma=5)
)

#.flatten(window_length=101): applies a Savitzky-Golay filter to remove long-term stellar variability and instrumental drifts
#.remove_outliers(sigma=5): removes non-physical single-cadence spikes (e.g., cosmic ray strikes) beyond 5σ


#extract transit signal parameters via BLS

period_grid = np.linspace(0.4, 5.0, 10000)
bls = lc_cleaned.to_periodogram(method='bls', period=period_grid, frequency_factor=5.0)

best_period = bls.period_at_max_power.value        # ays
best_t0 = bls.transit_time_at_max_power.value     #BTJD
best_duration = bls.duration_at_max_power.value   #days
best_depth = bls.depth_at_max_power.value         #relative flux dip

print(f"BLS Best Period:   {best_period:.5f} days")
print(f"BLS Best Epoch T0: {best_t0:.4f} BTJD")
print(f"BLS Best Duration: {best_duration * 24:.2f} hours")
print(f"BLS Best Depth:    {best_depth * 1e6:.1f} ppm\n")

#phase-fold light curve data

folded_lc = lc_cleaned.fold(period=best_period, epoch_time=best_t0)
#folds time series data on top of itself at interval P centered around transit epoch T_0, stacking all individual transit events into a single transit profile

time_fold_hours = folded_lc.time.value * 24
flux_fold = folded_lc.flux.value
#converts phase time from days to hours

#bin data points for cleaner visualisation
binned_lc = folded_lc.bin(time_bin_size=0.002) #~3 minute bins
time_bin_hours = binned_lc.time.value * 24
flux_bin = binned_lc.flux.value
#groups data points into ~3-minute time intervals and averages them to produce a cleaner, binned light curve overlay


#define transit dip model and fit curve
def box_transit_model(t, depth, duration_hours, center_offset=0.0, baseline=1.0):
    """Simple analytical box-dip model for exoplanet transit."""
    model = np.full_like(t, baseline)
    half_dur = duration_hours / 2.0
    in_transit = np.abs(t - center_offset) <= half_dur
    model[in_transit] = baseline - depth
    return model
#defines a mathematical step-function: creates an array initialized to baseline (1.0), finds timestamps within half_dur of the centre, and subtracts depth for those points


#initial guess parameter vector: [depth, duration (hrs), offset, baseline]
initial_guess = [best_depth, best_duration * 24, 0.0, 1.0]

#fit model using non-linear least squares
popt, pcov = curve_fit(box_transit_model, time_fold_hours, flux_fold, p0=initial_guess)
fit_depth, fit_duration_hours, fit_offset, fit_baseline = popt
fit_errors = np.sqrt(np.diag(pcov))
#runs curve_fit to adjust model parameters to best fit observed scatter:
    #popt: array containing optimal values for [depth, duration_hours, center_offset, baseline]
    #pcov: parameter covariance matrix, taking np.sqrt(np.diag(pcov)) computes standard 1σ uncertainties (fit_errors) for each fit parameter


#generate high-resolution model plot line
t_model = np.linspace(-6, 6, 1000)
flux_model = box_transit_model(t_model, *popt)
#creates a smooth 1,000-point timeline spanning -6 to +6 hours and calculates the fitted model values across it


#compute exoplanet physical parameters

#Kepler-10 stellar constants
R_star_solar = 1.056       #R_sun
M_star_solar = 0.895       #M_sun
R_sun_to_R_earth = 109.076 #1 R_sun = 109.076 R_earth
AU_in_km = 1.496e8         #1 AU in km
R_earth_in_km = 6371.0     #Earth radius in km

#planet radius calculation
R_star_earth = R_star_solar * R_sun_to_R_earth
R_planet_earth = R_star_earth * np.sqrt(fit_depth)
R_planet_km = R_planet_earth * R_earth_in_km

#Semi-Major Axis via Kepler's 3rd Law
a_AU = (M_star_solar)**(1/3) * (best_period / 365.25)**(2/3)
a_km = a_AU * AU_in_km
#derives orbital semi-major axis a from Kepler’s Third Law (a∝M^(1/3) P^(2/3) in Astronomical Units (AU) and kilometers


#impact parameter estimation
fit_duration_days = fit_duration_hours / 24.0
R_star_AU = R_star_solar * 0.00465047
b_arg = 1.0 - ((np.pi * a_AU * fit_duration_days) / (best_period * R_star_AU))**2
b_impact = np.sqrt(max(0, b_arg))

print(f"Fitted Transit Depth:    {fit_depth * 1e6:.2f} ± {fit_errors[0]*1e6:.2f} ppm")
print(f"Fitted Transit Duration: {fit_duration_hours:.2f} ± {fit_errors[1]:.2f} hours")
print(f"Estimated Planet Radius: {R_planet_earth:.2f} R_Earth ({R_planet_km:.0f} km)")
print(f"Semi-Major Axis (a):     {a_AU:.5f} AU ({a_km:.0f} km)")
print(f"Impact Parameter (b):    {b_impact:.2f} (0 = central transit, 1 = grazing)")

#plot
fig, ax = plt.subplots(figsize=(10, 6))

#plot raw phase-folded data points
ax.scatter(time_fold_hours, flux_fold, color='gray', alpha=0.25, s=6, label='Phase-folded Data')

#plot binned points
ax.errorbar(time_bin_hours, flux_bin, yerr=np.std(flux_bin)/10, fmt='o', color='navy', 
            markersize=4, alpha=0.8, label='Binned Data (~3 min)')

#plot fitted transit model line
ax.plot(t_model, flux_model, color='crimson', linewidth=2.5, 
        label=f'Fitted Model\n(Depth: {fit_depth*1e6:.1f} ppm)')

#axis labels and annotations
ax.set_xlim(-5, 5)
ax.set_xlabel("Time from Mid-Transit (Hours)", fontsize=12)
ax.set_ylabel("Normalized Relative Flux", fontsize=12)
ax.set_title(f"Step 4 — Transit Model Fit & Parameter Extraction (Kepler-10b)\n"
             f"P = {best_period:.4f} d | R_p = {R_planet_earth:.2f} R_Earth | a = {a_AU:.4f} AU", 
             fontsize=13, fontweight='bold')
ax.legend(loc='lower left', frameon=True)
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
