import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

#set aesthetic style for plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')



#download Kepler-10 Quarter 2 light curve
search_result = lk.search_lightcurve("Kepler-10", mission="Kepler", quarter=2)
lc_raw = search_result.download()

#apply cleaning steps from Step 2
lc_cleaned = (
    lc_raw
    .remove_nans()
    .normalize()
    .flatten(window_length=101)
    .remove_outliers(sigma=5)
)

#.remove_nans(): gets rid of invalid data points (missing sensor readings)
#.normalize(): converts raw electron counts per second (e ^−/s) into relative flux centered around 1.0
#.flatten(window_length=101): applies a Savitzky-Golay filter across sliding window of 101 data cadences to smooth out low-frequency stellar variability and instrumental drifts without removing short transit dips
#.remove_outliers(sigma=5): removes statistical anomalies (e.g. cosmic ray hits) that lie more than 5 standard deviations (5σ) away from local median trend line

print(f"Cleaned Light Curve ready with {len(lc_cleaned)} points.")



#define trial period grid (Kepler-10b has an ultra-short period ~0.83 days)
period_grid = np.linspace(0.4, 5.0, 10000) #generates array of 10,000 evenly spaced trial orbital periods ranging from 0.4 days to 5.0 days

#compute BLS periodogram
bls = lc_cleaned.to_periodogram(
    method='bls', 
    period=period_grid, 
    frequency_factor=5.0
)
#runs Box Least Squares (BLS) algorithm on lc_cleaned, fits box-shaped dips across all specified trial periods in period_grid and evaluates signal power (statistical likelihood of a real transit)
#BLS is a statistical algorithm specifically designed to search for periodic, box-shaped signals in time-series data
#standard method used to detect exoplanet transits in light curves collected by telescopes like Kepler, K2, and TESS

#extract detected transit parameters
best_period = bls.period_at_max_power #extracts specific trial orbital period (P) corresponding to highest peak (maximum power signal) in BLS periodogram
best_t0 = bls.transit_time_at_max_power #extracts estimated mid-transit time / epoch (T_0), which is the timestamp marking centre of primary transit event
best_duration = bls.duration_at_max_power #extracts estimated transit duration (d), which represents how long planet takes to cross face of star
best_depth = bls.depth_at_max_power #extracts estimated transit depth (ΔF/F), representing fractional drop in stellar brightness caused by transiting planet

#print summary statistics
print(f"Orbital Period (P):  {best_period.value:.5f} days")
print(f"Transit Epoch (T0):  {best_t0.value:.4f} BTJD")
print(f"Transit Duration:    {best_duration.value * 24:.2f} hours")
print(f"Transit Depth:       {best_depth.value * 1e6:.1f} ppm (parts per million)")



fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

#plot periodogram power spectrum
bls.plot(ax=ax, color='navy', linewidth=1)

#highlight detected candidate period
ax.axvline(
    best_period.value, 
    color='red', 
    linestyle='--', 
    alpha=0.7, 
    label=f'Peak Period: {best_period.value:.5f} d'
)
#draws vertical dashed red line at detected candidate period (P≈0.837 days) to highlight strongest detected periodic signal

ax.set_title("LS Periodogram — Power vs Orbital Period", fontsize=12, fontweight='bold')
ax.set_xlabel("Period (days)", fontsize=10)
ax.set_ylabel("BLS Power", fontsize=10)
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()



#phase-fold light curve using detected period and mid-transit epoch
folded_lc = lc_cleaned.fold(period=best_period, epoch_time=best_t0)
#performs phase folding: takes  entire multi-day time series and wraps all time points modulo the detected period (P), stacking every individual transit on top of each other centered at phase 0.0 (T_0)

#bin folded data to average out high-frequency noise
binned_lc = folded_lc.bin(time_bin_size=0.01)
#bins folded observations in time intervals of 0.01 days (≈14.4 minutes) by taking average flux within each bin:reduces high-frequency photometric noise and brings out characteristic U-shaped transit curve

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
#plots all raw individual folded data points as small, semi-transparent grey dots in background to show observational scatter

#plot raw folded points in background
folded_lc.scatter(ax=ax, color='gray', alpha=0.3, s=5, label='Folded Data')

#overlay binned points on top
binned_lc.scatter(ax=ax, color='crimson', s=25, label='Binned Transit Profile')
#overlays binned data points as prominent red markers on top of raw data to clearly delineate transit profile

#focus window on mid-transit
ax.set_xlim(-0.25, 0.25)
ax.set_title(
    f"Phase-Folded Light Curve (Period = {best_period.value:.5f} days)", 
    fontsize=12, 
    fontweight='bold'
)
ax.set_xlabel("Phase (Days from Mid-Transit)", fontsize=10)
ax.set_ylabel("Normalized Flux", fontsize=10)
ax.legend(loc='lower right')

plt.tight_layout()
plt.show()


