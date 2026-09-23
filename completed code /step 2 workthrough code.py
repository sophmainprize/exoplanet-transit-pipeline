import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

#load target light curve (Kepler-10, Quarter 2)
search_result = lk.search_lightcurve("Kepler-10", mission="Kepler", quarter=2)
lc_raw = search_result.download()

print("Loaded Raw Light Curve:")
print(f"Number of cadence points: {len(lc_raw)}")



#remove NaN values from flux and time arrays
lc_nans_removed = lc_raw.remove_nans() #drop any invalid or undefined flux measurements (NaNs) caused by instrument drops

nans_count = len(lc_raw) - len(lc_nans_removed) #subtracts cleaned light curve length from original length to count how many NaN points were removed
print(f"Removed {nans_count} NaN values.")
print(f"Remaining data points: {len(lc_nans_removed)}")



#normalise flux relative to baseline median
lc_norm = lc_nans_removed.normalize() #divides flux measurements by median baseline value, rescales light curve so that relative brightness is centered around 1.0

print(f"Raw median flux: {np.nanmedian(lc_nans_removed.flux):.2f}")
print(f"Normalised median flux: {np.nanmedian(lc_norm.flux):.2f}")

#visualise Normalised Data
lc_norm.plot(title="2.: Normalised Light Curve", color="black", alpha=0.7);



#detrend/flatten using Savitzky-Golay filter
lc_flat = lc_norm.flatten(window_length=101) #applies sliding-window Savitzky-Golay polynomial filter (window_length=101 cadences ≈ ~2 days) to fit and divide out long-term stellar variability and instrumental thermal drift without flattening out short-duration exoplanet transit dips

print("Detrended Light Curve (flattened):")
print(f"Flattened median flux: {np.nanmedian(lc_flat.flux):.2f}")

#visualise Detrended Data
lc_flat.plot(title="Step 3.: Detrended (Flattened) Light Curve", color="black", alpha=0.7); #plots flattened light curve to verify that long-term continuous waves have been removed



#remove cosmic rays and extreme single-point anomalies
lc_cleaned = lc_flat.remove_outliers(sigma_upper=5, sigma_lower=3) #asymmetric sigma clipping

outliers_count = len(lc_flat) - len(lc_cleaned)
print(f"Removed {outliers_count} outlier data points.")
print(f"Final dataset length: {len(lc_cleaned)}")



#create side-by-side or stacked comparative plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

#top Plot: Raw Light Curve
axes[0].plot(lc_raw.time.value, lc_raw.flux.value, color="black", linewidth=0.5, label="Raw Light Curve")
axes[0].set_title("Raw Light Curve (Quarter 2)")
axes[0].set_ylabel("Flux (electrons/sec)")
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend()

#bottom Plot: Cleaned & Detrended Light Curve
axes[1].plot(lc_cleaned.time.value, lc_cleaned.flux.value, color="darkblue", linewidth=0.5, label="Cleaned & Preprocessed Light Curve")
axes[1].set_title("Cleaned Light Curve (Normalized, Detrended, Outliers Removed)")
axes[1].set_xlabel("Time (BJD - 2454833)")
axes[1].set_ylabel("Normalized Flux")
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend()

plt.tight_layout()
plt.show()
