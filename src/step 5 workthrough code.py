import lightkurve as lk
import numpy as np
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

#download Kepler-10 light curve data
search_result = lk.search_lightcurve('Kepler-10', mission='Kepler', cadence='long') #searches NASA's archive for long-cadence light curve observations of the target star Kepler-10
lc = search_result[0].download().remove_nans().normalize() #downloads the first light curve file, removes invalid missing data (NaNs), and normalizes the light flux around 1.0

#estimated transit parameters for Kepler-10b
period_est = 0.837495  # days
epoch_est = 200.57     # BKJD
duration_est = 0.075   # days


def calculate_snr(time, flux, period, epoch, duration):
    """Calculates the transit Signal-to-Noise Ratio (SNR)."""
    phase = ((time - epoch + 0.5 * period) % period) - 0.5 * period
    #folds the time-series data using modulo arithmetic (%). This aligns all repeating transits on top of each other centered at phase 0.0
    
    in_transit = np.abs(phase) < (duration / 2.0)
    out_transit = ~in_transit
    #creates boolean masks: in_transit identifies points occurring during the transit window; out_transit identifies points outside the transit
    
    baseline = np.median(flux[out_transit])
    depth = baseline - np.mean(flux[in_transit])
    noise = np.std(flux[out_transit])
    n_in = np.sum(in_transit)
    #calculates the average baseline brightness, the transit depth (δ), the noise standard deviation (σ), and counts the number of data points inside the transit (N_in)
    
    snr = (depth / noise) * np.sqrt(n_in) if noise > 0 else 0.0
    return float(snr), float(depth), float(noise)
#computes the Signal-to-Noise Ratio (SNR = depth/noise ×sqrt(N_(in))) and returns values as standard Python floats

def run_false_positive_checks(time, flux, period, epoch, duration):
    """Evaluates odd/even transit depth differences to rule out eclipsing binaries."""
    phase = ((time - epoch + 0.5 * period) % period) - 0.5 * period
    transit_number = np.floor((time - epoch + 0.5 * period) / period).astype(int)
    #phase-folds time data and calculates an index integer (transit_number) for every individual transit event
    
    in_transit = np.abs(phase) < (duration / 2.0)
    odd_mask = in_transit & (transit_number % 2 != 0)
    even_mask = in_transit & (transit_number % 2 == 0)
    out_mask = ~in_transit
    #isolates data points belonging specifically to odd-numbered transits vs. even-numbered transits
    
    baseline = np.median(flux[out_mask])
    odd_depth = baseline - np.mean(flux[odd_mask])
    even_depth = baseline - np.mean(flux[even_mask])
    #computes separate transit depths for odd and even subsets
    
    depth_diff = np.abs(odd_depth - even_depth) / np.std(flux[out_mask])
    is_false_positive = depth_diff > 3.0
    
    return is_false_positive, float(depth_diff)
#calculates depth discrepancy in units of standard deviations (σ). If difference exceeds 3σ, it flags a potential false positive (common in eclipsing binary stars)

def check_known_catalogs(target_name, detected_period, tolerance=0.01):
    """Queries NASA Exoplanet Archive for confirmed signals matching target and period."""
    try:
        catalog = NasaExoplanetArchive.query_criteria(
            table="pscomppars", 
            where=f"hostname = '{target_name}'"
        )
        #uses astroquery to query NASA's Exoplanet Archive database table (pscomppars) for target host star
        
        if len(catalog) > 0:
            known_periods = catalog['pl_orbper'].value
            for p in known_periods:
                if np.isclose(detected_period, p, rtol=tolerance):
                    return 1.0, f"Matched confirmed planet with period {p:.5f} d"
            return 0.7, "Host star in catalog, but period mismatch."
        #if star exists in database, it iterates through known planetary orbital periods. If period matches a known planet within 1% tolerance, it returns a high confidence factor (1.0)
        
        else:
            return 0.5, "New candidate (not in confirmed catalog)."
    except Exception as e:
        return 0.5, f"Catalog query unverified ({str(e)})"
#returns fallback confidence factors (0.5) if star is unlisted or if network request fails

def compute_confidence_score(snr, fp_flag, catalog_score, snr_threshold=7.1):
    """Combines validation metrics into a single score between 0.0 and 1.0."""
    snr_component = 1.0 / (1.0 + np.exp(-0.8 * (snr - snr_threshold)))
    fp_penalty = 0.5 if fp_flag else 1.0
    final_score = snr_component * fp_penalty * catalog_score
    return float(np.clip(final_score, 0.0, 1.0))
#combines all indicators: applies a logistic sigmoid curve to SNR centered at Kepler's 7.1σ detection limit, applies a 50% penalty if flagged as a false positive, multiplies by catalog score, and clips the final result between 0.0 and 1.0

#extract raw values from LightCurve object
time_vals = lc.time.value
flux_vals = lc.flux.value
#extracts raw numerical NumPy arrays from Lightkurve's unit-aware time and flux objects

# 1.compute SNR
snr_val, depth_val, noise_val = calculate_snr(
    time_vals, flux_vals, period_est, epoch_est, duration_est
)

# 2.run false positive checks
fp_flag, depth_diff_sigma = run_false_positive_checks(
    time_vals, flux_vals, period_est, epoch_est, duration_est
)

# 3.cross-reference catalogs
catalog_score, catalog_msg = check_known_catalogs("Kepler-10", period_est)

# 4.calculate final confidence score
confidence = compute_confidence_score(snr_val, fp_flag, catalog_score)
#sequentially passes light curve data into all 4 functions to calculate each validation metric


#output Summary Table
validation_summary = pd.DataFrame([{
    "Target": "Kepler-10",
    "Period (days)": period_est,
    "SNR": round(snr_val, 2),
    "False Positive Flag": fp_flag,
    "Catalog Match Score": catalog_score,
    "Confidence Score": round(confidence, 4),
    "Status": "CONFIRMED PLANET" if confidence > 0.85 else "CANDIDATE"
}])

print(validation_summary.to_string(index=False))
#formats final pipeline metrics into a Pandas DataFrame and prints a clean summary table without index numbers
