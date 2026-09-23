import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np

search_result = lk.search_lightcurve("Kepler-10", mission="Kepler") #Kepler-10: specifies target star, mission="Kepler": restricts search specifically to observations collected by Kepler Space Telescope
search_result #evaluates variable at end of cell to display a table listing all available data products found for Kepler-10 



lc = search_result[0].download() #selects first search result entry (search_result[0], corresponding to Quarter 2 observations) and calls its .download() method to fetch actual FITS file from archive into memory as a KeplerLightCurve object named lc
lc #evaluates lc to display an overview table of light curve object, showing its metadata, length, and columns

lc.columns #accesses column names property of KeplerLightCurve object lc to inspect all available data fields (such as time, flux, flux_err, cadenceno, pdcsap_flux, etc.)

time = lc.time.value #extracts timestamp values from lc.time into standard NumPy array named time
flux = lc.flux.value #extracts observed stellar brightness values into NumPy array named flux
flux_err = lc.flux_err.value #extracts measurement uncertainties (errors) associated with each flux reading into NumPy array named flux_err

print("Time array shape:", time.shape)
print("Flux array shape:", flux.shape)
print("Flux error shape:", flux_err.shape)



plt.figure(figsize=(12, 5))
plt.plot(time, flux, linewidth=0.5, color="black")
plt.xlabel("Time (BJD - 2454833)")
plt.ylabel("Flux (electrons/sec)")
plt.title("Kepler-10 — Raw Stellar Light Curve (Quarter 2)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

flux_values = lc.flux.value #re-extracts raw numerical flux values into NumPy array named flux_values

print("Min flux:", np.min(flux_values))
print("Max flux:", np.max(flux_values))
print("Mean flux:", np.mean(flux_values))
print("Standard deviation:", np.std(flux_values))

