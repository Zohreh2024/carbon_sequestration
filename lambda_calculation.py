import numpy as np
import rasterio

Revised_M_file_path = r"N:/Current-Users/ZOHREH-KALAHROUDI/M_FPI_P90/New_M_2019_aligned_2.tif"
Formula_M_path      = r"N:/Current-Users/ZOHREH-KALAHROUDI/Original_M/Original_M.tif"
output_raster_path  = r"N:/Current-Users/ZOHREH-KALAHROUDI/M_FPI_P90/lambda_f.tif"

def read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        prof = src.meta.copy()
        nd = src.nodata
    return arr, prof, nd

num, profile, nd_num = read_band(Revised_M_file_path)
den, _,      nd_den  = read_band(Formula_M_path)

# Fallback if nodata is missing in metadata
if nd_num is None: nd_num = -9999.0
if nd_den is None: nd_den = -9999.0

# Build masks for valid data
valid_num = np.isfinite(num) & (np.abs(num - nd_num) > 1e-6)
valid_den = np.isfinite(den) & (np.abs(den - nd_den) > 1e-6) & (den != 0)

# Report diagnostics (robust stats, ignoring invalids)
def robust_stats(a, mask):
    a2 = a[mask]
    if a2.size == 0:
        return None
    return dict(
        n=int(a2.size),
        min=float(np.nanmin(a2)),
        p1=float(np.nanpercentile(a2, 1)),
        p50=float(np.nanpercentile(a2, 50)),
        p99=float(np.nanpercentile(a2, 99)),
        max=float(np.nanmax(a2)),
        zeros=int(np.sum(a2 == 0))
    )

print("NUM (revised M) stats:", robust_stats(num, valid_num))
print("DEN (formula M) stats:", robust_stats(den, valid_den))

# Adaptive near-zero threshold for denominator:
# use the 1st percentile of |den|, but not smaller than 1e-3 (tune if needed)
abs_den = np.abs(den[valid_den])
if abs_den.size == 0:
    raise ValueError("No valid denominator pixels found.")
thr = max(1e-3, float(np.nanpercentile(abs_den, 1)) * 0.1)  # 10% of p1
print(f"Near-zero threshold for denominator: {thr:g}")

safe_den = valid_den & (np.abs(den) >= thr)
valid = valid_num & safe_den

# Compute lambda
out = np.full(num.shape, np.float32(nd_num), dtype="float32")
with np.errstate(divide='ignore', invalid='ignore'):
    out[valid] = num[valid] / den[valid]
    # keep nodata where result is not finite
    bad = ~np.isfinite(out)
    out[bad] = nd_num

# Update profile and save
profile.update(dtype="float32", nodata=float(nd_num), count=1, compress="lzw")
with rasterio.open(output_raster_path, "w", **profile) as dst:
    dst.write(out, 1)

print("Done. λ written to:", output_raster_path)
