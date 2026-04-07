import os
import re
import glob
import csv
import math
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

# paths
WHOLE_MASK_DIR  = r"H:\Data\Quantification\FAIR_masks"
CORTEX_MASK_DIR = r"H:\Data\Quantification\FAIR_cortex_masks"
OUT_DIR         = r"H:\Data\Analysis\STRUCT_BIOMARKERS"
os.makedirs(OUT_DIR, exist_ok=True)
WHOLE_SUFFIX  = "_FAIRmask_refined.nii.gz"
CORTEX_SUFFIX = "_CORTEX_FAIRmask.nii.gz"

OUT_CSV = os.path.join(OUT_DIR, "structural_biomarkers.csv")

#qc thresholds
QC_MIN_AREA_MM2          = 500.0      # very small mask
QC_MAX_COMPONENTS        = 5          # fragmentation warning if too many
QC_LARGEST_FRAC_MIN      = 0.85       # fragmentation warning if largest comp too small
QC_LR_RATIO_MIN          = 0.33       # asymmetry flag
QC_LR_RATIO_MAX          = 3.0

# Remove tiny speckles before computing components (in voxels)
MIN_SPECKLE_VOXELS = 20

# helpers
def read_mask(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  
    spacing = img.GetSpacing()      
    return img, arr, spacing

def cleanup_binary(mask, min_vox=MIN_SPECKLE_VOXELS):
    """Remove tiny connected components."""
    if mask.sum() == 0:
        return mask
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    counts = np.bincount(lab.ravel())
    keep = np.zeros(n + 1, dtype=bool)
    keep[0] = False
    for k in range(1, n + 1):
        if counts[k] >= min_vox:
            keep[k] = True
    return keep[lab]

def split_left_right(mask_bin):
    """
    Split a binary mask into left/right by connected components.
    If 2+ components, take two largest and assign by x-centroid.
    """
    out = {'left': np.zeros_like(mask_bin, dtype=bool),
           'right': np.zeros_like(mask_bin, dtype=bool)}

    if mask_bin.sum() == 0:
        return out, False

    lab, n = ndi.label(mask_bin)
    if n < 2:
        # cannot split
        return out, False

    # component sizes (skip background=0)
    counts = np.bincount(lab.ravel())
    comps = [(k, counts[k]) for k in range(1, n + 1)]
    comps.sort(key=lambda t: t[1], reverse=True)

    # pick two largest components
    k1, k2 = comps[0][0], comps[1][0]
    m1 = (lab == k1)
    m2 = (lab == k2)

    # decide left vs right by x centroid in array coords (z,y,x)
    x1 = np.mean(np.where(m1)[2])
    x2 = np.mean(np.where(m2)[2])

    if x1 < x2:
        out['left'] = m1
        out['right'] = m2
    else:
        out['left'] = m2
        out['right'] = m1

    return out, True

def extract_lr_masks(arr):
    arr = np.asarray(arr)

    # normalize dimensions
    if arr.ndim == 2:
        arr = arr[None, ...]  # z=1

    vals = np.unique(arr[arr != 0])
    masks = {'left': None, 'right': None, 'total': None}

    total = (arr != 0)
    total = cleanup_binary(total)
    masks['total'] = total

    # label case
    if set(vals.tolist()) >= {1} and (2 in vals):
        m1 = cleanup_binary(arr == 1)
        m2 = cleanup_binary(arr == 2)

        # assign left/right by x centroid (robust to swapped labels)
        def xcent(m):
            if m.sum() == 0: return np.inf
            return np.mean(np.where(m)[2])

        if xcent(m1) < xcent(m2):
            masks['left'], masks['right'] = m1, m2
        else:
            masks['left'], masks['right'] = m2, m1
        return masks

    # binary case
    lr, ok = split_left_right(total)
    if ok:
        masks['left'] = cleanup_binary(lr['left'])
        masks['right'] = cleanup_binary(lr['right'])
    else:
        masks['left'] = np.zeros_like(total, dtype=bool)
        masks['right'] = np.zeros_like(total, dtype=bool)

    return masks

def area_mm2(mask, spacing):
    # spacing = (sx, sy, sz)
    sx, sy = float(spacing[0]), float(spacing[1])
    # single slice or multi-slice
    vox = int(mask.sum())
    return vox * sx * sy, vox

def approx_perimeter_mm(mask2d, sx, sy):
    """
    Approx boundary length in mm:
    perimeter pixels = boundary = mask XOR erode(mask)
    perimeter_mm = count * mean_pixel_edge
    """
    if mask2d.sum() == 0:
        return 0.0
    er = ndi.binary_erosion(mask2d)
    boundary = mask2d ^ er
    n = int(boundary.sum())
    mean_edge = 0.5 * (sx + sy)
    return n * mean_edge

def pca_axes(mask2d, sx, sy):
    """Return (major_mm, minor_mm, eccentricity) using PCA on physical coords."""
    if mask2d.sum() < 10:
        return (np.nan, np.nan, np.nan)

    yy, xx = np.where(mask2d)  # y,x
    # physical coords (mm), center them
    X = np.column_stack((xx * sx, yy * sy)).astype(np.float64)
    X -= X.mean(axis=0, keepdims=True)

    C = np.cov(X.T)
    vals, _ = np.linalg.eigh(C)  # ascending
    vals = np.sort(vals)[::-1]   # descending

    # standard deviation along principal axes; scale to "diameter-ish"
    major = 4.0 * math.sqrt(max(vals[0], 0.0))
    minor = 4.0 * math.sqrt(max(vals[1], 0.0))

    if major <= 0:
        ecc = np.nan
    else:
        ecc = math.sqrt(max(0.0, 1.0 - (minor * minor) / (major * major))) if minor == minor else np.nan

    return (major, minor, ecc)

def components_stats(mask2d):
    """connected components, largest fraction"""
    if mask2d.sum() == 0:
        return 0, np.nan
    lab, n = ndi.label(mask2d)
    if n == 0:
        return 0, np.nan
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    largest = counts.max() if counts.size else 0
    frac = float(largest) / float(mask2d.sum()) if mask2d.sum() > 0 else np.nan
    return int(n), frac

def hole_fraction(mask2d):
    """(filled - original) / filled"""
    if mask2d.sum() == 0:
        return np.nan
    filled = ndi.binary_fill_holes(mask2d)
    a0 = float(mask2d.sum())
    a1 = float(filled.sum())
    if a1 <= 0:
        return np.nan
    return (a1 - a0) / a1

def compactness(area, perimeter):
    if area <= 0 or perimeter <= 0:
        return np.nan
    return 4.0 * math.pi * area / (perimeter * perimeter)

def lr_ratio(aL, aR):
    if aL is None or aR is None or aR <= 0:
        return np.nan
    return aL / aR

def asymmetry_percent(aL, aR):
    if aL is None or aR is None:
        return np.nan
    denom = (aL + aR) / 2.0
    if denom <= 0:
        return np.nan
    return abs(aL - aR) / denom * 100.0

def get_subject_id(fname):
    # try "Neo01170" style
    m = re.search(r"(Neo\d{5})", fname)
    return m.group(1) if m else os.path.basename(fname).split()[0]

# Main
whole_files = sorted(glob.glob(os.path.join(WHOLE_MASK_DIR, f"*{WHOLE_SUFFIX}")))
if not whole_files:
    raise RuntimeError(f"No whole-kidney masks found in {WHOLE_MASK_DIR} with suffix {WHOLE_SUFFIX}")

rows = []
for wf in whole_files:
    base = os.path.basename(wf)
    subj = get_subject_id(base)

    cortex_candidates = glob.glob(os.path.join(CORTEX_MASK_DIR, f"*{subj}*{CORTEX_SUFFIX}"))
    cf = cortex_candidates[0] if cortex_candidates else None

    # load whole kidney
    try:
        _, w_arr, spacing = read_mask(wf)
    except Exception as e:
        print(f"[SKIP] {subj} failed reading whole mask: {e}")
        continue

    # load cortex 
    c_arr = None
    if cf and os.path.exists(cf):
        try:
            _, c_arr, _ = read_mask(cf)
        except Exception as e:
            print(f"[WARN] {subj} failed reading cortex mask: {e}")
            c_arr = None

    # extract L/R/total masks
    w_masks = extract_lr_masks(w_arr)
    c_masks = extract_lr_masks(c_arr) if c_arr is not None else None

    sx, sy = float(spacing[0]), float(spacing[1])

    def compute_for(mask3d, prefix, kidney_name):
        # collapse to 2D if single-slice; for multi-slice, compute per-slice and sum where needed
        if mask3d is None or mask3d.sum() == 0:
            return {
                f"{prefix}_{kidney_name}_vox": 0,
                f"{prefix}_{kidney_name}_area_mm2": 0.0,
                f"{prefix}_{kidney_name}_perimeter_mm": 0.0,
                f"{prefix}_{kidney_name}_compactness": np.nan,
                f"{prefix}_{kidney_name}_n_components": 0,
                f"{prefix}_{kidney_name}_largest_comp_frac": np.nan,
                f"{prefix}_{kidney_name}_hole_frac": np.nan,
                f"{prefix}_{kidney_name}_major_mm": np.nan,
                f"{prefix}_{kidney_name}_minor_mm": np.nan,
                f"{prefix}_{kidney_name}_eccentricity": np.nan,
            }

        area, vox = area_mm2(mask3d, spacing)

        # pick slice with max pixels for shape-based metrics
        z_sums = mask3d.sum(axis=(1, 2))
        z = int(np.argmax(z_sums))
        m2 = mask3d[z].astype(bool)

        perim = approx_perimeter_mm(m2, sx, sy)
        comp = compactness(area, perim)
        ncc, lfrac = components_stats(m2)
        hfrac = hole_fraction(m2)
        major, minor, ecc = pca_axes(m2, sx, sy)

        return {
            f"{prefix}_{kidney_name}_vox": int(vox),
            f"{prefix}_{kidney_name}_area_mm2": float(area),
            f"{prefix}_{kidney_name}_perimeter_mm": float(perim),
            f"{prefix}_{kidney_name}_compactness": float(comp) if comp == comp else np.nan,
            f"{prefix}_{kidney_name}_n_components": int(ncc),
            f"{prefix}_{kidney_name}_largest_comp_frac": float(lfrac) if lfrac == lfrac else np.nan,
            f"{prefix}_{kidney_name}_hole_frac": float(hfrac) if hfrac == hfrac else np.nan,
            f"{prefix}_{kidney_name}_major_mm": float(major) if major == major else np.nan,
            f"{prefix}_{kidney_name}_minor_mm": float(minor) if minor == minor else np.nan,
            f"{prefix}_{kidney_name}_eccentricity": float(ecc) if ecc == ecc else np.nan,
        }

    row = {
        "subject": subj,
        "whole_mask_file": os.path.basename(wf),
        "cortex_mask_file": os.path.basename(cf) if cf else "",
        "spacing_x_mm": sx,
        "spacing_y_mm": sy,
        "spacing_z_mm": float(spacing[2]) if len(spacing) > 2 else np.nan,
    }

    # whole kidney
    row.update(compute_for(w_masks["total"], "whole", "total"))
    row.update(compute_for(w_masks["left"],  "whole", "left"))
    row.update(compute_for(w_masks["right"], "whole", "right"))

    # cortex
    if c_masks is not None:
        row.update(compute_for(c_masks["total"], "cortex", "total"))
        row.update(compute_for(c_masks["left"],  "cortex", "left"))
        row.update(compute_for(c_masks["right"], "cortex", "right"))
    else:
        for k in ["total", "left", "right"]:
            row.update(compute_for(None, "cortex", k))

    # Derived metrics
    wL = row["whole_left_area_mm2"]
    wR = row["whole_right_area_mm2"]
    cL = row["cortex_left_area_mm2"]
    cR = row["cortex_right_area_mm2"]

    row["whole_lr_ratio"] = lr_ratio(wL, wR)
    row["whole_asymmetry_percent"] = asymmetry_percent(wL, wR)

    row["cortex_lr_ratio"] = lr_ratio(cL, cR)
    row["cortex_asymmetry_percent"] = asymmetry_percent(cL, cR)

    # cortex fraction (total)
    wT = row["whole_total_area_mm2"]
    cT = row["cortex_total_area_mm2"]
    row["cortex_fraction_total"] = (cT / wT) if (wT and wT > 0) else np.nan

    # qc flags
    qc_flags = []
    if row["whole_total_area_mm2"] < QC_MIN_AREA_MM2:
        qc_flags.append("whole_too_small")
    if row["cortex_total_area_mm2"] > 0 and row["cortex_total_area_mm2"] < QC_MIN_AREA_MM2:
        qc_flags.append("cortex_too_small")

    # fragmentation 
    if row["cortex_total_n_components"] > QC_MAX_COMPONENTS:
        qc_flags.append("cortex_fragmented_many_components")
    lfrac = row["cortex_total_largest_comp_frac"]
    if lfrac == lfrac and lfrac < QC_LARGEST_FRAC_MIN:
        qc_flags.append("cortex_fragmented_low_largest_frac")

    # asymmetry (wk)
    lrw = row["whole_lr_ratio"]
    if lrw == lrw and (lrw < QC_LR_RATIO_MIN or lrw > QC_LR_RATIO_MAX):
        qc_flags.append("whole_lr_asymmetry")

    row["qc_flags"] = ";".join(qc_flags)
    row["qc_flagged"] = (len(qc_flags) > 0)

    rows.append(row)

# CSV
fieldnames = list(rows[0].keys()) if rows else []
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"Saved: {OUT_CSV}")
print(f"Processed: {len(rows)} subjects")
print(f"QC flagged: {sum(1 for r in rows if r['qc_flagged'])}")