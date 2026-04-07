import os, glob, re, csv
import numpy as np
import nibabel as nib

PERF_DIR = r"H:\Data\Quantification\RBF_CORTEX"           
MASK_DIR = r"H:\Data\Quantification\FAIR_cortex_masks"   
OUT_CSV  = os.path.join(PERF_DIR, "RBF_Cortex_filtered.csv")

PERF_SUFFIX = "_cortex_rBF.nii.gz"     
MASK_SUFFIX = "_CORTEX_FAIRmask.nii.gz"  

# Filtering
LOW, HIGH = 0.0, 500.0    
MIN_VALID_VOXELS = 200     
DROP_NONFINITE = True

def load_nii(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data

def subject_id_from_fname(fname):
    base = os.path.basename(fname)
    m = re.search(r"(Neo\d+)", base)
    return m.group(1) if m else os.path.splitext(os.path.splitext(base)[0])[0]

def summarize(values):
    return {
        "n_valid": int(values.size),
        "mean": float(np.mean(values)) if values.size else np.nan,
        "std": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
        "median": float(np.median(values)) if values.size else np.nan,
        "p10": float(np.percentile(values, 10)) if values.size else np.nan,
        "p90": float(np.percentile(values, 90)) if values.size else np.nan,
        "p99": float(np.percentile(values, 99)) if values.size else np.nan,
        "min": float(np.min(values)) if values.size else np.nan,
        "max": float(np.max(values)) if values.size else np.nan,
    }

def main():
    perf_files = sorted(glob.glob(os.path.join(PERF_DIR, f"*{PERF_SUFFIX}")))
    if not perf_files:
        raise RuntimeError(f"No perfusion files found in {PERF_DIR} matching *{PERF_SUFFIX}")

    rows = []
    missing_masks = 0

    for pf in perf_files:
        sid = subject_id_from_fname(pf)
        base = os.path.basename(pf).replace(PERF_SUFFIX, "")

        mf = os.path.join(MASK_DIR, base + MASK_SUFFIX)
        if not os.path.exists(mf):
            candidates = glob.glob(os.path.join(MASK_DIR, f"*{sid}*{MASK_SUFFIX}"))
            if candidates:
                mf = candidates[0]
            else:
                missing_masks += 1
                rows.append({
                    "subject": sid,
                    "perf_file": os.path.basename(pf),
                    "mask_file": "",
                    "status": "MISSING_MASK",
                })
                continue

        perf = load_nii(pf)
        mask = load_nii(mf)

        if perf.shape != mask.shape:
            rows.append({
                "subject": sid,
                "perf_file": os.path.basename(pf),
                "mask_file": os.path.basename(mf),
                "status": f"SHAPE_MISMATCH perf{perf.shape} mask{mask.shape}",
            })
            continue

        vox = perf[mask > 0]

        n_total = int(vox.size)
        if n_total == 0:
            rows.append({
                "subject": sid,
                "perf_file": os.path.basename(pf),
                "mask_file": os.path.basename(mf),
                "status": "EMPTY_MASK",
                "n_total": 0,
                "n_valid": 0,
            })
            continue

        if DROP_NONFINITE:
            finite = np.isfinite(vox)
            n_nonfinite = int(np.sum(~finite))
            vox = vox[finite]
        else:
            n_nonfinite = 0

        # Count how many are below/above physiologic range before filtering
        n_below = int(np.sum(vox < LOW))
        n_above = int(np.sum(vox > HIGH))

        # Apply filtering (drop out-of-range voxels)
        valid = vox[(vox >= LOW) & (vox <= HIGH)]

        stats_dict = summarize(valid)

        frac_valid = (stats_dict["n_valid"] / max(1, (n_total - n_nonfinite))) * 100.0
        frac_below = (n_below / max(1, (n_total - n_nonfinite))) * 100.0
        frac_above = (n_above / max(1, (n_total - n_nonfinite))) * 100.0

        status = "OK"
        if stats_dict["n_valid"] < MIN_VALID_VOXELS:
            status = "LOW_VALID_VOXELS"

        rows.append({
            "subject": sid,
            "perf_file": os.path.basename(pf),
            "mask_file": os.path.basename(mf),
            "status": status,
            "n_total": n_total,
            "n_nonfinite": n_nonfinite,
            "n_below": n_below,
            "n_above": n_above,
            "pct_valid": frac_valid,
            "pct_below": frac_below,
            "pct_above": frac_above,
            "phys_low": LOW,
            "phys_high": HIGH,
            **stats_dict
        })

    
    all_keys = sorted({k for r in rows for k in r.keys()})
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)

    print(f"Done. Wrote: {OUT_CSV}")
    print(f"Perf files: {len(perf_files)} | Missing masks: {missing_masks}")
    print("Tip: check 'status', 'pct_valid', 'pct_below', 'pct_above' for QC.")

if __name__ == "__main__":
    main()
