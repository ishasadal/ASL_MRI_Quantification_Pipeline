import os
import re
import csv
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates


MC_DIR      = r"H:\Data\Quantification\MC"
M0_DIR      = r"H:\Data\Quantification\Raw_M0"
MASK_DIR    = r"H:\Data\Quantification\FAIR_cortex_masks"  
OUT_DIR     = r"H:\Data\Quantification\RBF_CORTEX" 
os.makedirs(OUT_DIR, exist_ok=True)

# constants (from formula)
lambda_bt = 0.9
T1b = 1.650
TI  = 1.400
TI1 = 1.200
alpha = 0.95 * (0.93 ** 2)  # nBS=2
SCALE = 6000.0
scale_factor = (SCALE * lambda_bt * np.exp(TI / T1b)) / (2.0 * alpha * TI1)

# figure settings
MAKE_FIGS = True
SMOOTH_SIGMA_VOX = 1.0
DISPLAY_MIN = 0
DISPLAY_MAX = 300
OVERLAY_ALPHA = 0.85
ROT90_K = 1  

def disp(a):
    return np.rot90(a, k=ROT90_K)

# helpers
def norm_name(s: str) -> str:
    s = os.path.basename(s)
    s = s.replace(".nii.gz", "").replace(".nii", "")
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    return s

def parse_parts(name: str):
    neo = re.search(r"(Neo\d{5}[A-Za-z]{0,2})", name)  
    idt = re.search(r"__\d+__", name)
    dt  = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    neo = neo.group(1) if neo else None
    idt = idt.group(0) if idt else None
    dt  = dt.group(1) if dt else None
    return neo, idt, dt

def key_full(neo, idt, dt):
    if not (neo and idt and dt):
        return None
    return f"{neo}{idt}{dt.replace('-', '_')}"

def key_neo_id(neo, idt):
    if not (neo and idt):
        return None
    return f"{neo}{idt}"

def key_neo_date(neo, dt):
    if not (neo and dt):
        return None
    return f"{neo}{dt.replace('-', '_')}"

def find_match(folder: str, keys: list, must_contain: str | None = None) -> str | None:
    must_norm = norm_name(must_contain) if must_contain else None
    for fn in os.listdir(folder):
        if not (fn.endswith(".nii") or fn.endswith(".nii.gz")):
            continue
        n = norm_name(fn)
        if must_norm and must_norm not in n:
            continue
        for k in keys:
            if k and (k in n):
                return os.path.join(folder, fn)
    return None


def load_2d(path):
    img = nib.load(path)
    arr = np.squeeze(img.get_fdata()).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D after squeeze for {path}. Got {arr.shape}")
    return img, arr

def summarise(vals):
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
    }

# resampling helper (M0 2D slice -> PWI grid)
def resample_2d_to_shape(img2d, out_shape):
    """
    Resample img2d (H,W) to out_shape (H2,W2) using bilinear interpolation.
    Assumes same FOV roughly; pragmatic fix for 128->112 cases.
    """
    in_h, in_w = img2d.shape
    out_h, out_w = out_shape

    ys = np.linspace(0, in_h - 1, out_h, dtype=np.float32)
    xs = np.linspace(0, in_w - 1, out_w, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    coords = np.vstack([yy.ravel(), xx.ravel()])
    out = map_coordinates(img2d, coords, order=1, mode="nearest").reshape(out_h, out_w)
    return out.astype(np.float32)

# cortex specific settings
CORTEX_LEFT_LABEL  = 1
CORTEX_RIGHT_LABEL = 2

# Perfusion voxel filtering 
MIN_PERF = 1.0      
MAX_PERF = 1000.0   

# batch loop
dm_files = sorted([f for f in os.listdir(MC_DIR)
                   if "FAIR_PWI_deltaM" in f and f.endswith((".nii", ".nii.gz"))])
print(f"Found {len(dm_files)} deltaM_robust files in {MC_DIR}")

rows = []
failed = 0

for dm_fn in dm_files:
    try:
        neo, idt, dt = parse_parts(dm_fn)
        if not neo:
            print(f"Could not parse Neo ID from {dm_fn} -> skip")
            failed += 1
            continue

        k_full = key_full(neo, idt, dt)
        k_ni   = key_neo_id(neo, idt)
        k_nd   = key_neo_date(neo, dt)
        keys   = [k_full, k_ni, k_nd, neo]

        dm_path = os.path.join(MC_DIR, dm_fn)

        meanctrl_path = find_match(MC_DIR, keys, must_contain="meanControl")
        m0_path       = find_match(M0_DIR, keys, must_contain="_M0")

        # CORTEX mask search: refined first, then any cortex FAIRmask, then fallback Neo+date
        cortex_mask_path = find_match(MASK_DIR, keys, must_contain="CORTEX")
        if cortex_mask_path is None:
            cortex_mask_path = find_match(MASK_DIR, keys, must_contain="cortex")
        if cortex_mask_path is None:
            cortex_mask_path = find_match(MASK_DIR, keys, must_contain="FAIRmask")
        if cortex_mask_path is None and dt is not None:
            cortex_mask_path = find_match(MASK_DIR, [key_neo_date(neo, dt), neo], must_contain="cortex")

        if m0_path is None:
            print(f"M0 not found for {dm_fn} -> skip")
            failed += 1
            continue
        if cortex_mask_path is None:
            print(f"cortex mask not found for {dm_fn} -> skip")
            failed += 1
            continue

        dm_img, PWI = load_2d(dm_path)
        _, cortex_mask = load_2d(cortex_mask_path)
        cortex_mask = cortex_mask.astype(np.int16)

        cortex_roi = cortex_mask > 0
        if not np.any(cortex_roi):
            print(f"Empty cortex mask for {dm_fn} -> skip")
            failed += 1
            continue

        # Load M0 
        m0_img = nib.load(m0_path)
        m0 = m0_img.get_fdata().astype(np.float32)
        if m0.ndim != 3:
            print(f"M0 not 3D for {dm_fn}: {m0.shape} -> skip")
            failed += 1
            continue

        # If mismatch, resample each slice to PWI grid
        if m0.shape[:2] != PWI.shape:
            m0_res = np.zeros((PWI.shape[0], PWI.shape[1], m0.shape[2]), dtype=np.float32)
            for z in range(m0.shape[2]):
                m0_res[:, :, z] = resample_2d_to_shape(m0[:, :, z], PWI.shape)
            m0 = m0_res

        # Choose best slice by cortex ROI mean
        roi_means = [float(np.mean(m0[:, :, z][cortex_roi])) for z in range(m0.shape[2])]
        z_best = int(np.argmax(roi_means))
        M0_xy = np.maximum(m0[:, :, z_best], 1e-6)

        # Valid cortex voxels:
        # 1) inside cortex ROI
        # 2) M0 above floor (5th percentile inside cortex)
        m0_floor = float(np.percentile(M0_xy[cortex_roi], 5))
        valid = cortex_roi & (M0_xy > m0_floor)

        # Compute rBF
        rbf = np.zeros_like(PWI, dtype=np.float32)
        rbf[valid] = scale_factor * (PWI[valid] / M0_xy[valid])

        # Exclude non-physiologic perfusion values (for summary stats only)
        stats_mask = valid & (rbf > MIN_PERF) & (rbf < MAX_PERF)

        base = (dm_fn
                .replace("_FAIR_ASL_deltaM_robust.nii.gz", "")
                .replace("_FAIR_ASL_deltaM_robust.nii", ""))

        out_rbf = os.path.join(OUT_DIR, f"{base}_CORTEX_rBF.nii.gz")
        nib.save(nib.Nifti1Image(rbf[:, :, None], dm_img.affine, dm_img.header), out_rbf)

        # Stats: cortex total + left/right
        cortex_all = summarise(rbf[stats_mask])

        left_mask  = (cortex_mask == CORTEX_LEFT_LABEL) & stats_mask
        right_mask = (cortex_mask == CORTEX_RIGHT_LABEL) & stats_mask

        cortex_L = summarise(rbf[left_mask]) if np.any(left_mask) else None
        cortex_R = summarise(rbf[right_mask]) if np.any(right_mask) else None

        # QC flag
        qc_flag = ""
        if cortex_all and cortex_all["mean"] < 30:
            qc_flag = "LOW_MEAN(<30)"
        if cortex_all and cortex_all["mean"] > 600:
            qc_flag = (qc_flag + "; " if qc_flag else "") + "HIGH_MEAN(>600)"

        rows.append({
            "subject": base,
            "neo": neo or "",
            "idtoken": idt or "",
            "date": dt or "",
            "dm_path": dm_path,
            "m0_path": m0_path,
            "cortex_mask_path": cortex_mask_path,
            "meanctrl_path": meanctrl_path or "",
            "z_m0": z_best,
            "m0_floor_p5": m0_floor,
            "cortex_n": cortex_all["n"] if cortex_all else 0,
            "cortex_mean": cortex_all["mean"] if cortex_all else np.nan,
            "cortex_std": cortex_all["std"] if cortex_all else np.nan,
            "cortex_median": cortex_all["median"] if cortex_all else np.nan,
            "cortex_p10": cortex_all["p10"] if cortex_all else np.nan,
            "cortex_p90": cortex_all["p90"] if cortex_all else np.nan,
            "L_n": cortex_L["n"] if cortex_L else 0,
            "L_mean": cortex_L["mean"] if cortex_L else np.nan,
            "R_n": cortex_R["n"] if cortex_R else 0,
            "R_mean": cortex_R["mean"] if cortex_R else np.nan,
            "qc_flag": qc_flag,
            "out_rbf": out_rbf,
        })

        # figure 
        if MAKE_FIGS and meanctrl_path:
            _, meanControl = load_2d(meanctrl_path)

            rbf_k = rbf.copy()
            rbf_k[~valid] = np.nan
            rbf_s = gaussian_filter(np.nan_to_num(rbf_k, nan=0.0), sigma=SMOOTH_SIGMA_VOX)
            rbf_s[~valid] = np.nan
            rbf_vis = np.clip(rbf_s, DISPLAY_MIN, DISPLAY_MAX)

            out_fig = os.path.join(OUT_DIR, f"{base}_CORTEX_rBF_overlay.png")
            plt.figure(figsize=(6, 6))
            plt.imshow(disp(meanControl), cmap="gray", origin="lower")
            im = plt.imshow(disp(rbf_vis), cmap="hot", origin="lower",
                            alpha=OVERLAY_ALPHA, vmin=DISPLAY_MIN, vmax=DISPLAY_MAX)
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label("ml/100g/min")
            plt.axis("off")
            plt.title("Cortex perfusion")
            plt.tight_layout()
            plt.savefig(out_fig, dpi=180)
            plt.close()

        print(f"✓ {base} | z={z_best} | cortex_mean={(cortex_all['mean'] if cortex_all else np.nan):.1f} {qc_flag}")

    except Exception as e:
        print(f"❌ Failed {dm_fn}: {e}")
        failed += 1

# summary CSV
csv_path = os.path.join(OUT_DIR, "rbf_cortex_summary.csv")
with open(csv_path, "w", newline="") as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    else:
        f.write("subject\n")

print(f"\nDone. OK={len(rows)}  Failed={failed}")
print("Saved summary CSV:", csv_path)
print("Outputs in:", OUT_DIR)
