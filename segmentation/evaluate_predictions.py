import os
from glob import glob
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

DATA_ROOT = r"H:/Data/Kidney_Segmentation"
MASK_DIR  = os.path.join(DATA_ROOT, "Final_Predicted_Masks")
PATTERN   = "*.nii*"

OUT_FEATURES = os.path.join(DATA_ROOT, "segmentation_QC_features_fixed.csv")
OUT_CATA     = os.path.join(DATA_ROOT, "segmentation_QC_catastrophic_excluded.csv")
OUT_SUSP     = os.path.join(DATA_ROOT, "segmentation_QC_suspicious.csv")

PLOT_DIR = os.path.join(DATA_ROOT, "_QC_PLOTS_AUTO")
os.makedirs(PLOT_DIR, exist_ok=True)

# labels
LEFT_LABEL = 1   
RIGHT_LABEL  = 2

# settings
CONNECTIVITY = 1

# hard fails
MIN_TOTAL_VOX_HARD   = 12000
MIN_KIDNEY_VOX_HARD  = 4000

# catastrophic tail 
EXCLUDE_TOP_PERCENT_NONZERO = 5.0

# soft scoring
VOL_LOW_Q  = 0.01
VOL_HIGH_Q = 0.99
RATIO_LOW  = 0.33
RATIO_HIGH = 3.0
FRAG_SOFT_START = 0.05
FRAG_SOFT_END   = 0.30

# how to define suspicious:
#   "any_soft"  -> suspicious if badness_score > 0 
#   "top_pct"   -> suspicious = top SUSPICIOUS_TOP_PERCENT of non-catastrophic
SUSPICIOUS_MODE = "any_soft"
SUSPICIOUS_TOP_PERCENT = 10.0 


# helpers
def load_mask_uint8(path):
    nii = nib.load(path)
    data = np.asanyarray(nii.dataobj).astype(np.uint8)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Unexpected ndim={data.ndim}, shape={data.shape}")
    return data

def connected_components(binary_3d, connectivity=1):
    struct = ndi.generate_binary_structure(3, connectivity)
    lab, n = ndi.label(binary_3d.astype(bool), structure=struct)
    return lab, int(n)

def frac_not_lcc(binary_3d, connectivity=1):
    total = int(binary_3d.sum())
    if total == 0:
        return 0, np.nan
    lab, n = connected_components(binary_3d, connectivity)
    if n <= 1:
        return n, 0.0
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    largest = counts.max()
    return n, float((total - largest) / total)

def clip01(x):
    return float(np.clip(x, 0.0, 1.0))

def soft_score_volume(v, vlow, vhigh):
    if not np.isfinite(v) or not np.isfinite(vlow) or not np.isfinite(vhigh) or vlow <= 0:
        return 0.0
    if vlow <= v <= vhigh:
        return 0.0
    if v < vlow:
        return clip01((vlow - v) / vlow)
    return clip01((v - vhigh) / vhigh)

def soft_score_ratio(r):
    if not np.isfinite(r):
        return 1.0
    if RATIO_LOW <= r <= RATIO_HIGH:
        return 0.0
    if r < RATIO_LOW:
        return clip01((RATIO_LOW - r) / RATIO_LOW)
    return clip01((r - RATIO_HIGH) / RATIO_HIGH)

def soft_score_frag(frac):
    if not np.isfinite(frac):
        return 1.0
    if frac <= FRAG_SOFT_START:
        return 0.0
    if frac >= FRAG_SOFT_END:
        return 1.0
    return float((frac - FRAG_SOFT_START) / (FRAG_SOFT_END - FRAG_SOFT_START))

def save_hist(series, out_png, title):
    v = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if len(v) == 0:
        return
    plt.figure()
    plt.hist(v, bins=60)
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# features

paths = sorted(glob(os.path.join(MASK_DIR, PATTERN)))
records = []

for p in paths:
    sid = os.path.basename(p).replace(".nii.gz", "").replace(".nii", "")
    try:
        m = load_mask_uint8(p)
        u = set(np.unique(m).tolist())
        if not u.issubset({0, 1, 2}):
            raise ValueError(f"Unexpected labels {sorted(u)}")

        right = (m == RIGHT_LABEL)
        left  = (m == LEFT_LABEL)

        rv = int(right.sum())
        lv = int(left.sum())
        tv = rv + lv
        ratio_rl = np.inf if lv == 0 else (rv / max(lv, 1))

        ncc_r, frac_r = frac_not_lcc(right, CONNECTIVITY)
        ncc_l, frac_l = frac_not_lcc(left,  CONNECTIVITY)

        records.append({
            "subj_id": sid,
            "path": p,
            "right_vol": rv,
            "left_vol": lv,
            "total_vol": tv,
            "ratio_right_left": ratio_rl,
            "ncomp_right": ncc_r,
            "ncomp_left": ncc_l,
            "frac_not_lcc_right": frac_r,
            "frac_not_lcc_left": frac_l,
            "error": ""
        })
    except Exception as e:
        records.append({"subj_id": sid, "path": p, "error": str(e)})

df = pd.DataFrame(records)
print("Total subjects:", len(df))
df.to_csv(OUT_FEATURES, index=False)
print("Saved QC features:", OUT_FEATURES)

# hard fails

df["hard_fail"] = False
df.loc[df["error"].astype(str) != "", "hard_fail"] = True
df.loc[df["right_vol"].fillna(0) == 0, "hard_fail"] = True
df.loc[df["left_vol"].fillna(0) == 0, "hard_fail"] = True
df.loc[df["total_vol"].fillna(0) < MIN_TOTAL_VOX_HARD, "hard_fail"] = True
df.loc[df["right_vol"].fillna(0) < MIN_KIDNEY_VOX_HARD, "hard_fail"] = True
df.loc[df["left_vol"].fillna(0) < MIN_KIDNEY_VOX_HARD, "hard_fail"] = True

# badness score

valid_vol = df.loc[~df["hard_fail"], "total_vol"].replace([np.inf, -np.inf], np.nan).dropna()
vlow  = float(valid_vol.quantile(VOL_LOW_Q)) if len(valid_vol) > 20 else np.nan
vhigh = float(valid_vol.quantile(VOL_HIGH_Q)) if len(valid_vol) > 20 else np.nan
print(f"Volume quantiles (soft): low={vlow:.0f}, high={vhigh:.0f}")

scores = []
for _, r in df.iterrows():
    if r["hard_fail"]:
        scores.append(np.nan)
        continue
    s_vol = soft_score_volume(r["total_vol"], vlow, vhigh)
    s_rat = soft_score_ratio(r["ratio_right_left"])
    s_frg = max(soft_score_frag(r["frac_not_lcc_right"]),
                soft_score_frag(r["frac_not_lcc_left"]))
    bad = 0.45 * s_vol + 0.30 * s_rat + 0.25 * s_frg
    scores.append(float(bad))
df["badness_score"] = scores

# catastrophic exclusions

df["catastrophic_exclude"] = df["hard_fail"].copy()

cand = df.loc[(~df["hard_fail"]) & (df["badness_score"] > 0), "badness_score"].dropna()
if len(cand) > 0:
    thr_cata = float(np.percentile(cand.to_numpy(), 100.0 - EXCLUDE_TOP_PERCENT_NONZERO))
    df.loc[df["badness_score"] > thr_cata, "catastrophic_exclude"] = True
    print(f"Catastrophic tail cutoff (top {EXCLUDE_TOP_PERCENT_NONZERO:.1f}% of non-zero): {thr_cata:.4f}")
else:
    thr_cata = np.nan
    print("No non-zero badness scores found — no tail exclusion applied.")

cata = df[df["catastrophic_exclude"]].copy().sort_values(["hard_fail","badness_score"], ascending=[False, False])
cata.to_csv(OUT_CATA, index=False)
print(f"Catastrophic excluded: {len(cata)} / {len(df)}")
print("Saved catastrophic list:", OUT_CATA)

# suspicious cases
non_cata = df[~df["catastrophic_exclude"]].copy()

if SUSPICIOUS_MODE == "any_soft":
    # suspicious = any soft deviation at all
    suspicious = non_cata[non_cata["badness_score"].fillna(0) > 0].copy()
    susp_rule = "badness_score > 0 (any soft deviation)"
elif SUSPICIOUS_MODE == "top_pct":
    cand2 = non_cata["badness_score"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(cand2) > 0:
        thr_susp = float(np.percentile(cand2.to_numpy(), 100.0 - SUSPICIOUS_TOP_PERCENT))
        suspicious = non_cata[non_cata["badness_score"] >= thr_susp].copy()
        susp_rule = f"top {SUSPICIOUS_TOP_PERCENT:.1f}% by badness_score"
    else:
        suspicious = non_cata.iloc[0:0].copy()
        susp_rule = "no candidates"
else:
    raise ValueError("SUSPICIOUS_MODE must be 'any_soft' or 'top_pct'")

suspicious = suspicious.sort_values("badness_score", ascending=False)
suspicious.to_csv(OUT_SUSP, index=False)

print(f"Suspicious (review) cases: {len(suspicious)} / {len(df)}  using rule: {susp_rule}")
print("Saved suspicious list:", OUT_SUSP)

print("\nTop 20 worst (non-catastrophic):")
print(
    non_cata.sort_values("badness_score", ascending=False)
    .head(20)[["subj_id", "total_vol", "ratio_right_left",
               "frac_not_lcc_right", "frac_not_lcc_left", "badness_score"]]
    .to_string(index=False)
)
