import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_ROOT = r"H:/Data/Kidney_Segmentation"
FEATURES_CSV     = os.path.join(DATA_ROOT, "cortex_segmentation_QC_features.csv")
CATASTROPHIC_CSV = os.path.join(DATA_ROOT, "cortex_segmentation_QC_catastrophic_excluded.csv")
SUSPICIOUS_CSV   = os.path.join(DATA_ROOT, "cortex_segmentation_QC_suspicious.csv")

OUT_DIR = os.path.join(DATA_ROOT, "QC_PLOTS_CORTEX")
os.makedirs(OUT_DIR, exist_ok=True)

# load data
df_all = pd.read_csv(FEATURES_CSV)
df_cat = pd.read_csv(CATASTROPHIC_CSV)
df_sus = pd.read_csv(SUSPICIOUS_CSV)


# Create matching base_id across all tables.
def to_base_id(s):
    s = str(s)
    s = s.replace("_MOLLI_Native_cortex_pred", "")
    s = s.replace("_MOLLI_Native_mask_pred", "")  
    s = s.replace("_MOLLI_Native_mask_pred_3class", "")
    return s

for df in (df_all, df_cat, df_sus):
    if "subj_id" not in df.columns:
        raise ValueError("Expected a 'subj_id' column in all QC CSVs.")
    df["base_id"] = df["subj_id"].apply(to_base_id)

# excluded cases
excluded_ids = set(df_cat["base_id"]) | set(df_sus["base_id"])
df_excl = df_all[df_all["base_id"].isin(excluded_ids)].copy()

print(f"Total cortex subjects: {len(df_all)}")
print(f"Excluded cortex subjects (catastrophic ∪ suspicious): {len(df_excl)}")

# histogram: TOTAL CORTEX VOLUME
plt.figure(figsize=(8, 5))
plt.hist(df_all["total_vol"].dropna(), bins=50, edgecolor="black")
plt.xlabel("Total cortex volume (voxels)")
plt.ylabel("Number of subjects")
plt.title("Distribution of total cortex volume")

plt.tight_layout()
hist_path = os.path.join(OUT_DIR, "total_cortex_volume_histogram.png")
plt.savefig(hist_path, dpi=300)
plt.close()

print("Saved:", hist_path)

# bar plot: EXCLUSION SUBGROUP 
# thresholds consistent with cortex QC logic
LR_LOW, LR_HIGH = 0.33, 3.0
VOL_LOW_Q, VOL_HIGH_Q = 0.01, 0.99

vol_low  = df_all["total_vol"].quantile(VOL_LOW_Q)
vol_high = df_all["total_vol"].quantile(VOL_HIGH_Q)

def primary_reason(row):
    # Missing side
    if row.get("right_vol", 0) == 0 or row.get("left_vol", 0) == 0:
        return "missing_cortex_side"

    # LR asymmetry
    r = row.get("ratio_right_left", np.nan)
    if np.isfinite(r) and (r < LR_LOW or r > LR_HIGH):
        return "lr_asymmetry"

    # Size outliers
    tv = row.get("total_vol", np.nan)
    if np.isfinite(tv) and tv < vol_low:
        return "small_volume"
    if np.isfinite(tv) and tv > vol_high:
        return "large_volume"

    # Fragmentation (use your saved features if present)
    fr = row.get("frac_not_lcc_right", np.nan)
    fl = row.get("frac_not_lcc_left", np.nan)
    if (np.isfinite(fr) and fr > 0.30) or (np.isfinite(fl) and fl > 0.30):
        return "fragmentation"

    return "other"

df_excl["primary_reason"] = df_excl.apply(primary_reason, axis=1)

reason_order = [
    "lr_asymmetry",
    "small_volume",
    "large_volume",
    "missing_cortex_side",
    "fragmentation",
    "other"
]

reason_counts = (
    df_excl["primary_reason"]
    .value_counts()
    .reindex(reason_order)
    .dropna()
)

plt.figure(figsize=(9, 5))
reason_counts.plot(kind="bar", edgecolor="black")
plt.ylabel("Number of excluded subjects")
plt.xlabel("Primary exclusion reason")
plt.title("Cortex segmentation QC exclusion reasons")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()
bar_path = os.path.join(OUT_DIR, "cortex_exclusion_reason_barplot.png")
plt.savefig(bar_path, dpi=300)
plt.close()

print("Saved:", bar_path)

print("\nExclusion reason counts:")
print(reason_counts)

#  exclusion list you plotted
df_excl_out = df_excl[["base_id", "subj_id", "total_vol", "right_vol", "left_vol",
                      "ratio_right_left", "frac_not_lcc_right", "frac_not_lcc_left",
                      "primary_reason"]].copy()

out_csv = os.path.join(OUT_DIR, "cortex_excluded_subjects_with_reasons.csv")
df_excl_out.to_csv(out_csv, index=False)
print("Saved:", out_csv)
