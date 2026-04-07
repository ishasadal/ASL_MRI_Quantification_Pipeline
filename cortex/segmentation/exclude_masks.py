import os
import shutil
import pandas as pd


DATA_ROOT = r"H:/Data/Kidney_Segmentation"
MASK_DIR = os.path.join(DATA_ROOT, "Final_Predicted_Cortex_Masks")
EXCL_DIR = os.path.join(DATA_ROOT, "Final_Predicted_Cortex_Masks_EXCLUDED")
os.makedirs(EXCL_DIR, exist_ok=True)

CATASTROPHIC_CSV = os.path.join(DATA_ROOT, "cortex_segmentation_QC_catastrophic_excluded.csv")
SUSPICIOUS_CSV   = os.path.join(DATA_ROOT, "cortex_segmentation_QC_suspicious.csv")

# Load excluded subjects
df_cat = pd.read_csv(CATASTROPHIC_CSV)
df_sus = pd.read_csv(SUSPICIOUS_CSV)

def to_base_id(s):
    s = str(s)
    return s.replace("_MOLLI_Native_cortex_pred", "")

excluded_ids = set(
    df_cat["subj_id"].apply(to_base_id)
) | set(
    df_sus["subj_id"].apply(to_base_id)
)

print(f"Total excluded cortex subjects: {len(excluded_ids)}")

# Move files
moved = 0
skipped = 0

for fname in os.listdir(MASK_DIR):
    if not fname.endswith((".nii", ".nii.gz")):
        continue
    base_id = fname
    base_id = base_id.replace("_MOLLI_Native_cortex_pred.nii.gz", "")
    base_id = base_id.replace("_MOLLI_Native_cortex_pred.nii", "")

    src = os.path.join(MASK_DIR, fname)

    if base_id in excluded_ids:
        dst = os.path.join(EXCL_DIR, fname)
        shutil.move(src, dst)
        moved += 1
    else:
        skipped += 1

print(f"Moved cortex masks: {moved}")
print(f"Kept cortex masks: {skipped}")
print("Done.")
