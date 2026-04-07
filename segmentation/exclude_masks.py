import os
import shutil
import pandas as pd


DATA_ROOT = r"H:/Data/Kidney_Segmentation"
MASK_DIR = os.path.join(DATA_ROOT, "Final_Predicted_Masks")
EXCL_DIR = os.path.join(DATA_ROOT, "Final_Predicted_Masks_EXCLUDED")
os.makedirs(EXCL_DIR, exist_ok=True)

# CSVs from QC pipeline
CATASTROPHIC_CSV = os.path.join(DATA_ROOT, "segmentation_QC_catastrophic_excluded.csv")
SUSPICIOUS_CSV   = os.path.join(DATA_ROOT, "segmentation_QC_suspicious.csv")

# load excluded subjects
df_cat = pd.read_csv(CATASTROPHIC_CSV)
df_sus = pd.read_csv(SUSPICIOUS_CSV)

excluded_ids = set(df_cat["subj_id"]) | set(df_sus["subj_id"])
print(f"Total excluded subjects: {len(excluded_ids)}")

# move files
moved = 0
missing = 0

for fname in os.listdir(MASK_DIR):
    if not fname.endswith((".nii", ".nii.gz")):
        continue

    # match subject id 
    for sid in excluded_ids:
        if sid in fname:
            src = os.path.join(MASK_DIR, fname)
            dst = os.path.join(EXCL_DIR, fname)

            if os.path.exists(src):
                shutil.move(src, dst)
                moved += 1
            else:
                missing += 1
            break

print(f"Moved files: {moved}")
print(f"Missing files (listed but not found): {missing}")
print("Done.")
