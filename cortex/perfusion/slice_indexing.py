import os
import csv
import numpy as np
import nibabel as nib

DATA_ROOT = r"H:/Data/Kidney_Segmentation"
MASK_DIR  = os.path.join(DATA_ROOT, "Masks_cortex")
MASK_SUFFIX = "_MOLLI_Native_cortex_mask.nii.gz"

out_csv = os.path.join(DATA_ROOT, "cortex_labelled_slice_index.csv")

rows = []
problems = []

for fn in os.listdir(MASK_DIR):
    if not fn.endswith(MASK_SUFFIX):
        continue

    sid = fn.replace(MASK_SUFFIX, "")
    m = nib.load(os.path.join(MASK_DIR, fn)).get_fdata().astype(np.int16)

    if m.ndim != 3:
        problems.append((sid, "mask_not_3d", str(m.shape)))
        continue

    zs = []
    areas = []
    for z in range(m.shape[2]):
        cortex = (m[..., z] == 1) | (m[..., z] == 2)
        a = int(np.sum(cortex))
        if a > 0:
            zs.append(z)
            areas.append(a)

    if len(zs) == 1:
        rows.append((sid, zs[0]))
    elif len(zs) == 2:
        # pick slice with larger cortex area
        best = zs[int(np.argmax(areas))]
        rows.append((sid, best))
        print(f"[INFO] {sid}: two slices {zs}, picked {best} (areas={areas})")
    else:
        problems.append((sid, f"nonzero_slices={len(zs)}", str(zs)))

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["subject_id", "z"])
    w.writerows(rows)

print("Wrote:", out_csv)
print("OK subjects:", len(rows))
print("Problems:", len(problems))
if problems:
    print("Problems list:")
    for p in problems:
        print("  ", p)
