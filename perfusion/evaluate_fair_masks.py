import os, glob, csv, shutil
import numpy as np
import SimpleITK as sitk

MC_DIR          = r"H:\Data\Quantification\MC"
WHOLE_FAIR_DIR  = r"H:\Data\Quantification\FAIR_masks"
CORTEX_FAIR_DIR = r"H:\Data\Quantification\FAIR_cortex_masks"

OUT_DIR         = r"H:\Data\Quantification\QC_masks"
os.makedirs(OUT_DIR, exist_ok=True)

WHOLE_SUFFIX   = "_FAIRmask_refined.nii.gz"
MC_SUFFIX      = "_FAIR_ASL_meanControl.nii.gz"
CORTEX_SUFFIX  = "_CORTEX_FAIRmask.nii.gz"

QC_CSV = os.path.join(OUT_DIR, "wholekidney_fair_mask_qc.csv")
BAD_BASES_TXT = os.path.join(OUT_DIR, "bad_bases_qc.txt")

BAD_WHOLE_DIR  = os.path.join(WHOLE_FAIR_DIR, "_BAD_FAIRMASKS_QC")
BAD_CORTEX_DIR = os.path.join(CORTEX_FAIR_DIR, "_BAD_FAIRMASKS_CORTEX_QC")
os.makedirs(BAD_WHOLE_DIR, exist_ok=True)
os.makedirs(BAD_CORTEX_DIR, exist_ok=True)

# Settings
SMOOTH_SIGMA   = 1.0
ROI_DILATE     = 18      # pixels
FG_PERCENTILE  = 65      # intensity threshold percentile inside ROI
FG_MIN_OVERLAP = 0.55    

# Hard fail thresholds
MIN_VOX_HARD   = 80
MAX_COMP_HARD  = 6

# Helpers
def find_exact(folder, filename):
    p = os.path.join(folder, filename)
    return p if os.path.exists(p) else None

def extract_2d(img: sitk.Image, z: int = 0) -> sitk.Image:
    if img.GetDimension() == 2:
        return img
    size = list(img.GetSize())
    z = int(np.clip(z, 0, size[2]-1))
    return sitk.Extract(img, [size[0], size[1], 0], [0, 0, z])

def normalize_to_index_space_2d(img2: sitk.Image) -> sitk.Image:
    out = sitk.Image(img2)
    out.SetOrigin((0.0, 0.0))
    out.SetSpacing((1.0, 1.0))
    out.SetDirection((1.0, 0.0,
                      0.0, 1.0))
    return out

def count_components(bin_u8: sitk.Image) -> int:
    cc = sitk.ConnectedComponent(bin_u8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    return stats.GetNumberOfLabels()

def edge_score(fair2i: sitk.Image, mask2i_bin: sitk.Image) -> float:
    fair_s = sitk.SmoothingRecursiveGaussian(fair2i, SMOOTH_SIGMA) if SMOOTH_SIGMA > 0 else fair2i
    edge = sitk.GradientMagnitude(fair_s)
    edge_np = sitk.GetArrayFromImage(edge).astype(np.float32)

    boundary = sitk.BinaryContour(mask2i_bin)  # 1px
    roi = sitk.BinaryDilate(mask2i_bin, [ROI_DILATE, ROI_DILATE])

    bnd = sitk.GetArrayFromImage(boundary).astype(np.uint8) > 0
    roi_np = sitk.GetArrayFromImage(roi).astype(np.uint8) > 0

    m = bnd & roi_np
    if m.sum() == 0:
        return -1e9
    return float(edge_np[m].sum())

def foreground_overlap(fair2i: sitk.Image, mask2i_bin: sitk.Image) -> float:
    fair_np = sitk.GetArrayFromImage(fair2i).astype(np.float32)
    mask_np = sitk.GetArrayFromImage(mask2i_bin).astype(np.uint8) > 0
    if mask_np.sum() == 0:
        return 0.0

    roi = sitk.BinaryDilate(mask2i_bin, [ROI_DILATE, ROI_DILATE])
    roi_np = sitk.GetArrayFromImage(roi).astype(np.uint8) > 0

    vals = fair_np[roi_np]
    if vals.size < 50:
        return 0.0

    thr = float(np.percentile(vals, FG_PERCENTILE))
    fg = (fair_np > thr) & roi_np

    overlap = np.logical_and(mask_np, fg).sum()
    return float(overlap / (mask_np.sum() + 1e-9))

def safe_move(src_path: str, dst_dir: str) -> bool:
    if src_path is None or (not os.path.exists(src_path)):
        return False
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    if os.path.exists(dst_path):
        return True  # already moved
    shutil.move(src_path, dst_path)
    return True

# Main (qc + move bad)
whole_paths = sorted(glob.glob(os.path.join(WHOLE_FAIR_DIR, f"*{WHOLE_SUFFIX}")))
print(f"Found {len(whole_paths)} whole-kidney FAIR masks")

rows = []
edge_scores = []
sizes = []

missing_mc = 0
missing_cortex = 0

for p in whole_paths:
    fn = os.path.basename(p)
    base = fn[:-len(WHOLE_SUFFIX)]

    mc_path = find_exact(MC_DIR, base + MC_SUFFIX)
    if mc_path is None:
        missing_mc += 1
        rows.append({"base": base, "status": "missing_mc", "flags": "missing_mc",
                     "n_vox": 0, "n_comp": 0, "edge_score": np.nan, "fg_overlap": np.nan,
                     "has_cortex_mask": False,
                     "whole_path": p, "mc_path": ""})
        continue

    fair = sitk.ReadImage(mc_path, sitk.sitkFloat32)
    mask = sitk.ReadImage(p, sitk.sitkUInt8)

    fair2 = normalize_to_index_space_2d(extract_2d(fair, 0))
    mask2 = normalize_to_index_space_2d(extract_2d(mask, 0))
    mask_bin = sitk.Cast(mask2 > 0, sitk.sitkUInt8)

    n_vox = int(np.count_nonzero(sitk.GetArrayFromImage(mask_bin)))
    n_comp = count_components(mask_bin) if n_vox > 0 else 0
    sc = edge_score(fair2, mask_bin) if n_vox > 0 else -1e9
    fg = foreground_overlap(fair2, mask_bin) if n_vox > 0 else 0.0

    cortex_path = find_exact(CORTEX_FAIR_DIR, base + CORTEX_SUFFIX)
    has_cortex = bool(cortex_path is not None)
    if not has_cortex:
        missing_cortex += 1

    rows.append({
        "base": base,
        "status": "pending",
        "flags": "",
        "n_vox": n_vox,
        "n_comp": n_comp,
        "edge_score": float(sc),
        "fg_overlap": float(fg),
        "has_cortex_mask": has_cortex,
        "whole_path": p,
        "mc_path": mc_path
    })

    if n_vox > 0:
        edge_scores.append(sc)
        sizes.append(n_vox)

# Adaptive thresholds
edge_lo = float(np.percentile(edge_scores, 10)) if len(edge_scores) >= 50 else (float(min(edge_scores)) if edge_scores else -1e9)
size_lo = int(np.percentile(sizes, 1)) if len(sizes) >= 50 else MIN_VOX_HARD
size_hi = int(np.percentile(sizes, 99)) if len(sizes) >= 50 else (int(max(sizes)) if sizes else 10**9)

print("\nThresholds (applied):")
print(f"  edge_score < {edge_lo:.1f} (10th percentile)")
print(f"  size outside [{size_lo}, {size_hi}] (1st–99th percentile)")
print(f"  fg_overlap < {FG_MIN_OVERLAP:.2f} (hard)")

bad_bases = []
flag_counts = {}

for r in rows:
    if r["status"] != "pending":
        continue

    flags = []
    if r["n_vox"] < MIN_VOX_HARD:
        flags.append("empty_or_tiny")
    if r["n_comp"] > MAX_COMP_HARD:
        flags.append("too_many_components")
    if r["n_vox"] < size_lo or r["n_vox"] > size_hi:
        flags.append("size_outlier")
    if np.isfinite(r["edge_score"]) and r["edge_score"] < edge_lo:
        flags.append("low_edge_support")
    if np.isfinite(r["fg_overlap"]) and r["fg_overlap"] < FG_MIN_OVERLAP:
        flags.append("low_fg_overlap_shifted")

    if flags:
        r["status"] = "bad"
        r["flags"] = "|".join(flags)
        bad_bases.append(r["base"])
        for f in flags:
            flag_counts[f] = flag_counts.get(f, 0) + 1
    else:
        r["status"] = "ok"
        r["flags"] = ""

# QC CSV
with open(QC_CSV, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["base","status","flags","n_vox","n_comp","edge_score","fg_overlap","has_cortex_mask","whole_path","mc_path"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fieldnames})

# bad bases list
with open(BAD_BASES_TXT, "w", encoding="utf-8") as f:
    for b in bad_bases:
        f.write(b + "\n")

# Move bad whole masks + dependent cortex masks
moved_whole = 0
moved_cortex = 0
missing_whole_files = 0
missing_cortex_files = 0

for base in bad_bases:
    whole_src = os.path.join(WHOLE_FAIR_DIR, base + WHOLE_SUFFIX)
    cortex_src = os.path.join(CORTEX_FAIR_DIR, base + CORTEX_SUFFIX)

    if not safe_move(whole_src, BAD_WHOLE_DIR):
        missing_whole_files += 1
    else:
        moved_whole += 1

    if os.path.exists(cortex_src):
        if safe_move(cortex_src, BAD_CORTEX_DIR):
            moved_cortex += 1
    else:
        missing_cortex_files += 1

# Summary
n_ok = sum(r["status"] == "ok" for r in rows)
n_bad = sum(r["status"] == "bad" for r in rows)
n_missing_mc = sum(r["status"] == "missing_mc" for r in rows)

print("\n==================== QC + MOVE SUMMARY ====================")
print(f"Total whole masks scanned:   {len(rows)}")
print(f"Missing meanControl:         {n_missing_mc}")
print(f"OK (kept):                   {n_ok}")
print(f"BAD (excluded):              {n_bad}")
print("\nBreakdown of BAD flags:")
for k in sorted(flag_counts.keys()):
    print(f"  {k}: {flag_counts[k]}")
print("\nMoved files:")
print(f"  Whole masks moved:         {moved_whole} -> {BAD_WHOLE_DIR}")
print(f"  Cortex masks moved:        {moved_cortex} -> {BAD_CORTEX_DIR}")
print(f"  Missing whole file (rare): {missing_whole_files}")
print(f"  Missing cortex file:       {missing_cortex_files}")
print(f"\nQC CSV saved:                {QC_CSV}")
print(f"Bad bases list saved:        {BAD_BASES_TXT}")
print("===========================================================")
