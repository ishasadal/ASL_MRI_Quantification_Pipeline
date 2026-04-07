import os
import re
import glob
import csv
import traceback
import numpy as np
import SimpleITK as sitk

WHOLE_MOLLI_MASK_DIR = r"H:\Data\Kidney_Segmentation\Masks"  
M0_DIR               = r"H:\Data\Quantification\Raw_M0"
OUT_ROOT             = r"H:\Data\Quantification\M0_masks_GT"
os.makedirs(OUT_ROOT, exist_ok=True)

ORIENT = "RAI"
# slab projection
STEP_MM = 1.0
EXTRA_MARGIN_MM = 8.0

# translation refinement
DO_TRANSLATION_REFINE = True
MAX_TRANSLATION_MM = 15.0          # clamp and search range
SEARCH_STEP_MM = 1.0               # 1 mm grid
ASSUME_LEFT_RIGHT_LABELS = True

# Helpers
def reorient(img, orient=ORIENT):
    f = sitk.DICOMOrientImageFilter()
    f.SetDesiredCoordinateOrientation(orient)
    return f.Execute(img)

def slice_axis_of(img):
    sz = img.GetSize()  # (x,y,z)
    return int(np.argmin(sz))

def physical_axis_vector(img, axis_index):
    D = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    v = D[:, axis_index]
    n = np.linalg.norm(v) + 1e-12
    return v / n

def project_label_binary(molli_bin, fixed_ref, axis_vec, offsets_mm):
    acc = sitk.Image(fixed_ref.GetSize(), sitk.sitkUInt8)
    acc.CopyInformation(fixed_ref)
    for d in offsets_mm:
        t = sitk.TranslationTransform(3)
        t.SetOffset(tuple((-d * axis_vec).tolist()))
        moved = sitk.Resample(molli_bin, fixed_ref, t, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        acc = sitk.Maximum(acc, moved)
    return sitk.Cast(acc > 0, sitk.sitkUInt8)

def find_subject_prefix(mask_path):
    base = os.path.basename(mask_path)
    m = re.match(r"(.+?)_MOLLI_Native.*\.nii(\.gz)?$", base)
    return m.group(1) if m else None

def count_vox(img_u8):
    return int(np.count_nonzero(sitk.GetArrayViewFromImage(img_u8)))

def boundary_from_mask(mask_u8):
    # boundary = mask - erode(mask)
    m = sitk.Cast(mask_u8 > 0, sitk.sitkUInt8)
    er = sitk.BinaryErode(m, (1, 1, 0))  # in-plane only, keep z
    b = sitk.Cast(m & sitk.Cast(er == 0, sitk.sitkUInt8), sitk.sitkUInt8)
    return b

def gradmag_m0(m0_img):
    # in-plane gradients only
    m0f = sitk.Cast(m0_img, sitk.sitkFloat32)
    m0s = sitk.SmoothingRecursiveGaussian(m0f, 1.0)
    gx = sitk.Derivative(m0s, direction=0, order=1)
    gy = sitk.Derivative(m0s, direction=1, order=1)
    g = sitk.Sqrt(gx*gx + gy*gy)
    return g

def resample_with_tx_nn(mask_u8, ref, tx):
    return sitk.Resample(mask_u8, ref, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

def score_edge_alignment(mask_u8, gmag):
    # sum of gradient magnitude on mask boundary
    b = boundary_from_mask(mask_u8)
    b_arr = sitk.GetArrayViewFromImage(b).astype(np.float32)
    g_arr = sitk.GetArrayViewFromImage(gmag).astype(np.float32)
    return float((b_arr * g_arr).sum())

def refine_translation_by_edges(moving_u8, ref_m0, gmag,
                               max_mm=MAX_TRANSLATION_MM, step_mm=SEARCH_STEP_MM):
    # search in mm in x/y only
    best_score = -1e30
    best_tx = sitk.TranslationTransform(3)
    best_tx.SetOffset((0.0, 0.0, 0.0))

    # convert mm grid directly to physical-space translation 
    grid = np.arange(-max_mm, max_mm + 1e-6, step_mm, dtype=np.float64)

    # precompute baseline to speed up comparison
    for dx in grid:
        for dy in grid:
            tx = sitk.TranslationTransform(3)
            tx.SetOffset((float(dx), float(dy), 0.0))
            moved = resample_with_tx_nn(moving_u8, ref_m0, tx)
            sc = score_edge_alignment(moved, gmag)
            if sc > best_score:
                best_score = sc
                best_tx = tx

    return best_tx, best_score

# Per-case processing
def process_case(molli_mask_path):
    prefix = find_subject_prefix(molli_mask_path)
    if prefix is None:
        raise ValueError(f"Unrecognized MOLLI filename: {os.path.basename(molli_mask_path)}")

    m0_path = os.path.join(M0_DIR, f"{prefix}_M0.nii.gz")
    if not os.path.exists(m0_path):
        raise FileNotFoundError(f"M0 not found: {m0_path}")

    out_path = os.path.join(OUT_ROOT, f"{prefix}_MOLLI_to_M0_mask.nii.gz")

    m0    = reorient(sitk.ReadImage(m0_path), ORIENT)
    molli = reorient(sitk.ReadImage(molli_mask_path), ORIENT)

    ax = slice_axis_of(m0)
    ax_vec = physical_axis_vector(m0, ax)

    ms = molli.GetSize()
    sp = molli.GetSpacing()
    slab_mm = (ms[ax] - 1) * sp[ax]
    half_mm = 0.5 * slab_mm + EXTRA_MARGIN_MM
    offsets_mm = np.arange(-half_mm, half_mm + 1e-6, STEP_MM, dtype=np.float64)

    # project into M0 space
    if ASSUME_LEFT_RIGHT_LABELS:
        right_bin = sitk.Cast(molli == 1, sitk.sitkUInt8)
        left_bin  = sitk.Cast(molli == 2, sitk.sitkUInt8)

        right_proj = project_label_binary(right_bin, m0, ax_vec, offsets_mm)
        left_proj  = project_label_binary(left_bin,  m0, ax_vec, offsets_mm)
        union_proj = sitk.Cast((right_proj > 0) | (left_proj > 0), sitk.sitkUInt8)
    else:
        union_proj = project_label_binary(sitk.Cast(molli > 0, sitk.sitkUInt8), m0, ax_vec, offsets_mm)
        right_proj = None
        left_proj = None

    if count_vox(union_proj) < 50:
        out = np.zeros(sitk.GetArrayViewFromImage(m0).shape, dtype=np.uint8)
        out_img = sitk.GetImageFromArray(out); out_img.CopyInformation(m0)
        sitk.WriteImage(out_img, out_path)
        return {"prefix": prefix, "molli_mask": molli_mask_path, "out_path": out_path,
                "dx_mm": 0.0, "dy_mm": 0.0, "n_right": 0, "n_left": 0, "mode": "skipped_empty"}

    dx = dy = 0.0
    mode = "project_only"

    # translation refinement using M0 edges
    if DO_TRANSLATION_REFINE:
        gmag = gradmag_m0(m0)
        tx, best_sc = refine_translation_by_edges(union_proj, m0, gmag,
                                                  max_mm=MAX_TRANSLATION_MM, step_mm=SEARCH_STEP_MM)
        dx, dy, dz = tx.GetOffset()
        mode = f"project+edge_refine(sc={best_sc:.1f})"

        if ASSUME_LEFT_RIGHT_LABELS:
            right_proj = resample_with_tx_nn(right_proj, m0, tx)
            left_proj  = resample_with_tx_nn(left_proj,  m0, tx)
        else:
            union_proj = resample_with_tx_nn(union_proj, m0, tx)

    # write output labels
    if ASSUME_LEFT_RIGHT_LABELS:
        rA = sitk.GetArrayViewFromImage(right_proj) > 0
        lA = sitk.GetArrayViewFromImage(left_proj)  > 0
        out = np.zeros(rA.shape, dtype=np.uint8)
        out[rA] = 1
        out[lA] = 2
        n_right = int(np.count_nonzero(out == 1))
        n_left  = int(np.count_nonzero(out == 2))
    else:
        out = sitk.GetArrayFromImage(union_proj).astype(np.uint8)
        n_right = int(np.count_nonzero(out > 0))
        n_left = 0

    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(m0)
    sitk.WriteImage(out_img, out_path)

    return {"prefix": prefix, "molli_mask": molli_mask_path, "out_path": out_path,
            "dx_mm": float(dx), "dy_mm": float(dy), "n_right": n_right, "n_left": n_left, "mode": mode}

# main batch
def main():
    molli_paths = sorted(glob.glob(os.path.join(WHOLE_MOLLI_MASK_DIR, "*.nii")) +
                         glob.glob(os.path.join(WHOLE_MOLLI_MASK_DIR, "*.nii.gz")))
    molli_paths = [p for p in molli_paths if "_MOLLI_Native" in os.path.basename(p)]

    if len(molli_paths) == 0:
        print(f"[WARN] No MOLLI whole-kidney masks found in {WHOLE_MOLLI_MASK_DIR}")
        return

    print(f"[INFO] Found {len(molli_paths)} MOLLI whole-kidney masks")

    report_csv = os.path.join(OUT_ROOT, "batch_report_wholekidney_molli_to_m0.csv")
    err_txt    = os.path.join(OUT_ROOT, "batch_errors_wholekidney_molli_to_m0.txt")

    ok = 0
    bad = 0

    with open(report_csv, "w", newline="", encoding="utf-8") as fcsv, \
         open(err_txt, "w", encoding="utf-8") as ferr:

        writer = csv.DictWriter(
            fcsv,
            fieldnames=["prefix", "molli_mask", "out_path", "mode", "dx_mm", "dy_mm", "n_right", "n_left"]
        )
        writer.writeheader()

        for i, p in enumerate(molli_paths, 1):
            print(f"\n[{i}/{len(molli_paths)}] {os.path.basename(p)}")
            try:
                row = process_case(p)
                writer.writerow(row)
                ok += 1
                print(f"  -> OK | {row['mode']} dx={row['dx_mm']:.2f} dy={row['dy_mm']:.2f} | R={row['n_right']} L={row['n_left']}")
            except Exception as e:
                bad += 1
                ferr.write(f"\n=== FAILED: {p} ===\n")
                ferr.write(str(e) + "\n")
                ferr.write(traceback.format_exc() + "\n")
                print(f"  -> FAIL: {e}")

    print("\n====================")
    print(f"Done. OK={ok}  FAIL={bad}")
    print(f"Report: {report_csv}")
    print(f"Errors: {err_txt}")
    print(f"Output: {OUT_ROOT}")

if __name__ == "__main__":
    main()