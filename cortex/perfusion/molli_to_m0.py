import os
import re
import glob
import csv
import traceback
import numpy as np
import SimpleITK as sitk

CORTEX_MASK_DIR   = r"H:\Data\Kidney_Segmentation\Final_Predicted_Cortex_Masks"
M0_DIR            = r"H:\Data\Quantification\Raw_M0"
WHOLE_M0_MASK_DIR = r"H:\Data\Quantification\M0_masks"

OUT_ROOT          = r"H:\Data\Quantification\M0_cortex_masks"
os.makedirs(OUT_ROOT, exist_ok=True)

# knobs
ORIENT = "RAI"

# slab projection
STEP_MM = 1.0
EXTRA_MARGIN_MM = 8.0

# distance-map registration
DO_ROTATION = False          # True to see rotation
MAX_ITERS = 250
SHRINK_FACTORS = [4, 2, 1]
SMOOTH_SIGMAS  = [2.0, 1.0, 0.0]

MAX_TRANSLATION_MM = 15.0

# helpers
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

def signed_distance(u8):
    b = sitk.Cast(u8 > 0, sitk.sitkUInt8)
    return sitk.SignedMaurerDistanceMap(b, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)

def clamp_translation(tx, max_mm=MAX_TRANSLATION_MM):
    # TranslationTransform: params = (dx,dy,dz)
    if isinstance(tx, sitk.TranslationTransform):
        dx, dy, dz = tx.GetOffset()
        dx = float(np.clip(dx, -max_mm, max_mm))
        dy = float(np.clip(dy, -max_mm, max_mm))
        dz = 0.0
        tx.SetOffset((dx, dy, dz))
        return tx

    # Euler3D: params = (rx,ry,rz, tx,ty,tz)
    p = list(tx.GetParameters())
    if len(p) >= 6:
        p[3] = float(np.clip(p[3], -max_mm, max_mm))
        p[4] = float(np.clip(p[4], -max_mm, max_mm))
        p[5] = 0.0
        tx.SetParameters(tuple(p))
    return tx

def whole_mask_path_for_prefix(prefix):
    path = os.path.join(WHOLE_M0_MASK_DIR, f"{prefix}_MOLLI_to_M0_mask.nii.gz")
    return path if os.path.exists(path) else None

def resample_mask_moving_to_fixed(moving_mask_u8, fixed_ref, moving_to_fixed_tx):
    inv = moving_to_fixed_tx.GetInverse()
    return sitk.Resample(moving_mask_u8, fixed_ref, inv, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

# distance-map registration

def register_masks_distance(fixed_u8, moving_u8, do_rotation=False):
    fixed_d  = sitk.Cast(signed_distance(fixed_u8), sitk.sitkFloat32)
    moving_d = sitk.Cast(signed_distance(moving_u8), sitk.sitkFloat32)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetInterpolator(sitk.sitkLinear)

    if do_rotation:
        tx0 = sitk.CenteredTransformInitializer(
            fixed_d, moving_d, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        R.SetInitialTransform(tx0, inPlace=False)
    else:
        tx0 = sitk.TranslationTransform(3)
        R.SetInitialTransform(tx0, inPlace=False)

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-3,
        numberOfIterations=MAX_ITERS,
        relaxationFactor=0.5
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel(SHRINK_FACTORS)
    R.SetSmoothingSigmasPerLevel(SMOOTH_SIGMAS)
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    tx_moving_to_fixed = R.Execute(fixed_d, moving_d)
    tx_moving_to_fixed = clamp_translation(tx_moving_to_fixed, MAX_TRANSLATION_MM)
    return tx_moving_to_fixed

# per-case processing
def process_case(cortex_mask_path):
    prefix = find_subject_prefix(cortex_mask_path)
    if prefix is None:
        raise ValueError(f"Unrecognized cortex filename: {os.path.basename(cortex_mask_path)}")

    m0_path = os.path.join(M0_DIR, f"{prefix}_M0.nii.gz")
    if not os.path.exists(m0_path):
        raise FileNotFoundError(f"M0 not found: {m0_path}")

    whole_path = whole_mask_path_for_prefix(prefix)
    if whole_path is None:
        raise FileNotFoundError(f"Whole-kidney mask not found: {prefix}_MOLLI_to_M0_mask.nii.gz")

    out_path = os.path.join(OUT_ROOT, f"{prefix}_CORTEX_MOLLI_to_M0.nii.gz")

    m0     = reorient(sitk.ReadImage(m0_path), ORIENT)
    cortex = reorient(sitk.ReadImage(cortex_mask_path), ORIENT)
    whole  = reorient(sitk.ReadImage(whole_path), ORIENT)

    # slab axis + vector from M0
    ax = slice_axis_of(m0)
    ax_vec = physical_axis_vector(m0, ax)

    # offsets from cortex mask slab thickness 
    molli_sz = cortex.GetSize()
    molli_sp = cortex.GetSpacing()
    slab_mm = (molli_sz[ax] - 1) * molli_sp[ax]
    half_mm = 0.5 * slab_mm + EXTRA_MARGIN_MM
    offsets_mm = np.arange(-half_mm, half_mm + 1e-6, STEP_MM, dtype=np.float64)

    # cortex labels in MOLLI space
    right_bin_molli = sitk.Cast(cortex == 1, sitk.sitkUInt8)
    left_bin_molli  = sitk.Cast(cortex == 2, sitk.sitkUInt8)

    # project into M0 space
    right_proj = project_label_binary(right_bin_molli, m0, ax_vec, offsets_mm)
    left_proj  = project_label_binary(left_bin_molli,  m0, ax_vec, offsets_mm)
    comb_proj  = sitk.Cast((right_proj > 0) | (left_proj > 0), sitk.sitkUInt8)

    # fixed whole-kidney in M0 space
    whole_u8 = sitk.Cast(whole > 0, sitk.sitkUInt8)
    # resample whole to m0 if headers differ
    if (whole_u8.GetSize() != m0.GetSize() or
        whole_u8.GetSpacing() != m0.GetSpacing() or
        whole_u8.GetOrigin() != m0.GetOrigin() or
        whole_u8.GetDirection() != m0.GetDirection()):
        whole_u8 = sitk.Resample(whole_u8, m0, sitk.Transform(3, sitk.sitkIdentity),
                                 sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    if count_vox(comb_proj) < 50:
        # write empty
        out = np.zeros(sitk.GetArrayViewFromImage(m0).shape, dtype=np.uint8)
        out_img = sitk.GetImageFromArray(out)
        out_img.CopyInformation(m0)
        sitk.WriteImage(out_img, out_path)
        return {"prefix": prefix, "dx_mm": 0.0, "dy_mm": 0.0, "n_right": 0, "n_left": 0, "tx_type": "skipped_empty",
                "cortex_mask": cortex_mask_path, "whole_mask": whole_path, "out_path": out_path}

    # Register moving (cortex union) -> fixed (whole kidney)
    tx_m2f = register_masks_distance(whole_u8, comb_proj, do_rotation=DO_ROTATION)

    # use inverse transform when resampling onto fixed grid
    right_al = resample_mask_moving_to_fixed(right_proj, m0, tx_m2f)
    left_al  = resample_mask_moving_to_fixed(left_proj,  m0, tx_m2f)

    # keep cortex inside whole kidney 
    right_al = sitk.Cast(right_al & whole_u8, sitk.sitkUInt8)
    left_al  = sitk.Cast(left_al  & whole_u8, sitk.sitkUInt8)

    # recombine
    rA = sitk.GetArrayViewFromImage(right_al) > 0
    lA = sitk.GetArrayViewFromImage(left_al)  > 0
    out = np.zeros(rA.shape, dtype=np.uint8)
    out[rA] = 1
    out[lA] = 2

    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(m0)
    sitk.WriteImage(out_img, out_path)

    # report dx/dy (moving->fixed)
    dx = dy = 0.0
    if isinstance(tx_m2f, sitk.TranslationTransform):
        dx, dy, dz = tx_m2f.GetOffset()
    else:
        p = list(tx_m2f.GetParameters())
        if len(p) >= 6:
            dx, dy, dz = p[3], p[4], p[5]

    return {
        "prefix": prefix,
        "cortex_mask": cortex_mask_path,
        "whole_mask": whole_path,
        "out_path": out_path,
        "tx_type": "Euler3D" if DO_ROTATION else "Translation",
        "dx_mm": float(dx),
        "dy_mm": float(dy),
        "n_right": int(np.count_nonzero(out == 1)),
        "n_left": int(np.count_nonzero(out == 2)),
    }

# main
def main():
    cortex_paths = sorted(glob.glob(os.path.join(CORTEX_MASK_DIR, "*.nii")) +
                          glob.glob(os.path.join(CORTEX_MASK_DIR, "*.nii.gz")))
    cortex_paths = [p for p in cortex_paths if "_MOLLI_Native" in os.path.basename(p)]

    if len(cortex_paths) == 0:
        print(f"[WARN] No cortex masks found in {CORTEX_MASK_DIR}")
        return

    print(f"[INFO] Found {len(cortex_paths)} cortex masks")

    report_csv = os.path.join(OUT_ROOT, "batch_report_cortex_to_m0.csv")
    err_txt    = os.path.join(OUT_ROOT, "batch_errors_cortex_to_m0.txt")

    ok = 0
    bad = 0

    with open(report_csv, "w", newline="", encoding="utf-8") as fcsv, \
         open(err_txt, "w", encoding="utf-8") as ferr:

        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "prefix", "cortex_mask", "whole_mask", "out_path",
                "tx_type", "dx_mm", "dy_mm", "n_right", "n_left"
            ]
        )
        writer.writeheader()

        for i, p in enumerate(cortex_paths, 1):
            print(f"\n[{i}/{len(cortex_paths)}] {os.path.basename(p)}")
            try:
                row = process_case(p)
                writer.writerow(row)
                ok += 1
                print(f"  -> OK | {row['tx_type']} dx={row['dx_mm']:.2f} dy={row['dy_mm']:.2f} | R={row['n_right']} L={row['n_left']}")
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
