import os
import glob
import csv
import numpy as np
import SimpleITK as sitk

M0_MASKS_DIR = r"H:\Data\Quantification\M0_masks"
MC_DIR       = r"H:\Data\Quantification\MC"
OUT_DIR      = r"H:\Data\Quantification\FAIR_masks_good"
os.makedirs(OUT_DIR, exist_ok=True)

MASK_SUFFIX = "_MOLLI_to_M0_mask.nii.gz"
MC_SUFFIX   = "_FAIR_ASL_meanControl.nii.gz"
CSV_PATH    = os.path.join(OUT_DIR, "_refine_params.csv")

# settings
DX_RANGE = 12
DY_RANGE = 12
COARSE_STEP = 2          # coarse translation step (pixels)
FINE_STEP   = 1          # fine translation step (pixels)

# rotation search (degrees) (0 to disable rotation search)
ANGLE_RANGE_DEG = 6
ANGLE_STEP_DEG  = 2

# ROI around mask (limits scoring to local area)
ROI_DILATE = 15

# Smooth FAIR before edge detection (reduces noise)
SMOOTH_SIGMA = 1.0

# Helpers
def find_file_exact(folder: str, filename: str) -> str | None:
    p = os.path.join(folder, filename)
    return p if os.path.exists(p) else None

def extract_2d(img: sitk.Image, z: int = 0) -> sitk.Image:
    if img.GetDimension() == 2:
        return img
    size = list(img.GetSize())  # [x,y,z]
    z = int(np.clip(z, 0, size[2] - 1))
    return sitk.Extract(img, [size[0], size[1], 0], [0, 0, z])

def resample_mask_to_reference(mask: sitk.Image, ref: sitk.Image) -> sitk.Image:
    """
    Put mask into ref grid using identity transform in physical space.
    """
    # ensure both are same dimension
    if ref.GetDimension() == 2 and mask.GetDimension() == 3:
        # choose best z of mask by area
        arr = sitk.GetArrayFromImage(mask)  # [z,y,x]
        z = int(np.argmax([np.count_nonzero(arr[k] > 0) for k in range(arr.shape[0])])) if arr.ndim == 3 else 0
        mask = extract_2d(mask, z)
    if ref.GetDimension() == 3 and mask.GetDimension() == 2:
        mask = sitk.JoinSeries(mask)

    tx = sitk.Transform(ref.GetDimension(), sitk.sitkIdentity)
    return sitk.Resample(mask, ref, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

def normalize_to_index_space_2d(img2: sitk.Image) -> sitk.Image:
    out = sitk.Image(img2)
    out.SetOrigin((0.0, 0.0))
    out.SetSpacing((1.0, 1.0))
    out.SetDirection((1.0, 0.0,
                      0.0, 1.0))
    return out

def apply_rigid_2d(mask2: sitk.Image, dx: float, dy: float, angle_deg: float) -> sitk.Image:
    """
    Apply rotation around image center + translation in pixel space.
    """
    size = mask2.GetSize()  # [x,y]
    cx = (size[0] - 1) / 2.0
    cy = (size[1] - 1) / 2.0

    t = sitk.Euler2DTransform()
    t.SetCenter((cx, cy))
    t.SetAngle(np.deg2rad(angle_deg))
    t.SetTranslation((dx, dy))

    out = sitk.Resample(mask2, mask2, t, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return out

def score_edge_boundary(edge_np: np.ndarray, boundary_np: np.ndarray, roi_np: np.ndarray) -> float:
    m = (roi_np > 0) & (boundary_np > 0)
    if m.sum() == 0:
        return -1e9
    return float(edge_np[m].sum())

# Main
def main():
    mask_paths = sorted(glob.glob(os.path.join(M0_MASKS_DIR, f"*{MASK_SUFFIX}")))
    print(f"Found {len(mask_paths)} masks")

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["base", "dx_px", "dy_px", "angle_deg", "score", "status"])

        ok = 0
        fail = 0

        for i, mask_path in enumerate(mask_paths, start=1):
            base = os.path.basename(mask_path)[:-len(MASK_SUFFIX)]
            mc_path = find_file_exact(MC_DIR, base + MC_SUFFIX)
            if mc_path is None:
                print(f"[{i}/{len(mask_paths)}] {base} -> missing FAIR")
                w.writerow([base, "", "", "", "", "missing_fair"])
                fail += 1
                continue

            try:
                fair = sitk.ReadImage(mc_path, sitk.sitkFloat32)
                mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)

                # Resample mask into FAIR grid (real FAIR-space mask)
                mask_on_fair = resample_mask_to_reference(mask, fair)

                # Work in 2D for refinement 
                fair2 = extract_2d(fair, 0)
                mask2 = extract_2d(mask_on_fair, 0)

                # Make index-space copies so dx/dy are in pixels and stable
                fair2i = normalize_to_index_space_2d(fair2)
                mask2i = normalize_to_index_space_2d(mask2)

                # Edge image from FAIR
                fair_s = sitk.SmoothingRecursiveGaussian(fair2i, SMOOTH_SIGMA) if SMOOTH_SIGMA > 0 else fair2i
                edge = sitk.GradientMagnitude(fair_s)
                edge_np = sitk.GetArrayFromImage(edge).astype(np.float32)  # [y,x]

                # Boundary of mask + ROI around it (limits scoring to kidney region)
                mask_bin = sitk.Cast(mask2i > 0, sitk.sitkUInt8)
                boundary = sitk.BinaryContour(mask_bin)  # 1-pixel contour
                roi = sitk.BinaryDilate(mask_bin, [ROI_DILATE, ROI_DILATE])

                boundary_np = sitk.GetArrayFromImage(boundary).astype(np.uint8)
                roi_np = sitk.GetArrayFromImage(roi).astype(np.uint8)

                # Baseline score (no change)
                best_dx, best_dy, best_ang = 0.0, 0.0, 0.0
                best_mask = mask2i
                best_score = score_edge_boundary(edge_np, boundary_np, roi_np)

                # Coarse translation search (angle=0) 
                for dy in range(-DY_RANGE, DY_RANGE + 1, COARSE_STEP):
                    for dx in range(-DX_RANGE, DX_RANGE + 1, COARSE_STEP):
                        cand = apply_rigid_2d(mask2i, dx, dy, 0.0)
                        bnd = sitk.BinaryContour(sitk.Cast(cand > 0, sitk.sitkUInt8))
                        bnd_np = sitk.GetArrayFromImage(bnd).astype(np.uint8)
                        sc = score_edge_boundary(edge_np, bnd_np, roi_np)
                        if sc > best_score:
                            best_score = sc
                            best_dx, best_dy, best_ang = float(dx), float(dy), 0.0
                            best_mask = cand

                # Fine translation search around best 
                for dy in range(int(best_dy) - COARSE_STEP, int(best_dy) + COARSE_STEP + 1, FINE_STEP):
                    for dx in range(int(best_dx) - COARSE_STEP, int(best_dx) + COARSE_STEP + 1, FINE_STEP):
                        cand = apply_rigid_2d(mask2i, dx, dy, 0.0)
                        bnd = sitk.BinaryContour(sitk.Cast(cand > 0, sitk.sitkUInt8))
                        bnd_np = sitk.GetArrayFromImage(bnd).astype(np.uint8)
                        sc = score_edge_boundary(edge_np, bnd_np, roi_np)
                        if sc > best_score:
                            best_score = sc
                            best_dx, best_dy, best_ang = float(dx), float(dy), 0.0
                            best_mask = cand

                # small rotation search
                if ANGLE_RANGE_DEG > 0 and ANGLE_STEP_DEG > 0:
                    for ang in range(-ANGLE_RANGE_DEG, ANGLE_RANGE_DEG + 1, ANGLE_STEP_DEG):
                        cand = apply_rigid_2d(mask2i, best_dx, best_dy, float(ang))
                        bnd = sitk.BinaryContour(sitk.Cast(cand > 0, sitk.sitkUInt8))
                        bnd_np = sitk.GetArrayFromImage(bnd).astype(np.uint8)
                        sc = score_edge_boundary(edge_np, bnd_np, roi_np)
                        if sc > best_score:
                            best_score = sc
                            best_ang = float(ang)
                            best_mask = cand

                # Put refined 2D mask back into FAIR geometry + save
                refined2 = sitk.Image(best_mask)
                refined2.CopyInformation(fair2)  # back to FAIR slice geometry

                # If FAIR is 3D (z=1), return a 3D mask to match it
                if fair.GetDimension() == 3:
                    refined3 = sitk.JoinSeries(refined2)
                    refined3.CopyInformation(fair)
                    out_mask = refined3
                else:
                    out_mask = refined2

                out_path = os.path.join(OUT_DIR, base + "_FAIRmask_refined.nii.gz")
                sitk.WriteImage(out_mask, out_path)

                ok += 1
                w.writerow([base, best_dx, best_dy, best_ang, best_score, "ok"])
                if ok % 50 == 0 or i == 1:
                    print(f"[{i}/{len(mask_paths)}] {base} -> dx={best_dx:.1f}, dy={best_dy:.1f}, ang={best_ang:.1f}")

            except Exception as e:
                print(f"[{i}/{len(mask_paths)}] {base} -> FAILED: {e}")
                w.writerow([base, "", "", "", "", f"exception:{e}"])
                fail += 1

    print(f"\nDONE. OK={ok} FAIL={fail}")
    print(f"Output: {OUT_DIR}")
    print(f"Log: {CSV_PATH}")

if __name__ == "__main__":
    main()
