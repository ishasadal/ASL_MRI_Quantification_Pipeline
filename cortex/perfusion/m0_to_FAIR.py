import os
import glob
import csv
import numpy as np
import SimpleITK as sitk


M0_CORTEX_DIR   = r"H:\Data\Quantification\M0_cortex_masks" 
MC_DIR          = r"H:\Data\Quantification\MC"  
FAIR_WHOLE_DIR  = r"H:\Data\Quantification\FAIR_masks"  
REFINE_CSV      = os.path.join(FAIR_WHOLE_DIR, "_refine_params.csv") 

OUT_DIR         = r"H:\Data\Quantification\FAIR_cortex_masks"
os.makedirs(OUT_DIR, exist_ok=True)

# suffixes
MC_SUFFIX          = "_FAIR_ASL_meanControl.nii.gz"
FAIR_WHOLE_SUFFIX  = "_FAIRmask_refined.nii.gz"    
CORTEX_GLOB        = "*CORTEX*to_M0*.nii*"   

LEFT_LABEL  = 1
RIGHT_LABEL = 2

# helpers
def find_file_exact(folder: str, filename: str) -> str | None:
    p = os.path.join(folder, filename)
    return p if os.path.exists(p) else None

def extract_2d(img: sitk.Image, z: int = 0) -> sitk.Image:
    if img.GetDimension() == 2:
        return img
    size = list(img.GetSize())  # [x,y,z]
    z = int(np.clip(z, 0, size[2] - 1))
    return sitk.Extract(img, [size[0], size[1], 0], [0, 0, z])

def choose_best_z_by_area(mask3: sitk.Image) -> int:
    arr = sitk.GetArrayFromImage(mask3)  # [z,y,x]
    if arr.ndim != 3:
        return 0
    areas = [int(np.count_nonzero(arr[k] > 0)) for k in range(arr.shape[0])]
    return int(np.argmax(areas)) if len(areas) else 0

def resample_mask_to_reference(mask: sitk.Image, ref: sitk.Image) -> sitk.Image:
    """
    Put mask into ref grid using identity transform in PHYSICAL space.
    Handles 2D/3D mismatch (FAIR is 2D with z=1).
    """
    if ref.GetDimension() == 2 and mask.GetDimension() == 3:
        z = choose_best_z_by_area(mask)
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
    t.SetAngle(np.deg2rad(float(angle_deg)))
    t.SetTranslation((float(dx), float(dy)))

    return sitk.Resample(mask2, mask2, t, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

def load_refine_params(csv_path: str) -> dict:
    """
    Reads _refine_params.csv produced by the whole-kidney script.
    Returns dict: base -> (dx,dy,ang) for rows with status 'ok'
    """
    params = {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Refine CSV not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            base = (row.get("base", "") or "").strip()
            status = (row.get("status", "") or "").strip()
            if not base or status != "ok":
                continue
            dx = float(row["dx_px"])
            dy = float(row["dy_px"])
            ang = float(row["angle_deg"])
            params[base] = (dx, dy, ang)
    return params

def constrain_labels_to_whole(cortex_lbl_u8: sitk.Image, whole_bin_u8: sitk.Image) -> sitk.Image:
    """
    Clip cortex labels (1/2) to inside whole kidney mask without collapsing labels.
    Both inputs 2D, index-space aligned.
    """
    whole_bin_u8 = sitk.Cast(whole_bin_u8 > 0, sitk.sitkUInt8)

    right = sitk.Cast(cortex_lbl_u8 == RIGHT_LABEL, sitk.sitkUInt8)
    left  = sitk.Cast(cortex_lbl_u8 == LEFT_LABEL,  sitk.sitkUInt8)

    right = sitk.Cast(right & whole_bin_u8, sitk.sitkUInt8)
    left  = sitk.Cast(left  & whole_bin_u8, sitk.sitkUInt8)

    r = sitk.GetArrayFromImage(right) > 0
    l = sitk.GetArrayFromImage(left) > 0

    out = np.zeros(r.shape, dtype=np.uint8)
    out[r] = RIGHT_LABEL
    out[l] = LEFT_LABEL

    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(cortex_lbl_u8)
    return out_img

def derive_base_from_cortex_filename(fname: str) -> str:
    if "_CORTEX" in fname:
        return fname.split("_CORTEX")[0]
    # fallback: strip extension
    if fname.endswith(".nii.gz"):
        return fname[:-7]
    if fname.endswith(".nii"):
        return fname[:-4]
    return os.path.splitext(fname)[0]

# main
def main():
    refine = load_refine_params(REFINE_CSV)
    print(f"Loaded {len(refine)} whole-kidney refinement transforms from: {REFINE_CSV}")

    cortex_paths = sorted(glob.glob(os.path.join(M0_CORTEX_DIR, CORTEX_GLOB)))
    print(f"Found {len(cortex_paths)} cortex masks in: {M0_CORTEX_DIR}")

    report_csv = os.path.join(OUT_DIR, "_cortex_to_fair_report.csv")
    err_txt    = os.path.join(OUT_DIR, "_cortex_to_fair_errors.txt")

    ok = 0
    fail = 0

    with open(report_csv, "w", newline="") as fcsv, open(err_txt, "w") as ferr:
        w = csv.writer(fcsv)
        w.writerow(["base", "dx_px", "dy_px", "angle_deg", "status", "out_path"])

        for i, cortex_path in enumerate(cortex_paths, 1):
            fname = os.path.basename(cortex_path)
            base = derive_base_from_cortex_filename(fname)

            if base not in refine:
                print(f"[{i}/{len(cortex_paths)}] {base} -> missing refine params in CSV")
                w.writerow([base, "", "", "", "missing_refine_params", ""])
                fail += 1
                continue

            mc_path = find_file_exact(MC_DIR, base + MC_SUFFIX)
            if mc_path is None:
                print(f"[{i}/{len(cortex_paths)}] {base} -> missing FAIR meanControl")
                w.writerow([base, "", "", "", "missing_fair", ""])
                fail += 1
                continue

            whole_fair_path = find_file_exact(FAIR_WHOLE_DIR, base + FAIR_WHOLE_SUFFIX)
            if whole_fair_path is None:
                print(f"[{i}/{len(cortex_paths)}] {base} -> missing FAIR whole mask")
                w.writerow([base, "", "", "", "missing_whole_fair_mask", ""])
                fail += 1
                continue

            try:
                dx_px, dy_px, ang_deg = refine[base]

                fair = sitk.ReadImage(mc_path, sitk.sitkFloat32)
                cortex_m0 = sitk.ReadImage(cortex_path, sitk.sitkUInt8)
                whole_fair = sitk.ReadImage(whole_fair_path, sitk.sitkUInt8)

                # 1) Cortex M0 -> FAIR grid (physical identity)
                cortex_on_fair = resample_mask_to_reference(cortex_m0, fair)

                # 2) Work in 2D
                fair2   = extract_2d(fair, 0)
                cortex2 = extract_2d(cortex_on_fair, 0)
                whole2  = extract_2d(whole_fair, 0)

                # 3) Index-space for pixel dx/dy
                cortex2i = normalize_to_index_space_2d(cortex2)
                whole2i  = normalize_to_index_space_2d(whole2)

                # 4) Apply same rigid correction as whole kidney
                cortex2i_aligned = apply_rigid_2d(cortex2i, dx_px, dy_px, ang_deg)

                # 5) Constrain to whole kidney without label collapse
                cortex2i_aligned = constrain_labels_to_whole(cortex2i_aligned, whole2i)

                # 6) Back to FAIR space
                refined2 = sitk.Image(cortex2i_aligned)
                refined2.CopyInformation(fair2)

                if fair.GetDimension() == 3:
                    refined3 = sitk.JoinSeries(refined2)
                    refined3.CopyInformation(fair)
                    out_mask = refined3
                else:
                    out_mask = refined2

                out_path = os.path.join(OUT_DIR, base + "_CORTEX_FAIRmask.nii.gz")
                sitk.WriteImage(out_mask, out_path)

                ok += 1
                w.writerow([base, dx_px, dy_px, ang_deg, "ok", out_path])
                if ok % 50 == 0 or i == 1:
                    print(f"[{i}/{len(cortex_paths)}] {base} -> OK dx={dx_px:.1f}, dy={dy_px:.1f}, ang={ang_deg:.1f}")

            except Exception as e:
                fail += 1
                ferr.write(f"\n=== FAILED: {cortex_path} ===\n{e}\n")
                print(f"[{i}/{len(cortex_paths)}] {base} -> FAILED: {e}")
                w.writerow([base, "", "", "", f"exception:{e}", ""])

    print("\n====================")
    print(f"DONE. OK={ok} FAIL={fail}")
    print(f"Output: {OUT_DIR}")
    print(f"Report: {report_csv}")
    print(f"Errors: {err_txt}")

if __name__ == "__main__":
    main()