import os
import numpy as np
import nibabel as nib
try:
    from scipy import ndimage as ndi
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    from skimage.measure import label as sklabel
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False


IN_MASK_DIR  = r"H:\Data\Kidney_Segmentation\Predicted_Masks"
OUT_MASK_DIR = r"H:\Data\Kidney_Segmentation\Final_Predicted_Masks"
os.makedirs(OUT_MASK_DIR, exist_ok=True)

KEEP_ONLY_TWO_LARGEST = True


# Helpers
# ==========================
def connected_components_3d(binary):
    if HAVE_SCIPY:
        struct = ndi.generate_binary_structure(rank=3, connectivity=1)  # 6-connectivity
        lab, ncomp = ndi.label(binary, structure=struct)
        return lab.astype(np.int32), int(ncomp)
    if HAVE_SKIMAGE:
        lab = sklabel(binary, connectivity=1)
        return lab.astype(np.int32), int(lab.max())
    raise RuntimeError("Need either scipy or scikit-image installed for connected components.")

def world_x_of_voxels(affine, ijk):
    # homogeneous coords
    ones = np.ones((ijk.shape[0], 1), dtype=np.float64)
    ijk1 = np.concatenate([ijk.astype(np.float64), ones], axis=1)  # (N,4)
    xyz = ijk1 @ affine.T  # (N,4) -> world coords
    return xyz[:, 0]       # x in RAS: increases to anatomical RIGHT

def centroid_world_x(mask_bool, affine):
    """Compute world-x centroid of a boolean mask."""
    ijk = np.argwhere(mask_bool)
    if ijk.size == 0:
        return np.nan
    x = world_x_of_voxels(affine, ijk)
    return float(x.mean())


def enforce_anatomical_right_left(mask_path, out_path):
    nii = nib.load(mask_path)
    data = np.asanyarray(nii.dataobj).astype(np.uint8)

    u = set(np.unique(data).tolist())
    if not u.issubset({0, 1, 2}):
        raise ValueError(f"Unexpected labels {sorted(u)} in {os.path.basename(mask_path)}")

    kidney = (data > 0)

    # If empty, save as is
    if kidney.sum() == 0:
        out_nii = nib.Nifti1Image(data, nii.affine, header=nii.header)
        out_nii.set_data_dtype(np.uint8)
        nib.save(out_nii, out_path)
        return

    lab, ncomp = connected_components_3d(kidney)

    # If only one component (merged kidneys / only one kidney visible):
    # split kidney voxels by median world-x.
    if ncomp < 2:
        ijk = np.argwhere(kidney)
        x = world_x_of_voxels(nii.affine, ijk)
        x_med = np.median(x)

        out = np.zeros_like(data, dtype=np.uint8)
        # anatomical RIGHT = larger x -> label 1
        right_idx = ijk[x > x_med]
        left_idx  = ijk[x <= x_med]

        out[right_idx[:, 0], right_idx[:, 1], right_idx[:, 2]] = 1
        out[left_idx[:, 0],  left_idx[:, 1],  left_idx[:, 2]]  = 2

        out_nii = nib.Nifti1Image(out, nii.affine, header=nii.header)
        out_nii.set_data_dtype(np.uint8)
        nib.save(out_nii, out_path)
        return

    # Component sizes
    counts = np.bincount(lab.ravel())
    comp_ids = np.argsort(counts)[::-1]
    comp_ids = [cid for cid in comp_ids if cid != 0 and counts[cid] > 0]

    if len(comp_ids) < 2:
        raise RuntimeError("Connected components returned <2 non-zero components unexpectedly.")

    # Two largest blobs
    kA_id, kB_id = comp_ids[0], comp_ids[1]
    kA = (lab == kA_id)
    kB = (lab == kB_id)

    # Determine anatomical side using world-x centroid (RAS)
    xA = centroid_world_x(kA, nii.affine)
    xB = centroid_world_x(kB, nii.affine)

    # anatomical right kidney has larger world-x
    right_kidney = kA if xA > xB else kB
    left_kidney  = kB if xA > xB else kA

    out = np.zeros_like(data, dtype=np.uint8)
    out[left_kidney] = 1  
    out[right_kidney]  = 2 

    if not KEEP_ONLY_TWO_LARGEST:
        leftover = kidney & (~(right_kidney | left_kidney))
        if leftover.any():
            # classify leftover voxels by world-x median threshold between the two kidneys
            xR = centroid_world_x(right_kidney, nii.affine)
            xL = centroid_world_x(left_kidney, nii.affine)
            thresh = 0.5 * (xR + xL)

            ijk = np.argwhere(leftover)
            x = world_x_of_voxels(nii.affine, ijk)
            r_idx = ijk[x > thresh]
            l_idx = ijk[x <= thresh]
            out[r_idx[:, 0], r_idx[:, 1], r_idx[:, 2]] = 1
            out[l_idx[:, 0], l_idx[:, 1], l_idx[:, 2]] = 2

    out_nii = nib.Nifti1Image(out, nii.affine, header=nii.header)
    out_nii.set_data_dtype(np.uint8)
    nib.save(out_nii, out_path)


# Batch
if __name__ == "__main__":
    masks = [f for f in os.listdir(IN_MASK_DIR) if f.endswith((".nii", ".nii.gz"))]
    masks.sort()
    print(f"Found {len(masks)} masks")

    n_ok, n_err = 0, 0
    for fname in masks:
        in_path  = os.path.join(IN_MASK_DIR, fname)
        out_path = os.path.join(OUT_MASK_DIR, fname)

        try:
            enforce_anatomical_right_left(in_path, out_path)
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"[OK] {n_ok}/{len(masks)}")
        except Exception as e:
            n_err += 1
            print(f"[ERROR] {fname}: {e}")

    print("\nDone.")
    print(f"OK: {n_ok}, ERROR: {n_err}")
    print("0=background, 1=anatomical RIGHT kidney, 2=anatomical LEFT kidney")
