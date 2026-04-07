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

IN_MASK_DIR  = r"H:\Data\Kidney_Segmentation\Predicted_Cortex_Masks"
OUT_MASK_DIR = r"H:\Data\Kidney_Segmentation\Final_Predicted_Cortex_Masks"
os.makedirs(OUT_MASK_DIR, exist_ok=True)

#  remove tiny islands after relabeling
REMOVE_SMALL_COMPONENTS = True
MIN_COMP_VOX = 200  # cortex islands smaller than this are removed 

# helpers
def connected_components_3d(binary):
    if HAVE_SCIPY:
        struct = ndi.generate_binary_structure(rank=3, connectivity=1) 
        lab, ncomp = ndi.label(binary, structure=struct)
        return lab.astype(np.int32), int(ncomp)
    if HAVE_SKIMAGE:
        lab = sklabel(binary, connectivity=1)
        return lab.astype(np.int32), int(lab.max())
    raise RuntimeError("Need either scipy or scikit-image installed.")

def world_x_of_voxels(affine, ijk):
    ones = np.ones((ijk.shape[0], 1), dtype=np.float64)
    ijk1 = np.concatenate([ijk.astype(np.float64), ones], axis=1)  # (N,4)
    xyz = ijk1 @ affine.T
    return xyz[:, 0]  

def centroid_world_x(mask_bool, affine):
    ijk = np.argwhere(mask_bool)
    if ijk.size == 0:
        return np.nan
    x = world_x_of_voxels(affine, ijk)
    return float(x.mean())

def remove_small_cc(label_mask_bool, min_vox):
    """Keep only components >= min_vox."""
    if not label_mask_bool.any():
        return label_mask_bool
    lab, ncomp = connected_components_3d(label_mask_bool)
    if ncomp <= 1:
        return label_mask_bool
    counts = np.bincount(lab.ravel())
    keep = np.zeros_like(label_mask_bool, dtype=bool)
    for cid in range(1, len(counts)):
        if counts[cid] >= min_vox:
            keep |= (lab == cid)
    return keep


def enforce_right_left_cortex(mask_path, out_path):
    nii = nib.load(mask_path)
    data = np.asanyarray(nii.dataobj).astype(np.uint8)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {data.shape}")

    u = set(np.unique(data).tolist())
    if not u.issubset({0, 1, 2}):
        raise ValueError(f"Unexpected labels {sorted(u)} in {os.path.basename(mask_path)}")

    # cortex = any non-zero
    cortex = (data > 0)
    if cortex.sum() == 0:
        out = np.zeros_like(data, dtype=np.uint8)
        nib.save(nib.Nifti1Image(out, nii.affine, header=nii.header), out_path)
        return

    # compute centroids and threshold between them.
    left_seed = (data == 1)
    right_seed  = (data == 2)

    xR = centroid_world_x(right_seed, nii.affine)
    xL = centroid_world_x(left_seed, nii.affine)

    ijk_all = np.argwhere(cortex)
    x_all = world_x_of_voxels(nii.affine, ijk_all)

    if np.isfinite(xR) and np.isfinite(xL):
        thresh = 0.5 * (xR + xL)
    else:
        # fallback
        thresh = float(np.median(x_all))

    # Assign all cortex voxels by side
    out = np.zeros_like(data, dtype=np.uint8)
    right_idx = ijk_all[x_all > thresh]
    left_idx  = ijk_all[x_all <= thresh]

    out[right_idx[:, 0], right_idx[:, 1], right_idx[:, 2]] = 1  # RIGHT
    out[left_idx[:, 0],  left_idx[:, 1],  left_idx[:, 2]]  = 2  # LEFT

    # remove tiny islands per side
    if REMOVE_SMALL_COMPONENTS:
        r = (out == 1)
        l = (out == 2)
        r2 = remove_small_cc(r, MIN_COMP_VOX)
        l2 = remove_small_cc(l, MIN_COMP_VOX)
        out[:] = 0
        out[r2] = 1
        out[l2] = 2

    out_nii = nib.Nifti1Image(out, nii.affine, header=nii.header)
    out_nii.set_data_dtype(np.uint8)
    nib.save(out_nii, out_path)

# batch
if __name__ == "__main__":
    masks = [f for f in os.listdir(IN_MASK_DIR) if f.endswith((".nii", ".nii.gz"))]
    masks.sort()
    print(f"Found {len(masks)} cortex masks")

    n_ok, n_err = 0, 0
    for fname in masks:
        in_path  = os.path.join(IN_MASK_DIR, fname)
        out_path = os.path.join(OUT_MASK_DIR, fname)
        try:
            enforce_right_left_cortex(in_path, out_path)
            n_ok += 1
            if n_ok % 100 == 0:
                print(f"[OK] {n_ok}/{len(masks)}")
        except Exception as e:
            n_err += 1
            print(f"[ERROR] {fname}: {e}")

    print("\nDone.")
    print(f"OK: {n_ok}, ERROR: {n_err}")
    print("0=bg, 1=RIGHT cortex, 2=LEFT cortex")
