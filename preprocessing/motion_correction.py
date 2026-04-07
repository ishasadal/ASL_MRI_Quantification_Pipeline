import os, glob
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd


RAW_FAIR_DIR = r"H:\Data\Quantification\Raw_FAIR_ASL"   # 2D+time NIfTI: (X,Y,T)
OUT_DIR      = r"H:\Data\Quantification\MC"
os.makedirs(OUT_DIR, exist_ok=True)

PATTERN = "*.nii*"

#Settings
ORDER = "control_first"   

# 2D registration settings
N_ITER = 200
MI_BINS = 50
SAMPLE_PCT = 0.20
SHRINK = [4, 2, 1]
SMOOTH = [2, 1, 0]

def safe_base(name: str) -> str:
    if name.endswith(".nii.gz"): return name[:-7]
    if name.endswith(".nii"):    return name[:-4]
    return os.path.splitext(name)[0]

def is_2d_time_nifti(path: str) -> bool:
    try:
        img = nib.load(path)
        shp = img.shape
        return (len(shp) == 3) and (shp[2] >= 2) and (shp[0] > 8) and (shp[1] > 8)
    except:
        return False

def build_reg_method_2d():
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=MI_BINS)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(SAMPLE_PCT)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetShrinkFactorsPerLevel(SHRINK)
    reg.SetSmoothingSigmasPerLevel(SMOOTH)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=N_ITER
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    return reg

def rigid2d_mc_to_ref(fixed2d: sitk.Image, moving2d: sitk.Image) -> sitk.Image:
    reg = build_reg_method_2d()
    init = sitk.CenteredTransformInitializer(
        fixed2d, moving2d, sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    reg.SetInitialTransform(init, inPlace=False)
    tx = reg.Execute(fixed2d, moving2d)
    out = sitk.Resample(moving2d, fixed2d, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    return out

def split_control_label_2dtime(arr_xyT: np.ndarray, order: str):
    T = arr_xyT.shape[2]
    if order == "control_first":
        control = arr_xyT[:, :, 0:T-1:2]
        label   = arr_xyT[:, :, 1:T:2]
    elif order == "label_first":
        label   = arr_xyT[:, :, 0:T-1:2]
        control = arr_xyT[:, :, 1:T:2]
    else:
        raise ValueError("ORDER must be control_first or label_first")
    n = min(control.shape[2], label.shape[2])
    return control[:, :, :n], label[:, :, :n]

# Main
paths_all = sorted(glob.glob(os.path.join(RAW_FAIR_DIR, PATTERN)))
paths = [p for p in paths_all if is_2d_time_nifti(p)]
print(f"Glob found {len(paths_all)} files; kept {len(paths)} 2D+time files in: {RAW_FAIR_DIR}")

log_rows = []
processed = 0

for p in paths:
    base = safe_base(os.path.basename(p))

    out_meanC = os.path.join(OUT_DIR, base + "_FAIR_meanControl.nii.gz")
    out_delta = os.path.join(OUT_DIR, base + "_FAIR_deltaM_dim.nii.gz")      
    out_pwi   = os.path.join(OUT_DIR, base + "_FAIR_PWI_deltaM.nii.gz")

    try:
        nii = nib.load(p)
        arr = nii.get_fdata().astype(np.float32)  
        X, Y, T = arr.shape
        if T < 2:
            raise ValueError("Too few timepoints")

        # Reference: mean over time (2D)
        ref_np = np.mean(arr, axis=2)
        fixed = sitk.GetImageFromArray(ref_np.T.astype(np.float32))  
        fixed.SetSpacing((1.0, 1.0))

        # 2D motion correction for each timepoint
        mc = np.zeros_like(arr, dtype=np.float32)
        for t in range(T):
            moving = sitk.GetImageFromArray(arr[:, :, t].T.astype(np.float32))
            moving.SetSpacing((1.0, 1.0))
            out = rigid2d_mc_to_ref(fixed, moving)
            mc[:, :, t] = sitk.GetArrayFromImage(out).T

        control, label = split_control_label_2dtime(mc, ORDER)  
        delta = control - label                                  
        meanC = np.mean(control, axis=2)                       
        pwi   = np.mean(delta, axis=2)                         

    
        # Save outputs using original affine
        aff = nii.affine
        hdr = nii.header.copy()

        meanC_3d = meanC[:, :, None]   # (X,Y,1)
        pwi_3d   = pwi[:, :, None]     # (X,Y,1)

        hdr3_1 = hdr.copy()
        hdr3_1.set_data_shape(meanC_3d.shape)
        nib.save(nib.Nifti1Image(meanC_3d.astype(np.float32), aff, hdr3_1), out_meanC)

        hdr3_2 = hdr.copy()
        hdr3_2.set_data_shape(pwi_3d.shape)
        nib.save(nib.Nifti1Image(pwi_3d.astype(np.float32), aff, hdr3_2), out_pwi)

        # delta is already (X,Y,N)
        hdr_d = hdr.copy()
        hdr_d.set_data_shape(delta.shape)  # (X,Y,N)
        nib.save(nib.Nifti1Image(delta.astype(np.float32), aff, hdr_d), out_delta)


        processed += 1
        print(f"[OK] {base}  shape={arr.shape}  pairs={delta.shape[2]}")

        log_rows.append({
            "base": base,
            "shape": str(arr.shape),
            "n_pairs": int(delta.shape[2]),
            "pwi_min": float(np.min(pwi)),
            "pwi_mean": float(np.mean(pwi)),
            "pwi_max": float(np.max(pwi)),
        })

    except Exception as e:
        print(f"[ERR] {base}: {e}")
        log_rows.append({"base": base, "error": str(e)})

log_csv = os.path.join(OUT_DIR, "_mc_pwi_log_2Dtime_minimal.csv")
pd.DataFrame(log_rows).to_csv(log_csv, index=False)
print("\n====================")
print(f"Processed: {processed} / {len(paths)}")
print("Log saved:", log_csv)
