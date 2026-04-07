import os
import pydicom
import numpy as np
import nibabel as nib

PATIENT_ROOT = r"Z:\Database"
OUTPUT_DIR = r"H:\Data\Kidney_Segmentation\Raw_M0"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_molli_native_dicom(dicom_dir):
    if not os.path.isdir(dicom_dir):
        return None

    for fname in os.listdir(dicom_dir):
        upper = fname.upper()
        if "M0" in upper:
            return os.path.join(dicom_dir, fname)

    return None

def process_patient(patient_dir):
    patient_name = os.path.basename(patient_dir)
    dicom_dir = os.path.join(patient_dir, "Dicom")

    molli_path = find_molli_native_dicom(dicom_dir)
    if molli_path is None:
        print(f"[SKIP] {patient_name}: no M0 DICOM found")
        return
    print(f"Processing {patient_name} -> {molli_path}")

    ds = pydicom.dcmread(molli_path)
    pixel_data = ds.pixel_array  #(num_slices, rows, cols)
    print("  Original pixel_data shape:", pixel_data.shape)

    # 3D
    if pixel_data.ndim == 2:
        # Single slice (rows, cols) -> (rows, cols, 1)
        vol = pixel_data[:, :, np.newaxis]
    elif pixel_data.ndim == 3:
        # (num_slices, rows, cols) -> (rows, cols, num_slices)
        vol = np.transpose(pixel_data, (1, 2, 0))
    else:
        print(f"  [ERROR] Unexpected DICOM shape: {pixel_data.shape}")
        return

    # rotate in-plane if needed (keep axes 0,1 as in-plane)
    vol = np.rot90(vol, k=3, axes=(0, 1))

    # Make it 3D NIfTI 
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(vol.astype(np.float32), affine)

    output_file = os.path.join(OUTPUT_DIR, f"{patient_name}_M0.nii.gz")
    nib.save(nifti_img, output_file)

    print(f"  Saved NIfTI: {output_file}")
    print(f"  Shape of NIfTI: {vol.shape}")


def main():
    patients = [
        d for d in os.listdir(PATIENT_ROOT)
        if os.path.isdir(os.path.join(PATIENT_ROOT, d))
    ]
    print(f"Found {len(patients)} potential patient folders\n")

    for pname in sorted(patients):
        patient_dir = os.path.join(PATIENT_ROOT, pname)
        process_patient(patient_dir)

if __name__ == "__main__":
    main()
