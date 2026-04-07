import os
import SimpleITK as sitk

PATIENT_ROOT = r"Z:\Database"
OUTPUT_DIR   = r"H:\Data\Quantification\Raw_FAIR_ASL"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_fair_asl_file(dicom_dir):
    if not os.path.isdir(dicom_dir):
        return None

    for fname in os.listdir(dicom_dir):
        u = fname.upper()

        # match pattern
        if "WIP_SOURCE_-_FAIR_FB" in u:
            return os.path.join(dicom_dir, fname)

    return None

def process_patient(patient_dir):
    patient_name = os.path.basename(patient_dir)
    dicom_dir = os.path.join(patient_dir, "Dicom")

    fair_path = find_fair_asl_file(dicom_dir)
    if fair_path is None:
        print(f"[SKIP] {patient_name}: no FAIR ASL DICOM file found")
        return

    print(f"[PT] {patient_name}")
    print(f"     Using file: {os.path.basename(fair_path)}")

    # Read single DICOM file (often multiframe)
    r = sitk.ImageFileReader()
    r.SetFileName(fair_path)
    r.LoadPrivateTagsOn()
    img = r.Execute()

    out_path = os.path.join(OUTPUT_DIR, f"{patient_name}_FAIR_ASL.nii.gz")
    sitk.WriteImage(img, out_path)

    print(f"[OK]  Saved: {out_path}")
    print("     size     ", img.GetSize())
    print("     spacing  ", img.GetSpacing())
    print("     origin   ", img.GetOrigin())
    print("     direction", img.GetDirection())

def main():
    patients = [d for d in os.listdir(PATIENT_ROOT) if os.path.isdir(os.path.join(PATIENT_ROOT, d))]
    print(f"PATIENT_ROOT: {PATIENT_ROOT}")
    print(f"Found {len(patients)} patient folders")

    for pname in sorted(patients):
        process_patient(os.path.join(PATIENT_ROOT, pname))

if __name__ == "__main__":
    main()
