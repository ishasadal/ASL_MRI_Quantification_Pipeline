import os
import csv
import SimpleITK as sitk
from datetime import datetime

PATIENT_ROOT = r"Z:\Database"
OUT_CSV      = r"H:\Data\Quantification\subjects.csv"
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
FAIR_KEY = "WIP_SOURCE_-_FAIR_FB"   


def find_fair_asl_file(dicom_dir):
    if not os.path.isdir(dicom_dir):
        return None
    for fname in os.listdir(dicom_dir):
        if FAIR_KEY in fname.upper():
            return os.path.join(dicom_dir, fname)
    return None


def has_tag(reader, tag):
    """Check if tag exists in reader metadata."""
    try:
        return tag in reader.GetMetaDataKeys()
    except Exception:
        return False


def get_tag(reader, tag):
    """Get tag value as string, or None."""
    try:
        if has_tag(reader, tag):
            v = reader.GetMetaData(tag)
            return v if v != "" else None
    except Exception:
        pass
    return None


def parse_patient_age(age_str):
    """'044Y' -> 44.0, '018M' -> 1.5, '120D' -> 0.33"""
    if not age_str:
        return None
    s = str(age_str).strip()
    if len(s) < 2:
        return None
    try:
        val = float(s[:-1])
        unit = s[-1].upper()
        if unit == "Y":
            return val
        if unit == "M":
            return val / 12.0
        if unit == "D":
            return val / 365.25
    except Exception:
        return None
    return None


def parse_yyyymmdd(d):
    if not d:
        return None
    s = str(d).strip()
    try:
        return datetime.strptime(s, "%Y%m%d").date().isoformat()
    except Exception:
        return None


def safe_float(x):
    if x is None:
        return None
    try:
        return float(str(x).strip())
    except Exception:
        return None


rows = []
patients = [d for d in os.listdir(PATIENT_ROOT) if os.path.isdir(os.path.join(PATIENT_ROOT, d))]
print(f"PATIENT_ROOT: {PATIENT_ROOT}")
print(f"Found {len(patients)} patient folders")

for pname in sorted(patients):
    patient_dir = os.path.join(PATIENT_ROOT, pname)
    dicom_dir = os.path.join(patient_dir, "Dicom")

    fair_path = find_fair_asl_file(dicom_dir)
    if fair_path is None:
        continue

    r = sitk.ImageFileReader()
    r.SetFileName(fair_path)
    r.LoadPrivateTagsOn()
    r.ReadImageInformation()

    # Standard DICOM tags
    patient_id  = get_tag(r, "0010|0020")  # PatientID
    sex         = get_tag(r, "0010|0040")  # PatientSex
    age_raw     = get_tag(r, "0010|1010")  # PatientAge
    weight_raw  = get_tag(r, "0010|1030")  # PatientWeight (kg)
    height_raw  = get_tag(r, "0010|1020")  # PatientSize (m)
    study_date  = get_tag(r, "0008|0020")  # StudyDate
    series_desc = get_tag(r, "0008|103e")  # SeriesDescription
    protocol    = get_tag(r, "0018|1030")  # ProtocolName

    age_years = parse_patient_age(age_raw)
    weight_kg = safe_float(weight_raw)
    height_m  = safe_float(height_raw)

    bmi = None
    if weight_kg is not None and height_m is not None and height_m > 0:
        bmi = weight_kg / (height_m ** 2)

    rows.append({
        "subject_id": pname,
        "scan_date": parse_yyyymmdd(study_date),
        "sex": sex,
        "age_years": age_years,
        "weight_kg": weight_kg,
        "height_m": height_m,
        "bmi": bmi,
        "patient_id_in_dicom": patient_id,
        "series_description": series_desc,
        "protocol_name": protocol,
        "fair_dicom_filename": os.path.basename(fair_path),
    })

    print(f"[OK] {pname} | date={parse_yyyymmdd(study_date)} sex={sex} age={age_years} "
          f"w={weight_kg} h={height_m} | {os.path.basename(fair_path)}")

# CSV
fieldnames = [
    "subject_id","scan_date","sex","age_years","weight_kg","height_m","bmi",
    "patient_id_in_dicom","series_description","protocol_name","fair_dicom_filename"
]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"\n Saved subjects.csv: {OUT_CSV}")
print(f"Rows written: {len(rows)}")
