import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_ROOT  = r"H:/Data/Kidney_Segmentation"
MOLLI_DIR  = os.path.join(DATA_ROOT, "Raw_MOLLI")  
OUT_DIR    = os.path.join(DATA_ROOT, "Predicted_Masks")
QC_DIR     = os.path.join(OUT_DIR, "_qc_png")  

MODEL_PATH = os.path.join(DATA_ROOT, "unet_molli_kidney_3class_best.pth")

IMAGE_SUFFIX = "_MOLLI_Native.nii.gz"
CROP_SIZE = (256, 256)  # must match training
N_CLASSES = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(QC_DIR, exist_ok=True)

# Model (same as training)
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.inc   = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1   = Up(1024, 256)
        self.up2   = Up(512, 128)
        self.up3   = Up(256, 64)
        self.up4   = Up(128, 64)

        self.outc  = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)  # logits (B,3,H,W)


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Same 1–99 percentile clipping + scaling as training."""
    vol = vol.astype(np.float32)
    p1 = np.percentile(vol, 1)
    p99 = np.percentile(vol, 99)
    vol = np.clip(vol, p1, p99)
    vol = (vol - p1) / (p99 - p1 + 1e-8)
    return vol

def list_subject_ids(molli_dir: str, suffix: str):
    ids = []
    for fn in os.listdir(molli_dir):
        if fn.endswith(suffix):
            ids.append(fn.replace(suffix, ""))
    return sorted(list(set(ids)))

def read_id_file(path):
    ids = set()
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    ids.add(s)
    return ids

# Optional QC 
def save_qc_png(img2d, lab2d, out_png):
    try:
        import imageio.v2 as imageio
    except Exception:
        return 

    # normalize image for viewing
    x = img2d.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x = (255 * x).astype(np.uint8)

    # make RGB overlay: left=red, right=blue
    rgb = np.stack([x, x, x], axis=-1)
    left = (lab2d == 1)
    right = (lab2d == 2)
    rgb[left, 0] = 255
    rgb[left, 1] = 0
    rgb[left, 2] = 0
    rgb[right, 0] = 0
    rgb[right, 1] = 0
    rgb[right, 2] = 255

    imageio.imwrite(out_png, rgb)

# inference
def predict_subject(subj_id: str, model: nn.Module, qc_slices: int = 3):
    molli_path = os.path.join(MOLLI_DIR, f"{subj_id}{IMAGE_SUFFIX}")
    out_path   = os.path.join(OUT_DIR,  f"{subj_id}_MOLLI_Native_mask_pred.nii.gz")

    if not os.path.exists(molli_path):
        print(f"[WARN] Missing MOLLI for {subj_id}, skipping.")
        return

    nii = nib.load(molli_path)
    vol = nii.get_fdata().astype(np.float32)

    # Expect (H,W,Z)
    if vol.ndim != 3:
        print(f"[WARN] {subj_id}: expected 3D MOLLI, got shape {vol.shape}, skipping.")
        return

    H, W, Z = vol.shape
    vol_n = normalize_volume(vol)

    pred_lab = np.zeros((H, W, Z), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for z in range(Z):
            slice2d = vol_n[..., z]  # (H,W)

            t = torch.from_numpy(slice2d)[None, None, ...]  # (1,1,H,W)
            t = F.interpolate(t, size=CROP_SIZE, mode="bilinear", align_corners=False)
            t = t.to(DEVICE, dtype=torch.float32)

            logits = model(t)                 # (1,3,256,256)
            lab = torch.argmax(logits, dim=1) # (1,256,256), values 0/1/2

            # resize label map back to original HxW using nearest
            lab = lab.unsqueeze(1).float()    # (1,1,256,256)
            lab = F.interpolate(lab, size=(H, W), mode="nearest")
            lab2d = lab[0, 0].byte().cpu().numpy()

            pred_lab[..., z] = lab2d

    # Save with original affine + header 
    out_nii = nib.Nifti1Image(pred_lab, nii.affine, header=nii.header)
    out_nii.set_data_dtype(np.uint8)
    nib.save(out_nii, out_path)
    print("Saved:", out_path)

    # QC overlays
    if qc_slices > 0:
        zs = list(range(Z))
        np.random.shuffle(zs)
        zs = zs[:min(qc_slices, Z)]
        for z in zs:
            png_path = os.path.join(QC_DIR, f"{subj_id}_z{z:03d}.png")
            save_qc_png(vol_n[..., z], pred_lab[..., z], png_path)

def main():
    # Load model
    model = UNet2D(in_channels=1, out_channels=N_CLASSES)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)

    # Decide which subjects to run:
    # all MOLLI subjects minus manual train/val/test ids 
    manual_ids = set()
    manual_ids |= read_id_file(os.path.join(DATA_ROOT, "train_ids.txt"))
    manual_ids |= read_id_file(os.path.join(DATA_ROOT, "val_ids.txt"))
    manual_ids |= read_id_file(os.path.join(DATA_ROOT, "test_ids.txt"))

    all_ids = list_subject_ids(MOLLI_DIR, IMAGE_SUFFIX)
    pred_ids = [sid for sid in all_ids if sid not in manual_ids]

    print("Device:", DEVICE)
    print("Total MOLLI subjects:", len(all_ids))
    print("Manual-labelled subjects:", len(manual_ids))
    print("Subjects to predict:", len(pred_ids))

    # Run
    for sid in pred_ids:
        try:
            predict_subject(sid, model, qc_slices=3)
        except Exception as e:
            print(f"[ERROR] {sid}: {e}")

if __name__ == "__main__":
    main()
