import os
import csv
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = r"H:/Data/Kidney_Segmentation"
MOLLI_DIR = os.path.join(DATA_ROOT, "Raw_MOLLI")
MASK_DIR  = os.path.join(DATA_ROOT, "Masks_cortex")

IMAGE_SUFFIX = "_MOLLI_Native.nii.gz"
MASK_SUFFIX  = "_MOLLI_Native_cortex_mask.nii.gz"

# split files during cortex training
VAL_IDS  = os.path.join(DATA_ROOT, "Cortex_UNet_3class", "val_ids_CORTEX.txt")
TEST_IDS = os.path.join(DATA_ROOT, "Cortex_UNet_3class", "test_ids_CORTEX.txt")

# labelled slice map
SLICE_CSV = os.path.join(DATA_ROOT, "cortex_labelled_slice_index.csv")

# model checkpoint
MODEL_PATH = os.path.join(DATA_ROOT, "Cortex_UNet_3class", "unet_molli_cortex_3class_best.pth")

# must match training
CROP_SIZE = (256, 256)
N_CLASSES = 3
BATCH_SIZE = 8
NUM_WORKERS = 0 if os.name == "nt" else 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

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
        return self.outc(x)

# Helpers 
def normalize_volume(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    p1 = np.percentile(vol, 1)
    p99 = np.percentile(vol, 99)
    vol = np.clip(vol, p1, p99)
    vol = (vol - p1) / (p99 - p1 + 1e-8)
    return vol

def load_ids(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_slice_map(csv_path: str) -> dict:
    m = {}
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            m[row["subject_id"]] = int(row["z"])
    return m

SLICE_MAP = load_slice_map(SLICE_CSV)

def dice_for_class(pred, target, cls, eps=1e-6):
    pred_c = (pred == cls)
    tgt_c  = (target == cls)
    inter = (pred_c & tgt_c).sum().item()
    den = pred_c.sum().item() + tgt_c.sum().item() + eps
    if den < eps * 10:
        return 1.0
    return float(2.0 * inter / den)

# Dataset
class CortexEvalDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        if sid not in SLICE_MAP:
            raise KeyError(f"{sid} missing from slice map: {SLICE_CSV}")

        img_path  = os.path.join(MOLLI_DIR, f"{sid}{IMAGE_SUFFIX}")
        mask_path = os.path.join(MASK_DIR,  f"{sid}{MASK_SUFFIX}")

        img = nib.load(img_path).get_fdata().astype(np.float32)
        msk = nib.load(mask_path).get_fdata().astype(np.int64)

        if img.ndim != 3 or msk.ndim != 3:
            raise ValueError(f"{sid}: expected 3D img+mask, got img={img.shape}, mask={msk.shape}")

        z = int(SLICE_MAP[sid])
        if z < 0 or z >= img.shape[2] or z >= msk.shape[2]:
            raise ValueError(f"{sid}: z={z} out of range img={img.shape} mask={msk.shape}")

        img_n = normalize_volume(img)
        img2d = img_n[..., z]
        m2d   = msk[..., z]

        img_t = torch.from_numpy(img2d)[None, ...].float()     # (1,H,W)
        m_t   = torch.from_numpy(m2d).long()                   # (H,W)

        # Resize to training size
        img_t = F.interpolate(img_t.unsqueeze(0), size=CROP_SIZE, mode="bilinear", align_corners=False).squeeze(0)
        m_f   = F.interpolate(m_t[None, None].float(), size=CROP_SIZE, mode="nearest")
        m_t   = m_f[0, 0].long()

        return img_t, m_t, sid

# Evaluation
def eval_split(ids_path, split_name, model):
    ids = load_ids(ids_path)
    if len(ids) == 0:
        print(f"[INFO] {split_name} IDs file not found or empty: {ids_path} -> skipping.")
        return

    ds = CortexEvalDataset(ids)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

    ce = nn.CrossEntropyLoss()

    losses = []
    d_right = []
    d_left  = []
    d_binary = []
    bad_subjects = 0

    model.eval()
    with torch.no_grad():
        for imgs, masks, sids in loader:
            imgs  = imgs.to(DEVICE, dtype=torch.float32)
            masks = masks.to(DEVICE, dtype=torch.long)

            logits = model(imgs)
            loss = ce(logits, masks)
            losses.append(loss.item())

            pred = torch.argmax(logits, dim=1)

            pred_cpu = pred.cpu()
            mask_cpu = masks.cpu()

            for i in range(pred_cpu.shape[0]):
                pr = pred_cpu[i]
                gt = mask_cpu[i]

                d_right.append(dice_for_class(pr, gt, 1))
                d_left.append(dice_for_class(pr, gt, 2))

                pb = (pr > 0)
                mb = (gt > 0)
                inter = (pb & mb).sum().item()
                den = pb.sum().item() + mb.sum().item() + 1e-6
                d_binary.append(float(2.0 * inter / den) if den > 0 else 1.0)

    losses = np.array(losses, dtype=float)
    d_right = np.array(d_right, dtype=float)
    d_left  = np.array(d_left, dtype=float)
    d_mean  = 0.5 * (d_right + d_left)
    d_binary = np.array(d_binary, dtype=float)

    print(f"\n=== {split_name} (labelled-slice eval) ===")
    print(f"Subjects: {len(d_mean)}")
    print(f"CE loss (mean over batches): {losses.mean():.4f}")

    print(f"Dice RIGHT (label 1): mean={d_right.mean():.4f}, median={np.median(d_right):.4f}, "
          f"std={d_right.std():.4f}, min={d_right.min():.4f}, max={d_right.max():.4f}")
    print(f"Dice LEFT  (label 2): mean={d_left.mean():.4f}, median={np.median(d_left):.4f}, "
          f"std={d_left.std():.4f}, min={d_left.min():.4f}, max={d_left.max():.4f}")
    print(f"Dice MEAN (avg L/R):  mean={d_mean.mean():.4f}, median={np.median(d_mean):.4f}, "
          f"std={d_mean.std():.4f}, min={d_mean.min():.4f}, max={d_mean.max():.4f}")

    print(f"Dice BINARY (cortex vs bg): mean={d_binary.mean():.4f}")

    # Save arrays
    np.save(os.path.join(DATA_ROOT, f"cortex_{split_name.lower()}_dice_right.npy"), d_right)
    np.save(os.path.join(DATA_ROOT, f"cortex_{split_name.lower()}_dice_left.npy"), d_left)
    np.save(os.path.join(DATA_ROOT, f"cortex_{split_name.lower()}_dice_mean.npy"), d_mean)
    np.save(os.path.join(DATA_ROOT, f"cortex_{split_name.lower()}_ce_loss_batches.npy"), losses)

    print(f"Saved .npy files to {DATA_ROOT}")

def main():
    model = UNet2D(in_channels=1, out_channels=N_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    print("Device:", DEVICE)
    print("Model :", MODEL_PATH)
    print("Evaluating cortex 3-class U-Net on val/test (labelled slice only)...")

    eval_split(VAL_IDS,  "Val",  model)
    eval_split(TEST_IDS, "Test", model)

if __name__ == "__main__":
    main()
