import os
import random
import csv
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_ROOT  = r"H:/Data/Kidney_Segmentation"
MOLLI_DIR  = os.path.join(DATA_ROOT, "Raw_MOLLI")
MASK_DIR   = os.path.join(DATA_ROOT, "Masks_cortex")

OUT_DIR    = os.path.join(DATA_ROOT, "Cortex_UNet_3class")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_BEST = os.path.join(OUT_DIR, "unet_molli_cortex_3class_best.pth")
MODEL_LAST = os.path.join(OUT_DIR, "unet_molli_cortex_3class_last.pth")

IMAGE_SUFFIX = "_MOLLI_Native.nii.gz"
MASK_SUFFIX  = "_MOLLI_Native_cortex_mask.nii.gz"

CROP_SIZE = (256, 256)
N_CLASSES = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EPOCHS = 120
BATCH_SIZE = 8
LR = 1e-3
NUM_WORKERS = 0 if os.name == "nt" else 4

SLICE_CSV = os.path.join(DATA_ROOT, "cortex_labelled_slice_index.csv")

def load_slice_map(csv_path: str) -> dict:
    m = {}
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sid = row["subject_id"]
            z = int(row["z"])
            m[sid] = z
    return m

if not os.path.exists(SLICE_CSV):
    raise FileNotFoundError(f"Missing slice index CSV: {SLICE_CSV}")

SLICE_MAP = load_slice_map(SLICE_CSV)
print("Loaded slice indices:", len(SLICE_MAP))

# model
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
        return self.outc(x)


# preprocess

def normalize_volume(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    p1 = np.percentile(vol, 1)
    p99 = np.percentile(vol, 99)
    vol = np.clip(vol, p1, p99)
    vol = (vol - p1) / (p99 - p1 + 1e-8)
    return vol

def random_augment(img: torch.Tensor, mask: torch.Tensor):
    # flips
    if random.random() < 0.5:
        img = torch.flip(img, dims=[2])
        mask = torch.flip(mask, dims=[1])
    if random.random() < 0.5:
        img = torch.flip(img, dims=[1])
        mask = torch.flip(mask, dims=[0])

    # small rotation
    if random.random() < 0.5:
        angle = random.uniform(-12, 12)
        theta = torch.tensor([
            [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0.0],
            [np.sin(np.deg2rad(angle)),  np.cos(np.deg2rad(angle)), 0.0]
        ], dtype=torch.float32, device=img.device).unsqueeze(0)

        H, W = img.shape[1], img.shape[2]
        grid = F.affine_grid(theta, size=(1, 1, H, W), align_corners=False)
        img = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear",
                            padding_mode="border", align_corners=False).squeeze(0)

        mask_f = mask.unsqueeze(0).unsqueeze(0).float()
        mask_f = F.grid_sample(mask_f, grid, mode="nearest",
                               padding_mode="zeros", align_corners=False)
        mask = mask_f.squeeze(0).squeeze(0).long()

    # intensity jitter
    if random.random() < 0.5:
        img = img * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)
        img = torch.clamp(img, 0.0, 1.0)

    return img, mask

# dataset

class CortexSliceDataset(Dataset):
    def __init__(self, ids, augment=False):
        self.ids = ids
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        img_path  = os.path.join(MOLLI_DIR, f"{sid}{IMAGE_SUFFIX}")
        mask_path = os.path.join(MASK_DIR,  f"{sid}{MASK_SUFFIX}")

        if sid not in SLICE_MAP:
            raise KeyError(f"{sid} not found in slice map CSV: {SLICE_CSV}")

        img_nii  = nib.load(img_path)
        mask_nii = nib.load(mask_path)

        img  = img_nii.get_fdata().astype(np.float32)
        mask = mask_nii.get_fdata().astype(np.int64)

        if img.ndim != 3 or mask.ndim != 3:
            raise ValueError(f"{sid}: expected 3D image+mask, got img={img.shape}, mask={mask.shape}")

        img_n = normalize_volume(img)

        # Use the labelled slice index 
        z = int(SLICE_MAP[sid])
        if z < 0 or z >= img.shape[2] or z >= mask.shape[2]:
            raise ValueError(f"{sid}: z={z} out of range. img={img.shape}, mask={mask.shape}")

        img2d  = img_n[..., z]
        mask2d = mask[..., z]

        img_t  = torch.from_numpy(img2d)[None, ...].float()  # (1,H,W)
        mask_t = torch.from_numpy(mask2d).long()             # (H,W)

        img_t = F.interpolate(img_t.unsqueeze(0), size=CROP_SIZE, mode="bilinear", align_corners=False).squeeze(0)
        mask_f = mask_t.unsqueeze(0).unsqueeze(0).float()
        mask_f = F.interpolate(mask_f, size=CROP_SIZE, mode="nearest")
        mask_t = mask_f.squeeze(0).squeeze(0).long()

        if self.augment:
            img_t, mask_t = random_augment(img_t, mask_t)

        return img_t, mask_t, sid

# metrics
def dice_per_class_from_logits(logits, target, num_classes=3, eps=1e-6):
    pred = torch.argmax(logits, dim=1)
    dices = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        d = (2 * inter + eps) / (denom + eps)
        dices.append(d.mean().item())
    return dices

class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes=3, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dices = []
        for c in range(1, self.num_classes):
            p = probs[:, c]
            t = target_oh[:, c]
            inter = (p * t).sum(dim=(1, 2))
            denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice = (2 * inter + self.eps) / (denom + self.eps)
            dices.append(dice.mean())
        return 1.0 - torch.stack(dices).mean()

def estimate_class_weights(train_ids, max_subjects=200):
    ids = train_ids[:]
    random.shuffle(ids)
    ids = ids[:min(max_subjects, len(ids))]
    counts = np.zeros(N_CLASSES, dtype=np.float64)

    for sid in ids:
        mask_path = os.path.join(MASK_DIR, f"{sid}{MASK_SUFFIX}")
        m = nib.load(mask_path).get_fdata().astype(np.int64)

        if sid not in SLICE_MAP:
            continue
        z = int(SLICE_MAP[sid])
        if z < 0 or z >= m.shape[2]:
            continue

        m2 = m[..., z]
        for c in range(N_CLASSES):
            counts[c] += np.sum(m2 == c)

    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / inv.mean()
    w[0] = min(w[0], 0.5)
    return torch.tensor(w, dtype=torch.float32)

# helpers

def list_ids_from_masks(mask_dir, mask_suffix):
    ids = []
    for fn in os.listdir(mask_dir):
        if fn.endswith(mask_suffix):
            ids.append(fn.replace(mask_suffix, ""))
    return sorted(list(set(ids)))

def split_ids(ids, train=0.7, val=0.15, test=0.15):
    ids = ids[:]
    random.shuffle(ids)
    n = len(ids)
    n_train = int(round(train * n))
    n_val   = int(round(val * n))
    train_ids = ids[:n_train]
    val_ids   = ids[n_train:n_train + n_val]
    test_ids  = ids[n_train + n_val:]
    return train_ids, val_ids, test_ids

def save_ids(path, ids):
    with open(path, "w") as f:
        for s in ids:
            f.write(s + "\n")


# training

def run_epoch(loader, model, ce_loss, dice_loss, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    dices_accum = np.zeros(N_CLASSES, dtype=np.float64)
    n_batches = 0

    for img, mask, _sid in loader:
        img = img.to(DEVICE)
        mask = mask.to(DEVICE)

        logits = model(img)

        loss_ce = ce_loss(logits, mask)
        loss_d  = dice_loss(logits, mask)
        loss = 0.5 * loss_ce + 0.5 * loss_d

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        dices = dice_per_class_from_logits(logits.detach(), mask.detach(), num_classes=N_CLASSES)
        dices_accum += np.array(dices)
        n_batches += 1

    return total_loss / max(n_batches, 1), dices_accum / max(n_batches, 1)

def main():
    all_ids = list_ids_from_masks(MASK_DIR, MASK_SUFFIX)

    # Keep only IDs that are in the slice map
    all_ids = [sid for sid in all_ids if sid in SLICE_MAP]

    if len(all_ids) < 10:
        raise RuntimeError(f"Found only {len(all_ids)} valid masks/IDs. Check paths/suffix/CSV.")

    train_ids, val_ids, test_ids = split_ids(all_ids, 0.7, 0.15, 0.15)

    save_ids(os.path.join(OUT_DIR, "train_ids_CORTEX.txt"), train_ids)
    save_ids(os.path.join(OUT_DIR, "val_ids_CORTEX.txt"),   val_ids)
    save_ids(os.path.join(OUT_DIR, "test_ids_CORTEX.txt"),  test_ids)

    print("Device:", DEVICE)
    print("Total labelled:", len(all_ids))
    print("Train/Val/Test:", len(train_ids), len(val_ids), len(test_ids))

    ds_train = CortexSliceDataset(train_ids, augment=True)
    ds_val   = CortexSliceDataset(val_ids, augment=False)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    dl_val   = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

    model = UNet2D(in_channels=1, out_channels=N_CLASSES).to(DEVICE)

    class_w = estimate_class_weights(train_ids).to(DEVICE)
    print("Estimated class weights [bg, right, left]:", class_w.detach().cpu().numpy().round(3).tolist())

    ce_loss = nn.CrossEntropyLoss(weight=class_w)
    dice_loss = SoftDiceLoss(num_classes=N_CLASSES)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = -1.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_d = run_epoch(dl_train, model, ce_loss, dice_loss, optimizer=optimizer)
        va_loss, va_d = run_epoch(dl_val,   model, ce_loss, dice_loss, optimizer=None)

        tr_bg, tr_r, tr_l = tr_d.tolist()
        va_bg, va_r, va_l = va_d.tolist()
        va_mean_cortex = (va_r + va_l) / 2.0

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss {tr_loss:.4f} | Val loss {va_loss:.4f} | "
            f"Val Dice R {va_r:.3f} L {va_l:.3f} Mean {va_mean_cortex:.3f}"
        )

        if va_mean_cortex > best_val:
            best_val = va_mean_cortex
            torch.save(model.state_dict(), MODEL_BEST)
            print(f"  -> saved BEST: {MODEL_BEST}")

        torch.save(model.state_dict(), MODEL_LAST)

    print("Training done.")
    print("Best Val mean cortex Dice:", best_val)

if __name__ == "__main__":
    main()
