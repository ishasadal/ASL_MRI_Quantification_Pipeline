import os
import random
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

DATA_ROOT  = r"H:/Data/Kidney_Segmentation"
IMAGES_DIR = r"H:/Data/Kidney_Segmentation/Images"
MASKS_DIR  = r"H:/Data/Kidney_Segmentation/Masks"

TRAIN_IDS = os.path.join(DATA_ROOT, "train_ids.txt")
VAL_IDS   = os.path.join(DATA_ROOT, "val_ids.txt")
SAVE_BEST_PATH = os.path.join(DATA_ROOT, "unet_molli_kidney_3class_best.pth")


# training parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP_SIZE  = (256, 256)
BATCH_SIZE = 8
EPOCHS   = 30
LR         = 1e-4
SEED = 42
NUM_WORKERS = 4

IMAGE_SUFFIX = "_MOLLI_Native.nii.gz"
MASK_SUFFIX  = "_MOLLI_Native_mask.nii.gz"
N_CLASSES = 3  # 0=background/1=left/2=right


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

### dataset
class MolliKidneySubjectDataset(Dataset):
    def __init__(self, ids_file, mode="train", crop_size=(256, 256)):
        self.mode = mode
        self.crop_size = crop_size

        with open(ids_file, "r") as f:
            ids = [line.strip() for line in f if line.strip()]

        self.ids = []
        missing = 0
        for subj_id in ids:
            img_path  = os.path.join(IMAGES_DIR, f"{subj_id}{IMAGE_SUFFIX}")
            mask_path = os.path.join(MASKS_DIR,  f"{subj_id}{MASK_SUFFIX}")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.ids.append(subj_id)
            else:
                missing += 1

        print(f"[{mode}] Loaded {len(self.ids)} subjects (skipped missing: {missing})")
        if len(self.ids) == 0:
            raise RuntimeError(f"[{mode}] No subjects found. Check dirs/suffixes.")

    def _load_nifti(self, path):
        return nib.load(path).get_fdata()

    def _normalize_volume(self, vol):
        vol = vol.astype(np.float32)
        p1 = np.percentile(vol, 1)
        p99 = np.percentile(vol, 99)
        vol = np.clip(vol, p1, p99)
        return (vol - p1) / (p99 - p1 + 1e-8)

    def _resize_img(self, slice2d, size):
        t = torch.from_numpy(slice2d.astype(np.float32))[None, None, ...]
        t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
        return t[0, 0].numpy()

    def _resize_mask(self, slice2d, size):
        # nearest-neighbor to preserve labels
        t = torch.from_numpy(slice2d.astype(np.int64))[None, None, ...].float()
        t = F.interpolate(t, size=size, mode="nearest")
        return t[0, 0].numpy().astype(np.int64)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        subj_id = self.ids[idx]
        img_path  = os.path.join(IMAGES_DIR, f"{subj_id}{IMAGE_SUFFIX}")
        mask_path = os.path.join(MASKS_DIR,  f"{subj_id}{MASK_SUFFIX}")

        img_3d  = self._normalize_volume(self._load_nifti(img_path))
        mask_3d = self._load_nifti(mask_path).astype(np.int64)
        mask_3d = np.clip(mask_3d, 0, 2)

        _, _, Z = img_3d.shape
        imgs, msks = [], []

        for z in range(Z):
            img_slice  = img_3d[..., z]
            mask_slice = mask_3d[..., z]

            # keep only slices with kidney present
            if np.max(mask_slice) == 0:
                continue

            img_res  = self._resize_img(img_slice, self.crop_size)
            mask_res = self._resize_mask(mask_slice, self.crop_size)

            # augmentation
            if self.mode == "train":
                if random.random() < 0.5:
                    img_res  = np.fliplr(img_res)
                    mask_res = np.fliplr(mask_res)
                if random.random() < 0.5:
                    img_res  = np.flipud(img_res)
                    mask_res = np.flipud(mask_res)

            imgs.append(torch.from_numpy(img_res.copy()).unsqueeze(0))   
            msks.append(torch.from_numpy(mask_res.copy()).long())      

        if len(imgs) == 0:
            img_res  = np.zeros(self.crop_size, dtype=np.float32)
            mask_res = np.zeros(self.crop_size, dtype=np.int64)
            imgs = [torch.from_numpy(img_res)[None, ...]]
            msks = [torch.from_numpy(mask_res).long()]

        images = torch.stack(imgs, dim=0)  
        masks  = torch.stack(msks, dim=0)   
        return images, masks, subj_id

class SliceDataset(Dataset):
    """Flatten subject dataset into slice samples"""
    def __init__(self, subj_dataset):
        self.samples = []
        for images, masks, _ in subj_dataset:
            for i in range(images.shape[0]):
                self.samples.append((images[i], masks[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


### model (2D U-Net, 3-class output)
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


### metrics: Dice for left/right + mean kidney dice
def dice_for_class(pred, target, cls, eps=1e-6):
    pred_c = (pred == cls)
    tgt_c  = (target == cls)
    inter = (pred_c & tgt_c).sum().item()
    den = pred_c.sum().item() + tgt_c.sum().item() + eps
    return (2.0 * inter) / den

def evaluate(model, loader):
    model.eval()
    dL, dR = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(DEVICE, dtype=torch.float32)
            masks = masks.to(DEVICE, dtype=torch.long)

            logits = model(imgs)
            pred = torch.argmax(logits, dim=1)  # (B,H,W)

            dL.append(dice_for_class(pred.cpu(), masks.cpu(), 1))
            dR.append(dice_for_class(pred.cpu(), masks.cpu(), 2))

    dL = float(np.mean(dL)) if len(dL) else 0.0
    dR = float(np.mean(dR)) if len(dR) else 0.0
    dMean = 0.5 * (dL + dR)
    return dL, dR, dMean


### TRAINING
def main():
    seed_everything(SEED)

    print("Device:", DEVICE)
    print("Images:", IMAGES_DIR)
    print("Masks :", MASKS_DIR)
    print("Save best:", SAVE_BEST_PATH)

    train_subj = MolliKidneySubjectDataset(TRAIN_IDS, mode="train", crop_size=CROP_SIZE)
    val_subj   = MolliKidneySubjectDataset(VAL_IDS,   mode="val",   crop_size=CROP_SIZE)

    train_ds = SliceDataset(train_subj)
    val_ds   = SliceDataset(val_subj)

    print(f"Train slices: {len(train_ds)}, Val slices: {len(val_ds)}")

    pin = (DEVICE == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    model = UNet2D(in_channels=1, out_channels=N_CLASSES).to(DEVICE)

    # 3-class loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_mean_dice = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for imgs, masks in train_loader:
            imgs  = imgs.to(DEVICE, dtype=torch.float32)
            masks = masks.to(DEVICE, dtype=torch.long)   
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)                        
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))

        dL, dR, dMean = evaluate(model, val_loader)

        print(f"Epoch {epoch:03d} | Train CE loss: {train_loss:.4f} | "
              f"Val Dice L: {dL:.4f} R: {dR:.4f} Mean: {dMean:.4f}")

        if dMean > best_mean_dice:
            best_mean_dice = dMean
            torch.save(model.state_dict(), SAVE_BEST_PATH)
            print("New best model saved:", SAVE_BEST_PATH)

    print("Training finished. Best mean kidney Dice:", best_mean_dice)

if __name__ == "__main__":
    main()
