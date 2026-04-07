import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from U_NET_TRAINING import (
    DATA_ROOT, CROP_SIZE, DEVICE,
    MolliKidneySubjectDataset, SliceDataset,
    UNet2D
)

VAL_IDS  = os.path.join(DATA_ROOT, "val_ids.txt")
TEST_IDS = os.path.join(DATA_ROOT, "test_ids.txt")

MODEL_PATH = os.path.join(DATA_ROOT, "unet_molli_kidney_3class_best.pth")

BATCH_SIZE = 8
NUM_WORKERS = 4
N_CLASSES = 3  

def dice_for_class(pred, target, cls, eps=1e-6):
    """
    pred, target: torch tensors (H,W) with integer labels
    """
    pred_c = (pred == cls)
    tgt_c  = (target == cls)
    inter = (pred_c & tgt_c).sum().item()
    den = pred_c.sum().item() + tgt_c.sum().item() + eps
    # if both empty-> treat as perfect
    if den < eps * 10:
        return 1.0
    return float(2.0 * inter / den)

def eval_split(ids_path, split_name, model):
    if not os.path.exists(ids_path):
        print(f"[INFO] {split_name} IDs file not found: {ids_path} -> skipping.")
        return

    subj_ds  = MolliKidneySubjectDataset(ids_path, mode="val", crop_size=CROP_SIZE)
    slice_ds = SliceDataset(subj_ds)

    if len(slice_ds) == 0:
        print(f"[WARN] No slices found for {split_name} split.")
        return

    loader = DataLoader(slice_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))

    ce = nn.CrossEntropyLoss()

    losses = []
    d_right = []
    d_left  = []
    d_binary = []

    model.eval()
    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(DEVICE, dtype=torch.float32)  # (B,1,H,W)
            masks = masks.to(DEVICE, dtype=torch.long)    # (B,H,W), values 0/1/2

            logits = model(imgs)                          # (B,3,H,W)
            loss = ce(logits, masks)
            losses.append(loss.item())

            pred = torch.argmax(logits, dim=1)            # (B,H,W)

            # Slice-wise metrics
            pred_cpu = pred.cpu()
            mask_cpu = masks.cpu()

            for i in range(pred_cpu.shape[0]):
                # Per-kidney Dice 
                d_right.append(dice_for_class(pred_cpu[i], mask_cpu[i], 1))  
                d_left.append(dice_for_class(pred_cpu[i], mask_cpu[i], 2))   

                # binary kidney-vs-background Dice (for comparison)
                pb = (pred_cpu[i] > 0)
                mb = (mask_cpu[i] > 0)
                inter = (pb & mb).sum().item()
                den = pb.sum().item() + mb.sum().item() + 1e-6
                d_binary.append(float(2.0 * inter / den) if den > 0 else 1.0)

    losses = np.array(losses, dtype=float)
    d_right = np.array(d_right, dtype=float)
    d_left  = np.array(d_left, dtype=float)
    d_mean  = 0.5 * (d_right + d_left)
    d_binary = np.array(d_binary, dtype=float)

    print(f"\n=== {split_name} (slice-wise) ===")
    print(f"Slices: {len(d_mean)}")

    print(f"CE loss (mean over batches): {losses.mean():.4f}")

    print(f"Dice RIGHT (label 1): mean={d_right.mean():.4f}, median={np.median(d_right):.4f}, "
          f"std={d_right.std():.4f}, min={d_right.min():.4f}, max={d_right.max():.4f}")
    print(f"Dice LEFT  (label 2): mean={d_left.mean():.4f}, median={np.median(d_left):.4f}, "
          f"std={d_left.std():.4f}, min={d_left.min():.4f}, max={d_left.max():.4f}")
    print(f"Dice MEAN (avg L/R):  mean={d_mean.mean():.4f}, median={np.median(d_mean):.4f}, "
          f"std={d_mean.std():.4f}, min={d_mean.min():.4f}, max={d_mean.max():.4f}")

    print(f"Dice BINARY (kidney vs bg): mean={d_binary.mean():.4f}")

    # Save arrays 
    np.save(os.path.join(DATA_ROOT, f"{split_name.lower()}_dice_right.npy"), d_right)
    np.save(os.path.join(DATA_ROOT, f"{split_name.lower()}_dice_left.npy"), d_left)
    np.save(os.path.join(DATA_ROOT, f"{split_name.lower()}_dice_mean.npy"), d_mean)
    np.save(os.path.join(DATA_ROOT, f"{split_name.lower()}_ce_loss_batches.npy"), losses)

    print(f"Saved .npy files to {DATA_ROOT}")

def main():
    model = UNet2D(in_channels=1, out_channels=N_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    print("Device:", DEVICE)
    print("Model :", MODEL_PATH)
    print("Evaluating 3-class U-Net on val/test...")

    eval_split(VAL_IDS,  "Val",  model)
    eval_split(TEST_IDS, "Test", model)

if __name__ == "__main__":
    main()
