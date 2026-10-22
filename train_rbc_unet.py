"""
Train a lightweight U-Net for binary RBC semantic segmentation.

Architecture : encoder [32, 64, 128, 256], bottleneck 512, bilinear decoder
Loss         : 0.5 * DiceLoss + 0.5 * BCE  (pos_weight=10 for sparse RBCs)
Augmentation : albumentations 2.x
Split        : 85/15 stratified by tile to prevent data leakage
Hardware     : single GPU (CUDA 12.4 / RTX 6000 Ada)
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp


# ─── config ────────────────────────────────────────────────────────────────────
TRAINING_DATA  = Path("/home/caleb/Desktop/rbcs/training_data_tiff_combined")
CKPT_DIR       = Path("/home/caleb/Desktop/rbcs/checkpoints")
CORRECTED_LOG  = Path("/home/caleb/Desktop/rbcs/training_data_tiff_combined/corrected_patches_combined.json")
VAL_SAMPLES   = CKPT_DIR / "val_samples"

FEATURES    = (32, 64, 128, 256)
PATCH_SIZE  = 256    # crop size used during training
BATCH_SIZE  = 16
NUM_WORKERS = 4
EPOCHS      = 50
LR          = 1e-3
WEIGHT_DECAY = 1e-4
VAL_FRAC    = 0.10
SEED        = 42
POS_WEIGHT  = 1  # RBCs are ~0.3% of pixels; upweight to prevent background collapse

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]


# # ─── model ─────────────────────────────────────────────────────────────────────
# class DoubleConv(nn.Module):
#     def __init__(self, in_ch: int, out_ch: int) -> None:
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.block(x)


# class Down(nn.Module):
#     def __init__(self, in_ch: int, out_ch: int) -> None:
#         super().__init__()
#         self.pool = nn.MaxPool2d(2)
#         self.conv = DoubleConv(in_ch, out_ch)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.conv(self.pool(x))


# class Up(nn.Module):
#     def __init__(self, in_ch: int, out_ch: int) -> None:
#         super().__init__()
#         self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
#         self.conv = DoubleConv(in_ch, out_ch)

#     def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
#         x = self.up(x)
#         # Pad x to match skip's spatial size (handles odd-sized inputs)
#         dy = skip.shape[2] - x.shape[2]
#         dx = skip.shape[3] - x.shape[3]
#         x  = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
#         return self.conv(torch.cat([skip, x], dim=1))


# class RBCUNet(nn.Module):
#     def __init__(
#         self,
#         in_ch: int = 3,
#         features: Tuple[int, ...] = FEATURES,
#     ) -> None:
#         super().__init__()
#         f = features
#         self.inc   = DoubleConv(in_ch, f[0])
#         self.down1 = Down(f[0], f[1])
#         self.down2 = Down(f[1], f[2])
#         self.down3 = Down(f[2], f[3])
#         self.down4 = Down(f[3], f[3] * 2)          # bottleneck
#         self.up1   = Up(f[3] * 2 + f[3], f[3])
#         self.up2   = Up(f[3]     + f[2], f[2])
#         self.up3   = Up(f[2]     + f[1], f[1])
#         self.up4   = Up(f[1]     + f[0], f[0])
#         self.head  = nn.Conv2d(f[0], 1, kernel_size=1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x  = self.up1(x5, x4)
#         x  = self.up2(x,  x3)
#         x  = self.up3(x,  x2)
#         x  = self.up4(x,  x1)
#         return self.head(x)   # raw logits, shape (B, 1, H, W)




# ─── loss ──────────────────────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight: float = POS_WEIGHT, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
        prob   = torch.sigmoid(logits)
        flat_p = prob.view(logits.shape[0], -1)
        flat_t = targets.view(logits.shape[0], -1)
        inter  = (flat_p * flat_t).sum(dim=1)
        dice   = 1.0 - (2.0 * inter + self.smooth) / (
            flat_p.sum(dim=1) + flat_t.sum(dim=1) + self.smooth
        )
        return 0.5 * bce + 0.5 * dice.mean()


# ─── dataset ───────────────────────────────────────────────────────────────────
class RBCDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        transform: A.Compose,
    ) -> None:
        self.pairs     = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, msk_path = self.pairs[idx]
        img  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        out      = self.transform(image=img, mask=mask)
        img_t    = out["image"]                # (3, H, W) float tensor
        mask_t   = out["mask"].unsqueeze(0)    # (1, H, W) float tensor
        return img_t, mask_t


# ─── data split ────────────────────────────────────────────────────────────────
def stratified_split_by_tile(
    pairs: List[Tuple[Path, Path]],
    val_frac: float = VAL_FRAC,
    seed: int = SEED,
) -> Tuple[List, List]:
    """
    Split patch pairs 85/15 at the tile level.
    Tile key = everything before the last two _NNN segments.
    Ensures both control and AKI tiles appear in val when possible.
    """
    control_groups: dict = defaultdict(list)
    aki_groups:     dict = defaultdict(list)

    for img_p, msk_p in pairs:
        parts    = img_p.stem.rsplit("_", 2)
        tile_key = parts[0] if len(parts) == 3 else img_p.stem
        if "aki" in tile_key.lower():
            aki_groups[tile_key].append((img_p, msk_p))
        else:
            control_groups[tile_key].append((img_p, msk_p))

    def _split_group(groups: dict, n_val: int) -> Tuple[List, List]:
        keys = sorted(groups.keys())
        rng  = random.Random(seed)
        rng.shuffle(keys)
        val_keys = set(keys[:n_val])
        train = [p for k in keys if k not in val_keys for p in groups[k]]
        val   = [p for k in val_keys for p in groups[k]]
        return train, val

    n_val_ctrl = max(1, math.ceil(len(control_groups) * val_frac)) if control_groups else 0
    n_val_aki  = max(1, math.ceil(len(aki_groups)     * val_frac)) if aki_groups     else 0

    train_ctrl, val_ctrl = _split_group(control_groups, n_val_ctrl)
    train_aki,  val_aki  = _split_group(aki_groups,     n_val_aki)

    print(f"  Split — control: {len(control_groups)} tiles "
          f"({len(control_groups) - n_val_ctrl} train / {n_val_ctrl} val)")
    print(f"  Split — AKI:     {len(aki_groups)} tiles "
          f"({len(aki_groups) - n_val_aki} train / {n_val_aki} val)")

    return train_ctrl + train_aki, val_ctrl + val_aki


# ─── transforms ────────────────────────────────────────────────────────────────
def build_transforms() -> Tuple[A.Compose, A.Compose]:
    train_tf = A.Compose([
        A.RandomCrop(PATCH_SIZE, PATCH_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(
            brightness=(0.8, 1.2), contrast=(0.8, 1.2),
            saturation=(0.8, 1.2), hue=(-0.1, 0.1), p=0.5,
        ),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.3),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.5,
        ),
        A.RGBShift(
            r_shift_limit=15,
            g_shift_limit=10,
            b_shift_limit=10,
            p=0.3,
        ),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.CenterCrop(PATCH_SIZE, PATCH_SIZE),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])
    return train_tf, val_tf


# ─── training loop ─────────────────────────────────────────────────────────────
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    total = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            logits = model(imgs)
            loss   = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / max(len(loader), 1)


def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.autocast(device_type="cuda"):
                logits = model(imgs)
                loss   = criterion(logits, masks)
            total += loss.item()
    return total / max(len(loader), 1)


# ─── post-training QC ──────────────────────────────────────────────────────────
def save_val_samples(
    model: nn.Module,
    val_dataset: RBCDataset,
    out_dir: Path,
    n: int = 4,
    threshold: float = 0.5,
    device: torch.device = torch.device("cuda"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    mean = np.array(NORM_MEAN, dtype=np.float32)
    std  = np.array(NORM_STD,  dtype=np.float32)
    indices = np.random.choice(len(val_dataset), min(n, len(val_dataset)), replace=False)

    for i, idx in enumerate(indices):
        img_t, _ = val_dataset[int(idx)]
        with torch.no_grad():
            logit = model(img_t.unsqueeze(0).to(device))
            prob  = torch.sigmoid(logit).squeeze().cpu().numpy()

        pred_mask = (prob > threshold).astype(np.uint8)
        img_np    = (img_t.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        img_uint8 = (img_np * 255).astype(np.uint8)

        overlay   = img_uint8.copy()
        contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

        from PIL import Image as _PIL
        _PIL.fromarray(overlay).save(out_dir / f"val_sample_{i:02d}.png")

    print(f"Val sample overlays saved to: {out_dir}")


# ─── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Train RBC U-Net.")
    parser.add_argument(
        "--corrected_only", action="store_true",
        help=f"Use only patches listed in {CORRECTED_LOG} (human-corrected masks).",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Collect all (image, mask) pairs
    img_dir = TRAINING_DATA / "images"
    msk_dir = TRAINING_DATA / "masks"
    img_paths = sorted(img_dir.glob("*.png"))
    if not img_paths:
        raise FileNotFoundError(
            f"No patches found in {img_dir}.  Run prepare_training_data.py first."
        )
    pairs = [(p, msk_dir / p.name) for p in img_paths if (msk_dir / p.name).exists()]
    print(f"Total patches with masks: {len(pairs)}")

    if args.corrected_only:
        if not CORRECTED_LOG.exists():
            raise FileNotFoundError(f"Corrected patches log not found: {CORRECTED_LOG}")
        corrected = set(json.loads(CORRECTED_LOG.read_text()))
        pairs = [(p, m) for p, m in pairs if p.stem in corrected]
        print(f"  --corrected_only: filtered to {len(pairs)} corrected patches.")

    if not pairs:
        raise RuntimeError(
            "No patches to train on. "
            + ("corrected_patches.json stems don't match images in TRAINING_DATA — "
               "check that TRAINING_DATA points at the same dataset that was corrected."
               if args.corrected_only else
               "No paired image+mask files found in TRAINING_DATA.")
        )

    train_pairs, val_pairs = stratified_split_by_tile(pairs)
    print(f"  Train: {len(train_pairs)} patches   Val: {len(val_pairs)} patches")
    if len(val_pairs) < 5:
        print("  Warning: very few val patches — consider generating more masks.")

    train_tf, val_tf = build_transforms()
    train_ds = RBCDataset(train_pairs, train_tf)
    val_ds   = RBCDataset(val_pairs,   val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # Model
    # model = RBCUNet(in_ch=3, features=FEATURES).to(device)
    model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1).to(device)
    try:
        if int(torch.__version__.split(".")[0]) >= 2:
            model = torch.compile(model)
            print("torch.compile enabled.")
    except Exception as e:
        print(f"torch.compile skipped: {e}")

    criterion = DiceBCELoss(pos_weight=POS_WEIGHT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt     = CKPT_DIR / "best_rbc_unet.pth"

    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'LR':>12}")
    print("─" * 50)

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        vl_loss = val_epoch(model,   val_loader,   criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"{epoch:>6}  {tr_loss:>12.6f}  {vl_loss:>10.6f}  {current_lr:>12.2e}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            # Strip _orig_mod prefix so the checkpoint loads into a non-compiled model
            raw_state = {
                k.replace("_orig_mod.", "").replace("module.", ""): v
                for k, v in model.state_dict().items()
            }
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    raw_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss":            best_val_loss,
                "config": {
                    "encoder_name":    "efficientnet-b0",
                    "in_channels":     3,
                    "classes":         1,
                    "patch_size":      PATCH_SIZE,
                    "normalize_mean":  NORM_MEAN,
                    "normalize_std":   NORM_STD,
                },
            }, best_ckpt)

    print(f"\nBest val loss: {best_val_loss:.6f}  →  {best_ckpt}")

    # Load best weights back for val sample generation
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    # eval_model = RBCUNet(in_ch=3, features=FEATURES).to(device)
    eval_model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1).to(device)
    eval_model.load_state_dict(ckpt["model_state_dict"])

    save_val_samples(eval_model, val_ds, VAL_SAMPLES, n=4, device=device)


if __name__ == "__main__":
    main()
