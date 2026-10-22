"""
Full-slide RBC segmentation on a flat Visium HD TIFF.

Tiles the TIFF on-the-fly at the training MPP, runs the U-Net, stitches
predictions back at native TIFF resolution, and writes a pyramidal TIFF
+ GeoJSON suitable for QuPath overlay.

Accumulation arrays are memory-mapped to disk to avoid OOM.

Usage
-----
python predict_tif.py \\
    --slide         "/path/to/Ctrl_1A2.tif" \\
    --scalefactors  "/path/to/square_002um/spatial/scalefactors_json.json" \\
    --ckpt          "/path/to/best_rbc_unet.pth" \\
    --out            /path/to/output_mask.tif \\
    --threshold      0.5 \\
    --tile_size      2048 \\
    --overlap        256
"""

import argparse
import gc
import json
import os
import tempfile
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import segmentation_models_pytorch as smp


# sample
sample_id = "Ctrl_1A4" # Ctrl_1A2 Ctrl_1A4 IR_2A2 IR_2A4


# ─── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SLIDE      = Path(
    f"/home/caleb/Desktop/3D/data/kidney_rabb_visiumhd/VisiumHD/{sample_id}/outs/spatial/{sample_id}.tif"
)

DEFAULT_SCALEFACTORS = Path(
    f"/home/caleb/Desktop/3D/data/kidney_rabb_visiumhd/VisiumHD/{sample_id}/outs/binned_outputs/square_002um/spatial/scalefactors_json.json",
)

DEFAULT_CKPT       = Path("/home/caleb/Desktop/rbcs/checkpoints/best_rbc_unet.pth")
DEFAULT_OUT        = Path(f"/home/caleb/Desktop/rbcs/unet_predictions_tiff/{sample_id}_rbc_mask.tif")
DEFAULT_THRESHOLD  = 0.2
DEFAULT_TILE_SIZE  = 2048
DEFAULT_OVERLAP    = 256
TARGET_MPP         = 0.5
N_PYRAMID_LEVELS   = 1
MIN_RBC_PX         = 30  

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]


# # ─── model (must match train_rbc_unet.py) ──────────────────────────────────────
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
#         x  = self.up(x)
#         dy = skip.shape[2] - x.shape[2]
#         dx = skip.shape[3] - x.shape[3]
#         x  = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
#         return self.conv(torch.cat([skip, x], dim=1))


# class RBCUNet(nn.Module):
#     def __init__(
#         self,
#         in_ch: int = 3,
#         features: Tuple[int, ...] = (32, 64, 128, 256),
#     ) -> None:
#         super().__init__()
#         f = features
#         self.inc   = DoubleConv(in_ch, f[0])
#         self.down1 = Down(f[0], f[1])
#         self.down2 = Down(f[1], f[2])
#         self.down3 = Down(f[2], f[3])
#         self.down4 = Down(f[3], f[3] * 2)
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
#         return self.head(x)


# ─── post-processing ──────────────────────────────────────────────────────────
def filter_small_rbcs(binary: np.ndarray, min_px: int) -> np.ndarray:
    """Remove connected components smaller than min_px pixels from a binary mask."""
    if min_px <= 0:
        return binary
    binary_01 = (binary > 0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_01, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    big   = np.where(areas >= min_px)[0]
    big   = big[big > 0]                          # exclude background label 0
    keep  = np.isin(labels, big).astype(np.uint8)
    return (keep * (255 if binary.max() > 1 else 1)).astype(binary.dtype)


# ─── model loading ─────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt   = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = smp.Unet(
    encoder_name=ckpt["config"]["encoder_name"],
    encoder_weights=None,  # loading from checkpoint, not pretrained
    in_channels=ckpt["config"]["in_channels"],
    classes=ckpt["config"]["classes"]).to(device)
    state = {
        k.replace("_orig_mod.", "").replace("module.", ""): v
        for k, v in ckpt["model_state_dict"].items()
    }
    model.load_state_dict(state)
    model.eval()
    return model


# ─── TIFF helpers ─────────────────────────────────────────────────────────────
def load_mpp(scalefactors_path: Path) -> float:
    """Read microns_per_pixel from Visium HD scalefactors_json.json."""
    data = json.loads(scalefactors_path.read_text())
    mpp  = float(data["microns_per_pixel"])
    print(f"  Source MPP: {mpp:.6f} µm/px")
    return mpp


def load_tiff_array(tiff_path: Path) -> np.ndarray:
    """Memory-map the TIFF (falls back to full load if mmap fails)."""
    try:
        arr = tifffile.memmap(str(tiff_path), mode="r")
        print(f"  Memory-mapped TIFF: {arr.shape}, dtype={arr.dtype}")
    except Exception:
        arr = tifffile.imread(str(tiff_path))
        print(f"  Loaded TIFF into RAM: {arr.shape}, dtype={arr.dtype}")
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def read_tile_tiff(
    arr: np.ndarray,
    x0: int,
    y0: int,
    tile_size_native: int,
    tile_size_out: int,
) -> np.ndarray:
    """Crop a native tile and resize to tile_size_out × tile_size_out (uint8 RGB)."""
    h, w = arr.shape[:2]
    x_end = min(x0 + tile_size_native, w)
    y_end = min(y0 + tile_size_native, h)
    crop  = arr[y0:y_end, x0:x_end]
    if crop.shape[0] < tile_size_native or crop.shape[1] < tile_size_native:
        pad = np.full((tile_size_native, tile_size_native, 3), 255, dtype=np.uint8)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        crop = pad
    img = Image.fromarray(crop)
    if img.size != (tile_size_out, tile_size_out):
        img = img.resize((tile_size_out, tile_size_out), Image.LANCZOS)
    return np.asarray(img)


def tile_grid(
    slide_w: int,
    slide_h: int,
    tile_size_l0: int,
    overlap_l0: int,
) -> Iterator[Tuple[int, int]]:
    step = tile_size_l0 - overlap_l0
    for y in range(0, slide_h, step):
        for x in range(0, slide_w, step):
            yield x, y


def has_tissue(tile_rgb: np.ndarray, bg_thresh: int = 220) -> bool:
    """True if the tile contains tissue (not all background)."""
    return float(tile_rgb.reshape(-1, 3).mean(axis=0).min()) <= bg_thresh


# ─── sliding-window inference ──────────────────────────────────────────────────
def sliding_window_predict(
    model: nn.Module,
    tile_rgb: np.ndarray,
    patch_size: int = 512,
    overlap: int = 128,
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    """Return float32 prob map (H, W) for the tile."""
    H, W = tile_rgb.shape[:2]
    step = patch_size - overlap

    pad_h = max(0, patch_size - H)
    pad_w = max(0, patch_size - W)
    if pad_h > 0 or pad_w > 0:
        tile_padded = np.pad(tile_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        tile_padded = tile_rgb
    pH, pW = tile_padded.shape[:2]

    extra_h = (step - (pH - patch_size) % step) % step
    extra_w = (step - (pW - patch_size) % step) % step
    if extra_h > 0 or extra_w > 0:
        tile_padded = np.pad(
            tile_padded, ((0, extra_h), (0, extra_w), (0, 0)), mode="reflect"
        )
    pH, pW = tile_padded.shape[:2]

    prob_acc   = np.zeros((pH, pW), dtype=np.float32)
    weight_acc = np.zeros((pH, pW), dtype=np.float32)
    ramp       = np.hanning(patch_size).astype(np.float32)
    weight_2d  = np.outer(ramp, ramp)

    mean = np.array(NORM_MEAN, dtype=np.float32)
    std  = np.array(NORM_STD,  dtype=np.float32)

    with torch.no_grad():
        for y in range(0, pH - patch_size + 1, step):
            for x in range(0, pW - patch_size + 1, step):
                patch   = tile_padded[y : y + patch_size, x : x + patch_size]
                patch_f = (patch.astype(np.float32) / 255.0 - mean) / std
                tensor  = torch.from_numpy(
                    patch_f.transpose(2, 0, 1)
                ).unsqueeze(0).to(device)
                logit = model(tensor)
                prob = torch.sigmoid(logit).squeeze().cpu().numpy()
                prob_acc  [y : y + patch_size, x : x + patch_size] += prob * weight_2d
                weight_acc[y : y + patch_size, x : x + patch_size] += weight_2d

    prob_map = np.where(weight_acc > 0, prob_acc / weight_acc, 0.0)
    return prob_map[:H, :W]


def sliding_window_predict_tta(
    model: nn.Module,
    tile_rgb: np.ndarray,
    patch_size: int = 512,
    overlap: int = 128,
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    """Sliding window inference with test-time augmentation (8 orientations)."""

    augmentations = [
        lambda x: x,                                           # original
        lambda x: np.fliplr(x),                               # flip horizontal
        lambda x: np.flipud(x),                               # flip vertical
        lambda x: np.rot90(x, 1),                             # rotate 90
        lambda x: np.rot90(x, 2),                             # rotate 180
        lambda x: np.rot90(x, 3),                             # rotate 270
        lambda x: np.fliplr(np.rot90(x, 1)),                  # flip + rot90
        lambda x: np.flipud(np.rot90(x, 1)),                  # flip + rot90
    ]
    inverses = [
        lambda x: x,
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: np.rot90(x, -1),
        lambda x: np.rot90(x, -2),
        lambda x: np.rot90(x, -3),
        lambda x: np.rot90(np.fliplr(x), -1),
        lambda x: np.rot90(np.flipud(x), -1),
    ]

    H, W = tile_rgb.shape[:2]
    prob_sum = np.zeros((H, W), dtype=np.float32)

    for aug, inv in zip(augmentations, inverses):
        aug_tile = aug(tile_rgb)
        # pad back to original shape if rotation changed dims
        prob = sliding_window_predict(model, aug_tile, patch_size, overlap, device)
        prob_sum += inv(prob)

    return prob_sum / len(augmentations)


# ─── GeoJSON export ───────────────────────────────────────────────────────────
from shapely.geometry import Polygon as ShapePolygon

def write_geojson(
    binary_full: np.ndarray,
    out_path: Path,
    target_ds: float,
) -> None:
    print("  Building GeoJSON contours ...")
    contours, _ = cv2.findContours(
        (binary_full > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    features = []
    skipped = 0
    for i, contour in enumerate(contours):
        try:
            if len(contour) < 4:
                skipped += 1
                continue

            # Simplify — same epsilon as tubule pipeline
            contour = cv2.approxPolyDP(contour, 0.5, closed=True)
            if len(contour) < 3:
                skipped += 1
                continue

            # Scale to level-0 coords, round to 1dp
            pts = contour[:, 0, :].astype(np.float64) * target_ds
            raw_coords = [[round(float(x), 1), round(float(y), 1)] for x, y in pts]

            if len(raw_coords) < 3:
                skipped += 1
                continue

            # Fix geometry with Shapely buffer(0) — same trick as tubule pipeline
            shape = ShapePolygon(raw_coords).buffer(0)
            if shape.is_empty or not shape.is_valid:
                skipped += 1
                continue
            if shape.geom_type == "MultiPolygon":
                shape = max(shape.geoms, key=lambda g: g.area)
            if shape.is_empty:
                skipped += 1
                continue

            # Cap at 500 coords for QuPath compatibility
            tol = 0.5
            while len(shape.exterior.coords) > 500 and tol < 50:
                shape = shape.simplify(tol, preserve_topology=True)
                if shape.geom_type == "MultiPolygon":
                    shape = max(shape.geoms, key=lambda g: g.area)
                tol *= 2
            if shape.is_empty:
                skipped += 1
                continue

            final_coords = [list(pt) for pt in shape.exterior.coords]

            features.append({
                "type": "Feature",
                "id": "PathAnnotationObject",        # exact string QuPath expects
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [final_coords],
                },
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": "RBC",
                        "colorRGB": -65536,           # packed ARGB int for red
                    },
                    "isLocked": True,
                    "measurements": [],               # empty array, not {}
                },
            })

        except Exception:
            skipped += 1
            continue

    # Stream write — same as save_geojson_streaming in tubule pipeline
    geojson_path = out_path.with_suffix(".geojson")
    with open(geojson_path, "w", encoding="utf-8") as f:
        f.write('{"type":"FeatureCollection","features":[\n')
        for i, feat in enumerate(features):
            separator = ",\n" if i > 0 else ""
            f.write(separator + json.dumps(feat))
        f.write("\n]}")

    print(f"  GeoJSON — {len(features):,} annotations ({skipped} skipped) → {geojson_path}")


    
# ─── pyramidal TIFF writing ────────────────────────────────────────────────────
def write_pyramidal_tiff(
    prob_acc: np.ndarray,
    weight_acc: np.ndarray,
    out_path: Path,
    threshold: float,
    out_h: int,
    out_w: int,
    target_ds: float = 1.0,
    slide_w0: int = 0,    # add
    slide_h0: int = 0,    # add
    min_rbc_px: int = MIN_RBC_PX,
    n_levels: int = N_PYRAMID_LEVELS,
    chunk_rows: int = 2048,
) -> None:
    """Normalize accumulator, threshold, filter small objects, write BigTIFF + GeoJSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("  Normalizing probability accumulator ...")
    binary_full = np.zeros((out_h, out_w), dtype=np.uint8)
    for y in range(0, out_h, chunk_rows):
        y_end = min(y + chunk_rows, out_h)
        p     = prob_acc  [y:y_end]
        w     = weight_acc[y:y_end]
        prob_norm = np.where(w > 0, p / w, 0.0)
        binary_full[y:y_end] = (prob_norm > threshold).astype(np.uint8) * 255

    print(f"  Filtering components < {min_rbc_px} px ...")
    binary_full = filter_small_rbcs(binary_full, min_rbc_px)

    print(f"  Writing pyramidal TIFF ({n_levels} levels) → {out_path}")
    with tifffile.TiffWriter(str(out_path), bigtiff=True) as tif:
        tif.write(
            binary_full,
            tile=(256, 256),
            compression="zlib",
            compressionargs={"level": 6},
            photometric="minisblack",
            subifds=n_levels - 1,
            metadata={"axes": "YX"},
        )
        current = binary_full
        for lvl in range(1, n_levels):
            scale = 2 ** lvl
            h_l   = out_h // scale
            w_l   = out_w // scale
            down  = cv2.resize(current, (w_l, h_l), interpolation=cv2.INTER_AREA)
            tif.write(
                down,
                tile=(256, 256),
                compression="zlib",
                photometric="minisblack",
                subfiletype=1,   # FILETYPE_REDUCEDIMAGE
            )
            current = down
            del down
            gc.collect()

    write_geojson(binary_full, out_path, target_ds)

    del binary_full
    gc.collect()
    # Save metadata sidecar so QuPath script can use exact scale
    meta = {
        "target_ds": float(target_ds),
        "out_w": int(out_w),
        "out_h": int(out_h),
        "slide_w0": int(slide_w0),
        "slide_h0": int(slide_h0),
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved → {meta_path}")


# ─── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Full-slide RBC prediction on a Visium HD TIFF.")
    parser.add_argument("--slide",        type=Path,  default=DEFAULT_SLIDE)
    parser.add_argument("--scalefactors", type=Path,  default=DEFAULT_SCALEFACTORS,
                        help="Path to scalefactors_json.json (Visium HD binned output).")
    parser.add_argument("--ckpt",         type=Path,  default=DEFAULT_CKPT)
    parser.add_argument("--out",          type=Path,  default=DEFAULT_OUT)
    parser.add_argument("--threshold",    type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--tile_size",    type=int,   default=DEFAULT_TILE_SIZE)
    parser.add_argument("--overlap",      type=int,   default=DEFAULT_OVERLAP)
    parser.add_argument("--min_rbc_px",   type=int,   default=MIN_RBC_PX,
                        help="Remove predicted RBC regions smaller than this many pixels.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device       : {device}")
    print(f"Slide        : {args.slide}")
    print(f"Scalefactors : {args.scalefactors}")
    print(f"Ckpt         : {args.ckpt}\n")

    assert args.slide.exists(),        f"TIFF not found: {args.slide}"
    assert args.scalefactors.exists(), f"scalefactors_json not found: {args.scalefactors}"

    source_mpp = load_mpp(args.scalefactors)
    # Scale factor consistent with extract_tiles_tiff.py
    scale_factor      = TARGET_MPP / source_mpp
    tile_size_native  = int(round(args.tile_size / scale_factor))
    overlap_native    = int(round(args.overlap   / scale_factor))

    print(f"  scale_factor={scale_factor:.4f}  "
          f"tile_native={tile_size_native}px → out={args.tile_size}px")

    arr = load_tiff_array(args.slide)
    tiff_h, tiff_w = arr.shape[:2]
    print(f"  TIFF dims: {tiff_w}×{tiff_h}")

    # Output mask at native TIFF resolution (1:1 with input → direct QuPath overlay)
    out_w = tiff_w
    out_h = tiff_h
    print(f"  Output mask : {out_w}×{out_h}  (native TIFF resolution)")

    total_tiles = sum(1 for _ in tile_grid(tiff_w, tiff_h, tile_size_native, overlap_native))
    print(f"  Total tiles : {total_tiles}\n")

    model = load_model(args.ckpt, device)

    # Hanning weight in native tile space for smooth stitching
    tile_ramp   = np.hanning(tile_size_native).astype(np.float32)
    tile_weight = np.outer(tile_ramp, tile_ramp)

    tmp_dir = Path(tempfile.mkdtemp(prefix="rbc_tif_"))
    print(f"Temp dir for memmaps: {tmp_dir}")
    prob_acc   = np.memmap(tmp_dir / "prob_acc.dat",   dtype=np.float32,
                           mode="w+", shape=(out_h, out_w))
    weight_acc = np.memmap(tmp_dir / "weight_acc.dat", dtype=np.float32,
                           mode="w+", shape=(out_h, out_w))

    rbc_px   = 0
    total_px = 0
    n_done   = 0

    for x0, y0 in tile_grid(tiff_w, tiff_h, tile_size_native, overlap_native):
        n_done += 1

        tile_rgb = read_tile_tiff(arr, x0, y0, tile_size_native, args.tile_size)
        if not has_tissue(tile_rgb):
            print(f"\r  [{n_done}/{total_tiles}] skip (background)", end="", flush=True)
            continue

        prob_tile = sliding_window_predict(model, tile_rgb, device=device)
        # prob_tile = sliding_window_predict_tta(model, tile_rgb, device=device)

        # Resize prob map back to native tile size before accumulating
        ph = min(tile_size_native, out_h - y0)
        pw = min(tile_size_native, out_w - x0)
        if ph <= 0 or pw <= 0:
            continue

        prob_native = cv2.resize(prob_tile, (tile_size_native, tile_size_native),
                                 interpolation=cv2.INTER_LINEAR)

        tw = tile_weight[:ph, :pw]
        prob_acc  [y0 : y0 + ph, x0 : x0 + pw] += prob_native[:ph, :pw] * tw
        weight_acc[y0 : y0 + ph, x0 : x0 + pw] += tw

        binary_tile = (prob_native[:ph, :pw] > args.threshold).astype(np.uint8)
        rbc_px   += int(binary_tile.sum())
        total_px += binary_tile.size

        print(
            f"\r  [{n_done}/{total_tiles}]  "
            f"running RBC frac: {rbc_px / max(total_px, 1) * 100:.2f}%",
            end="", flush=True,
        )

    print()
    del arr  # free TIFF memmap

    # target_ds=1.0 → GeoJSON coords are already in native TIFF pixel space
    write_pyramidal_tiff(
        prob_acc, weight_acc, args.out,
        threshold=args.threshold,
        out_h=out_h, out_w=out_w,
        target_ds=1.0,
        slide_w0=tiff_w,
        slide_h0=tiff_h,
        min_rbc_px=args.min_rbc_px,
    )

    del prob_acc, weight_acc
    gc.collect()
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()

    print(f"\nOverall RBC area fraction: {rbc_px / max(total_px, 1) * 100:.3f}%")
    print(f"Pyramidal TIFF saved to  : {args.out}")


if __name__ == "__main__":
    main()


