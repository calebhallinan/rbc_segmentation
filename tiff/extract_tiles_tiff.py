"""
Extract tiles from a flat Visium HD TIFF at a target MPP.

The TIFF has no image pyramid, so we read native-resolution regions and
resize each tile to the target MPP before saving.

Strategy:
  1. Read microns_per_pixel from the companion scalefactors_json.json.
  2. Load the full TIFF as a memory-mapped array.
  3. Build a low-res thumbnail for Otsu tissue detection.
  4. Sample N tile positions from tissue-positive regions.
  5. Read each native tile, resize to TARGET_MPP, save PNG + CSV log.
"""

import csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_objects


# ─── config ────────────────────────────────────────────────────────────────────
CTRL_TIFF = Path(
    "/home/caleb/Desktop/3D/data/kidney_rabb_visiumhd/VisiumHD/Ctrl_1A4/outs/spatial/Ctrl_1A4.tif"
)
CTRL_SCALEFACTORS = Path(
    "/home/caleb/Desktop/3D/data/kidney_rabb_visiumhd/VisiumHD/Ctrl_1A4/outs/"
    "binned_outputs/square_002um/spatial/scalefactors_json.json"
)
CTRL_OUT_DIR = Path(
    "/home/caleb/Desktop/3D/data/kidney_rabb_visiumhd/rbc_tiles_tiff_ctrl_1A4"
)

AKI_TIFF = Path(
    "/home/caleb/Desktop/3D/data/kidney_rabb_visiumhd/VisiumHD/IR_2A4/outs/spatial/IR_2A4.tif"
)
AKI_SCALEFACTORS = Path(
    "/home/caleb/Desktop/3D/data/kidney_rabb_visiumhd/VisiumHD/IR_2A4/outs/"
    "binned_outputs/square_002um/spatial/scalefactors_json.json"
)
AKI_OUT_DIR = Path(
    "/home/caleb/Desktop/3D/data/kidney_rabb_visiumhd/rbc_tiles_tiff_aki_2A4"
)

# Which slide to run — change here or call process_slide() directly
ACTIVE = "aki"   # "ctrl" | "aki"

TARGET_MPP      = 0.5    # µm/px (20x equivalent)
TILE_SIZE       = 2048   # output tile size in pixels at TARGET_MPP
N_TILES         = 12
MIN_TISSUE_FRAC = 0.60
RANDOM_SEED     = 42
THUMB_MAX_DIM   = 2048   # max thumbnail dimension for tissue detection


# ─── helpers ───────────────────────────────────────────────────────────────────
def load_mpp(scalefactors_path: Path) -> float:
    data = json.loads(scalefactors_path.read_text())
    mpp  = float(data["microns_per_pixel"])
    print(f"  Source MPP from scalefactors: {mpp:.6f} µm/px")
    return mpp


def load_tiff_array(tiff_path: Path) -> np.ndarray:
    """Load TIFF as a memory-mapped array where possible, else read fully."""
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


def tissue_mask_from_array(
    arr: np.ndarray,
    thumb_max_dim: int = THUMB_MAX_DIM,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Otsu tissue mask on a downscaled thumbnail. Returns (mask, thumb_scale, thumb_rgb)."""
    H, W = arr.shape[:2]
    scale = max(H, W) / thumb_max_dim
    th = int(H / scale)
    tw = int(W / scale)

    thumb_pil  = Image.fromarray(arr).resize((tw, th), Image.LANCZOS)
    thumb_rgb  = np.asarray(thumb_pil)
    gray       = rgb2gray(thumb_rgb)

    try:
        t = threshold_otsu(gray)
    except Exception:
        t = 0.8
    tissue = (gray < t) & (gray > 0.05)
    tissue = binary_closing(tissue, disk(3))
    tissue = remove_small_objects(tissue, min_size=200)

    print(f"  Thumbnail: {tw}×{th}, tissue fraction: {tissue.mean():.3f}")
    return tissue, scale, thumb_rgb


def sample_tile_coords(
    tissue: np.ndarray,
    thumb_scale: float,
    tile_size_native: int,
    img_h: int,
    img_w: int,
    n_tiles: int,
    min_tissue_frac: float,
    rng: np.random.Generator,
) -> list:
    """Sample native (x0, y0) tile origins on tissue-positive regions."""
    h_thumb, w_thumb = tissue.shape
    tile_thumb = max(1, int(tile_size_native / thumb_scale))

    ys, xs = np.where(tissue)
    if len(xs) == 0:
        raise RuntimeError("No tissue found in thumbnail.")

    picked = []
    tries  = 0
    while len(picked) < n_tiles and tries < n_tiles * 200:
        tries += 1
        idx        = rng.integers(0, len(xs))
        x_thumb    = xs[idx] - tile_thumb // 2
        y_thumb    = ys[idx] - tile_thumb // 2
        if x_thumb < 0 or y_thumb < 0:
            continue
        if x_thumb + tile_thumb >= w_thumb or y_thumb + tile_thumb >= h_thumb:
            continue

        patch = tissue[y_thumb : y_thumb + tile_thumb, x_thumb : x_thumb + tile_thumb]
        if patch.mean() < min_tissue_frac:
            continue

        x0 = int(x_thumb * thumb_scale)
        y0 = int(y_thumb * thumb_scale)
        if x0 + tile_size_native > img_w or y0 + tile_size_native > img_h:
            continue

        too_close = any(
            abs(x0 - px) < tile_size_native // 2 and abs(y0 - py) < tile_size_native // 2
            for px, py in picked
        )
        if too_close:
            continue

        picked.append((x0, y0))

    if len(picked) < n_tiles:
        print(f"  Warning: only found {len(picked)}/{n_tiles} tiles after {tries} tries.")
    return picked


def read_tile(
    arr: np.ndarray,
    x0: int,
    y0: int,
    tile_size_native: int,
    out_size: int,
) -> Image.Image:
    """Crop native tile from array and resize to out_size."""
    crop = arr[y0 : y0 + tile_size_native, x0 : x0 + tile_size_native]
    img  = Image.fromarray(crop)
    if img.size != (out_size, out_size):
        img = img.resize((out_size, out_size), Image.LANCZOS)
    return img


# ─── main extraction ───────────────────────────────────────────────────────────
def process_slide(
    tiff_path: Path,
    scalefactors_path: Path,
    out_dir: Path,
    n_tiles: int       = N_TILES,
    tile_size: int     = TILE_SIZE,
    target_mpp: float  = TARGET_MPP,
    min_tissue_frac: float = MIN_TISSUE_FRAC,
    seed: int          = RANDOM_SEED,
) -> None:
    assert tiff_path.exists(),        f"TIFF not found: {tiff_path}"
    assert scalefactors_path.exists(), f"scalefactors_json not found: {scalefactors_path}"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_mpp      = load_mpp(scalefactors_path)
    scale_factor    = target_mpp / source_mpp       # < 1 → output is smaller than native
    tile_size_native = int(round(tile_size / scale_factor))  # native px to read per tile

    print(f"  Source MPP={source_mpp:.4f}, target MPP={target_mpp}, "
          f"scale={scale_factor:.4f}")
    print(f"  Reading {tile_size_native}×{tile_size_native} native px → "
          f"resizing to {tile_size}×{tile_size}")

    arr = load_tiff_array(tiff_path)
    img_h, img_w = arr.shape[:2]
    print(f"  TIFF dims: {img_w}×{img_h}")

    tissue, thumb_scale, thumb_rgb = tissue_mask_from_array(arr)

    Image.fromarray(thumb_rgb).save(out_dir / "_thumbnail.png")
    Image.fromarray((tissue * 255).astype(np.uint8)).save(out_dir / "_tissue_mask.png")

    rng    = np.random.default_rng(seed)
    coords = sample_tile_coords(
        tissue, thumb_scale, tile_size_native, img_h, img_w,
        n_tiles, min_tissue_frac, rng,
    )

    log_rows = []
    for i, (x0, y0) in enumerate(coords):
        tile = read_tile(arr, x0, y0, tile_size_native, tile_size)
        name = f"tile_{i:02d}_x{x0}_y{y0}.png"
        tile.save(out_dir / name)
        log_rows.append({
            "filename":        name,
            "x0_native":       x0,
            "y0_native":       y0,
            "tile_size_px":    tile_size,
            "tile_size_native": tile_size_native,
            "source_mpp":      source_mpp,
            "target_mpp":      target_mpp,
        })
        print(f"  saved {name}")

    with open(out_dir / "tiles_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nDone. {len(log_rows)} tiles → {out_dir}")


def main() -> None:
    if ACTIVE == "ctrl":
        process_slide(CTRL_TIFF, CTRL_SCALEFACTORS, CTRL_OUT_DIR)
    elif ACTIVE == "aki":
        process_slide(AKI_TIFF, AKI_SCALEFACTORS, AKI_OUT_DIR)
    else:
        raise ValueError(f"Unknown ACTIVE={ACTIVE!r}. Set to 'ctrl' or 'aki'.")


if __name__ == "__main__":
    main()
