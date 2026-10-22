"""
Prepare U-Net training patches from Visium HD TIFF tile + classical-mask pairs.

Mirrors rbcs/prepare_training_data.py but points at the TIFF-derived
tile and mask directories.

Sources:
  Control : rbc_tiles_tiff_ctrl/  + rbc_qc_out_tiff_ctrl/
  AKI     : rbc_tiles_tiff_aki/   + rbc_qc_out_tiff_aki/

For each paired tile+mask, extract 512×512 patches at 256px stride.
Skip patches where the minimum per-channel mean exceeds BG_THRESH (background).
Save image/mask PNGs to training_data_tiff/images/ and training_data_tiff/masks/.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile
from PIL import Image


# ─── config ────────────────────────────────────────────────────────────────────
CTRL_TILE_DIR = Path(
    "/home/caleb/Desktop/3D/data/kidney_rabb_visiumhd/rbc_tiles_tiff_ctrl_1A4"
)
CTRL_MASK_DIR = Path("/home/caleb/Desktop/rbcs/rbc_qc_out_tiff_ctrl_1A4")

AKI_TILE_DIR  = Path(
    "/home/caleb/Desktop/3D/data/kidney_rabb_visiumhd/rbc_tiles_tiff_aki_2A4"
)
AKI_MASK_DIR  = Path("/home/caleb/Desktop/rbcs/rbc_qc_out_tiff_aki_2A4")

OUT_BASE   = Path("/home/caleb/Desktop/rbcs/training_data_tiff_ctrl_1A4_aki_2A4") 

PATCH_SIZE = 512
STRIDE     = 512
BG_THRESH  = 220   # skip patch if min channel mean > this


# ─── helpers ───────────────────────────────────────────────────────────────────
def find_paired_tiles(tile_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    """Return [(tile_png, mask_tif), ...] for tiles that have a classical mask."""
    pairs: List[Tuple[Path, Path]] = []
    for mask_path in sorted(mask_dir.glob("*_rbc_mask.tif")):
        tile_stem = mask_path.stem.removesuffix("_rbc_mask")
        tile_path = tile_dir / f"{tile_stem}.png"
        if tile_path.exists():
            pairs.append((tile_path, mask_path))
        else:
            print(f"  Warning: no tile PNG for {mask_path.name} (expected {tile_path})")
    return pairs


def is_background(patch_rgb: np.ndarray, thresh: int = BG_THRESH) -> bool:
    """True when the lowest per-channel mean exceeds thresh (nearly all white)."""
    return float(patch_rgb.reshape(-1, 3).mean(axis=0).min()) > thresh


def extract_patches(
    tile: np.ndarray,
    mask: np.ndarray,
    tile_stem: str,
    img_out: Path,
    msk_out: Path,
    patch_size: int = PATCH_SIZE,
    stride: int = STRIDE,
) -> Tuple[int, int, int]:
    """
    Slide a window over tile/mask and save non-background patches.

    Patch names: {tile_stem}_{row:03d}_{col:03d}.png
    Returns (total_saved, with_rbcs, without_rbcs).
    """
    H, W = tile.shape[:2]
    total = with_rbcs = 0

    row = 0
    for y in range(0, H - patch_size + 1, stride):
        col = 0
        for x in range(0, W - patch_size + 1, stride):
            img_patch  = tile[y : y + patch_size, x : x + patch_size]
            mask_patch = mask[y : y + patch_size, x : x + patch_size]

            if is_background(img_patch):
                col += 1
                continue

            name       = f"{tile_stem}_{row:03d}_{col:03d}.png"
            clean_mask = (mask_patch > 127).astype(np.uint8) * 255

            Image.fromarray(img_patch).save(img_out / name)
            Image.fromarray(clean_mask).save(msk_out / name)

            total += 1
            if clean_mask.any():
                with_rbcs += 1
            col += 1
        row += 1

    return total, with_rbcs, total - with_rbcs


# ─── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    img_out = OUT_BASE / "images"
    msk_out = OUT_BASE / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    sources = [
        ("control", CTRL_TILE_DIR, CTRL_MASK_DIR),
        ("aki",     AKI_TILE_DIR,  AKI_MASK_DIR),
    ]

    grand_total = grand_rbc = grand_bg = 0

    for label, tile_dir, mask_dir in sources:
        if not tile_dir.exists():
            print(f"\n[{label.upper()}] Skipping — tile dir not found: {tile_dir}")
            continue
        if not mask_dir.exists():
            print(f"\n[{label.upper()}] Skipping — mask dir not found: {mask_dir}")
            continue

        pairs = find_paired_tiles(tile_dir, mask_dir)
        if not pairs:
            print(f"\n[{label.upper()}] No paired tiles found.")
            continue

        print(f"\n[{label.upper()}] {len(pairs)} paired tiles")
        src_total = src_rbc = src_bg = 0

        for tile_path, mask_path in pairs:
            tile = np.asarray(Image.open(tile_path).convert("RGB"))
            mask = tifffile.imread(str(mask_path))
            if mask.ndim > 2:
                mask = mask[..., 0]

            n, r, b = extract_patches(tile, mask, tile_path.stem, img_out, msk_out)
            src_total += n
            src_rbc   += r
            src_bg    += b
            print(f"  {tile_path.name}: {n} patches  ({r} with RBCs, {b} bg-only)")

        print(f"  {label} subtotal — Total: {src_total} | With RBCs: {src_rbc} | Without: {src_bg}")
        grand_total += src_total
        grand_rbc   += src_rbc
        grand_bg    += src_bg

    print(f"\n{'─'*60}")
    print(f"Grand total — Total: {grand_total} | With RBCs: {grand_rbc} | Without: {grand_bg}")
    print(f"Patches saved to: {OUT_BASE}")


if __name__ == "__main__":
    main()
