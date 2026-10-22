"""
RBC classical segmentation on Visium HD TIFF tiles.

Identical logic to rbcs/rbc_segmentation_mask.py — Macenko normalization,
color deconvolution, thresholding — but with default paths pointing at the
TIFF-derived tile directories.

Outputs per tile (in OUT_DIR):
  *_rbc_mask.tif          — binary TIFF mask (0/255, zlib-compressed)
  *_qc.png                — 6-panel QC figure
  *_diagnostic.png        — 8-panel diagnostic figure
  *_normalized_fullres.png — full-res Macenko-normalized tile
  *_boundary_overlay.png   — red contour overlay on normalized tile
"""

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torchstain
from PIL import Image
from skimage.color import rgb2hed
from skimage.morphology import binary_opening, disk, remove_small_holes, remove_small_objects


# ─── config ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MPP    = 0.5   # target MPP used during tile extraction

E_THRESH       = 0.8  # TIFF scanner has lower eosin contrast than NDPI (p95≈0.67)
H_THRESH       = 0.25
RED_BLUE_RATIO = 1.15

MIN_RBC_DIAM_UM  = 0
MAX_HOLE_DIAM_UM = 0
OPENING_RADIUS_UM = 0

# ── switch between ctrl and aki here ──
ACTIVE = "ctrl"   # "ctrl" | "aki"

CTRL_TILE_DIR = Path(
    "/home/caleb/Desktop/3D/data/kidney_rabb_visiumhd/rbc_tiles_tiff_ctrl_1A4"
)
CTRL_OUT_DIR  = Path("/home/caleb/Desktop/rbcs/rbc_qc_out_tiff_ctrl_1A4")

AKI_TILE_DIR  = Path(
    "/home/caleb/Desktop/3D/data/kidney_rabb_visiumhd/rbc_tiles_tiff_aki_2A4"
)
AKI_OUT_DIR   = Path("/home/caleb/Desktop/rbcs/rbc_qc_out_tiff_aki_2A4")


# ─── Macenko normalizer ─────────────────────────────────────────────────────────
def build_normalizer(reference_rgb: np.ndarray):
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
    ref_t = torch.from_numpy(reference_rgb).permute(2, 0, 1).float().to(DEVICE)
    normalizer.HERef    = normalizer.HERef.to(DEVICE)
    normalizer.maxCRef  = normalizer.maxCRef.to(DEVICE)
    normalizer.fit(ref_t)
    return normalizer


def macenko_normalize(img: np.ndarray, normalizer) -> np.ndarray:
    t = torch.from_numpy(img).permute(2, 0, 1).float().to(DEVICE)
    norm, _, _ = normalizer.normalize(I=t, stains=False)
    out = norm.cpu().numpy().astype(np.uint8)
    if out.ndim == 3 and out.shape[0] == 3:
        out = out.transpose(1, 2, 0)
    return out


# ─── RBC mask ──────────────────────────────────────────────────────────────────
def rbc_mask(
    img_rgb: np.ndarray,
    img_rgb_original: Optional[np.ndarray] = None,
    mpp: float = MPP,
) -> dict:
    """
    img_rgb          : Macenko-normalized tile (used for HED stain rules)
    img_rgb_original : pre-normalization tile  (used for R/B ratio rule)
                       Falls back to img_rgb if not provided.
    Macenko collapses R/B ratio, so using the original for the red rule
    is critical for TIFF data.
    """
    src = img_rgb_original if img_rgb_original is not None else img_rgb

    img_f = img_rgb.astype(np.float32) / 255.0
    hed = rgb2hed(img_f)
    H, E = hed[..., 0], hed[..., 1]

    def norm01(x: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)

    Hn, En = norm01(H), norm01(E)
    R, B = src[..., 0], src[..., 2]

    rb_ratio   = R.astype(float) / (B.astype(float) + 1e-3)
    stain_rule = (En > E_THRESH) & (Hn < H_THRESH)
    red_rule   = rb_ratio > RED_BLUE_RATIO
    not_bg     = src.min(axis=-1) < 220

    print(f"    En>{E_THRESH}: {stain_rule.mean()*100:.2f}%  "
          f"R/B>{RED_BLUE_RATIO} (original): {red_rule.mean()*100:.2f}%  "
          f"combined: {(stain_rule & red_rule & not_bg).mean()*100:.3f}%")

    mask = stain_rule & red_rule & not_bg

    px_per_um = 1.0 / mpp
    min_area  = int(np.pi * (MIN_RBC_DIAM_UM  * px_per_um / 2) ** 2)
    max_hole  = int(np.pi * (MAX_HOLE_DIAM_UM  * px_per_um / 2) ** 2)
    open_r    = max(1, int(OPENING_RADIUS_UM   * px_per_um))

    mask = binary_opening(mask, disk(open_r))
    mask = remove_small_objects(mask, min_size=min_area)
    mask = remove_small_holes(mask, area_threshold=max_hole)

    return {"mask": mask, "H": Hn, "E": En}


# ─── QC figures ────────────────────────────────────────────────────────────────
def qc_figure(original: np.ndarray, normalized: np.ndarray, out: dict, save_path: Path) -> None:
    mask    = out["mask"]
    overlay = normalized.copy()
    overlay[mask] = (0.4 * overlay[mask] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(original);              axes[0, 0].set_title("Original")
    axes[0, 1].imshow(normalized);            axes[0, 1].set_title("Macenko normalized")
    axes[0, 2].imshow(out["E"], cmap="pink"); axes[0, 2].set_title("Eosin")
    axes[1, 0].imshow(out["H"], cmap="bone"); axes[1, 0].set_title("Hematoxylin")
    axes[1, 1].imshow(mask, cmap="gray");     axes[1, 1].set_title(
        f"RBC mask ({mask.mean()*100:.2f}%)"
    )
    axes[1, 2].imshow(overlay);               axes[1, 2].set_title("Overlay")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def diagnostic_figure(img_normalized: np.ndarray, save_path: Path) -> None:
    from skimage.color import rgb2hsv
    img_f = img_normalized.astype(np.float32) / 255.0
    hed   = rgb2hed(img_f)

    def norm01(x: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(x, 1), np.percentile(x, 99)
        return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)

    Hn, En = norm01(hed[..., 0]), norm01(hed[..., 1])
    hsv = rgb2hsv(img_f)
    sat, hue, val = hsv[..., 1], hsv[..., 0], hsv[..., 2]

    _E, _H, _S = 0.45, 0.25, 0.25
    combined = (
        (En > _E) & (Hn < _H) & (sat > _S)
        & ((hue < 0.08) | (hue > 0.92))
        & (img_normalized.min(axis=-1) < 220) & (val > 0.2)
    )

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes[0, 0].imshow(img_normalized);        axes[0, 0].set_title("Normalized")
    axes[0, 1].imshow(En, cmap="hot");        axes[0, 1].set_title(f"Eosin (>{_E})")
    axes[0, 2].imshow(sat, cmap="magma");     axes[0, 2].set_title(f"Saturation (>{_S})")
    axes[0, 3].imshow(hue, cmap="hsv");       axes[0, 3].set_title("Hue")
    axes[1, 0].imshow(En > _E, cmap="gray");  axes[1, 0].set_title("Eosin pass")
    axes[1, 1].imshow(Hn < _H, cmap="gray");  axes[1, 1].set_title("Hematoxylin pass")
    axes[1, 2].imshow(sat > _S, cmap="gray"); axes[1, 2].set_title("Saturation pass")
    axes[1, 3].imshow(combined, cmap="gray"); axes[1, 3].set_title("All rules")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_boundary_overlay(
    img_rgb: np.ndarray, mask: np.ndarray, save_path: Path,
    thickness: int = 1, color: tuple = (255, 0, 0),
) -> None:
    overlay   = img_rgb.copy()
    mask_u8   = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, thickness)
    Image.fromarray(overlay).save(save_path)


# ─── tile loader ───────────────────────────────────────────────────────────────
def load_tile(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        img = tifffile.imread(str(path))
    else:
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype(np.uint8)


# ─── main ──────────────────────────────────────────────────────────────────────
def process_tiles(tile_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = sorted(tile_dir.glob("tile_*.png"))
    if not tiles:
        raise FileNotFoundError(f"No tile PNGs found in {tile_dir}")
    print(f"Processing {len(tiles)} tiles from {tile_dir}")

    normalizer = build_normalizer(load_tile(tiles[0]))

    for p in tiles:
        original = load_tile(p)
        try:
            normalized = macenko_normalize(original, normalizer)
        except Exception as e:
            print(f"  [{p.name}] Macenko failed ({e}); using original")
            normalized = original

        out = rbc_mask(normalized, img_rgb_original=original)

        tifffile.imwrite(
            out_dir / f"{p.stem}_rbc_mask.tif",
            (out["mask"].astype(np.uint8) * 255),
            compression="zlib",
        )
        qc_figure(original, normalized, out, out_dir / f"{p.stem}_qc.png")
        diagnostic_figure(normalized, out_dir / f"{p.stem}_diagnostic.png")
        Image.fromarray(normalized).save(out_dir / f"{p.stem}_normalized_fullres.png")
        save_boundary_overlay(normalized, out["mask"], out_dir / f"{p.stem}_boundary_overlay.png")

        print(f"  [{p.name}] RBC area = {out['mask'].mean()*100:.3f}%")


def main() -> None:
    if ACTIVE == "ctrl":
        process_tiles(CTRL_TILE_DIR, CTRL_OUT_DIR)
    elif ACTIVE == "aki":
        process_tiles(AKI_TILE_DIR, AKI_OUT_DIR)
    else:
        raise ValueError(f"Unknown ACTIVE={ACTIVE!r}. Set to 'ctrl' or 'aki'.")


if __name__ == "__main__":
    main()
