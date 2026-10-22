"""
Combine multiple training data directories and their corrected-patches JSON files
into a single merged directory.

Edit the SOURCES and OUTPUT sections below, then run:
    python combine_training_data.py
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional


# ─── config ────────────────────────────────────────────────────────────────────
# Each entry: (training_data_dir, corrected_patches_json | None)
# Set the JSON path to None if that dataset has no corrected patches yet.
SOURCES = [
    (
        Path("/home/caleb/Desktop/rbcs/training_data_tiff"),
        Path("/home/caleb/Desktop/rbcs/training_data_tiff/corrected_patches.json"),
    ),
    (
        Path("/home/caleb/Desktop/rbcs/training_data_tiff_ctrl_1A4_aki_2A4"),
        Path("/home/caleb/Desktop/rbcs/training_data_tiff_ctrl_1A4_aki_2A4/corrected_patches_ctrl_1A4_aki_2A4.json"),
    ),
]

OUTPUT_DIR          = Path("/home/caleb/Desktop/rbcs/training_data_tiff_combined")
OUTPUT_CORRECTED    = OUTPUT_DIR / "corrected_patches_combined.json"

OVERWRITE_EXISTING  = True   # set True to re-run over an existing combined dir


# ─── helpers ───────────────────────────────────────────────────────────────────
def load_corrected(json_path: Optional[Path]) -> set:
    if json_path is None or not json_path.exists():
        return set()
    data = json.loads(json_path.read_text())
    return set(data) if isinstance(data, list) else set(data.keys())


def collect_files(data_dir: Path, subdir: str) -> Dict[str, Path]:
    """Return {filename: full_path} for all PNGs in data_dir/subdir."""
    d = data_dir / subdir
    if not d.exists():
        return {}
    return {p.name: p for p in d.glob("*.png")}


# ─── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    img_out = OUTPUT_DIR / "images"
    msk_out = OUTPUT_DIR / "masks"

    if OUTPUT_DIR.exists() and not OVERWRITE_EXISTING:
        existing_imgs = len(list(img_out.glob("*.png"))) if img_out.exists() else 0
        if existing_imgs > 0:
            print(f"Output dir already has {existing_imgs} images. "
                  f"Set OVERWRITE_EXISTING=True to re-run.")
            return

    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    # ── pass 1: check for filename conflicts across all sources ─────────────────
    all_img_names: Dict[str, Path] = {}
    conflicts: List[str] = []

    for data_dir, _ in SOURCES:
        imgs = collect_files(data_dir, "images")
        for name, path in imgs.items():
            if name in all_img_names:
                conflicts.append(
                    f"  CONFLICT: {name}\n"
                    f"    {all_img_names[name]}\n"
                    f"    {path}"
                )
            else:
                all_img_names[name] = path

    if conflicts:
        print("Filename conflicts detected — cannot safely merge:\n")
        for c in conflicts:
            print(c)
        return

    # ── pass 2: copy files and merge corrected-patches sets ────────────────────
    combined_corrected: set = set()
    grand_img = grand_mask = 0

    for data_dir, json_path in SOURCES:
        imgs  = collect_files(data_dir, "images")
        masks = collect_files(data_dir, "masks")

        print(f"\n{data_dir.name}")
        print(f"  images : {len(imgs):4d}   masks : {len(masks):4d}")

        for name, src in imgs.items():
            shutil.copy2(src, img_out / name)
        for name, src in masks.items():
            shutil.copy2(src, msk_out / name)

        corrected = load_corrected(json_path)
        print(f"  corrected patches from JSON: {len(corrected)}")
        combined_corrected |= corrected

        grand_img  += len(imgs)
        grand_mask += len(masks)

    # ── write combined corrected JSON ───────────────────────────────────────────
    OUTPUT_CORRECTED.write_text(
        json.dumps(sorted(combined_corrected), indent=2)
    )

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Combined output : {OUTPUT_DIR}")
    print(f"  images  : {grand_img}")
    print(f"  masks   : {grand_mask}")
    print(f"  corrected patches: {len(combined_corrected)} → {OUTPUT_CORRECTED.name}")

    # sanity: verify counts match on disk
    disk_imgs  = len(list(img_out.glob("*.png")))
    disk_masks = len(list(msk_out.glob("*.png")))
    if disk_imgs != grand_img or disk_masks != grand_mask:
        print(f"\n  WARNING: disk counts ({disk_imgs} imgs, {disk_masks} masks) "
              f"don't match expected ({grand_img}, {grand_mask})")
    else:
        print(f"  Verified: {disk_imgs} images and {disk_masks} masks on disk ✓")


if __name__ == "__main__":
    main()
