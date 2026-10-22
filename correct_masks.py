"""
Napari-based mask correction tool for RBC training patches.

Controls
--------
  n  — save corrected mask and advance to next patch
  s  — skip without saving
  q  — quit (progress is persisted to JSON log)

Corrected patch stems are recorded in corrected_patches.json so the tool
can be resumed at any time without re-annotating finished patches.
Masks are stored as uint8 PNG (0 = background, 255 = RBC), overwriting the
original bootstrapped mask.

Environment
-----------
Requires napari.  Install with:
  pip install "napari[all]"
or activate the misc conda environment if napari is already installed there.
X11 forwarding is required when running over SSH:  ssh -X <host>
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Set

import napari
import numpy as np
from PIL import Image

# ─── config ────────────────────────────────────────────────────────────────────
TRAINING_DATA = Path("/home/caleb/Desktop/rbcs/training_data")
LOG_PATH      = Path("/home/caleb/Desktop/rbcs/corrected_patches.json")


# ─── helpers ───────────────────────────────────────────────────────────────────
def load_corrected_log(log_path: Path) -> Set[str]:
    if log_path.exists():
        return set(json.loads(log_path.read_text()))
    return set()


def save_corrected_log(log_path: Path, corrected: Set[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(sorted(corrected), indent=2))


def annotate_patch(img_path: Path, mask_path: Path) -> Dict:
    """
    Open a napari viewer for one patch.

    Returns a dict:
      {'action': 'saved'|'skipped'|'quit', 'mask': ndarray (H,W) int|None}
    """
    img      = np.asarray(Image.open(img_path).convert("RGB"))
    mask_255 = np.asarray(Image.open(mask_path).convert("L"))
    mask_01  = (mask_255 > 127).astype(np.int32)

    result: Dict = {"action": "skipped", "mask": None}

    viewer = napari.Viewer(title=f"RBC Corrector — {img_path.name}")
    viewer.add_image(img, name="H&E")
    labels_layer = viewer.add_labels(mask_01, name="RBC mask")

    viewer.text_overlay.visible = True
    viewer.text_overlay.text = (
        "Label 1 = RBC (paint)   Label 0 = background (erase)\n"
        "n = save + next        s = skip        q = quit"
    )

    @viewer.bind_key("n")
    def _save_and_next(v: napari.Viewer) -> None:
        result["action"] = "saved"
        result["mask"]   = labels_layer.data.copy()
        v.close()

    @viewer.bind_key("s")
    def _skip(v: napari.Viewer) -> None:
        result["action"] = "skipped"
        v.close()

    @viewer.bind_key("q")
    def _quit(v: napari.Viewer) -> None:
        result["action"] = "quit"
        v.close()

    napari.run()   # blocks until the viewer window is closed
    return result


# ─── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correct RBC segmentation masks in napari."
    )
    parser.add_argument(
        "--start_from", type=int, default=0,
        help="0-based index of the patch to start from (useful for resuming).",
    )
    args = parser.parse_args()

    img_dir = TRAINING_DATA / "images"
    msk_dir = TRAINING_DATA / "masks"

    if not img_dir.exists():
        print(f"Image directory not found: {img_dir}")
        print("Run prepare_training_data.py first.")
        return

    patches = sorted(img_dir.glob("*.png"))
    if not patches:
        print(f"No patches found in {img_dir}.  Run prepare_training_data.py first.")
        return

    corrected = load_corrected_log(LOG_PATH)
    print(f"Found {len(patches)} patches.  {len(corrected)} already corrected.")
    print(f"Starting from index {args.start_from}.\n")

    for i, img_path in enumerate(patches[args.start_from:], start=args.start_from):
        if img_path.stem in corrected:
            print(f"[{i + 1}/{len(patches)}] Already corrected, skipping: {img_path.name}")
            continue

        mask_path = msk_dir / img_path.name
        if not mask_path.exists():
            print(f"[{i + 1}/{len(patches)}] No mask found for {img_path.name}, skipping.")
            continue

        print(f"[{i + 1}/{len(patches)}] Annotating: {img_path.name}")
        result = annotate_patch(img_path, mask_path)

        if result["action"] == "saved":
            corrected_mask = (result["mask"].astype(np.uint8) * 255)
            Image.fromarray(corrected_mask).save(mask_path)
            corrected.add(img_path.stem)
            save_corrected_log(LOG_PATH, corrected)
            print(f"  Saved. ({len(corrected)} corrected total)")

        elif result["action"] == "skipped":
            print("  Skipped.")

        elif result["action"] == "quit":
            print(f"\nQuit after {len(corrected)} corrections.  Log: {LOG_PATH}")
            return

    print(f"\nAll patches reviewed.  {len(corrected)} corrections saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
