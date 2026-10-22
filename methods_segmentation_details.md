# RBC segmentation — Methods reconstruction

Reconstructed by inspecting `/nvme/caleb/rbcs` (also reachable as
`/home/caleb/Desktop/rbcs`, a symlink to the same tree). Everything below is
taken from code, configs, logs, sidecar metadata, or output files in this
folder. Values that could not be found are marked `Not found in repository`.
A small number of summary statistics were recomputed from the repository's own
output files during this audit; those are labelled explicitly.

All paths are relative to `/nvme/caleb/rbcs` unless stated otherwise.

> **Note on the repository:** this audit covered the whole working folder. The
> git repository holds only the pipeline that produced the final masks (see
> `README.md`). Files cited here as plain text rather than links (`HANDOFF.md`,
> `partial_label/`, the NDPI-round and UNI scripts, `rbc_segmentation_mask.py`,
> `predict_rbc.py`, `clean_training_data.py`) stay in the working folder but are
> not part of the repository.

---

## 1. Project overview

### What the folder contains

The folder holds a red-blood-cell (RBC) semantic-segmentation project on H&E
kidney histology from a rabbit ischemia–reperfusion (AKI) study with Visium HD
spatial transcriptomics. It contains **three distinct efforts**, of which only
the second produced the four whole-slide masks:

| # | Effort | Location | Status |
|---|---|---|---|
| A | Classical (rule-based) RBC mask | `rbc_segmentation_mask.py`, `tiff/rbc_segmentation_mask_tiff.py`, `qc/` | Used **only to bootstrap training labels**, not to produce final masks |
| B | **U-Net (EfficientNet-B0) — the pipeline that produced the four final masks** | `train_rbc_unet.py`, `predict_tif.py`, `checkpoints/best_rbc_unet.pth`, `unet_predictions_tiff/` | **Final / deliverable** |
| C | Partial-label (positive-unlabelled) retraining | `partial_label/` | Experimental follow-up, applied to **one** slide (`Ctrl_1A2`) only; **not adopted** (`HANDOFF.md:344-355`) |

Two further sub-efforts exist and were **not** used for any reported result:

* An earlier NDPI whole-slide round (`extract_tiles.py`, `prepare_training_data.py`,
  `predict_wsi.py`, `training_data_ndpi/`, `unet_predictions_ndpi/`). It targets
  scanned NDPI slides, not the Visium HD TIFFs.
* A UNI (ViT-L histology foundation model) encoder variant
  (`train_rbc_uni.py:42` sets `ENCODER_BACKEND = "uni"`, `predict_rbc_uni.py`).
  No UNI checkpoint and no `uni_predictions/` directory exist on disk, so this
  branch was exploratory only.

### Main entry points

| Role | File | Notes |
|---|---|---|
| Tile extraction (preprocessing) | [tiff/extract_tiles_tiff.py](tiff/extract_tiles_tiff.py) | 12 tiles/slide, 2048², Otsu tissue sampling |
| Classical label bootstrap | [tiff/rbc_segmentation_mask_tiff.py](tiff/rbc_segmentation_mask_tiff.py) | Macenko + colour deconvolution + thresholds |
| Patch extraction | [tiff/prepare_training_data_tiff.py](tiff/prepare_training_data_tiff.py) | 512² patches, stride 512 |
| Manual mask correction | [correct_masks.py](correct_masks.py) | napari painting tool, logs corrected stems to JSON |
| Mask cleanup | `clean_training_data.py` | removes small connected components |
| Dataset merge | [combine_training_data.py](combine_training_data.py) | merges the two acquisition rounds |
| **Training** | [train_rbc_unet.py](train_rbc_unet.py) | smp U-Net, EfficientNet-B0, Dice+BCE |
| **Whole-slide inference (final)** | [predict_tif.py](predict_tif.py) | tiled inference → BigTIFF + QuPath GeoJSON |
| Tile-level inference (QC) | `predict_rbc.py` | per-tile PNG mask/prob/overlay |
| NDPI whole-slide inference | `predict_wsi.py` | earlier round, OpenSlide-based |
| Partial-label training | `partial_label/train_partial.py` | experimental |
| Partial-label inference | `partial_label/predict_partial.py` | experimental |
| Config (partial-label only) | `partial_label/config.py` | the **only** config file in the project |
| Documentation | `HANDOFF.md`, `partial_label/README.md` | detailed narrative record |

**Notebooks: `Not found in repository`** — the project contains no `.ipynb` files.
**Config files for the main pipeline: `Not found in repository`** — all constants
in efforts A and B are hard-coded module-level variables.

### High-level workflow (raw image → final RBC mask)

```
Visium HD flat RGB TIFF (0.441 µm/px, no pyramid)
  │
  ├─(1) tiff/extract_tiles_tiff.py
  │      Otsu tissue mask on thumbnail → sample 12 tiles/slide,
  │      read native 1806×1806 px → LANCZOS resize to 2048×2048 (0.5 µm/px)
  │
  ├─(2) tiff/rbc_segmentation_mask_tiff.py
  │      Macenko stain normalization (reference = first tile of the batch)
  │      → rgb2hed colour deconvolution → eosin-high AND haematoxylin-low
  │      AND R/B > 1.15 AND not-white → binary "classical" mask per tile
  │
  ├─(3) tiff/prepare_training_data_tiff.py
  │      512×512 patches, stride 512, drop near-white patches
  │
  ├─(4) correct_masks.py  (napari)
  │      human paints missed RBCs onto a subset of patches;
  │      corrected stems recorded in corrected_patches*.json
  │
  ├─(5) combine_training_data.py
  │      merge round 1 (Ctrl_1A2 + IR_2A2) with round 2 (Ctrl_1A4 + IR_2A4)
  │      → training_data_tiff_combined (739 patches, 190 corrected)
  │
  ├─(6) train_rbc_unet.py
  │      smp.Unet(efficientnet-b0, ImageNet weights), 256² random crops,
  │      0.5·Dice + 0.5·BCE, AdamW, 50 epochs, best-val-loss checkpoint
  │      → checkpoints/best_rbc_unet.pth  (best epoch 25, val loss 0.34979)
  │
  └─(7) predict_tif.py  (run once per slide, sample_id edited at line 41)
         2048 px tiles at 0.5 µm/px with 256 px overlap → 512 px sliding window
         with 128 px overlap → Hanning-weighted blending → resize to native
         → threshold 0.2 → remove components < 30 px
         → unet_predictions_tiff/<sample>_rbc_mask.tif + .geojson + .meta.json
```

---

## 2. Input data

### Number of histology images analyzed

**Four whole-slide images**, one per Visium HD capture area
(`HANDOFF.md:20-21`; four mask sets in `unet_predictions_tiff/`).

### Filenames, format, dimensions, resolution

Source images (outside this folder, at `/home/caleb/Desktop/histology_sdd/data/<sample>/outs/spatial/<sample>.tif`;
the original path `/home/caleb/Desktop/3D/data/kidney_rabb_visiumhd/VisiumHD/...`
hard-coded in [predict_tif.py:45-51](predict_tif.py#L45-L51) **no longer exists** —
`HANDOFF.md:44-58`):

| Sample | Group | File | Width × Height (px) | µm/px | Format |
|---|---|---|---|---|---|
| `Ctrl_1A2` | control | `Ctrl_1A2.tif` | 32,096 × 18,048 | 0.4408725 | flat uint8 RGB TIFF, uncompressed, single page (no pyramid) |
| `Ctrl_1A4` | control | `Ctrl_1A4.tif` | 30,592 × 17,160 | 0.4407405 | same |
| `IR_2A2` | ischemia–reperfusion (AKI) | `IR_2A2.tif` | 31,688 × 17,880 | 0.4408513 | same |
| `IR_2A4` | ischemia–reperfusion (AKI) | `IR_2A4.tif` | 30,336 × 17,160 | 0.4408973 | same |

Slide dimensions are recorded in the repository in
`unet_predictions_tiff/<sample>_rbc_mask.meta.json` (`slide_w0`/`slide_h0`); the
`µm/px` values come from each sample's
`outs/binned_outputs/square_002um/spatial/scalefactors_json.json`
(`microns_per_pixel`), read by [predict_tif.py:172-177](predict_tif.py#L172-L177).
Dimensions and MPP were re-verified against the source TIFFs during this audit
and match the sidecars exactly.

* **Modality:** brightfield H&E (`HANDOFF.md:19`; the classical
  pipeline performs H&E colour deconvolution, `rbc_segmentation_mask.py:19,74-79`).
* **Species / model:** rabbit kidney, control vs ischemia–reperfusion AKI,
  Visium HD (`HANDOFF.md:19-21`).
* **Nominal magnification:** the pipeline resamples everything to
  `TARGET_MPP = 0.5` µm/px, described in code as "20× equivalent"
  ([tiff/extract_tiles_tiff.py:54](tiff/extract_tiles_tiff.py#L54)). Native
  resolution is 0.441 µm/px, so the scale factor is ≈ 0.88.
* **Scanner:** `Not found in repository` for the TIFFs. (The unrelated earlier
  NDPI round used Hamamatsu NDPI files named `OTS-24-22043 - 2024-08-28 15.08.37.ndpi`
  (control) and `... 15.11.15.ndpi` (AKI), `extract_tiles.py:22`,
  `predict_wsi.py:43-46`.)

### Tiling / cropping strategy

**Training-tile extraction** ([tiff/extract_tiles_tiff.py:54-59, 111-176](tiff/extract_tiles_tiff.py#L54-L59)):

* Thumbnail (max dimension 2048 px) → greyscale → Otsu threshold →
  `binary_closing(disk(3))` → `remove_small_objects(min_size=200)` tissue mask.
* Sample `N_TILES = 12` tile origins per slide from tissue-positive pixels,
  requiring ≥ `MIN_TISSUE_FRAC = 0.60` tissue within the tile and enforcing a
  minimum centre separation of half a tile.
* Read `tile_size_native = round(2048 / (0.5 / source_mpp)) = 1806` px, LANCZOS
  resize to 2048 × 2048 at 0.5 µm/px.
* `RANDOM_SEED = 42`. Coordinates logged to `tiles_log.csv` in the tile output
  directory (those directories are on a path that no longer exists — see §7).
* Tile filenames encode native origin: `tile_<NN>_x<X0>_y<Y0>.png`.

**Patch extraction** ([tiff/prepare_training_data_tiff.py:37-39, 61-102](tiff/prepare_training_data_tiff.py#L37-L39)):
512 × 512 patches, `STRIDE = 512` (non-overlapping), skipping any patch whose
minimum per-channel mean exceeds `BG_THRESH = 220` (near-white background).
Patch names: `<tile_stem>_<row:03d>_<col:03d>.png`.

> **Conflict:** the earlier NDPI-round script used `STRIDE = 256` (overlapping
> patches) — `prepare_training_data.py:27-28`.
> The TIFF pipeline that produced the final dataset uses stride 512.

**Inference tiling** — see §5.

### Stain normalization / colour normalization / background removal

* **In the classical label-bootstrap step only:** Macenko stain normalization
  (`torchstain.normalizers.MacenkoNormalizer`, torch backend, GPU), fitted to a
  single reference tile — the first tile of the batch
  (`rbc_segmentation_mask.py:44-62, 288`). On
  failure the code falls back to the un-normalized tile
  (`rbc_segmentation_mask.py:150-153`).
* **In U-Net training and inference: no stain normalization is applied.** Images
  are only ImageNet-normalized (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`)
  ([train_rbc_unet.py:47-48, 243](train_rbc_unet.py#L47-L48),
  [predict_tif.py:62-63, 274](predict_tif.py#L62-L63)). Note that the U-Net was
  trained on the **un-normalized** tile PNGs; the Macenko-normalized images were
  used only to compute the bootstrap masks.
* **Background removal:** near-white patches dropped at patch extraction
  (`BG_THRESH = 220`), and at inference whole tiles are skipped when
  `has_tissue()` is false (minimum per-channel mean > 220)
  ([predict_tif.py:229-231, 567](predict_tif.py#L229-L231)).

### Manual annotations / ground truth

There is **no independent ground truth**. Labels are classical-algorithm output
partially corrected by hand:

| Dataset | Patches | Human-corrected patches | Corrected-patch log |
|---|---|---|---|
| `training_data_tiff/` (round 1: `Ctrl_1A2` + `IR_2A2`) | 365 | 115 | `corrected_patches.json` |
| `training_data_tiff_ctrl_1A4_aki_2A4/` (round 2: `Ctrl_1A4` + `IR_2A4`) | 374 | 75 | `corrected_patches_ctrl_1A4_aki_2A4.json` |
| **`training_data_tiff_combined/` (used for the final model)** | **739** | **190 (25.7%)** | `corrected_patches_combined.json` |
| `training_data_ndpi/` (earlier NDPI round, unused for the final model) | 1145 | 94 | `corrected_patches_ndpi.json` |

Correction was done in napari by painting label 1 = RBC / 0 = background over
the bootstrap mask, overwriting the mask PNG in place and appending the patch
stem to the JSON log ([correct_masks.py:49-89, 133-138](correct_masks.py#L49-L89)).
Masks are uint8 PNG, 0 = background, 255 = RBC.

Label quality was later quantified (`HANDOFF.md:65-101`,
`partial_label/README.md:20-29`):

| | n | empty masks | mean RBC px |
|---|---|---|---|
| human-corrected | 190 | 0.5% | 1.189% |
| classical (uncorrected) | 549 | 16.8% | 0.156% |

Within the same tile, corrected patches carry a **median 40× more** RBC pixels
than uncorrected ones (34/34 tiles). Corrections were overwhelmingly additions
(506,118 px added vs 47,159 px removed). **The labels are therefore
positive-unlabelled, not exhaustive.**

Round-1 slide identity is not written in any script (the extraction script was
edited in place for round 2). It was **verified during this audit** by
re-cropping the source TIFFs at the coordinates encoded in the tile filenames:
`tile_00_x18743_y3667` reproduces `Ctrl_1A2` exactly (mean absolute difference
0.000; 28–43 grey levels against the other three slides) and
`tile_00_x18133_y4022` reproduces `IR_2A2` exactly. Round 2 is self-identifying
from the directory names (`..._ctrl_1A4_aki_2A4`).

### Training / validation / test split

Split is performed at run time by
[train_rbc_unet.py:174-215](train_rbc_unet.py#L174-L215) — it is not stored on
disk. Re-running that function on `training_data_tiff_combined` during this
audit gives, deterministically (`SEED = 42`):

* 48 tiles total, **5 tiles held out** for validation.
* **660 training patches / 79 validation patches** (166 / 24 of them
  human-corrected).
* Held-out tiles: `tile_02_x7751_y9593`, `tile_05_x18660_y11852`,
  `tile_05_x9967_y11941`, `tile_06_x12786_y5213`, `tile_07_x18776_y3629`.
* **There is no test set.** Validation was used both for checkpoint selection
  and as the only quantitative read-out.

Split mechanics and two discrepancies worth reporting:

* The split is grouped **by tile**, so no two patches from the same 2048² tile
  land on opposite sides — this prevents patch-level leakage but **not
  slide-level leakage**.
* `VAL_FRAC = 0.10` ([train_rbc_unet.py:43](train_rbc_unet.py#L43)) while the
  module docstring and `unet_architecture.txt:126` both say "85/15 stratified by
  tile". The code is authoritative: `ceil(48 × 0.10) = 5` validation tiles
  (10.4% of tiles, 10.7% of patches).
* Stratification by disease group **silently fails on this dataset**: tiles are
  assigned to the AKI group by testing `"aki" in tile_key.lower()`
  ([train_rbc_unet.py:190-193](train_rbc_unet.py#L190-L193)), but TIFF-derived
  tile keys are of the form `tile_07_x18776_y3629` and never contain "aki". All
  48 tiles fall into the control branch, and the printed split summary reports
  "AKI: 0 tiles". Which slides the 5 validation tiles came from is therefore not
  controlled; by coordinate matching they are a mixture of both rounds.

### Were the four final images used for training or only inference?

**All four slides were used for training.** Twelve tiles were extracted from
each of `Ctrl_1A2`, `IR_2A2` (round 1) and `Ctrl_1A4`, `IR_2A4` (round 2), and
all 739 resulting patches went into `training_data_tiff_combined`. The same four
slides were then run through whole-slide inference. This is stated explicitly in
`HANDOFF.md:370-375`:

> "Everything here is in-sample: the original 739 training patches were cut from
> all four slides, so Ctrl_1A2 was seen in training by every model *including
> the baseline*. All reported numbers are optimistic. A fifth slide, or
> retraining with one slide fully excluded, is needed before any of this goes in
> a paper."

**There is no held-out slide.** This must be stated in the manuscript.

---

## 3. RBC segmentation algorithm

The final masks come from a **deep-learning** model. A **classical image
processing** step exists but only generated the initial (weak) training labels,
so the overall procedure is best described as a **hybrid weak-supervision
pipeline**: rule-based bootstrap labels → partial human correction → supervised
CNN.

### 3a. Deep-learning model (produced the four final masks)

| Property | Value | Source |
|---|---|---|
| Model class | `segmentation_models_pytorch.Unet` (smp 0.5.0) | [train_rbc_unet.py:392-396](train_rbc_unet.py#L392-L396) |
| Encoder / backbone | EfficientNet-B0 | same |
| Encoder weights | ImageNet (`encoder_weights="imagenet"`) | same |
| Decoder | standard smp U-Net decoder: bilinear upsample → concat skip → Conv-BN-ReLU ×2, 5 blocks (256→128→64→32→16 ch) | [unet_architecture.txt:38-71](unet_architecture.txt#L38-L71) |
| Encoder stages / skips | 6 feature levels (3, 32, 24, 40, 112, 320 ch); bottleneck 320 ch at stride 32 | [unet_architecture.txt:17-35](unet_architecture.txt#L17-L35) |
| Activations | encoder MBConv with SiLU; decoder ReLU | [unet_architecture.txt:24, 41](unet_architecture.txt#L24-L41) |
| Normalization layers | BatchNorm2d throughout (smp default) | [unet_architecture.txt:41](unet_architecture.txt#L41) |
| Dropout | none configured (smp defaults; no `aux_params`, no dropout argument) | [train_rbc_unet.py:392-396](train_rbc_unet.py#L392-L396) |
| Input channels | 3 (RGB) | same |
| Output channels / classes | 1 (binary) | same |
| Final activation | none in the model — raw logits; `sigmoid` applied at inference, then threshold | [predict_tif.py:279](predict_tif.py#L279) |
| Head | Conv2d(16 → 1, k3, pad 1) | [unet_architecture.txt:77](unet_architecture.txt#L77) |
| Total parameters | 6,251,469 (encoder 4,007,548 / decoder 2,243,776 / head 145) | [unet_architecture.txt:11-14](unet_architecture.txt#L11-L14) |
| Loss | `0.5 × soft Dice + 0.5 × BCE-with-logits`, `pos_weight = 1` | [train_rbc_unet.py:128-145](train_rbc_unet.py#L128-L145) |
| Frozen layers | none — the whole network is trainable (all parameters passed to the optimizer) | [train_rbc_unet.py:405](train_rbc_unet.py#L405) |
| Checkpoint | `checkpoints/best_rbc_unet.pth`, 72,240,575 bytes, sha256 `9ea753c7e8ec2ed5…` | computed during this audit |
| Checkpoint contents | `epoch = 25`, `val_loss = 0.3497944176197052`, `config = {encoder_name: efficientnet-b0, in_channels: 3, classes: 1, patch_size: 256, normalize_mean/std = ImageNet}` | read from the checkpoint during this audit |

> **Dead code:** a hand-written `RBCUNet` (features 32/64/128/256, bottleneck
> 512) is present but **fully commented out** in
> [train_rbc_unet.py:52-122](train_rbc_unet.py#L52-L122) and
> [predict_tif.py:67-136](predict_tif.py#L67-L136). The `FEATURES = (32,64,128,256)`
> constant at [train_rbc_unet.py:36](train_rbc_unet.py#L36) is unused. Do **not**
> describe that architecture in the Methods.

> **Conflict:** the module docstring at
> [train_rbc_unet.py:5](train_rbc_unet.py#L5) says `pos_weight=10`, but the code
> sets `POS_WEIGHT = 1` ([train_rbc_unet.py:45](train_rbc_unet.py#L45)), and that
> is the value passed to the loss ([train_rbc_unet.py:404](train_rbc_unet.py#L404)).
> `unet_architecture.txt:101` also records `pos_weight=1.0`. **The trained model
> used `pos_weight = 1`** — i.e. no class-imbalance re-weighting.

### 3b. Classical algorithm (bootstrap labels only)

Per tile ([tiff/rbc_segmentation_mask_tiff.py](tiff/rbc_segmentation_mask_tiff.py),
logic identical to `rbc_segmentation_mask.py:66-103`):

1. **Macenko stain normalization** to a reference tile (torchstain, GPU).
2. **Colour deconvolution** with `skimage.color.rgb2hed`; H and E channels each
   rescaled per tile to [0,1] using the 1st and 99th percentiles.
3. **Thresholds (ANDed):**
   * eosin high: `En > E_THRESH`
   * haematoxylin low: `Hn < H_THRESH = 0.25`
   * red-shift: `R / (B + 1e-3) > RED_BLUE_RATIO = 1.15`
   * not white background: `min(R,G,B) < 220`
4. **Morphology:** `binary_opening(disk(open_r))`, `remove_small_objects(min_size)`,
   `remove_small_holes(area_threshold)` — but all three physical size constants
   are **set to zero** (`MIN_RBC_DIAM_UM = MAX_HOLE_DIAM_UM = OPENING_RADIUS_UM = 0`,
   `rbc_segmentation_mask.py:38-40`,
   `tiff/rbc_segmentation_mask_tiff.py:38-40`). With radius clamped to
   `max(1, 0) = 1`, the effective operation is a single `disk(1)` opening;
   `min_size = 0` and `area_threshold = 0` make the other two no-ops.
5. **No watershed and no instance splitting anywhere in the project.** The output
   is a semantic (binary) mask; "objects" are 8-connected components.

**Threshold conflict — report all three:**

| `E_THRESH` | Where | Applies to |
|---|---|---|
| 0.9 | `rbc_segmentation_mask.py:33` | NDPI-round tiles (`qc/rbc_qc_out_ctrl`, `qc/rbc_qc_out_aki`) |
| **0.8** | `tiff/rbc_segmentation_mask_tiff.py:34` (comment: "TIFF scanner has lower eosin contrast than NDPI (p95≈0.67)") | **the four Visium HD TIFF slides — the labels used for the final model** |
| 0.45 | `rbc_segmentation_mask.py:206`, inside `diagnostic_figure()` | QC figure only; it therefore does **not** display the mask that was actually saved |

`E_THRESH = 0.9` is identified in `HANDOFF.md:80-83` as the
root cause of the label incompleteness ("requires eosin in the top 10% of each
tile's dynamic range, ANDed with three more conservative rules. High precision,
terrible recall by construction").

Also note `rbc_segmentation_mask.py` defines `main()`
**twice** (lines 141 and 234); the first definition is dead and the second — the
one that runs — additionally writes diagnostic figures, full-res normalized
tiles and boundary overlays (`HANDOFF.md:400-402`).

### 3c. Partial-label variant (experimental, not used for the four masks)

Same smp U-Net / EfficientNet-B0 / ImageNet backbone
(`partial_label/train_partial.py:53-59`),
with a different loss designed for positive-unlabelled data
(`partial_label/losses.py:33-116`):

`loss = 0.5 · Tversky + 0.5 · (0.5 · BCE_pos + 0.5 · BCE_neg)` where

* Tversky `α = 0.3` (false positives), `β = 0.7` (false negatives), with the FP
  term scaled per pixel by a provenance-dependent `neg_trust`;
* background target smoothed to `neg_smooth = 0.03` instead of 0;
* the highest-loss `drop_frac = 5%` of *unprotected* background pixels are
  discarded from the negative term;
* `neg_trust` = 1.0 (human-corrected), 0.08–0.60 (uncorrected classical),
  0.40–0.75 (pseudo-labels from the baseline model), 1.0 and "protected"
  (human-certified negative boxes)
  (`partial_label/config.py:74-95`,
  `HANDOFF.md:119-130`).

This branch is documented in detail in `partial_label/README.md` and
`HANDOFF.md`. Its conclusion (`HANDOFF.md:344-355`) is that
**no partial-label model beats the baseline on object-level fidelity** — recall
rose, but adjacent RBCs merged into blobs (object count fell from 8,130 to
~6,700 while median component size doubled from 151 px to ~290 px). If the
manuscript reports per-cell counts, the baseline masks in
`unet_predictions_tiff/` are the ones to use.

---

## 4. Training details (final model)

| Item | Value | Source |
|---|---|---|
| Training script | `train_rbc_unet.py` (no notebook) | — |
| Training data | `training_data_tiff_combined/` — 739 patches from 48 tiles across all 4 slides | [train_rbc_unet.py:31](train_rbc_unet.py#L31); counted on disk |
| Train / val patches | 660 / 79 (reproduced from the code with `SEED = 42`) | [train_rbc_unet.py:174-215](train_rbc_unet.py#L174-L215) |
| Patch size on disk | 512 × 512 | [tiff/prepare_training_data_tiff.py:37](tiff/prepare_training_data_tiff.py#L37) |
| Model input crop | **256 × 256** (`RandomCrop` train, `CenterCrop` val) | [train_rbc_unet.py:38, 221, 247](train_rbc_unet.py#L221) |
| Batch size | 16 | [train_rbc_unet.py:38](train_rbc_unet.py#L38) |
| Dataloader workers | 4 | [train_rbc_unet.py:39](train_rbc_unet.py#L39) |
| Epochs | 50 | [train_rbc_unet.py:40](train_rbc_unet.py#L40) |
| Optimizer | AdamW | [train_rbc_unet.py:405](train_rbc_unet.py#L405) |
| Learning rate | 1e-3 | [train_rbc_unet.py:41](train_rbc_unet.py#L41) |
| Weight decay | 1e-4 | [train_rbc_unet.py:42](train_rbc_unet.py#L42) |
| Scheduler | `CosineAnnealingLR(T_max=50)`, stepped per epoch | [train_rbc_unet.py:406, 419](train_rbc_unet.py#L406) |
| Loss | 0.5 · Dice + 0.5 · BCE(`pos_weight=1`) | [train_rbc_unet.py:128-145](train_rbc_unet.py#L128-L145) |
| Class weighting / imbalance handling | **none** (`pos_weight = 1`; no oversampling, no focal term) | [train_rbc_unet.py:45](train_rbc_unet.py#L45) |
| Mixed precision | `torch.autocast("cuda")` + `torch.amp.GradScaler` | [train_rbc_unet.py:268, 407](train_rbc_unet.py#L268) |
| Graph compilation | `torch.compile(model)` when torch ≥ 2 | [train_rbc_unet.py:397-402](train_rbc_unet.py#L397-L402) |
| Random seed | 42 (`torch.manual_seed`, `np.random.seed`, and the split RNG) | [train_rbc_unet.py:44, 340-341](train_rbc_unet.py#L340-L341) |
| Checkpointing | save whenever validation loss improves; single file overwritten | [train_rbc_unet.py:424-445](train_rbc_unet.py#L424-L445) |
| Best-checkpoint criterion | **minimum validation loss** (the combined Dice+BCE loss) | same |
| Selected checkpoint | **epoch 25 of 50, val loss 0.3497944** | read from `checkpoints/best_rbc_unet.pth` |

**Data augmentation** (albumentations 2.x,
[train_rbc_unet.py:219-251](train_rbc_unet.py#L219-L251)) — training:

| Transform | Parameters | p |
|---|---|---|
| `RandomCrop` | 256 × 256 | 1.0 |
| `HorizontalFlip` | — | 0.5 |
| `VerticalFlip` | — | 0.5 |
| `RandomRotate90` | — | 0.5 |
| `ColorJitter` | brightness/contrast/saturation (0.8, 1.2), hue (−0.1, 0.1) | 0.5 |
| `GaussianBlur` | blur_limit (3, 7), sigma (0.1, 2.0) | 0.3 |
| `GaussNoise` | std_range (0.02, 0.1) | 0.3 |
| `HueSaturationValue` | hue ±10, sat ±20, val ±20 | 0.5 |
| `RGBShift` | r ±15, g ±10, b ±10 | 0.3 |
| `Normalize` | ImageNet mean/std | 1.0 |

Validation: `CenterCrop(256)` + `Normalize` only.
**No scaling/zoom, no elastic deformation, no dedicated stain augmentation** in
the final model. (Stain augmentation `A.HEStain` appears only in the
experimental branch, `partial_label/data.py:214-216`.)

**Hardware:** not logged for this run. The module docstring says "single GPU
(CUDA 12.4 / RTX 6000 Ada)" ([train_rbc_unet.py:8](train_rbc_unet.py#L8)); the
machine has **2 × NVIDIA RTX 6000 Ada Generation, 48 GB each**, driver 550.144.03
(`HANDOFF.md:42`; confirmed with `nvidia-smi` during this audit).
Training time: `Not found in repository` (no training log for the final model —
`train_rbc_unet.py` prints to stdout and no capture exists).

**Software versions:** no environment file exists. The conda environment used
for the later work is `misc` (`/home/caleb/anaconda3/envs/misc`,
`HANDOFF.md:33-40`); its current contents are listed in §7.
The docstring's "CUDA 12.4" conflicts with the environment's current
`torch 2.7.1+cu126`; the environment may have been upgraded after the April 2026
training run, so the exact versions used to train `best_rbc_unet.pth` are
**`Not found in repository`**.

---

## 5. Inference details (the four final masks)

| Item | Value | Source |
|---|---|---|
| Script | `predict_tif.py` (run once per slide; `sample_id` edited at line 41) | [predict_tif.py:41](predict_tif.py#L41) |
| Checkpoint | `checkpoints/best_rbc_unet.pth` (epoch 25) | [predict_tif.py:53](predict_tif.py#L53) |
| Model reconstruction | `smp.Unet` rebuilt from the checkpoint's `config`, `encoder_weights=None`, `_orig_mod.`/`module.` prefixes stripped | [predict_tif.py:154-168](predict_tif.py#L154-L168) |
| Preprocessing at inference | resample native crop to 0.5 µm/px (LANCZOS); scale to [0,1]; ImageNet mean/std. **No stain normalization.** | [predict_tif.py:202-214, 274](predict_tif.py#L202-L214) |
| Outer tile size | 2048 px at 0.5 µm/px = 1806 native px | [predict_tif.py:56, 528](predict_tif.py#L528) |
| Outer tile overlap | 256 px at 0.5 µm/px (226 native px) | [predict_tif.py:57, 529](predict_tif.py#L529) |
| Tissue gating | skip tile when min per-channel mean > 220 | [predict_tif.py:229-231, 567](predict_tif.py#L229-L231) |
| Inner sliding window | 512 × 512 patches, 128 px overlap (step 384) | [predict_tif.py:238-239](predict_tif.py#L238-L239) |
| Batch size at inference | **1** (patches processed one at a time) | [predict_tif.py:276-278](predict_tif.py#L276-L278) |
| Edge handling | reflect-padding to a whole number of steps | [predict_tif.py:246-260](predict_tif.py#L246-L260) |
| Stitching | 2-D Hanning window (`np.outer(np.hanning(n), np.hanning(n))`) weighted accumulation, at both the patch level and the tile level; final probability = `prob_acc / weight_acc` | [predict_tif.py:262-284, 548-550, 584-585](predict_tif.py#L262-L284) |
| Memory strategy | `np.memmap` float32 accumulators (`prob_acc`, `weight_acc`) at native slide size in a temp dir | [predict_tif.py:552-557](predict_tif.py#L552-L557) |
| Resampling back to native | `cv2.resize(..., INTER_LINEAR)` of the tile probability map before accumulation | [predict_tif.py:580-581](predict_tif.py#L580-L581) |
| Probability threshold | **0.2** (`DEFAULT_THRESHOLD`) | [predict_tif.py:55](predict_tif.py#L55) |
| Test-time augmentation | **not used** — an 8-way TTA function exists but its call site is commented out | [predict_tif.py:287-326, 571-572](predict_tif.py#L571-L572) |
| Mask cleanup | remove 8-connected components < `MIN_RBC_PX = 30` px. **No hole filling, no dilation/erosion/opening/closing, no watershed splitting.** | [predict_tif.py:60, 140-150, 452-453](predict_tif.py#L140-L150) |
| Output resolution | native TIFF resolution, 1:1 with the input (`target_ds = 1.0`) | [predict_tif.py:538-541, 604](predict_tif.py#L538-L541) |

> **Threshold provenance caveat:** the baseline `*.meta.json` sidecars record
> only geometry (`target_ds`, `out_w/h`, `slide_w0/h0`) — **the threshold and
> size filter were not written to disk for these four masks.** The values 0.2
> and 30 px are the script defaults at the time the masks were written, and the
> partial-label audit adopted exactly that reading:
> `partial_label/comparison/Ctrl_1A2_summary.json` states
> `"threshold": 0.2, "min_rbc_px": 30, "note": "baseline meta.json recorded no
> threshold; value taken from predict_tif.py defaults at the time the mask was
> written"`. Timestamps are consistent: the checkpoint is dated Apr 29 12:51 and
> the four masks Apr 29 12:53–13:00.
>
> **Conflicting defaults elsewhere:** `predict_rbc.py:34` uses 0.1,
> `predict_wsi.py:49` uses 0.2, and `unet_architecture.txt:136` documents 0.5.
> Only `predict_tif.py` produced the four final masks.

### Output files produced for each of the four slides

In `unet_predictions_tiff/`, per sample (`Ctrl_1A2`, `Ctrl_1A4`, `IR_2A2`, `IR_2A4`):

| File | Content |
|---|---|
| `<sample>_rbc_mask.tif` | binary mask (0/255), BigTIFF, 256×256 tiles, zlib level 6, `photometric=minisblack`, single level (`N_PYRAMID_LEVELS = 1`), native slide dimensions |
| `<sample>_rbc_mask.geojson` | QuPath-ready `FeatureCollection`; one `PathAnnotationObject` polygon per component, classification `RBC` (colorRGB −65536), `isLocked: true`; contours simplified with `approxPolyDP(0.5)` then Shapely `buffer(0)`, capped at 500 vertices; coordinates in native TIFF pixels |
| `<sample>_rbc_mask.meta.json` | `target_ds`, `out_w`, `out_h`, `slide_w0`, `slide_h0` |

Per-tile QC PNGs (`*_prob.png`, `*_mask.png`, `*_boundary_overlay.png`) for 12
tiles are in `unet_predictions_tiff/` (from `predict_rbc.py`) — note these are
tile-level QC, not the whole-slide product. `unet_predictions_ndpi/` holds the
equivalent products for the abandoned NDPI round
(`wsi_rbc_mask.tif/.geojson/.meta.json`, `target_ds = 2.2692`, 11,908 features).

---

## 6. Quantification and evaluation

### Metrics computed during training

Only the **combined Dice+BCE validation loss** was tracked and used for
selection ([train_rbc_unet.py:278-293, 424](train_rbc_unet.py#L278-L293)).
Best value: **0.3497944 at epoch 25**. **No Dice, IoU, precision, recall, F1,
object-level accuracy, cell-count error or area error was computed against
ground truth for the final model** — `Not found in repository`. Note also that
this loss is measured against the same incomplete labels described in §2, so it
is not an accuracy estimate.

### Quantitative outputs that do exist

**Per-slide mask statistics.** These are not tabulated anywhere in the
repository; they were **computed during this audit** directly from
`unet_predictions_tiff/*_rbc_mask.tif` (component analysis with
`cv2.connectedComponentsWithStats`, connectivity 8) and from an Otsu tissue mask
on each source TIFF using the same procedure as
`partial_label/predict_partial.py:85-99`:

| Sample | Group | RBC px | % of slide | tissue % of slide | **% of tissue** | objects | median object (px) | mean object (px) |
|---|---|---|---|---|---|---|---|---|
| Ctrl_1A2 | control | 2,238,324 | 0.386% | 28.17% | **1.372%** | 8,130 | 151 | 275.3 |
| Ctrl_1A4 | control | 2,321,725 | 0.442% | 29.51% | **1.498%** | 8,520 | 154 | 272.5 |
| IR_2A2 | AKI | 2,128,567 | 0.376% | 32.34% | **1.162%** | 8,925 | 124 | 238.5 |
| IR_2A4 | AKI | 1,588,815 | 0.305% | 34.06% | **0.896%** | 6,899 | 119 | 230.3 |

Object counts match the `PathAnnotationObject` counts in the corresponding
GeoJSON files exactly (8,130 / 8,520 / 8,925 / 6,899), and the Ctrl_1A2 row
matches `partial_label/comparison/Ctrl_1A2_summary.json` and
`Ctrl_1A2_all_models.json` exactly, so the recomputation is consistent with the
repository's own numbers.

**Use % of tissue, not % of slide**, for cross-section comparison — tissue is
28–34% of the bounding box and that fraction differs per section
(`HANDOFF.md:198-200`).

**Scale reference:** one 7 µm RBC ≈ **198 px** at 0.4409 µm/px
(`rbc_area_px = 197.9976` in `partial_label/comparison/Ctrl_1A2_all_models.json`;
`HANDOFF.md:59-61`). The baseline's median object of ~151 px
is therefore slightly under one cell, and the `MIN_RBC_PX = 30` floor admits
objects at ~15% of a cell (`HANDOFF.md:398-399`).

**Partial-label comparison (Ctrl_1A2 only)** — from
`partial_label/comparison/*.json` and `HANDOFF.md:187-196`:

| model | % of tissue | objects | median px | % of claimed px red (R/B > 1.25) |
|---|---|---|---|---|
| baseline (`unet_predictions_tiff`) | 1.372% | 8,130 | 151 | 52.4% |
| permissive (`partial`, neg_trust 0.08) | 4.222% | 33,309 | 89 | — |
| `pseudo` @0.90 + 150 px | 1.456% | 5,553 | — | — |
| `B_nobox` @0.90 + 150 px | 2.165% | 6,820 | 289 | 42.3% |
| `A_raw` @0.90 + 150 px | 2.545% | 7,480 | 300 | 39.2% |
| `A_refined` @0.90 + 150 px + R/B > 1.10 | 1.509% | 6,224 | 261 | 56.5% |

Agreement of the first partial-label model with the baseline
(`Ctrl_1A2_summary.json`): IoU 0.531, 89.3% of baseline pixels retained,
1.57× fold change in claimed area.

Gain decomposition (`Ctrl_1A2_gain_analysis.json`): of 11,373 "novel" components
(no baseline overlap), only **652 (5.7%)** are at least one RBC in area; median
novel component 58 px vs 198 px for a real RBC.

Validation metrics for the partial-label runs are stored in the checkpoints and
in `partial_label/checkpoints/history_*.json` (per-epoch recall,
`precision_lower_bound`, `dice_contaminated`, F2, `pred_over_label`,
`n_components`, `frac_components_rbc_sized`) and echoed in `partial_label/*.log`.
Their precision is explicitly a **lower bound**, because an unlabelled true
positive counts as a false positive
(`partial_label/losses.py:147-165`,
`partial_label/README.md:64-82`).

### Manual vs quantitative assessment

Performance was assessed **visually**, not against ground truth. The repository
is explicit that "the only way to get true precision here is to audit by eye"
(`partial_label/README.md:77-79`) and that "the
stopping signal is the visual audit, not any number computed against the existing
labels" (`partial_label/README.md:168-171`).

### QC plots, overlays and visual inspection outputs

* `checkpoints/val_samples/val_sample_{00..03}.png` — 4 validation overlays with
  predicted contours at threshold 0.5 ([train_rbc_unet.py:297-328](train_rbc_unet.py#L297-L328)).
* `qc/rbc_qc_out_*/` — 6 directories × 60 files: per tile a binary mask TIFF, a
  6-panel QC figure, an 8-panel diagnostic figure, a full-res normalized tile and
  a boundary overlay (`rbc_segmentation_mask.py:234-276`).
* `unet_predictions_tiff/`, `unet_predictions_ndpi/` — per-tile probability maps,
  masks and red-contour overlays.
* `partial_label/comparison/` — `Ctrl_1A2_ALL.png` (six-model crop grid),
  `Ctrl_1A2_zoom_AB.png`, `Ctrl_1A2_zoom_refined.png`, `Ctrl_1A2_zoom_audit.png`,
  `Ctrl_1A2_overview.png`, `certified_qc.png`, five region crops, plus JSON
  threshold scans.

### Known failure modes recorded in the repository

From `HANDOFF.md` and `partial_label/README.md`:

1. **Labels are positive-unlabelled**; 74% of the training set asserts
   "background" over demonstrable RBCs (`HANDOFF.md:65-86`).
   The baseline model therefore converges to the annotation ceiling and "misses a
   lot of red blood cells in the tissue" (`HANDOFF.md:28-30`).
2. **Cluster merging** in every partial-label model: adjacent RBCs merge into
   blobs that swallow the inter-cell plasma; area rises 58% while object count
   falls and median component size doubles. Structural, not a thresholding
   artifact (`HANDOFF.md:204-228`).
3. **Tubule smear:** the model spreads onto eosinophilic tubular cytoplasm and
   brush border (`partial_label/README.md:140-145`).
   Shape does not separate these false positives from RBCs (median 66 px vs
   209 px, i.e. FPs are *smaller and more compact*), and colour barely does
   (R/B medians 1.235 vs 1.264) (`HANDOFF.md:259-268`).
4. **`MIN_RBC_PX = 30` is too small** — it assumes 0.5 µm/px, but at the true
   0.4409 µm/px one RBC is 198 px, so the floor admits objects at 15% of a cell
   (`HANDOFF.md:398-399`).
5. **`predict_tif.py:617` prints a wrong RBC area fraction** — it is computed
   from per-tile pre-blending probabilities, so it double-counts overlaps and
   ignores the size filter, and disagrees with the saved TIFF. Fixed in
   `predict_partial.py`, never fixed in the original
   (`HANDOFF.md:395-397`). **Do not quote the number that
   `predict_tif.py` prints**; use the mask.
6. **Everything is in-sample** — no held-out slide
   (`HANDOFF.md:370-375`).
7. Metric pathologies in the experimental branch: F-beta selection is biased
   conservative; "rbc-sized" selection rewards the merging failure mode; the
   slide-level tubule probe is blind to merging
   (`HANDOFF.md:231-279`).

---

## 7. Reproducibility details

### Environment files

**None.** `Not found in repository`: no `requirements.txt`, `environment.yml`,
`pyproject.toml`, `Dockerfile`, or lock file anywhere under `rbcs/`.

### Environment actually used

Conda environment `misc` at `/home/caleb/anaconda3/envs/misc`
(`HANDOFF.md:33-40`,
`partial_label/README.md:98`). Versions **queried
from that environment during this audit** (September 2026 state — the April 2026
training run may have used earlier versions):

| Package | Version |
|---|---|
| Python | 3.9.23 |
| torch | 2.7.1+cu126 |
| torchvision | 0.22.1+cu126 |
| segmentation-models-pytorch | 0.5.0 |
| albumentations | 2.0.8 |
| opencv (cv2) | 4.13.0 |
| numpy | 2.0.2 |
| scikit-image | 0.24.0 |
| scipy | 1.13.1 |
| tifffile | 2024.8.30 |
| shapely | 2.0.7 |
| Pillow | 11.2.1 |
| matplotlib | 3.9.4 |
| napari | 0.5.6 |
| torchstain | 1.3.0 |
| openslide-python | 1.4.2 |

Hardware: 2 × NVIDIA RTX 6000 Ada Generation (49,140 MiB each), driver
550.144.03, CUDA 12.6 runtime.

### Random seeds

| Seed | Where |
|---|---|
| 42 | `train_rbc_unet.py:44` — `torch.manual_seed`, `np.random.seed`, train/val split RNG |
| 42 | `tiff/extract_tiles_tiff.py:58` — tile sampling |
| 42 | `extract_tiles.py:29` — NDPI tile sampling |
| 42 | `partial_label/config.py:109` — partial-label training |
| 42 | `partial_label/harvest_pseudo.py` `--seed` default |

Note: dataloader worker seeding, cuDNN determinism flags and
`torch.use_deterministic_algorithms` are **not** set, so runs are not
bit-reproducible even with the seed fixed.

### Git

The parent repository `/nvme/caleb/3D` is at commit
**`5e53626b3fd917c8b0424849d8ee634daf00b842`** ("Initial commit"), but
**`rbcs/` is entirely untracked** (`git status` reports `?? rbcs/`). There is
therefore **no version history for any file described here**, and no commit hash
that identifies the state of the code that produced the masks. Timestamps are
the only provenance: scripts and outputs from 20–29 April 2026 (baseline), 13–14
August 2026 (partial-label branch). `train_partial.py` records `git_rev()` in
its checkpoints, which returns the parent-repo hash and is not informative about
`rbcs/` (`partial_label/train_partial.py:43-50`).
`HANDOFF.md:390-393` flags the untracked state as an outstanding issue.

> **Update (October 2026):** the folder was moved out of the `3D` repository to
> `/nvme/caleb/rbcs` and is now its own git repository. The audit above describes
> the state before that move; the October 2026 initial commit is the first
> versioned snapshot of this code.

### Key file paths

| Artifact | Path |
|---|---|
| Final model weights | `checkpoints/best_rbc_unet.pth` (sha256 `9ea753c7e8ec2ed5167b829644c6d894cf6ad64e25bf195b1787a6b4cbecf9c9`) |
| Final whole-slide masks | `unet_predictions_tiff/{Ctrl_1A2,Ctrl_1A4,IR_2A2,IR_2A4}_rbc_mask.{tif,geojson,meta.json}` |
| Training patches | `training_data_tiff_combined/{images,masks}/` + `corrected_patches_combined.json` |
| Classical bootstrap masks | `qc/rbc_qc_out_tiff_{ctrl,aki,ctrl_1A4,aki_2A4}/*_rbc_mask.tif` |
| Validation overlays | `checkpoints/val_samples/` |
| Experimental checkpoints | `partial_label/checkpoints/best_rbc_*.pth` (11 runs) + `history_*.json` |
| Experimental deliverable | `partial_label/predictions_A_boxes/Ctrl_1A2_rbc_mask.geojson` (6,224 objects) |
| Source slides | `/home/caleb/Desktop/histology_sdd/data/<sample>/outs/spatial/<sample>.tif` |
| Scalefactors | `/home/caleb/Desktop/histology_sdd/data/<sample>/outs/binned_outputs/square_002um/spatial/scalefactors_json.json` |

**Broken paths:** every script in efforts A and B hard-codes
`/home/caleb/Desktop/rbcs/...` (a working symlink to this folder) and
`/home/caleb/Desktop/3D/data/kidney_rabb_visiumhd/...` (**no longer exists** —
verified). The intermediate tile directories (`rbc_tiles_tiff_*`) and their
`tiles_log.csv` coordinate logs are therefore **not recoverable**, although the
tile origins survive inside the patch filenames and the classical masks survive
in `qc/`. `correct_masks.py:33-34` points at `training_data/`, a directory that
no longer exists (superseded by `training_data_tiff*`).

### Commands to reproduce

The scripts in efforts A and B take their inputs from module-level constants
that were **edited between runs** rather than from CLI arguments, so exact
invocations are `Not found in repository` (no shell history, no Makefile, no
run log for the baseline). Reconstructed sequence:

```bash
PY=/home/caleb/anaconda3/envs/misc/bin/python
cd /nvme/caleb/rbcs

# (1) tile extraction — edit ACTIVE = "ctrl" | "aki" and the four path
#     constants (Ctrl_1A2/IR_2A2 for round 1, Ctrl_1A4/IR_2A4 for round 2)
$PY tiff/extract_tiles_tiff.py

# (2) classical bootstrap masks — edit ACTIVE and the tile/out dirs
$PY tiff/rbc_segmentation_mask_tiff.py

# (3) 512x512 patches (edit CTRL_/AKI_ dirs + OUT_BASE per round)
$PY tiff/prepare_training_data_tiff.py

# (4) manual correction in napari (requires a display / ssh -X)
$PY correct_masks.py                      # edit TRAINING_DATA + LOG_PATH first
# optional: $PY clean_training_data.py    # edit MASK_DIR first

# (5) merge the two rounds
$PY combine_training_data.py

# (6) training  (add --corrected_only to train on the 190 corrected patches)
$PY train_rbc_unet.py

# (7) whole-slide inference — edit sample_id at predict_tif.py:41, or override:
for S in Ctrl_1A2 Ctrl_1A4 IR_2A2 IR_2A4; do
  $PY predict_tif.py \
    --slide        /home/caleb/Desktop/histology_sdd/data/$S/outs/spatial/$S.tif \
    --scalefactors /home/caleb/Desktop/histology_sdd/data/$S/outs/binned_outputs/square_002um/spatial/scalefactors_json.json \
    --ckpt         checkpoints/best_rbc_unet.pth \
    --out          unet_predictions_tiff/${S}_rbc_mask.tif \
    --threshold    0.2 --tile_size 2048 --overlap 256 --min_rbc_px 30
done
```

**Evaluation:** there is no evaluation script for the baseline. The
partial-label branch's comparison tooling can be pointed at the baseline masks
(`HANDOFF.md:331-339`):

```bash
cd /nvme/caleb/rbcs/partial_label   # scripts import config.py as a sibling
$PY compare_masks.py --sample Ctrl_1A2
$PY compare_crops.py --sample Ctrl_1A2 --crop 512 --n_tubule 2 --n_rich 2 \
  --mask baseline=../unet_predictions_tiff/Ctrl_1A2_rbc_mask.tif \
  --mask A_refined=predictions_A_boxes/Ctrl_1A2_rbc_mask.tif \
  --out comparison/whatever.png
```

Experimental branch (for completeness; `RBC_SLIDE_DIR` overrides the slide root):

```bash
cd /nvme/caleb/rbcs/partial_label
$PY harvest_pseudo.py --per_slide 400 --exclude Ctrl_1A2 --seed 42
$PY train_partial.py --epochs 60 --tag A_boxes --neg_trust_uncorrected 0.6 \
    --oversample_corrected 3 --use_pseudo --pseudo_neg_trust 0.4 --tversky_alpha 0.3
$PY predict_partial.py --sample Ctrl_1A2 --ckpt checkpoints/best_rbc_A_boxes.pth --save_prob
$PY retune_threshold.py --prob predictions_A_boxes/Ctrl_1A2_rbc_prob.tif \
    --red_min 1.10 --min_rbc_px 150 --write 0.90
```

---

## 8. Methods-section draft

### Red blood cell segmentation from histology images

Red blood cells (RBCs) were segmented from four H&E-stained rabbit kidney
sections imaged as part of a 10x Genomics Visium HD experiment (two control
sections, `Ctrl_1A2` and `Ctrl_1A4`; two ischemia–reperfusion injury sections,
`IR_2A2` and `IR_2A4`). Brightfield whole-section images were supplied as flat
RGB TIFFs of 30,336–32,096 × 17,160–18,048 pixels at 0.4407–0.4409 µm per pixel
(microns per pixel read from each sample's Visium HD `scalefactors_json.json`).
All processing was performed at a common working resolution of 0.5 µm per pixel
(≈20× equivalent); images were resampled between native and working resolution
with Lanczos (crops) or bilinear (probability maps) interpolation.

**Training data.** Because no exhaustive manual annotation was available, a
rule-based procedure was used to bootstrap labels. Twelve 2048 × 2048 pixel
fields (at 0.5 µm/px) were sampled from each section from tissue-positive regions
identified by Otsu thresholding of a downsampled greyscale thumbnail, followed by
binary closing (disk radius 3) and removal of objects smaller than 200 thumbnail
pixels; fields were required to be at least 60% tissue and to be separated by at
least half a field width (random seed 42). Each field was normalized to a
reference field with the Macenko method (torchstain) and decomposed into
haematoxylin and eosin channels by colour deconvolution (`skimage.color.rgb2hed`),
each channel rescaled per field between its 1st and 99th percentiles. A pixel was
labelled RBC when the eosin channel exceeded 0.8, the haematoxylin channel was
below 0.25, the red-to-blue intensity ratio exceeded 1.15, and the minimum
channel intensity was below 220 (excluding white background); the resulting mask
was cleaned with a single binary opening (disk radius 1). Fields were then cut
into non-overlapping 512 × 512 pixel patches, discarding patches whose minimum
per-channel mean exceeded 220. This produced 739 labelled patches across all four
sections. Because the rule-based mask has high precision but low recall, 190 of
these patches (25.7%) were subsequently reviewed and corrected by hand in napari,
where a human painted in RBCs the rule-based mask had missed; within the same
field, corrected patches contained a median of approximately 40-fold more RBC
pixels than uncorrected ones, and corrections were almost entirely additions
(506,118 pixels added versus 47,159 removed). The remaining 549 patches retain
their rule-based labels, so the training labels are best described as
positive-unlabelled: labelled RBC pixels are reliable, whereas unlabelled pixels
are not confirmed background.

**Model and training.** A U-Net (segmentation-models-pytorch) with an
EfficientNet-B0 encoder initialised from ImageNet weights was trained for binary
semantic segmentation (3 input channels, 1 output channel, 6.25 M parameters, all
layers trainable). Patches were split at the field level to prevent leakage
between training and validation, holding out 5 of 48 fields (660 training and 79
validation patches). Training used random 256 × 256 pixel crops with random
horizontal and vertical flips and 90° rotations (each p = 0.5), colour jitter
(brightness, contrast and saturation 0.8–1.2, hue ±0.1; p = 0.5), hue–saturation–
value shifts (p = 0.5), RGB channel shifts (p = 0.3), Gaussian blur (kernel 3–7,
sigma 0.1–2.0; p = 0.3) and Gaussian noise (standard deviation 0.02–0.1; p = 0.3),
followed by ImageNet channel normalization; validation used a central 256 × 256
crop with normalization only. The loss was the unweighted mean of soft Dice loss
and binary cross-entropy (positive-class weight 1). Optimization used AdamW
(learning rate 1 × 10⁻³, weight decay 1 × 10⁻⁴) with cosine-annealing of the
learning rate over 50 epochs, batch size 16, and automatic mixed precision, on a
single NVIDIA RTX 6000 Ada GPU. The random seed was 42. The checkpoint with the
lowest validation loss was retained (epoch 25 of 50; validation loss 0.3498).

**Whole-slide inference.** Each section was processed with the selected
checkpoint. The section was traversed in 2048 × 2048 pixel tiles at 0.5 µm/px
(1806 native pixels) with 256 pixel overlap; tiles whose minimum per-channel mean
exceeded 220 were skipped as background. Within each tile the network was applied
in a 512 × 512 pixel sliding window with 128 pixel overlap (batch size 1, no
test-time augmentation), and window predictions were combined by two-dimensional
Hanning-weighted averaging. Tile probability maps were resampled to native
resolution and blended into a whole-section probability accumulator with the same
Hanning weighting, held in a memory-mapped float32 array. The normalized
probability map was thresholded at 0.2 and 8-connected components smaller than 30
pixels were removed; no hole filling, morphological smoothing or watershed
splitting was applied. Masks were written as tiled, zlib-compressed BigTIFFs at
native resolution together with QuPath-compatible GeoJSON polygons (contours
simplified with a 0.5 pixel Douglas–Peucker tolerance, repaired with a zero-width
buffer and capped at 500 vertices per object).

**Quantification.** RBC burden is reported as the fraction of segmented tissue
area occupied by RBC pixels, where tissue was defined by Otsu thresholding of a
downsampled greyscale image of the section; fraction of tissue rather than
fraction of image is used because tissue occupies 28.2–34.1% of the image
bounding box and this fraction differs between sections. Object counts are
8-connected components of the final mask. Across the four sections the model
segmented 1,588,815–2,321,725 RBC pixels per section, corresponding to 0.90–1.50%
of tissue area and 6,899–8,925 objects, with median object areas of 119–154 pixels
against a nominal single-RBC area of ~198 pixels (7 µm diameter at 0.441 µm/px).

**Validation and limitations.** [No independent ground-truth annotation was
available; segmentation accuracy was not quantified against a reference standard.]
Model selection used the validation loss computed against the same partially
corrected labels used for training, and segmentation quality was additionally
assessed by visual inspection of contour overlays on validation patches and on
full-resolution fields from each section. [Dice, IoU, precision, recall and
object-level counting error not computed.] All four sections contributed patches
to training, so no section is fully held out and reported performance should be
regarded as in-sample. Because the rule-based labels under-call RBCs, the trained
model is expected to inherit a conservative bias and to miss faintly stained
cells. An exploratory retraining under a partial-label (positive-unlabelled)
objective increased recovered RBC area on one section (`Ctrl_1A2`) from 1.37% to
1.51–2.17% of tissue, but merged adjacent cells into larger objects (object count
falling from 8,130 to 6,224–6,820 while median object area rose from 151 to
261–289 pixels) and was therefore not used; the model described above was retained
for all reported results. [Inter-observer variability of the manual corrections
not assessed; number of annotators not recorded.]

*Placeholders to fill in from outside the repository:* [section thickness],
[staining protocol and scanner model], [animal numbers and study design],
[number of annotators and their expertise], [training wall-clock time].

---

## 9. Evidence table

| Detail | Value | Source file | Notes |
|---|---|---|---|
| Number of slides | 4 | `HANDOFF.md:20-21`; `unet_predictions_tiff/` (4 mask sets) | Ctrl_1A2, Ctrl_1A4, IR_2A2, IR_2A4 |
| Sample groups | 2 control, 2 ischemia–reperfusion (AKI) | `HANDOFF.md:19-21` | rabbit kidney, Visium HD |
| Modality | Brightfield H&E | `HANDOFF.md:19`; `rbc_segmentation_mask.py:74-79` (`rgb2hed`) | — |
| Image format | flat uint8 RGB TIFF, uncompressed, no pyramid | verified with `tifffile` during audit | `predict_tif.py:180-192` memory-maps it |
| Slide dimensions | 32096×18048, 30592×17160, 31688×17880, 30336×17160 | `unet_predictions_tiff/*_rbc_mask.meta.json` (`slide_w0/h0`) | re-verified against source TIFFs |
| Pixel size | 0.4408725, 0.4407405, 0.4408513, 0.4408973 µm/px | `scalefactors_json.json` → `predict_tif.py:172-177` | Ctrl_1A2 value also in `HANDOFF.md:59` |
| Working resolution | 0.5 µm/px (`TARGET_MPP`) | `predict_tif.py:58`; `tiff/extract_tiles_tiff.py:54` | "20× equivalent" |
| One RBC ≈ 198 px | `rbc_area_px = 197.9976` | `partial_label/comparison/Ctrl_1A2_all_models.json`; `HANDOFF.md:59-61` | 7 µm at 0.4409 µm/px |
| Tiles per slide | 12 (`N_TILES`) | `tiff/extract_tiles_tiff.py:56` | 12 masks per `qc/` dir confirms |
| Tile size | 2048 px at 0.5 µm/px (1806 native) | `tiff/extract_tiles_tiff.py:55`, `:196` | — |
| Tissue detection for tiling | Otsu + closing(disk 3) + remove_small_objects(200); ≥60% tissue | `tiff/extract_tiles_tiff.py:99-105`, `:57` | — |
| Tile sampling seed | 42 | `tiff/extract_tiles_tiff.py:58` | — |
| Stain normalization (labels) | Macenko, torchstain, reference = first tile | `rbc_segmentation_mask.py:44-52`, `:288` | fallback to raw on failure, `:150-153` |
| Stain normalization (CNN) | none | `train_rbc_unet.py:243`; `predict_tif.py:274` | ImageNet normalization only |
| Classical rule | `En>E_THRESH & Hn<0.25 & R/B>1.15 & min(RGB)<220` | `rbc_segmentation_mask.py:82-91` | — |
| `E_THRESH` (TIFF slides) | **0.8** | `tiff/rbc_segmentation_mask_tiff.py:34` | used for the final training labels |
| `E_THRESH` (NDPI round) | 0.9 | `rbc_segmentation_mask.py:33` | conflicting value; earlier round only |
| `E_THRESH` (QC figure) | 0.45 | `rbc_segmentation_mask.py:206` | figure disagrees with the saved mask |
| Classical morphology | opening disk(1); size/hole filters are no-ops | `rbc_segmentation_mask.py:38-40`, `:94-101` | all diameters set to 0 |
| Patch size / stride (final) | 512 / 512 | `tiff/prepare_training_data_tiff.py:37-38` | non-overlapping |
| Patch size / stride (NDPI round) | 512 / 256 | `prepare_training_data.py:27-28` | conflicting; unused for final model |
| Background patch rejection | min per-channel mean > 220 | `tiff/prepare_training_data_tiff.py:39`, `:56-58` | — |
| Total training patches | 739 | `training_data_tiff_combined/images` (counted) | 365 (round 1) + 374 (round 2) |
| Human-corrected patches | 190 | `training_data_tiff_combined/corrected_patches_combined.json` (counted) | 115 + 75 |
| Uncorrected patches | 549 | 739 − 190 | `HANDOFF.md:69-73` |
| Label incompleteness | corrected patches 1.189% RBC px vs 0.156%; median 40× within tile | `HANDOFF.md:67-77`; `partial_label/README.md:22-29` | 34/34 tiles |
| Corrections were additions | 506,118 px added vs 47,159 removed | `HANDOFF.md:90-96` | no erasure store to mine |
| Annotation tool | napari, keys n/s/q; masks 0/255 PNG | `correct_masks.py:49-89`, `:133-138` | log path in `correct_masks.py:34` is stale |
| Round-1 slide identity | `Ctrl_1A2` (ctrl) and `IR_2A2` (aki) | verified by re-cropping source TIFFs during audit | mean abs diff 0.000 vs 28–43 for others |
| Round-2 slide identity | `Ctrl_1A4`, `IR_2A4` | `tiff/extract_tiles_tiff.py:29-49` | directory names self-identify |
| Train/val split | tile-level, 5 of 48 tiles held out → 660 / 79 patches | `train_rbc_unet.py:174-215`; reproduced during audit | seed 42; deterministic |
| `VAL_FRAC` | 0.10 | `train_rbc_unet.py:43` | docstring `:7` and `unet_architecture.txt:126` say "85/15" — code wins |
| Group stratification | fails silently (no "aki" substring in tile keys) | `train_rbc_unet.py:190-193` | all 48 tiles treated as control |
| Test set | none | — | `Not found in repository` |
| Slide held out | none — all 4 slides in training | `HANDOFF.md:370-375` | results are in-sample |
| Model class | `smp.Unet` | `train_rbc_unet.py:392-396` | smp 0.5.0 |
| Encoder | EfficientNet-B0, ImageNet weights | `train_rbc_unet.py:393-394` | `unet_architecture.txt:4` |
| Decoder | 5 U-Net blocks, bilinear upsample + skip concat + Conv-BN-ReLU×2 | `unet_architecture.txt:38-71` | 256→128→64→32→16 ch |
| Head / final activation | Conv2d(16→1, k3); sigmoid at inference | `unet_architecture.txt:77-79`; `predict_tif.py:279` | raw logits from the model |
| Parameters | 6,251,469 | `unet_architecture.txt:11-14` | encoder 4.01 M / decoder 2.24 M |
| Dropout | none | `train_rbc_unet.py:392-396` | no `aux_params` |
| Frozen layers | none | `train_rbc_unet.py:405` | all params in optimizer |
| Loss | 0.5·Dice + 0.5·BCE | `train_rbc_unet.py:128-145` | — |
| `pos_weight` | 1 | `train_rbc_unet.py:45`, `:404`; `unet_architecture.txt:101` | docstring `:5` says 10 — **not** what ran |
| Train crop | 256×256 | `train_rbc_unet.py:38`, `:221` | inference runs at 512 — see `partial_label/config.py:59-61` |
| Batch size | 16 | `train_rbc_unet.py:38` | — |
| Epochs | 50 | `train_rbc_unet.py:40` | — |
| Optimizer | AdamW | `train_rbc_unet.py:405` | — |
| Learning rate | 1e-3 | `train_rbc_unet.py:41` | — |
| Weight decay | 1e-4 | `train_rbc_unet.py:42` | — |
| Scheduler | CosineAnnealingLR, T_max=50 | `train_rbc_unet.py:406` | stepped per epoch, `:419` |
| AMP | autocast + GradScaler | `train_rbc_unet.py:268`, `:407` | — |
| `torch.compile` | enabled if torch ≥ 2 | `train_rbc_unet.py:397-402` | prefix stripped on save, `:427-430` |
| Augmentations | flips, rot90, ColorJitter, GaussianBlur, GaussNoise, HSV, RGBShift | `train_rbc_unet.py:220-245` | no scaling, no elastic, no stain aug |
| Training seed | 42 | `train_rbc_unet.py:44`, `:340-341` | no cuDNN determinism flags |
| Checkpoint criterion | minimum validation loss | `train_rbc_unet.py:424` | — |
| Selected epoch / val loss | epoch 25, 0.3497944176197052 | `checkpoints/best_rbc_unet.pth` (read during audit) | — |
| Checkpoint sha256 | `9ea753c7e8ec2ed5167b829644c6d894cf6ad64e25bf195b1787a6b4cbecf9c9` | computed during audit | 72,240,575 bytes, dated Apr 29 12:51 |
| Inference script | `predict_tif.py` | `predict_tif.py:41` (`sample_id` edited per run) | — |
| Inference tile / overlap | 2048 / 256 px at 0.5 µm/px | `predict_tif.py:56-57` | native 1806 / 226 px |
| Sliding window / overlap | 512 / 128 px | `predict_tif.py:238-239` | step 384 |
| Inference batch size | 1 | `predict_tif.py:276-278` | one patch per forward pass |
| Stitching | 2-D Hanning weights at patch and tile level | `predict_tif.py:264-265`, `:548-550` | `prob_acc / weight_acc` |
| TTA | not used | `predict_tif.py:571-572` (call commented out) | 8-way function exists at `:287-326` |
| Threshold | 0.2 | `predict_tif.py:55` | not recorded in the mask sidecars |
| Threshold corroboration | "value taken from predict_tif.py defaults at the time the mask was written" | `partial_label/comparison/Ctrl_1A2_summary.json` | audit by the later branch |
| Conflicting thresholds | 0.1 / 0.2 / 0.5 | `predict_rbc.py:34`; `predict_wsi.py:49`; `unet_architecture.txt:136` | other scripts, not used for the 4 masks |
| Min object size | 30 px, 8-connected | `predict_tif.py:60`, `:140-150` | too small at 0.441 µm/px, `HANDOFF.md:398` |
| Other postprocessing | none (no fill, no morphology, no watershed) | `predict_tif.py:440-453` | — |
| Outputs per slide | `_rbc_mask.tif`, `.geojson`, `.meta.json` | `unet_predictions_tiff/`; `predict_tif.py:455-498` | BigTIFF, tiles 256², zlib-6 |
| GeoJSON polygonization | approxPolyDP(0.5) → Shapely buffer(0) → ≤500 vertices | `predict_tif.py:346-388` | class "RBC", colorRGB −65536 |
| Objects per slide | 8,130 / 8,520 / 8,925 / 6,899 | counted from `unet_predictions_tiff/*.geojson` and confirmed by component analysis | Ctrl_1A2 matches `Ctrl_1A2_summary.json` |
| RBC px per slide | 2,238,324 / 2,321,725 / 2,128,567 / 1,588,815 | computed from the mask TIFFs during audit | Ctrl_1A2 matches `Ctrl_1A2_summary.json` |
| RBC % of tissue | 1.372 / 1.498 / 1.162 / 0.896 | computed during audit (Otsu tissue mask) | Ctrl_1A2 value matches the repo exactly |
| Tissue % of slide | 28.17 / 29.51 / 32.34 / 34.06 | computed during audit; Ctrl_1A4/IR_2A2/IR_2A4 also in `partial_label/harvest.log` | Ctrl_1A2 in `Ctrl_1A2_summary.json` |
| Median object size | 151 / 154 / 124 / 119 px | computed during audit | vs ~198 px per RBC |
| Metrics vs ground truth | none computed | — | `Not found in repository` |
| QC overlays (training) | 4 validation overlays at threshold 0.5 | `train_rbc_unet.py:297-328`; `checkpoints/val_samples/` | — |
| QC figures (classical) | 6 dirs × 12 tiles × 5 files | `qc/`; `rbc_segmentation_mask.py:234-276` | mask, QC, diagnostic, full-res, overlay |
| Failure mode: label ceiling | 74% of the set asserts background over real RBCs | `HANDOFF.md:74-86` | root cause `E_THRESH` |
| Failure mode: cluster merging | area +58%, objects 8,130→~6,700, median 151→~290 px | `HANDOFF.md:204-216` | partial-label models only |
| Failure mode: tubule smear | model spreads onto eosinophilic tubular cytoplasm | `partial_label/README.md:140-143` | shape/colour do not separate, `HANDOFF.md:259-268` |
| Bug: printed area fraction | pre-blend, double-counts overlaps, ignores size filter | `predict_tif.py:617`; `HANDOFF.md:395-397` | use the mask, not the printout |
| Environment | conda `misc`, Python 3.9.23, torch 2.7.1+cu126, smp 0.5.0, albumentations 2.0.8 | `HANDOFF.md:33-40`; queried during audit | current state, not necessarily April 2026 |
| Hardware | 2 × RTX 6000 Ada 48 GB, driver 550.144.03 | `HANDOFF.md:42`; `nvidia-smi` during audit | docstring says CUDA 12.4; env has 12.6 |
| Environment file | none | `find` over the folder | `Not found in repository` |
| Git provenance | parent repo at `5e53626b…`; `rbcs/` untracked | `git status`, `git rev-parse` | `HANDOFF.md:390-393` |
| Notebooks | none | `find` over the folder | `Not found in repository` |
| Config files (main pipeline) | none | — | only `partial_label/config.py` exists |
| Exact run commands | not recorded | — | `Not found in repository`; constants edited per run |
| Partial-label loss | Tversky α 0.3 / β 0.7, neg_smooth 0.03, drop_frac 0.05 | `partial_label/config.py:98-101`; `partial_label/losses.py:33-116` | experimental branch |
| Partial-label trust | corrected 1.0, uncorrected 0.08–0.60, pseudo 0.40–0.75 | `partial_label/config.py:74-95`; `HANDOFF.md:121-127` | experimental branch |
| Partial-label training | 60 epochs, batch 12, AdamW 1e-3/1e-4, cosine, seed 42, patch 512 | `partial_label/config.py:61, 104-109` | 11 checkpoints in `partial_label/checkpoints/` |
| Pseudo-labels | 1,200 patches (400 each from Ctrl_1A4, IR_2A2, IR_2A4); Ctrl_1A2 excluded | `partial_label/harvest.log`; `partial_label/harvest_pseudo.py:53-54` | train-only, never validation |
| Certified negatives | human 79 boxes, mined 418, mined_pseudo 738 | `HANDOFF.md:153-157`; counted on disk | human boxes: 1.98 M px, no RBCs inside |
| Partial-label deliverable | `predictions_A_boxes/Ctrl_1A2_rbc_mask.geojson`, 6,224 objects, thr 0.90 + R/B>1.10 + 150 px | `partial_label/predictions_A_boxes/Ctrl_1A2_rbc_mask.meta.json` | 1.509% of tissue |
| Partial-label conclusion | no new model beats the baseline on object fidelity | `HANDOFF.md:344-355`, `:346-350` | use baseline for counts |
| UNI backbone variant | code exists, no checkpoint, no outputs | `train_rbc_uni.py:42`; `predict_rbc_uni.py` | exploratory, unused |
| NDPI round | separate tiles/masks/predictions | `extract_tiles.py`, `predict_wsi.py`, `training_data_ndpi/`, `unet_predictions_ndpi/` | superseded by the TIFF pipeline |
