# RBC segmentation in rabbit kidney H&E (Visium HD)

Semantic segmentation of red blood cells in four H&E-stained rabbit kidney
sections from a Visium HD ischemia–reperfusion study: two control
(`Ctrl_1A2`, `Ctrl_1A4`) and two ischemia–reperfusion (`IR_2A2`, `IR_2A4`).

The model is a U-Net with an ImageNet-pretrained EfficientNet-B0 encoder
(segmentation-models-pytorch), trained on rule-based labels that were partly
corrected by hand. Full methods, parameters and caveats are in
[methods_segmentation_details.md](methods_segmentation_details.md).

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1 | `tiff/extract_tiles_tiff.py` | 12 tiles per slide, 2048 × 2048 at 0.5 µm/px |
| 2 | `tiff/rbc_segmentation_mask_tiff.py` | rule-based starting labels (Macenko + colour deconvolution + thresholds) |
| 3 | `tiff/prepare_training_data_tiff.py` | 512 × 512 training patches |
| 4 | `correct_masks.py` | hand-corrected masks in napari (190 of 739 patches) |
| 5 | `combine_training_data.py` | `training_data_tiff_combined/` (both acquisition rounds merged) |
| 6 | `train_rbc_unet.py` | `checkpoints/best_rbc_unet.pth` (best validation loss, epoch 25 of 50) |
| 7 | `predict_tif.py` | whole-slide mask per slide: BigTIFF + QuPath GeoJSON + `meta.json` |

Steps 1–5 were run twice, once per acquisition round (`Ctrl_1A2` + `IR_2A2`,
then `Ctrl_1A4` + `IR_2A4`), by editing the path constants at the top of each
script. Those constants still hold the values from the last run.

`training_data_tiff_combined/corrected_patches_combined.json` lists which
patches were hand-corrected. `unet_predictions_tiff/*_rbc_mask.meta.json`
record the slide dimensions of each final mask.

## Data and weights (not in git)

The training patches, model weights and final masks are too large for git:

- `training_data_tiff_combined/{images,masks}/`: 739 patch pairs (~341 MB)
- `checkpoints/best_rbc_unet.pth`: final model (~72 MB)
- `unet_predictions_tiff/<sample>_rbc_mask.{tif,geojson}`: final masks (~26 MB)

Source slides are the Visium HD images
`<sample>/outs/spatial/<sample>.tif`, with pixel size read from
`<sample>/outs/binned_outputs/square_002um/spatial/scalefactors_json.json`.

## Running inference

```bash
python predict_tif.py \
  --slide        /path/to/<sample>/outs/spatial/<sample>.tif \
  --scalefactors /path/to/<sample>/outs/binned_outputs/square_002um/spatial/scalefactors_json.json \
  --ckpt         checkpoints/best_rbc_unet.pth \
  --out          unet_predictions_tiff/<sample>_rbc_mask.tif \
  --threshold 0.2 --tile_size 2048 --overlap 256 --min_rbc_px 30
```

Environment: Python 3.9, torch 2.7, segmentation-models-pytorch 0.5.0,
albumentations 2.0.8, opencv, tifffile, shapely, scikit-image, torchstain
(step 2) and napari (step 4).

## Caveats

- All four slides contributed training patches, so whole-slide results are
  in-sample. No slide was held out.
- Labels are positive-unlabelled: uncorrected patches under-call RBCs, so the
  model is conservative and misses faint cells.
- No accuracy metric was computed against independent ground truth.
- Report RBC burden as a fraction of tissue area, not of the image, and compute
  it from the saved mask. The area fraction that `predict_tif.py` prints is
  wrong (it counts tile overlaps twice and ignores the size filter).
