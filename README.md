# GeoAI-UrbanTree-RFDETR

Reproducible RF-DETR pipeline for urban tree detection from high-resolution geospatial imagery.

The repository supports:

- GeoTIFF + vector-label preparation to COCO format
- Standard train / validation / test training
- 5-fold cross-validation
- COCO AP / AR evaluation
- Precision / recall / F1 evaluation
- Transfer-city evaluation
- Image-level prediction
- Large-area GeoTIFF inference
- Export of geospatial predictions to JSON, GeoJSON, and Shapefile

The pipeline can be used either to reproduce the urban-tree experiments in this repository or to train RF-DETR on a user's own compatible dataset.

---

## Repository Structure

```text
GeoAI-UrbanTree-RFDETR/
├── configs/
│   ├── train.yaml
│   ├── cross_validation.yaml
│   ├── evaluate.yaml
│   ├── inference.yaml
│   └── dataset.yaml
│
├── datasets/
│   ├── geotiff_rgbn/
│   ├── labels/
│   └── coco/
│
├── outputs/
│
├── scripts/
│   ├── train.py
│   ├── cross_validate.py
│   ├── evaluate.py
│   ├── infer_geotiff.py
│   └── prepare_dataset.py
│
├── src/
│   └── geoai_tree/
│       ├── data/
│       ├── training/
│       ├── evaluation/
│       ├── inference/
│       ├── visualization/
│       └── utils/
│
├── weights/
│   ├── tvt/
│   └── cv/
│
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Installation

### Option 1: Conda environment

```bash
conda env create -f environment.yml
conda activate geoai-urbantree-rfdetr
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

The repository was tested with:

```text
Python 3.10.19
RF-DETR 1.5.0rc1
PyTorch 2.10.0
Torchvision 0.25.0
```

---

## Data Preparation

The repository can convert geospatial imagery and vector tree labels into a COCO object-detection dataset.

Example input:

```text
datasets/
├── geotiff_rgbn/
│   └── new_york_030/
│       ├── image_001.tif
│       ├── image_002.tif
│       └── ...
│
└── labels/
    └── new_york_030/
        ├── labels_new_york_030.shp
        ├── labels_new_york_030.shx
        ├── labels_new_york_030.dbf
        ├── labels_new_york_030.prj
        └── labels_new_york_030.cpg
```

Configure:

```text
configs/dataset.yaml
```

Then run:

```bash
PYTHONPATH=src python scripts/prepare_dataset.py \
  --config configs/dataset.yaml
```

The output follows the RF-DETR-compatible COCO structure:

```text
output_directory/
└── test/
    ├── images/
    ├── _annotations.coco.json
    └── preparation_summary.json
```

The preparation pipeline supports:

- multiple GeoTIFF files
- configurable RGB bands
- configurable tile size
- vector-label reprojection to raster CRS
- spatial-index label lookup
- geometry clipping
- geospatial-to-pixel bounding-box conversion
- minimum bounding-box area filtering
- optional empty-tile removal
- optional partial-edge tiles

---

## Using Your Own Dataset

Users can train RF-DETR on their own object-detection dataset.

The expected COCO structure is:

```text
dataset_root/
├── train/
│   ├── images/
│   └── _annotations.coco.json
│
├── valid/
│   ├── images/
│   └── _annotations.coco.json
│
└── test/
    ├── images/
    └── _annotations.coco.json
```

Update the dataset path in:

```text
configs/train.yaml
```

For example:

```yaml
dataset:
  type: tvt
  root: datasets/coco/my_dataset
  annotation_file: _annotations.coco.json
```

Then train with the same pipeline:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/train.yaml
```

---

## Standard Train / Validation / Test Training

Training settings are defined in:

```text
configs/train.yaml
```

Run:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/train.yaml
```

Default training settings used in this repository include:

```yaml
training:
  epochs: 50
  batch_size: 8
  grad_accum_steps: 4
  learning_rate: 0.0001
  num_workers: 8
  checkpoint_interval: 10
  run_test_during_training: false
```

Training outputs are written under:

```text
outputs/training/
```

---

## 5-Fold Cross-Validation

Cross-validation settings are defined in:

```text
configs/cross_validation.yaml
```

Expected structure:

```text
datasets/coco/cv/
├── fold_1/
│   ├── train/
│   └── valid/
├── fold_2/
│   ├── train/
│   └── valid/
├── fold_3/
├── fold_4/
└── fold_5/
```

Train all folds:

```bash
PYTHONPATH=src python scripts/cross_validate.py \
  --config configs/cross_validation.yaml
```

Train one fold:

```bash
PYTHONPATH=src python scripts/cross_validate.py \
  --config configs/cross_validation.yaml \
  --fold 3
```

Train selected folds:

```bash
PYTHONPATH=src python scripts/cross_validate.py \
  --config configs/cross_validation.yaml \
  --folds 1 3 5
```

---

## Model Weights

The repository uses EMA checkpoints for released model weights.

Expected layout:

```text
weights/
├── tvt/
│   └── checkpoint_best_ema.pth
│
└── cv/
    ├── fold_1/
    │   └── checkpoint_best_ema.pth
    ├── fold_2/
    │   └── checkpoint_best_ema.pth
    ├── fold_3/
    │   └── checkpoint_best_ema.pth
    ├── fold_4/
    │   └── checkpoint_best_ema.pth
    └── fold_5/
        └── checkpoint_best_ema.pth
```

Large pretrained weights and datasets can be distributed separately through Zenodo.

---

## Evaluation

Evaluation settings are defined in:

```text
configs/evaluate.yaml
```

### Train / Validation / Test Evaluation

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/evaluate.yaml \
  --mode tvt
```

### Cross-Validation Evaluation

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/evaluate.yaml \
  --mode cv
```

### Transfer-City Evaluation

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/evaluate.yaml \
  --mode transfer
```

The evaluation pipeline reports:

- COCO AP50:95
- AP50
- AP75
- AR
- precision
- recall
- F1
- TP / FP / FN
- per-image metrics
- prediction JSON
- ground-truth / prediction visualizations

---

## Large-Area GeoTIFF Inference

Large-area inference settings are defined in:

```text
configs/inference.yaml
```

Run:

```bash
PYTHONPATH=src python scripts/infer_geotiff.py \
  --config configs/inference.yaml
```

The inference pipeline:

```text
GeoTIFF
  ↓
256 × 256 raster tiles
  ↓
RF-DETR prediction
  ↓
pixel-coordinate bounding boxes
  ↓
map-coordinate geometries
  ↓
JSON / GeoJSON / Shapefile
```

Example outputs:

```text
outputs/inference/
└── miami_030/
    └── full_aoi/
        ├── miami_030_rfdetr_predictions.json
        ├── miami_030_rfdetr_predictions.geojson
        ├── miami_030_rfdetr_predictions.shp
        └── summary.json
```

A test mode is also available in `configs/inference.yaml` for validating the inference pipeline on a small number of raster tiles before running the full area.

---

## Configuration

The repository uses YAML configuration files to separate experiment settings from source code.

```text
configs/
├── dataset.yaml
├── train.yaml
├── cross_validation.yaml
├── evaluate.yaml
└── inference.yaml
```

This allows dataset paths, model settings, thresholds, output locations, and training parameters to be changed without modifying the Python modules.

---

## Main Python Package

Reusable functionality is implemented under:

```text
src/geoai_tree/
```

The package is organized into:

```text
data/
    COCO utilities
    geospatial utilities
    dataset preparation
    validation

training/
    shared RF-DETR trainer
    cross-validation training

evaluation/
    COCO metrics
    detection metrics
    evaluation pipeline

inference/
    RF-DETR prediction
    raster inference

visualization/
    detection visualization

utils/
    configuration
    paths
    checkpoints
```

---

## Citation

If you use this repository in research, please cite the associated publication and software release.

Citation metadata is provided in:

```text
CITATION.cff
```

A DOI for the archived software, datasets, and model weights can be added here after the Zenodo release is created.

---

## License

See:

```text
LICENSE
```

for license information.
