# GeoAI-UrbanTree-RFDETR

Reproducible RF-DETR pipeline for urban tree detection from high-resolution geospatial imagery.

This repository supports the complete workflow from geospatial data preparation to model training, cross-validation, evaluation, transfer testing, and large-area GeoTIFF inference. It can be used either to reproduce the urban-tree experiments in this project or to train and evaluate RF-DETR on a user's own compatible dataset.

---

## Categories

- [Quick Start](#quick-start)
- [Supported Workflows](#supported-workflows)
- [Dataset Types](#dataset-types)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Commands](#commands)
- [Configuration Files](#configuration-files)
  - [Dataset Preparation](#1-dataset-preparation-configsdatasetyaml)
  - [TVT Training](#2-tvt-training-configstrainyaml)
  - [Cross-Validation](#3-cross-validation-configscross_validationyaml)
  - [Evaluation](#4-evaluation-configsevaluateyaml)
  - [Large-Area Inference](#5-large-area-inference-configsinferenceyaml)
- [Using Your Own Dataset](#using-your-own-dataset)
- [Model Weights](#model-weights)
- [Outputs](#outputs)
- [Python Package](#python-package)
- [Citation](#citation)
- [License](#license)

---

## Overview

The pipeline is designed around high-resolution geospatial imagery and object-detection labels for individual trees.

Main capabilities include:

- GeoTIFF + vector-label preparation to COCO format
- configurable RGB band selection
- geospatial-to-pixel bounding-box conversion
- standard train / validation / test training
- 5-fold cross-validation
- COCO AP / AR evaluation
- precision / recall / F1 evaluation
- transfer-city evaluation
- image-level RF-DETR prediction through the Python API
- large-area GeoTIFF inference
- conversion of pixel detections back to geospatial coordinates
- export to JSON, GeoJSON, and Shapefile
- YAML-based experiment configuration
- reusable Python modules under `src/geoai_tree/`

---

## Quick Start

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate geoai-urbantree-rfdetr
```

or:

```bash
pip install -r requirements.txt
```

### 2. Prepare a geospatial dataset

```bash
PYTHONPATH=src python scripts/prepare_dataset.py \
  --config configs/dataset.yaml
```

### 3. Train the standard TVT model

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/train.yaml
```

### 4. Run 5-fold cross-validation

```bash
PYTHONPATH=src python scripts/cross_validate.py \
  --config configs/cross_validation.yaml
```

### 5. Evaluate

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/evaluate.yaml \
  --mode tvt
```

Available evaluation modes:

```text
tvt
cv
transfer
```

### 6. Run large-area GeoTIFF inference

```bash
PYTHONPATH=src python scripts/infer_geotiff.py \
  --config configs/inference.yaml
```

---

## Supported Workflows

```text
GeoTIFF imagery + vector tree labels
                │
                ▼
        Dataset preparation
                │
                ▼
          COCO dataset
                │
        ┌───────┴────────┐
        ▼                ▼
     TVT training    5-fold CV
        │                │
        └───────┬────────┘
                ▼
            Evaluation
       ┌────────┼─────────┐
       ▼        ▼         ▼
      TVT       CV     Transfer
                │
                ▼
        Trained EMA weights
                │
                ▼
       Large-area inference
                │
                ▼
   JSON / GeoJSON / Shapefile
```

---

## Dataset Types

The repository uses several dataset organizations depending on the task.

### 1. Raw geospatial imagery

Raw imagery is stored as GeoTIFFs.

```text
datasets/geotiff_rgbn/
├── miami_030/
├── new_york_030/
├── chicago_030/
├── dallas_060/
└── los_angeles_060/
```

The suffix indicates the nominal imagery resolution used in this project:

```text
030 → 0.30 m imagery
060 → 0.60 m imagery
```

The dataset-preparation pipeline reads three configurable bands, with RGB bands `1, 2, 3` used by default.

---

### 2. Vector tree labels

Tree-label files are stored separately from imagery.

```text
datasets/labels/
├── miami_030/
├── new_york_030/
├── chicago_030/
├── dallas_060/
└── los_angeles_060/
```

Example Shapefile components:

```text
labels_new_york_030.shp
labels_new_york_030.shx
labels_new_york_030.dbf
labels_new_york_030.prj
labels_new_york_030.cpg
```

The preparation pipeline automatically reprojects vector labels to the raster CRS when needed.

---

### 3. Standard train / validation / test COCO dataset

The main RF-DETR training dataset uses:

```text
datasets/coco/tvt/
├── train/
│   ├── images/
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   └── _annotations.coco.json
└── test/
    ├── images/
    └── _annotations.coco.json
```

This dataset is used by:

```text
scripts/train.py
scripts/evaluate.py --mode tvt
```

---

### 4. Cross-validation dataset

The 5-fold cross-validation dataset uses:

```text
datasets/coco/cv/
├── fold_1/
│   ├── train/
│   └── valid/
├── fold_2/
│   ├── train/
│   └── valid/
├── fold_3/
│   ├── train/
│   └── valid/
├── fold_4/
│   ├── train/
│   └── valid/
└── fold_5/
    ├── train/
    └── valid/
```

Each `train/` and `valid/` directory contains:

```text
images/
_annotations.coco.json
```

This dataset is used by:

```text
scripts/cross_validate.py
scripts/evaluate.py --mode cv
```

---

### 5. Transfer-city datasets

Transfer datasets evaluate a trained model on geographically independent cities.

```text
datasets/coco/transfer/
├── new_york_030/
│   └── test/
├── chicago_030/
│   └── test/
├── dallas_060/
│   └── test/
└── los_angeles_060/
    └── test/
```

Each `test/` directory contains:

```text
images/
_annotations.coco.json
```

This dataset is used by:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/evaluate.yaml \
  --mode transfer
```

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

The repository was tested with:

```text
Python       3.10.19
RF-DETR      1.5.0rc1
PyTorch      2.10.0
Torchvision  0.25.0
NumPy        2.2.6
Pandas       2.3.3
GeoPandas    1.1.3
Rasterio     1.4.4
Shapely      2.1.2
pycocotools  2.0.11
```

### Conda

```bash
conda env create -f environment.yml
conda activate geoai-urbantree-rfdetr
```

### pip

```bash
pip install -r requirements.txt
```

The examples below assume commands are run from the repository root with:

```bash
PYTHONPATH=src
```

---

## Commands

### Dataset preparation

```bash
PYTHONPATH=src python scripts/prepare_dataset.py \
  --config configs/dataset.yaml
```

### Standard TVT training

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/train.yaml
```

### Cross-validation

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

### Evaluation

TVT:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/evaluate.yaml \
  --mode tvt
```

Cross-validation:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/evaluate.yaml \
  --mode cv
```

Transfer cities:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/evaluate.yaml \
  --mode transfer
```

### Large-area inference

```bash
PYTHONPATH=src python scripts/infer_geotiff.py \
  --config configs/inference.yaml
```

---

# Configuration Files

All major workflows are controlled through YAML files so that dataset paths, training settings, evaluation thresholds, and output locations can be changed without editing the Python source code.

---

## 1. Dataset Preparation: `configs/dataset.yaml`

Example:

```yaml
project:
  name: GeoAI-UrbanTree-RFDETR

dataset:
  name: new_york_030
  split_name: test

input:
  raster_dir: datasets/geotiff_rgbn/new_york_030
  raster_pattern: "*.tif"
  labels: datasets/labels/new_york_030/labels_new_york_030.shp

processing:
  tile_size: 256
  bands:
    - 1
    - 2
    - 3
  min_bbox_area: 4.0
  skip_empty: false
  keep_partial: false

output:
  root: outputs/dataset_preparation/new_york_030
  clean: false
```

### Parameters

| Parameter | Meaning |
|---|---|
| `dataset.name` | Dataset identifier used in generated image filenames and metadata |
| `dataset.split_name` | Output COCO split name, such as `train`, `valid`, or `test` |
| `input.raster_dir` | Directory containing source GeoTIFF files |
| `input.raster_pattern` | Glob pattern used to discover raster files |
| `input.labels` | Vector label file used to create object-detection boxes |
| `processing.tile_size` | Raster tile size in pixels |
| `processing.bands` | Three 1-based raster bands exported as RGB |
| `processing.min_bbox_area` | Minimum retained bounding-box area in pixels² |
| `processing.skip_empty` | If `true`, patches with zero valid tree boxes are omitted |
| `processing.keep_partial` | If `true`, smaller edge tiles are retained |
| `output.root` | Output COCO dataset directory |
| `output.clean` | If `true`, remove the existing output directory before preparation |

The preparation workflow:

```text
GeoTIFF
  ↓
tile raster
  ↓
query intersecting vector labels
  ↓
clip geometry to tile
  ↓
convert map coordinates → pixel bbox
  ↓
filter small boxes
  ↓
save PNG
  ↓
write COCO JSON
```

Output:

```text
output.root/
└── <split_name>/
    ├── images/
    ├── _annotations.coco.json
    └── preparation_summary.json
```

---

## 2. TVT Training: `configs/train.yaml`

Example:

```yaml
project:
  name: GeoAI-UrbanTree-RFDETR
  experiment_name: tvt_baseline

dataset:
  type: tvt
  root: datasets/coco/tvt
  annotation_file: _annotations.coco.json
  splits:
    train: train
    valid: valid
    test: test

model:
  architecture: RFDETRBase
  pretrained: true

training:
  epochs: 50
  batch_size: 8
  grad_accum_steps: 4
  learning_rate: 0.0001
  num_workers: 8
  checkpoint_interval: 10
  run_test_during_training: false

output:
  root: outputs/training/tvt
  run_name: rfdetr_tvt_base

system:
  device: cuda
  seed: 42
```

### Parameters

| Parameter | Meaning |
|---|---|
| `project.name` | Repository / project name |
| `project.experiment_name` | Human-readable experiment identifier |
| `dataset.type` | Dataset workflow type; `tvt` for train/validation/test |
| `dataset.root` | Root directory containing the COCO splits |
| `dataset.annotation_file` | COCO annotation filename |
| `dataset.splits.train` | Training split directory name |
| `dataset.splits.valid` | Validation split directory name |
| `dataset.splits.test` | Test split directory name |
| `model.architecture` | RF-DETR architecture; current pipeline uses `RFDETRBase` |
| `model.pretrained` | Records the intended pretrained-model setup in the experiment configuration |
| `training.epochs` | Number of training epochs |
| `training.batch_size` | Number of images per training batch |
| `training.grad_accum_steps` | Number of gradient-accumulation steps |
| `training.learning_rate` | Optimizer learning rate |
| `training.num_workers` | Number of data-loader workers |
| `training.checkpoint_interval` | Save a regular checkpoint every N epochs |
| `training.run_test_during_training` | Whether RF-DETR evaluates the test split during training |
| `output.root` | Root training-output directory |
| `output.run_name` | Name of this training run |
| `system.device` | Intended compute device recorded in the configuration |
| `system.seed` | Experiment seed recorded for reproducibility |

Training output is written to:

```text
outputs/training/tvt/rfdetr_tvt_base/
```

The standard public model checkpoint is:

```text
checkpoint_best_ema.pth
```

---

## 3. Cross-Validation: `configs/cross_validation.yaml`

Example:

```yaml
project:
  name: GeoAI-UrbanTree-RFDETR
  experiment_name: cv_baseline

dataset:
  type: cross_validation
  root: datasets/coco/cv
  annotation_file: _annotations.coco.json
  fold_prefix: fold_
  num_folds: 5
  splits:
    train: train
    valid: valid

model:
  architecture: RFDETRBase
  pretrained: true

training:
  epochs: 50
  batch_size: 8
  grad_accum_steps: 4
  learning_rate: 0.0001
  num_workers: 8
  checkpoint_interval: 10
  run_test_during_training: false

output:
  root: outputs/cross_validation
  fold_prefix: fold_

system:
  device: cuda
  seed: 42
```

### Parameters

| Parameter | Meaning |
|---|---|
| `dataset.type` | Identifies the cross-validation workflow |
| `dataset.root` | Root directory containing fold directories |
| `dataset.annotation_file` | COCO annotation filename |
| `dataset.fold_prefix` | Prefix used for fold directories |
| `dataset.num_folds` | Number of cross-validation folds |
| `dataset.splits.train` | Training split name within each fold |
| `dataset.splits.valid` | Validation split name within each fold |
| `output.root` | Root directory for all CV training outputs |
| `output.fold_prefix` | Prefix used for CV output directories |

The remaining `model`, `training`, and `system` parameters have the same meanings as in `configs/train.yaml`.

Expected outputs:

```text
outputs/cross_validation/
├── fold_1/
├── fold_2/
├── fold_3/
├── fold_4/
└── fold_5/
```

---

## 4. Evaluation: `configs/evaluate.yaml`

Example:

```yaml
project:
  name: GeoAI-UrbanTree-RFDETR

model:
  architecture: RFDETRBase

checkpoint:
  tvt: weights/tvt/checkpoint_best_ema.pth
  cv_root: weights/cv
  cv_pattern: fold_{fold}/checkpoint_best_ema.pth

evaluation:
  coco_threshold: 0.001
  prediction_threshold: 0.30
  iou_threshold: 0.50
  max_detections: 200
  save_predictions: true
  save_visualizations: true

visualization:
  ground_truth_color: lime
  prediction_color: red
  box_width: 2
  panel_gap: 20

datasets:
  tvt:
    root: datasets/coco/tvt
    splits: [train, valid, test]

  cv:
    root: datasets/coco/cv
    num_folds: 5
    splits: [train, valid]

  transfer:
    root: datasets/coco/transfer
    cities:
      - new_york_030
      - chicago_030
      - dallas_060
      - los_angeles_060
    split: test

output:
  root: outputs/evaluation
  tvt: outputs/evaluation/tvt
  cv: outputs/evaluation/cv
  transfer: outputs/evaluation/transfer
```

### Parameters

| Parameter | Meaning |
|---|---|
| `checkpoint.tvt` | EMA checkpoint used for TVT and transfer evaluation |
| `checkpoint.cv_root` | Root directory containing CV checkpoints |
| `checkpoint.cv_pattern` | Fold-specific checkpoint path pattern |
| `evaluation.coco_threshold` | Low confidence threshold used to retain detections for COCO AP / AR evaluation |
| `evaluation.prediction_threshold` | Practical threshold used for precision, recall, F1, and visualization |
| `evaluation.iou_threshold` | IoU threshold used for TP / FP / FN matching |
| `evaluation.max_detections` | Maximum predictions retained per image |
| `evaluation.save_predictions` | Save COCO-format model predictions |
| `evaluation.save_visualizations` | Save ground-truth vs prediction images |
| `visualization.ground_truth_color` | Ground-truth box color |
| `visualization.prediction_color` | Prediction box color |
| `visualization.box_width` | Bounding-box line width |
| `visualization.panel_gap` | Pixel gap between GT and prediction panels |
| `datasets.tvt.root` | TVT dataset root |
| `datasets.tvt.splits` | TVT splits to evaluate |
| `datasets.cv.root` | CV dataset root |
| `datasets.cv.num_folds` | Number of CV folds |
| `datasets.cv.splits` | CV splits to evaluate |
| `datasets.transfer.root` | Transfer-city dataset root |
| `datasets.transfer.cities` | Transfer datasets to evaluate |
| `datasets.transfer.split` | Transfer split name |
| `output.tvt` | TVT evaluation output directory |
| `output.cv` | CV evaluation output directory |
| `output.transfer` | Transfer evaluation output directory |

### Metrics

The evaluation pipeline calculates:

```text
COCO AP50:95
AP50
AP75
AR@200

Precision
Recall
F1

TP
FP
FN
```

It also supports:

```text
per-image metrics CSV
summary JSON
summary TXT
COCO prediction JSON
GT / prediction visualizations
combined summary CSV
```

---

## 5. Large-Area Inference: `configs/inference.yaml`

Example:

```yaml
project:
  name: GeoAI-UrbanTree-RFDETR
  inference_name: miami_full_aoi

model:
  architecture: RFDETRBase

checkpoint:
  path: weights/tvt/checkpoint_best_ema.pth

input:
  raster: datasets/geotiff_rgbn/miami_030/NAIP_2023_Miami_RGBN_CLIP.tif
  bands: [1, 2, 3]

inference:
  tile_size: 256
  confidence_threshold: 0.30
  max_detections: 200
  test_mode: false
  max_test_tiles: 100
  save_test_tiles: false

output:
  root: outputs/inference/miami_030/full_aoi
  save_json: true
  save_geojson: true
  save_shapefile: true
```

### Parameters

| Parameter | Meaning |
|---|---|
| `project.inference_name` | Human-readable inference run name |
| `model.architecture` | RF-DETR model architecture |
| `checkpoint.path` | Trained model checkpoint |
| `input.raster` | Input GeoTIFF |
| `input.bands` | Three 1-based raster bands used as RGB |
| `inference.tile_size` | Inference tile size in pixels |
| `inference.confidence_threshold` | Minimum confidence retained in final predictions |
| `inference.max_detections` | Maximum predictions per tile |
| `inference.test_mode` | Run only a small central subset of tiles |
| `inference.max_test_tiles` | Maximum tiles processed when test mode is enabled |
| `inference.save_test_tiles` | Save test-mode RGB tiles for inspection |
| `output.root` | Output directory |
| `output.save_json` | Save tabular prediction records as JSON |
| `output.save_geojson` | Save geospatial predictions as GeoJSON |
| `output.save_shapefile` | Save geospatial predictions as an ESRI Shapefile |

Inference converts detections from:

```text
tile pixel coordinates
        ↓
full-raster pixel coordinates
        ↓
map coordinates
        ↓
geospatial polygons
```

---

## Using Your Own Dataset

There are two main ways to use your own data.

### Option A: Start from GeoTIFF + vector labels

Prepare:

```text
your imagery/*.tif
your labels.shp
```

Update:

```text
configs/dataset.yaml
```

and run:

```bash
PYTHONPATH=src python scripts/prepare_dataset.py \
  --config configs/dataset.yaml
```

This creates an RF-DETR-compatible COCO dataset.

---

### Option B: Start from an existing COCO dataset

If your dataset already uses COCO object-detection format, organize it as:

```text
my_dataset/
├── train/
│   ├── images/
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   └── _annotations.coco.json
└── test/
    ├── images/
    └── _annotations.coco.json
```

Then change:

```yaml
dataset:
  root: datasets/coco/my_dataset
```

in:

```text
configs/train.yaml
```

and train:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/train.yaml
```

The shared trainer validates the required directory structure before training starts.

---

## Model Weights

Released model weights use the EMA checkpoint:

```text
checkpoint_best_ema.pth
```

Expected structure:

```text
weights/
├── tvt/
│   └── checkpoint_best_ema.pth
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

Large model weights are not stored directly in the Git repository and can be distributed separately through Zenodo.

---

## Outputs

### Training

```text
outputs/training/tvt/
```

### Cross-validation

```text
outputs/cross_validation/
```

### Evaluation

```text
outputs/evaluation/
├── tvt/
├── cv/
└── transfer/
```

Typical evaluation split output:

```text
metrics_per_image.csv
summary.json
summary.txt
prediction COCO JSON
vis_gt_left__pred_right/
```

### Large-area inference

```text
outputs/inference/miami_030/full_aoi/
```

Typical outputs:

```text
miami_030_rfdetr_predictions.json
miami_030_rfdetr_predictions.geojson
miami_030_rfdetr_predictions.shp
miami_030_rfdetr_predictions.shx
miami_030_rfdetr_predictions.dbf
miami_030_rfdetr_predictions.prj
miami_030_rfdetr_predictions.cpg
summary.json
```

---

## Python Package

Reusable functionality is implemented under:

```text
src/geoai_tree/
```

### `data/`

```text
coco.py
geospatial.py
preparation.py
validation.py
```

Responsibilities:

- COCO parsing
- image-path resolution
- raster-window creation
- geospatial coordinate conversion
- GeoTIFF-to-COCO preparation
- dataset validation

### `training/`

```text
trainer.py
cross_validation.py
```

Responsibilities:

- shared RF-DETR training
- TVT training
- CV fold training
- CV orchestration

### `evaluation/`

```text
detection_metrics.py
coco_metrics.py
evaluator.py
```

Responsibilities:

- IoU matching
- TP / FP / FN
- precision / recall / F1
- COCO AP / AR
- split evaluation
- result summaries

### `inference/`

```text
predictor.py
raster_inference.py
```

Responsibilities:

- RF-DETR checkpoint loading
- image prediction
- confidence filtering
- COCO prediction conversion
- tiled raster inference
- map-coordinate prediction export

### `visualization/`

```text
detection.py
```

Responsibilities:

- bounding-box drawing
- GT / prediction side-by-side visualization

### `utils/`

```text
config.py
paths.py
checkpoints.py
```

Responsibilities:

- YAML configuration loading
- repository-relative path resolution
- model-checkpoint resolution

---

## Citation

If you use this repository in research, please cite the associated publication and software release.

Citation metadata is provided in:

```text
CITATION.cff
```

The Zenodo DOI for archived software, datasets, and model weights can be added after the public archive is created.

---

## License

This project is released under the MIT License.

See:

```text
LICENSE
```

for details.
