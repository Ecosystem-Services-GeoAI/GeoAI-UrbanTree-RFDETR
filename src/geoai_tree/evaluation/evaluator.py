#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Universal RF-DETR evaluator for GeoAI-UrbanTree-RFDETR.

This module provides the shared evaluation logic for:

- TVT train / valid / test evaluation
- cross-validation fold evaluation
- transfer-city evaluation

The workflow preserves the original project behavior:

1. Run RF-DETR once at a low confidence threshold.
2. Sort predictions by confidence.
3. Retain at most max_detections predictions.
4. Save low-threshold predictions for COCO AP / AR.
5. Apply a practical confidence threshold for:
   - TP / FP / FN
   - precision
   - recall
   - F1
   - visualization
6. Save per-image metrics and split-level summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from geoai_tree.data.coco import (
    get_annotation_path,
    get_image_records,
    load_coco_index,
    resolve_image_path,
)
from geoai_tree.evaluation.coco_metrics import (
    empty_coco_metrics,
    evaluate_coco_predictions,
)
from geoai_tree.evaluation.detection_metrics import (
    calculate_detection_metrics,
    precision_recall_f1,
)
from geoai_tree.inference.predictor import (
    filter_predictions,
    predict_image,
    predictions_to_coco,
)
from geoai_tree.visualization.detection import (
    create_gt_prediction_visualization,
)


# ============================================================
# Dataset helpers
# ============================================================

def get_primary_category_id(
    annotation_path: str | Path,
) -> int:
    """
    Return the primary COCO category ID.

    The urban-tree datasets used in this project are one-class
    detection datasets containing the category ``tree``.

    If multiple categories exist, the first category ID is used
    and a warning is printed, matching the behavior of the
    original transfer-city evaluation script.
    """

    annotation_path = Path(
        annotation_path
    )

    coco = COCO(
        str(annotation_path)
    )

    category_ids = coco.getCatIds()

    if len(category_ids) == 0:
        raise ValueError(
            f"No categories found in: {annotation_path}"
        )

    if len(category_ids) != 1:
        print(
            "Warning: multiple COCO category IDs found: "
            f"{category_ids}"
        )

    return int(
        category_ids[0]
    )


# ============================================================
# Output helpers
# ============================================================

def write_summary_text(
    summary: Dict,
    output_path: str | Path,
) -> Path:
    """
    Write a human-readable evaluation summary.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    maxdet = summary["maxdet"]

    lines = [
        f"Dataset: {summary['dataset']}",
        f"Split: {summary['split']}",
        f"Split directory: {summary['split_dir']}",
        f"Images: {summary['images']}",
        "",
        f"COCO threshold: {summary['coco_threshold']}",
        f"F1 / visualization threshold: "
        f"{summary['prediction_threshold']}",
        f"MAXDET: {maxdet}",
        f"IoU threshold: {summary['iou_threshold']}",
        "",
        f"TP: {summary['tp']}",
        f"FP: {summary['fp']}",
        f"FN: {summary['fn']}",
        f"Precision: {summary['precision']:.6f}",
        f"Recall: {summary['recall']:.6f}",
        f"F1: {summary['f1']:.6f}",
        "",
        f"AP_50_95: {summary['AP_50_95']:.6f}",
        f"AP50: {summary['AP50']:.6f}",
        f"AP75: {summary['AP75']:.6f}",
        f"AP_small: {summary['AP_small']:.6f}",
        f"AP_medium: {summary['AP_medium']:.6f}",
        f"AP_large: {summary['AP_large']:.6f}",
        "",
        f"AR_1: {summary['AR_1']:.6f}",
        f"AR_10: {summary['AR_10']:.6f}",
        f"AR_{maxdet}: "
        f"{summary[f'AR_{maxdet}']:.6f}",
        f"AR_small: {summary['AR_small']:.6f}",
        f"AR_medium: {summary['AR_medium']:.6f}",
        f"AR_large: {summary['AR_large']:.6f}",
        "",
    ]

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return output_path


# ============================================================
# Universal split evaluation
# ============================================================

def evaluate_split(
    model,
    dataset_name: str,
    split_name: str,
    split_dir: str | Path,
    output_dir: str | Path,
    annotation_file: str = "_annotations.coco.json",
    coco_threshold: float = 0.001,
    prediction_threshold: float = 0.30,
    iou_threshold: float = 0.50,
    max_detections: int = 200,
    save_predictions: bool = True,
    save_visualizations: bool = True,
    gt_color: str = "lime",
    pred_color: str = "red",
    box_width: int = 2,
    panel_gap: int = 20,
    print_coco_summary: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate one RF-DETR model on one COCO dataset split.

    Parameters
    ----------
    model
        Loaded RF-DETR model.

    dataset_name : str
        Logical dataset name.

        Examples:
            tvt
            fold_1
            new_york_030

    split_name : str
        Dataset split name.

        Examples:
            train
            valid
            test

    split_dir : str or Path
        COCO split directory containing:

            images/
            _annotations.coco.json

    output_dir : str or Path
        Directory where evaluation outputs are written.

    annotation_file : str
        COCO annotation filename.

    coco_threshold : float
        Low prediction threshold used for COCO AP / AR.

    prediction_threshold : float
        Practical threshold used for TP / FP / FN,
        precision / recall / F1, and visualization.

    iou_threshold : float
        IoU threshold for one-to-one detection matching.

    max_detections : int
        Maximum predictions retained per image.

    save_predictions : bool
        Whether to keep the COCO prediction JSON.

    save_visualizations : bool
        Whether to save GT-left / prediction-right images.

    Returns
    -------
    tuple
        (per_image_dataframe, summary_dictionary)
    """

    split_dir = Path(
        split_dir
    )

    output_dir = Path(
        output_dir
    )

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset split directory not found: {split_dir}"
        )

    annotation_path = get_annotation_path(
        split_dir=split_dir,
        annotation_file=annotation_file,
    )

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"COCO annotation file not found: "
            f"{annotation_path}"
        )

    if not 0.0 <= coco_threshold <= 1.0:
        raise ValueError(
            "coco_threshold must be between 0 and 1."
        )

    if not 0.0 <= prediction_threshold <= 1.0:
        raise ValueError(
            "prediction_threshold must be between 0 and 1."
        )

    if prediction_threshold < coco_threshold:
        raise ValueError(
            "prediction_threshold should be greater than or "
            "equal to coco_threshold so low-threshold "
            "predictions can be reused."
        )

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(
            "iou_threshold must be between 0 and 1."
        )

    if max_detections <= 0:
        raise ValueError(
            "max_detections must be greater than zero."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    visualization_dir = (
        output_dir
        / "vis_gt_left__pred_right"
    )

    if save_visualizations:
        visualization_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Load COCO indexes
    # --------------------------------------------------------

    coco_index = load_coco_index(
        annotation_path
    )

    category_id = get_primary_category_id(
        annotation_path
    )

    image_records = get_image_records(
        coco_index
    )

    # --------------------------------------------------------
    # Print evaluation information
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"Evaluating: {dataset_name}/{split_name}"
    )

    print(
        f"Split directory:       {split_dir}"
    )

    print(
        f"Annotation:            {annotation_path}"
    )

    print(
        f"Images:                {len(image_records)}"
    )

    print(
        f"COCO threshold:        {coco_threshold}"
    )

    print(
        f"F1 / VIS threshold:    {prediction_threshold}"
    )

    print(
        f"MAXDET:                {max_detections}"
    )

    print(
        f"IoU threshold:         {iou_threshold}"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # Evaluation containers
    # --------------------------------------------------------

    coco_predictions = []
    per_image_rows = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # --------------------------------------------------------
    # Image loop
    # --------------------------------------------------------

    for image_id, file_name in tqdm(
        image_records,
        desc=f"{dataset_name}/{split_name}",
    ):

        image_path = resolve_image_path(
            split_dir=split_dir,
            file_name=file_name,
        )

        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            # ------------------------------------------------
            # Run RF-DETR only once.
            #
            # Low threshold is required for COCO AP / AR.
            # ------------------------------------------------

            predictions = predict_image(
                model=model,
                image=image,
                threshold=coco_threshold,
                max_detections=max_detections,
            )

            # ------------------------------------------------
            # COCO AP / AR predictions
            # ------------------------------------------------

            image_coco_predictions = (
                predictions_to_coco(
                    predictions=predictions,
                    image_id=image_id,
                    category_id=category_id,
                )
            )

            coco_predictions.extend(
                image_coco_predictions
            )

            # ------------------------------------------------
            # Practical threshold for F1 + visualization
            # ------------------------------------------------

            practical_predictions = (
                filter_predictions(
                    predictions=predictions,
                    threshold=prediction_threshold,
                )
            )

            gt_boxes = (
                coco_index.gt_boxes.get(
                    int(image_id),
                    np.zeros(
                        (0, 4),
                        dtype=np.float32,
                    ),
                )
            )

            detection_metrics = (
                calculate_detection_metrics(
                    pred_boxes=(
                        practical_predictions.xyxy
                    ),
                    gt_boxes=gt_boxes,
                    iou_threshold=iou_threshold,
                )
            )

            total_tp += (
                detection_metrics["tp"]
            )

            total_fp += (
                detection_metrics["fp"]
            )

            total_fn += (
                detection_metrics["fn"]
            )

            # ------------------------------------------------
            # Per-image metrics
            # ------------------------------------------------

            base_name = Path(
                file_name
            ).name

            per_image_rows.append(
                {
                    "dataset": dataset_name,
                    "split": split_name,
                    "image": base_name,
                    "image_id": int(
                        image_id
                    ),
                    "gt_count": int(
                        len(gt_boxes)
                    ),
                    "pred_count": int(
                        len(
                            practical_predictions
                        )
                    ),
                    "tp": int(
                        detection_metrics[
                            "tp"
                        ]
                    ),
                    "fp": int(
                        detection_metrics[
                            "fp"
                        ]
                    ),
                    "fn": int(
                        detection_metrics[
                            "fn"
                        ]
                    ),
                    "precision": (
                        detection_metrics[
                            "precision"
                        ]
                    ),
                    "recall": (
                        detection_metrics[
                            "recall"
                        ]
                    ),
                    "f1": (
                        detection_metrics[
                            "f1"
                        ]
                    ),
                    "coco_threshold": (
                        coco_threshold
                    ),
                    "prediction_threshold": (
                        prediction_threshold
                    ),
                    "maxdet": (
                        max_detections
                    ),
                    "iou_threshold": (
                        iou_threshold
                    ),
                }
            )

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------

            if save_visualizations:

                patch_stem = Path(
                    base_name
                ).stem

                visualization_path = (
                    visualization_dir
                    / (
                        f"{dataset_name}_"
                        f"{split_name}_"
                        f"{patch_stem}_"
                        f"gt_left_pred_right.png"
                    )
                )

                create_gt_prediction_visualization(
                    image=image,
                    gt_boxes=gt_boxes,
                    pred_boxes=(
                        practical_predictions.xyxy
                    ),
                    output_path=(
                        visualization_path
                    ),
                    gt_color=gt_color,
                    pred_color=pred_color,
                    box_width=box_width,
                    panel_gap=panel_gap,
                )

    # ========================================================
    # Save COCO prediction JSON
    # ========================================================

    prediction_json = (
        output_dir
        / (
            f"{dataset_name}_"
            f"{split_name}_"
            f"predictions_"
            f"maxdet{max_detections}.coco.json"
        )
    )

    prediction_json.write_text(
        json.dumps(
            coco_predictions,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\nWrote predictions: "
        f"{prediction_json}"
    )

    print(
        "Number of detections for COCOeval: "
        f"{len(coco_predictions)}"
    )

    # ========================================================
    # COCO AP / AR
    # ========================================================

    if len(coco_predictions) == 0:

        print(
            "Warning: no detections. "
            "COCOeval metrics will be -1."
        )

        coco_metrics = (
            empty_coco_metrics(
                maxdet=max_detections
            )
        )

    else:

        coco_metrics = (
            evaluate_coco_predictions(
                ground_truth_json=(
                    annotation_path
                ),
                prediction_json=(
                    prediction_json
                ),
                maxdet=max_detections,
                print_summary=(
                    print_coco_summary
                ),
            )
        )

    # ========================================================
    # Overall precision / recall / F1
    # ========================================================

    (
        overall_precision,
        overall_recall,
        overall_f1,
    ) = precision_recall_f1(
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
    )

    # ========================================================
    # Per-image DataFrame
    # ========================================================

    per_image_df = pd.DataFrame(
        per_image_rows
    )

    metrics_csv = (
        output_dir
        / "metrics_per_image.csv"
    )

    per_image_df.to_csv(
        metrics_csv,
        index=False,
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "dataset": dataset_name,
        "split": split_name,
        "split_dir": str(
            split_dir
        ),
        "images": int(
            len(image_records)
        ),
        "num_coco_detections": int(
            len(coco_predictions)
        ),
        "tp": int(
            total_tp
        ),
        "fp": int(
            total_fp
        ),
        "fn": int(
            total_fn
        ),
        "precision": float(
            overall_precision
        ),
        "recall": float(
            overall_recall
        ),
        "f1": float(
            overall_f1
        ),
        "coco_threshold": float(
            coco_threshold
        ),
        "prediction_threshold": float(
            prediction_threshold
        ),
        "maxdet": int(
            max_detections
        ),
        "iou_threshold": float(
            iou_threshold
        ),
        **coco_metrics,
    }

    summary_json = (
        output_dir
        / "summary.json"
    )

    summary_json.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_txt = (
        output_dir
        / "summary.txt"
    )

    write_summary_text(
        summary=summary,
        output_path=summary_txt,
    )

    # --------------------------------------------------------
    # Optionally remove prediction file
    # --------------------------------------------------------

    if not save_predictions:

        if prediction_json.exists():
            prediction_json.unlink()

    # ========================================================
    # Final console summary
    # ========================================================

    print(
        "\n"
        + "-" * 100
    )

    print(
        f"{dataset_name}/{split_name}"
    )

    print(
        f"Precision: "
        f"{overall_precision:.4f}"
    )

    print(
        f"Recall:    "
        f"{overall_recall:.4f}"
    )

    print(
        f"F1:        "
        f"{overall_f1:.4f}"
    )

    print(
        f"AP50:95:   "
        f"{summary['AP_50_95']:.4f}"
    )

    print(
        f"AP50:      "
        f"{summary['AP50']:.4f}"
    )

    print(
        f"AP75:      "
        f"{summary['AP75']:.4f}"
    )

    print(
        f"AR@{max_detections}:   "
        f"{summary[f'AR_{max_detections}']:.4f}"
    )

    print(
        f"Per-image metrics: "
        f"{metrics_csv}"
    )

    print(
        f"Summary JSON:     "
        f"{summary_json}"
    )

    print(
        f"Summary TXT:      "
        f"{summary_txt}"
    )

    if save_visualizations:
        print(
            f"Visualizations:   "
            f"{visualization_dir}"
        )

    print(
        "-" * 100
    )

    return (
        per_image_df,
        summary,
    )


# ============================================================
# Multi-result summary helper
# ============================================================

def combine_evaluation_summaries(
    summaries: list[Dict],
) -> pd.DataFrame:
    """
    Combine multiple evaluation summary dictionaries into
    one DataFrame.

    This is useful for:

    - TVT train / valid / test
    - CV fold summaries
    - transfer-city summaries
    """

    if len(summaries) == 0:
        return pd.DataFrame()

    return pd.DataFrame(
        summaries
    )
