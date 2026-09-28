#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COCO AP/AR evaluation utilities for GeoAI-UrbanTree-RFDETR.

This module supports custom maximum detections per image, such as
maxDet = 200, while preserving the evaluation logic used in the
original project scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ============================================================
# Empty metric template
# ============================================================

def empty_coco_metrics(
    maxdet: int = 200,
) -> Dict[str, float]:
    """
    Return an empty COCO metric dictionary.

    Used when no detections are available and COCOeval cannot
    calculate normal AP/AR metrics.
    """

    return {
        "AP_50_95": -1.0,
        "AP50": -1.0,
        "AP75": -1.0,
        "AP_small": -1.0,
        "AP_medium": -1.0,
        "AP_large": -1.0,
        "AR_1": -1.0,
        "AR_10": -1.0,
        f"AR_{maxdet}": -1.0,
        "AR_small": -1.0,
        "AR_medium": -1.0,
        "AR_large": -1.0,
    }


# ============================================================
# Custom COCO metric extraction
# ============================================================

def summarize_coco_metrics_custom(
    evaluator: COCOeval,
    maxdet: int = 200,
) -> Dict[str, float]:
    """
    Extract COCO AP/AR metrics correctly for a custom maxDet.

    COCOeval precision array shape:

        precision[T, R, K, A, M]

    where:

        T = IoU thresholds
        R = recall thresholds
        K = categories
        A = area ranges
        M = maxDets

    COCOeval recall array shape:

        recall[T, K, A, M]

    Parameters
    ----------
    evaluator : COCOeval
        Evaluated and accumulated COCOeval object.

    maxdet : int
        Maximum detections per image.

    Returns
    -------
    dict
        Custom AP/AR metrics.
    """

    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]

    # COCO area-range indices
    area_all = 0
    area_small = 1
    area_medium = 2
    area_large = 3

    maxdets = list(
        evaluator.params.maxDets
    )

    if maxdet not in maxdets:
        raise ValueError(
            f"maxdet={maxdet} is not in "
            f"COCOeval maxDets={maxdets}"
        )

    maxdet_index = maxdets.index(
        maxdet
    )

    # --------------------------------------------------------
    # Precision helper
    # --------------------------------------------------------

    def mean_precision(
        iou_index: Optional[int] = None,
        area_index: int = 0,
    ) -> float:

        if iou_index is None:
            values = precision[
                :,
                :,
                :,
                area_index,
                maxdet_index,
            ]

        else:
            values = precision[
                iou_index,
                :,
                :,
                area_index,
                maxdet_index,
            ]

        # COCOeval uses -1 for unavailable values.
        values = values[
            values > -1
        ]

        if values.size == 0:
            return -1.0

        return float(
            values.mean()
        )

    # --------------------------------------------------------
    # Recall helper
    # --------------------------------------------------------

    def mean_recall(
        area_index: int = 0,
    ) -> float:

        values = recall[
            :,
            :,
            area_index,
            maxdet_index,
        ]

        values = values[
            values > -1
        ]

        if values.size == 0:
            return -1.0

        return float(
            values.mean()
        )

    # --------------------------------------------------------
    # Locate AP50 and AP75 IoU indices
    # --------------------------------------------------------

    iou_thresholds = (
        evaluator.params.iouThrs
    )

    iou_50_matches = np.where(
        np.isclose(
            iou_thresholds,
            0.50,
        )
    )[0]

    iou_75_matches = np.where(
        np.isclose(
            iou_thresholds,
            0.75,
        )
    )[0]

    if len(iou_50_matches) == 0:
        raise ValueError(
            "Could not find IoU threshold 0.50 "
            "in COCOeval."
        )

    if len(iou_75_matches) == 0:
        raise ValueError(
            "Could not find IoU threshold 0.75 "
            "in COCOeval."
        )

    iou_50_index = int(
        iou_50_matches[0]
    )

    iou_75_index = int(
        iou_75_matches[0]
    )

    return {
        "AP_50_95": mean_precision(
            iou_index=None,
            area_index=area_all,
        ),

        "AP50": mean_precision(
            iou_index=iou_50_index,
            area_index=area_all,
        ),

        "AP75": mean_precision(
            iou_index=iou_75_index,
            area_index=area_all,
        ),

        "AP_small": mean_precision(
            iou_index=None,
            area_index=area_small,
        ),

        "AP_medium": mean_precision(
            iou_index=None,
            area_index=area_medium,
        ),

        "AP_large": mean_precision(
            iou_index=None,
            area_index=area_large,
        ),

        f"AR_{maxdet}": mean_recall(
            area_index=area_all,
        ),

        "AR_small": mean_recall(
            area_index=area_small,
        ),

        "AR_medium": mean_recall(
            area_index=area_medium,
        ),

        "AR_large": mean_recall(
            area_index=area_large,
        ),
    }


# ============================================================
# AR at a specific maxDet
# ============================================================

def get_ar_at_maxdet(
    evaluator: COCOeval,
    maxdet_value: int,
    area_index: int = 0,
) -> float:
    """
    Extract average recall at a specific maxDet.

    Examples
    --------
    maxdet_value = 1
    maxdet_value = 10
    maxdet_value = 200
    """

    recall = evaluator.eval["recall"]

    maxdets = list(
        evaluator.params.maxDets
    )

    if maxdet_value not in maxdets:
        return -1.0

    maxdet_index = maxdets.index(
        maxdet_value
    )

    values = recall[
        :,
        :,
        area_index,
        maxdet_index,
    ]

    values = values[
        values > -1
    ]

    if values.size == 0:
        return -1.0

    return float(
        values.mean()
    )


# ============================================================
# Full COCO evaluation
# ============================================================

def evaluate_coco_predictions(
    ground_truth_json: str | Path,
    prediction_json: str | Path,
    maxdet: int = 200,
    print_summary: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a saved COCO prediction JSON against ground truth.

    Parameters
    ----------
    ground_truth_json : str or Path
        COCO ground-truth annotation file.

    prediction_json : str or Path
        COCO-format detection prediction file.

    maxdet : int
        Maximum detections per image.

    print_summary : bool
        Whether to call COCOeval.summarize().

    Returns
    -------
    dict
        AP and AR metrics.
    """

    ground_truth_json = Path(
        ground_truth_json
    )

    prediction_json = Path(
        prediction_json
    )

    if not ground_truth_json.exists():
        raise FileNotFoundError(
            f"Ground-truth COCO JSON not found: "
            f"{ground_truth_json}"
        )

    if not prediction_json.exists():
        raise FileNotFoundError(
            f"Prediction COCO JSON not found: "
            f"{prediction_json}"
        )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    coco_gt = COCO(
        str(ground_truth_json)
    )

    # --------------------------------------------------------
    # Check whether prediction JSON is empty
    # --------------------------------------------------------

    import json

    with open(
        prediction_json,
        "r",
        encoding="utf-8",
    ) as f:
        predictions = json.load(f)

    if len(predictions) == 0:
        print(
            "Warning: prediction JSON contains "
            "no detections."
        )

        return empty_coco_metrics(
            maxdet=maxdet
        )

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    coco_dt = coco_gt.loadRes(
        str(prediction_json)
    )

    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        iouType="bbox",
    )

    # Important:
    # Preserve original project behavior.
    evaluator.params.maxDets = [
        1,
        10,
        maxdet,
    ]

    evaluator.evaluate()
    evaluator.accumulate()

    if print_summary:
        print(
            "\nCOCOeval raw summarize output:"
        )

        evaluator.summarize()

    # --------------------------------------------------------
    # Extract custom metrics
    # --------------------------------------------------------

    custom_metrics = (
        summarize_coco_metrics_custom(
            evaluator=evaluator,
            maxdet=maxdet,
        )
    )

    metrics = {
        "AP_50_95":
            custom_metrics[
                "AP_50_95"
            ],

        "AP50":
            custom_metrics[
                "AP50"
            ],

        "AP75":
            custom_metrics[
                "AP75"
            ],

        "AP_small":
            custom_metrics[
                "AP_small"
            ],

        "AP_medium":
            custom_metrics[
                "AP_medium"
            ],

        "AP_large":
            custom_metrics[
                "AP_large"
            ],

        "AR_1":
            get_ar_at_maxdet(
                evaluator,
                1,
                area_index=0,
            ),

        "AR_10":
            get_ar_at_maxdet(
                evaluator,
                10,
                area_index=0,
            ),

        f"AR_{maxdet}":
            custom_metrics[
                f"AR_{maxdet}"
            ],

        "AR_small":
            custom_metrics[
                "AR_small"
            ],

        "AR_medium":
            custom_metrics[
                "AR_medium"
            ],

        "AR_large":
            custom_metrics[
                "AR_large"
            ],
    }

    return metrics
