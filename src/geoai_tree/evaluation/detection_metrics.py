#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detection metrics for GeoAI-UrbanTree-RFDETR.

This module provides reusable object-detection metrics used by:

- TVT evaluation
- cross-validation evaluation
- transfer-city evaluation

The implementation follows the logic used in the original project scripts:
one-to-one greedy matching between confidence-sorted predictions and
ground-truth boxes at a user-defined IoU threshold.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ============================================================
# IoU
# ============================================================

def iou_xyxy(
    box_a: np.ndarray,
    box_b: np.ndarray,
) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.

    Parameters
    ----------
    box_a : array-like
        Bounding box in [x1, y1, x2, y2] format.

    box_b : array-like
        Bounding box in [x1, y1, x2, y2] format.

    Returns
    -------
    float
        IoU value between 0 and 1.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(
        0.0,
        inter_x2 - inter_x1,
    )

    inter_height = max(
        0.0,
        inter_y2 - inter_y1,
    )

    intersection = (
        inter_width
        * inter_height
    )

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
        + 1e-9
    )

    return float(
        intersection / union
    )


# ============================================================
# Prediction / ground-truth matching
# ============================================================

def match_predictions_to_gt(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.50,
) -> Tuple[int, int, int]:
    """
    Match predicted boxes to ground-truth boxes using one-to-one
    greedy IoU matching.

    Important
    ---------
    Prediction boxes should already be sorted by confidence
    from highest to lowest before calling this function.

    Each ground-truth box can be matched only once.

    Parameters
    ----------
    pred_boxes : np.ndarray
        Predicted bounding boxes with shape (N, 4), using
        [x1, y1, x2, y2].

    gt_boxes : np.ndarray
        Ground-truth bounding boxes with shape (M, 4), using
        [x1, y1, x2, y2].

    iou_threshold : float
        Minimum IoU required for a true-positive match.

    Returns
    -------
    tuple
        (true_positives, false_positives, false_negatives)
    """

    if pred_boxes is None:
        pred_boxes = np.zeros(
            (0, 4),
            dtype=np.float32,
        )

    if gt_boxes is None:
        gt_boxes = np.zeros(
            (0, 4),
            dtype=np.float32,
        )

    pred_boxes = np.asarray(
        pred_boxes,
        dtype=np.float32,
    )

    gt_boxes = np.asarray(
        gt_boxes,
        dtype=np.float32,
    )

    num_predictions = len(pred_boxes)
    num_ground_truth = len(gt_boxes)

    # --------------------------------------------------------
    # Empty cases
    # --------------------------------------------------------

    if (
        num_predictions == 0
        and num_ground_truth == 0
    ):
        return 0, 0, 0

    if num_predictions == 0:
        return (
            0,
            0,
            num_ground_truth,
        )

    if num_ground_truth == 0:
        return (
            0,
            num_predictions,
            0,
        )

    # --------------------------------------------------------
    # Compute IoU matrix
    # --------------------------------------------------------

    iou_matrix = np.zeros(
        (
            num_predictions,
            num_ground_truth,
        ),
        dtype=np.float32,
    )

    for pred_index in range(
        num_predictions
    ):
        for gt_index in range(
            num_ground_truth
        ):
            iou_matrix[
                pred_index,
                gt_index,
            ] = iou_xyxy(
                pred_boxes[pred_index],
                gt_boxes[gt_index],
            )

    # --------------------------------------------------------
    # Greedy one-to-one matching
    # --------------------------------------------------------

    matched_gt = set()
    true_positives = 0

    for pred_index in range(
        num_predictions
    ):

        best_gt_index = int(
            np.argmax(
                iou_matrix[pred_index]
            )
        )

        best_iou = float(
            iou_matrix[
                pred_index,
                best_gt_index,
            ]
        )

        if (
            best_iou >= iou_threshold
            and best_gt_index
            not in matched_gt
        ):
            true_positives += 1
            matched_gt.add(
                best_gt_index
            )

    false_positives = (
        num_predictions
        - true_positives
    )

    false_negatives = (
        num_ground_truth
        - true_positives
    )

    return (
        int(true_positives),
        int(false_positives),
        int(false_negatives),
    )


# ============================================================
# Precision / Recall / F1
# ============================================================

def precision_recall_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.

    Parameters
    ----------
    true_positives : int
        Number of true-positive detections.

    false_positives : int
        Number of false-positive detections.

    false_negatives : int
        Number of missed ground-truth objects.

    Returns
    -------
    tuple
        (precision, recall, f1)
    """

    precision = (
        true_positives
        / (
            true_positives
            + false_positives
            + 1e-9
        )
    )

    recall = (
        true_positives
        / (
            true_positives
            + false_negatives
            + 1e-9
        )
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
            + 1e-9
        )
    )

    return (
        float(precision),
        float(recall),
        float(f1),
    )


# ============================================================
# Combined helper
# ============================================================

def calculate_detection_metrics(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.50,
) -> dict:
    """
    Convenience function that performs matching and returns all
    detection metrics in one dictionary.

    Prediction boxes must already be sorted by confidence.

    Returns
    -------
    dict
        Dictionary containing:

        - tp
        - fp
        - fn
        - precision
        - recall
        - f1
    """

    tp, fp, fn = match_predictions_to_gt(
        pred_boxes=pred_boxes,
        gt_boxes=gt_boxes,
        iou_threshold=iou_threshold,
    )

    precision, recall, f1 = (
        precision_recall_f1(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
