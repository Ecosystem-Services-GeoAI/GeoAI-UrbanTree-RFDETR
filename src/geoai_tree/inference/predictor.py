#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RF-DETR prediction utilities for GeoAI-UrbanTree-RFDETR.

This module provides one shared prediction interface for:

- TVT evaluation
- cross-validation evaluation
- transfer-city evaluation
- individual image prediction
- full-area GeoTIFF inference

The implementation preserves the RF-DETR inference behavior
used in the original project scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from rfdetr import RFDETRBase

from geoai_tree.utils.checkpoints import (
    validate_checkpoint,
)
from geoai_tree.utils.paths import (
    resolve_project_path,
)


# ============================================================
# Prediction result
# ============================================================

@dataclass
class PredictionResult:
    """
    Standardized RF-DETR prediction result.

    Attributes
    ----------
    xyxy : np.ndarray
        Bounding boxes with shape (N, 4) using:

            [x1, y1, x2, y2]

    confidence : np.ndarray
        Confidence scores with shape (N,).

    class_id : np.ndarray
        Predicted class IDs with shape (N,).
    """

    xyxy: np.ndarray
    confidence: np.ndarray
    class_id: np.ndarray

    def __len__(self) -> int:
        return len(self.xyxy)

    @classmethod
    def empty(cls) -> "PredictionResult":
        """
        Create an empty prediction result.
        """

        return cls(
            xyxy=np.zeros(
                (0, 4),
                dtype=np.float32,
            ),
            confidence=np.zeros(
                (0,),
                dtype=np.float32,
            ),
            class_id=np.zeros(
                (0,),
                dtype=np.int64,
            ),
        )


# ============================================================
# RF-DETR model loading
# ============================================================

def load_rfdetr_model(
    checkpoint_path: str | Path,
    architecture: str = "RFDETRBase",
    optimize: bool = True,
):
    """
    Load a trained RF-DETR model.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to trained RF-DETR checkpoint.

        Relative paths are interpreted relative to the
        repository root.

    architecture : str
        RF-DETR architecture name.

        Currently supported:

            RFDETRBase

    optimize : bool
        If True, call optimize_for_inference() after loading.

    Returns
    -------
    RFDETR model
        Loaded RF-DETR model.
    """

    checkpoint_path = resolve_project_path(
        checkpoint_path
    )

    checkpoint_path = validate_checkpoint(
        checkpoint_path
    )

    if architecture != "RFDETRBase":
        raise ValueError(
            "Unsupported RF-DETR architecture: "
            f"{architecture}. "
            "Currently supported: RFDETRBase"
        )

    print(
        f"Loading RF-DETR checkpoint: "
        f"{checkpoint_path}"
    )

    model = RFDETRBase(
        pretrain_weights=str(
            checkpoint_path
        )
    )

    if optimize:
        model.optimize_for_inference()

    return model


# ============================================================
# Prediction processing
# ============================================================

def process_rfdetr_detections(
    detections,
    max_detections: Optional[int] = 200,
) -> PredictionResult:
    """
    Convert RF-DETR detections into the project's standardized
    prediction format.

    Predictions are sorted by confidence from highest to lowest.

    Parameters
    ----------
    detections
        Object returned by RF-DETR model.predict().

    max_detections : int or None
        Maximum number of detections to retain.

        If None, all detections are retained.

    Returns
    -------
    PredictionResult
        Confidence-sorted predictions.
    """

    if detections is None:
        return PredictionResult.empty()

    if len(detections) == 0:
        return PredictionResult.empty()

    confidence = np.asarray(
        detections.confidence,
        dtype=np.float32,
    )

    xyxy = np.asarray(
        detections.xyxy,
        dtype=np.float32,
    )

    # Some workflows may not require class IDs, but RF-DETR
    # normally provides them.
    if hasattr(
        detections,
        "class_id",
    ):
        class_id = np.asarray(
            detections.class_id,
            dtype=np.int64,
        )
    else:
        class_id = np.zeros(
            len(confidence),
            dtype=np.int64,
        )

    # Highest confidence first.
    order = np.argsort(
        -confidence
    )

    if max_detections is not None:

        if max_detections <= 0:
            return PredictionResult.empty()

        order = order[
            :max_detections
        ]

    return PredictionResult(
        xyxy=xyxy[
            order
        ].astype(
            np.float32
        ),

        confidence=confidence[
            order
        ].astype(
            np.float32
        ),

        class_id=class_id[
            order
        ].astype(
            np.int64
        ),
    )


# ============================================================
# Prediction
# ============================================================

def predict_image(
    model,
    image: Image.Image,
    threshold: float = 0.30,
    max_detections: Optional[int] = 200,
) -> PredictionResult:
    """
    Run RF-DETR prediction on one image.

    Parameters
    ----------
    model
        Loaded RF-DETR model.

    image : PIL.Image.Image
        Input image.

    threshold : float
        RF-DETR confidence threshold.

    max_detections : int or None
        Maximum detections retained after sorting by confidence.

    Returns
    -------
    PredictionResult
        Standardized prediction result.
    """

    if not isinstance(
        image,
        Image.Image,
    ):
        raise TypeError(
            "image must be a PIL.Image.Image."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1. "
            f"Received: {threshold}"
        )

    # Ensure consistent RGB input.
    image = image.convert(
        "RGB"
    )

    detections = model.predict(
        image,
        threshold=threshold,
    )

    return process_rfdetr_detections(
        detections=detections,
        max_detections=max_detections,
    )


def predict_image_path(
    model,
    image_path: str | Path,
    threshold: float = 0.30,
    max_detections: Optional[int] = 200,
) -> PredictionResult:
    """
    Run RF-DETR prediction on an image file.

    Parameters
    ----------
    model
        Loaded RF-DETR model.

    image_path : str or Path
        Input image path.

    threshold : float
        Confidence threshold.

    max_detections : int or None
        Maximum number of detections.

    Returns
    -------
    PredictionResult
        Standardized prediction result.
    """

    image_path = Path(
        image_path
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(
        image_path
    ) as image:

        image = image.convert(
            "RGB"
        )

        return predict_image(
            model=model,
            image=image,
            threshold=threshold,
            max_detections=max_detections,
        )


# ============================================================
# Threshold filtering
# ============================================================

def filter_predictions(
    predictions: PredictionResult,
    threshold: float,
) -> PredictionResult:
    """
    Apply an additional confidence threshold to predictions.

    This is especially useful for the project's evaluation
    workflow:

        COCO AP/AR prediction threshold = 0.001

    while:

        F1 / visualization threshold = 0.30

    This allows one low-threshold model prediction to be reused
    for both purposes.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1. "
            f"Received: {threshold}"
        )

    if len(predictions) == 0:
        return PredictionResult.empty()

    keep = (
        predictions.confidence
        >= threshold
    )

    return PredictionResult(
        xyxy=predictions.xyxy[
            keep
        ].copy(),

        confidence=predictions.confidence[
            keep
        ].copy(),

        class_id=predictions.class_id[
            keep
        ].copy(),
    )


# ============================================================
# COCO conversion
# ============================================================

def predictions_to_coco(
    predictions: PredictionResult,
    image_id: int,
    category_id: int = 1,
) -> list[dict]:
    """
    Convert predictions to COCO detection-result format.

    RF-DETR xyxy:

        [x1, y1, x2, y2]

    becomes COCO:

        [x, y, width, height]

    Parameters
    ----------
    predictions : PredictionResult
        RF-DETR predictions.

    image_id : int
        COCO image ID.

    category_id : int
        COCO category ID.

    Returns
    -------
    list of dict
        COCO-format detections.
    """

    coco_results = []

    for box_xyxy, score in zip(
        predictions.xyxy,
        predictions.confidence,
    ):

        x1, y1, x2, y2 = (
            box_xyxy
        )

        width = float(
            max(
                0.0,
                x2 - x1,
            )
        )

        height = float(
            max(
                0.0,
                y2 - y1,
            )
        )

        if (
            width <= 0
            or height <= 0
        ):
            continue

        coco_results.append(
            {
                "image_id": int(
                    image_id
                ),
                "category_id": int(
                    category_id
                ),
                "bbox": [
                    float(x1),
                    float(y1),
                    width,
                    height,
                ],
                "score": float(
                    score
                ),
            }
        )

    return coco_results
