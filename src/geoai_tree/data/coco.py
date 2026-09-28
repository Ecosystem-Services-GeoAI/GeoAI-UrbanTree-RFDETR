#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COCO dataset utilities for GeoAI-UrbanTree-RFDETR.

This module provides reusable functions for:

- loading COCO annotation JSON files
- indexing image IDs and filenames
- indexing ground-truth bounding boxes
- converting COCO xywh boxes to xyxy format
- resolving image paths robustly

The functions are shared by training validation, evaluation,
visualization, transfer testing, and inference workflows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


# ============================================================
# COCO data structure
# ============================================================

@dataclass
class CocoIndex:
    """
    Indexed representation of a COCO annotation file.

    Attributes
    ----------
    id_to_file : dict
        image_id -> COCO file_name

    base_to_id : dict
        image basename -> image_id

    gt_boxes : dict
        image_id -> NumPy array of bounding boxes in xyxy format

    categories : dict
        category_id -> category name
    """

    id_to_file: Dict[int, str]
    base_to_id: Dict[str, int]
    gt_boxes: Dict[int, np.ndarray]
    categories: Dict[int, str]


# ============================================================
# Basic COCO loading
# ============================================================

def load_coco_json(annotation_path: str | Path) -> dict:
    """
    Load a COCO annotation JSON file.

    Parameters
    ----------
    annotation_path : str or Path
        Path to the COCO JSON file.

    Returns
    -------
    dict
        Parsed COCO JSON dictionary.
    """

    annotation_path = Path(annotation_path)

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"COCO annotation file not found: {annotation_path}"
        )

    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    return coco


# ============================================================
# Bounding-box conversion
# ============================================================

def coco_bbox_to_xyxy(bbox_xywh: List[float]) -> List[float]:
    """
    Convert a COCO bounding box from xywh to xyxy.

    COCO format
    -----------
    [x, y, width, height]

    Output
    ------
    [x1, y1, x2, y2]
    """

    if len(bbox_xywh) != 4:
        raise ValueError(
            f"COCO bbox must contain 4 values, got: {bbox_xywh}"
        )

    x, y, w, h = bbox_xywh

    return [
        float(x),
        float(y),
        float(x + w),
        float(y + h),
    ]


# ============================================================
# COCO indexing
# ============================================================

def load_coco_index(annotation_path: str | Path) -> CocoIndex:
    """
    Build reusable indexes from a COCO annotation file.

    Parameters
    ----------
    annotation_path : str or Path
        Path to _annotations.coco.json.

    Returns
    -------
    CocoIndex
        Indexed COCO dataset information.
    """

    coco = load_coco_json(annotation_path)

    id_to_file: Dict[int, str] = {}
    base_to_id: Dict[str, int] = {}

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    for image_record in coco.get("images", []):
        image_id = int(image_record["id"])
        file_name = str(image_record["file_name"])

        id_to_file[image_id] = file_name
        base_to_id[Path(file_name).name] = image_id

    # --------------------------------------------------------
    # Ground-truth boxes
    # --------------------------------------------------------

    boxes_by_image: Dict[int, List[List[float]]] = {}

    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])

        bbox_xyxy = coco_bbox_to_xyxy(
            annotation["bbox"]
        )

        boxes_by_image.setdefault(
            image_id,
            []
        ).append(bbox_xyxy)

    gt_boxes: Dict[int, np.ndarray] = {}

    # Include every image, including images with zero annotations.
    for image_id in id_to_file.keys():

        boxes = boxes_by_image.get(
            image_id,
            []
        )

        if boxes:
            gt_boxes[image_id] = np.asarray(
                boxes,
                dtype=np.float32,
            )
        else:
            gt_boxes[image_id] = np.zeros(
                (0, 4),
                dtype=np.float32,
            )

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    categories: Dict[int, str] = {}

    for category in coco.get("categories", []):
        category_id = int(category["id"])
        category_name = str(category["name"])

        categories[category_id] = category_name

    return CocoIndex(
        id_to_file=id_to_file,
        base_to_id=base_to_id,
        gt_boxes=gt_boxes,
        categories=categories,
    )


# ============================================================
# Image path resolution
# ============================================================

def resolve_image_path(
    split_dir: str | Path,
    file_name: str,
) -> Path:
    """
    Resolve the actual image path referenced by a COCO file.

    This supports both common COCO filename styles:

        patch_123.png

    and

        images/patch_123.png

    Expected directory structure:

        split_dir/
            images/
            _annotations.coco.json

    Parameters
    ----------
    split_dir : str or Path
        Dataset split directory.

    file_name : str
        file_name stored in the COCO JSON.

    Returns
    -------
    Path
        Existing image path.

    Raises
    ------
    FileNotFoundError
        If the referenced image cannot be located.
    """

    split_dir = Path(split_dir)

    # Normalize Windows-style separators if they appear in JSON.
    file_name = str(file_name).replace("\\", "/")

    candidates = [
        # Case 1:
        # COCO already contains "images/..."
        split_dir / file_name,

        # Case 2:
        # COCO contains only the image filename
        split_dir / "images" / file_name,

        # Case 3:
        # Defensive fallback using basename only
        split_dir / "images" / Path(file_name).name,
    ]

    # Avoid testing duplicate paths multiple times.
    unique_candidates = []

    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    attempted_paths = "\n".join(
        f"  - {path}"
        for path in unique_candidates
    )

    raise FileNotFoundError(
        f"Image referenced by COCO JSON was not found.\n"
        f"file_name: {file_name}\n"
        f"split_dir: {split_dir}\n"
        f"Tried:\n{attempted_paths}"
    )


# ============================================================
# Convenience helpers
# ============================================================

def get_annotation_path(
    split_dir: str | Path,
    annotation_file: str = "_annotations.coco.json",
) -> Path:
    """
    Return the expected COCO annotation path for a dataset split.
    """

    return Path(split_dir) / annotation_file


def get_image_records(
    coco_index: CocoIndex,
):
    """
    Return image records as:

        [(image_id, file_name), ...]

    sorted by image ID for deterministic processing.
    """

    return sorted(
        coco_index.id_to_file.items(),
        key=lambda item: item[0],
    )
