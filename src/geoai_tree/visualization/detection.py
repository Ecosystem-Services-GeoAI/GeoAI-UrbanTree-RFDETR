#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detection visualization utilities for GeoAI-UrbanTree-RFDETR.

Shared by:

- TVT evaluation
- cross-validation evaluation
- transfer-city evaluation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def draw_boxes(
    image: Image.Image,
    boxes_xyxy: np.ndarray,
    color: str,
    width: int = 2,
) -> Image.Image:
    """
    Draw bounding boxes on an image.

    Parameters
    ----------
    image : PIL.Image.Image
        Input RGB image.

    boxes_xyxy : np.ndarray
        Bounding boxes with shape (N, 4) in
        [x1, y1, x2, y2] format.

    color : str
        PIL-compatible box color.

    width : int
        Rectangle line width.

    Returns
    -------
    PIL.Image.Image
        Copy of the image with boxes drawn.
    """

    output = image.copy()
    draw = ImageDraw.Draw(output)

    if boxes_xyxy is None:
        return output

    boxes_xyxy = np.asarray(
        boxes_xyxy,
        dtype=np.float32,
    )

    if len(boxes_xyxy) == 0:
        return output

    for x1, y1, x2, y2 in boxes_xyxy:

        draw.rectangle(
            [
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            ],
            outline=color,
            width=width,
        )

    return output


def make_side_by_side(
    left_image: Image.Image,
    right_image: Image.Image,
    gap: int = 20,
    background_color: str = "white",
) -> Image.Image:
    """
    Combine two images side-by-side.

    Parameters
    ----------
    left_image : PIL.Image.Image
        Left panel.

    right_image : PIL.Image.Image
        Right panel.

    gap : int
        Pixel gap between the two panels.

    background_color : str
        Background color for the combined image.

    Returns
    -------
    PIL.Image.Image
        Combined side-by-side image.
    """

    left_width, left_height = (
        left_image.size
    )

    right_width, right_height = (
        right_image.size
    )

    output_width = (
        left_width
        + gap
        + right_width
    )

    output_height = max(
        left_height,
        right_height,
    )

    combined = Image.new(
        "RGB",
        (
            output_width,
            output_height,
        ),
        background_color,
    )

    combined.paste(
        left_image,
        (0, 0),
    )

    combined.paste(
        right_image,
        (
            left_width + gap,
            0,
        ),
    )

    return combined


def save_side_by_side(
    left_image: Image.Image,
    right_image: Image.Image,
    output_path: str | Path,
    gap: int = 20,
    background_color: str = "white",
) -> Path:
    """
    Save two images as one side-by-side image.

    Returns
    -------
    Path
        Saved output path.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined = make_side_by_side(
        left_image=left_image,
        right_image=right_image,
        gap=gap,
        background_color=background_color,
    )

    combined.save(
        output_path
    )

    return output_path


def create_gt_prediction_visualization(
    image: Image.Image,
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    output_path: str | Path,
    gt_color: str = "lime",
    pred_color: str = "red",
    box_width: int = 2,
    panel_gap: int = 20,
) -> Path:
    """
    Create the standard project visualization:

        Left  = ground truth
        Right = prediction

    No title is added, matching the original project figures.
    """

    gt_image = draw_boxes(
        image=image,
        boxes_xyxy=gt_boxes,
        color=gt_color,
        width=box_width,
    )

    pred_image = draw_boxes(
        image=image,
        boxes_xyxy=pred_boxes,
        color=pred_color,
        width=box_width,
    )

    return save_side_by_side(
        left_image=gt_image,
        right_image=pred_image,
        output_path=output_path,
        gap=panel_gap,
    )
