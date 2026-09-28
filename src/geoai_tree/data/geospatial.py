#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geospatial raster utilities for GeoAI-UrbanTree-RFDETR.

This module centralizes shared raster and coordinate helpers used by:

- COCO dataset preparation
- transfer-city dataset preparation
- GeoTIFF quality-control workflows
- full-area RF-DETR inference

The functions preserve the behavior of the original project scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import rowcol, xy
from rasterio.windows import Window
from shapely.geometry import box


# ============================================================
# Raster normalization
# ============================================================

def normalize_dataset_patch(
    patch: np.ndarray,
) -> np.ndarray:
    """
    Convert a raster patch to uint8 image format for dataset export.

    This preserves the normalization used in the original
    transfer-city COCO preparation script.

    Input
    -----
    patch:
        Array with shape:

            (bands, height, width)

    Output
    ------
    np.ndarray
        Array with shape:

            (height, width, bands)

        and dtype uint8.

    Behavior
    --------
    - uint8 input is preserved.
    - Non-uint8 input is stretched using the global
      1st and 99th percentiles of finite values.
    """

    patch = np.asarray(patch)

    if patch.ndim != 3:
        raise ValueError(
            "Raster patch must have shape "
            "(bands, height, width). "
            f"Received shape: {patch.shape}"
        )

    patch = np.transpose(
        patch,
        (1, 2, 0),
    )

    if patch.dtype != np.uint8:

        patch = patch.astype(
            np.float32
        )

        valid = np.isfinite(
            patch
        )

        if valid.sum() > 0:

            p_min = np.nanpercentile(
                patch[valid],
                1,
            )

            p_max = np.nanpercentile(
                patch[valid],
                99,
            )

        else:
            p_min = 0.0
            p_max = 1.0

        if p_max > p_min:

            patch = (
                (patch - p_min)
                / (p_max - p_min)
                * 255.0
            )

        patch = np.nan_to_num(
            patch,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        patch = np.clip(
            patch,
            0,
            255,
        ).astype(
            np.uint8
        )

    return patch


def normalize_inference_tile(
    tile: np.ndarray,
) -> np.ndarray:
    """
    Convert a raster tile to uint8 RGB for model inference.

    This preserves the normalization used by the original
    Miami full-AOI inference script.

    Input
    -----
    tile:
        Array with shape:

            (bands, height, width)

    Output
    ------
    np.ndarray
        Array with shape:

            (height, width, bands)

        and dtype uint8.

    Behavior
    --------
    If all finite raster values already fall inside 0–255,
    values are simply clipped.

    Otherwise, each band is independently stretched using
    its 2nd and 98th percentiles.
    """

    tile = np.asarray(
        tile
    )

    if tile.ndim != 3:
        raise ValueError(
            "Raster tile must have shape "
            "(bands, height, width). "
            f"Received shape: {tile.shape}"
        )

    tile = tile.astype(
        np.float32
    )

    finite = np.isfinite(tile)

    if not finite.any():
        return np.zeros(
            (
                tile.shape[1],
                tile.shape[2],
                tile.shape[0],
            ),
            dtype=np.uint8,
        )

    finite_values = tile[finite]

    if (
        np.max(finite_values) <= 255
        and np.min(finite_values) >= 0
    ):

        tile = np.clip(
            tile,
            0,
            255,
        )

    else:

        output = np.zeros_like(
            tile,
            dtype=np.float32,
        )

        for band_index in range(
            tile.shape[0]
        ):

            band = tile[
                band_index
            ]

            valid = np.isfinite(
                band
            )

            if valid.sum() == 0:

                output[
                    band_index
                ] = np.zeros_like(
                    band
                )

                continue

            p2, p98 = np.percentile(
                band[valid],
                (2, 98),
            )

            if p98 > p2:

                stretched = (
                    (band - p2)
                    / (p98 - p2)
                    * 255.0
                )

            else:

                stretched = np.zeros_like(
                    band
                )

            output[
                band_index
            ] = np.clip(
                stretched,
                0,
                255,
            )

        tile = output

    tile = np.nan_to_num(
        tile,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    tile_rgb = np.transpose(
        tile,
        (1, 2, 0),
    ).astype(
        np.uint8
    )

    return tile_rgb


# ============================================================
# Raster windows
# ============================================================

def build_full_tile_windows(
    width: int,
    height: int,
    tile_size: int = 256,
    keep_partial: bool = False,
) -> List[Window]:
    """
    Build non-overlapping raster tile windows.

    By default, edge tiles smaller than tile_size are skipped,
    matching the original project workflow.

    Parameters
    ----------
    width : int
        Raster width in pixels.

    height : int
        Raster height in pixels.

    tile_size : int
        Tile size in pixels.

    keep_partial : bool
        If True, include smaller edge tiles.

    Returns
    -------
    list of rasterio.windows.Window
    """

    if width <= 0 or height <= 0:
        raise ValueError(
            "Raster width and height must be positive."
        )

    if tile_size <= 0:
        raise ValueError(
            "tile_size must be positive."
        )

    windows: List[Window] = []

    for row_off in range(
        0,
        height,
        tile_size,
    ):

        for col_off in range(
            0,
            width,
            tile_size,
        ):

            window_width = min(
                tile_size,
                width - col_off,
            )

            window_height = min(
                tile_size,
                height - row_off,
            )

            if (
                not keep_partial
                and (
                    window_width != tile_size
                    or window_height != tile_size
                )
            ):
                continue

            windows.append(
                Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=window_width,
                    height=window_height,
                )
            )

    return windows


def build_center_test_windows(
    width: int,
    height: int,
    tile_size: int = 256,
    max_tiles: int = 100,
    radius: int = 5,
) -> List[Window]:
    """
    Build full-size test windows around the center of a raster.

    This preserves the logic used in the original Miami
    full-AOI inference script.

    It is useful for testing the complete inference pipeline
    before processing the entire GeoTIFF.
    """

    if max_tiles <= 0:
        return []

    center_x = width // 2
    center_y = height // 2

    candidate_windows = []

    for dy in range(
        -radius,
        radius + 1,
    ):

        for dx in range(
            -radius,
            radius + 1,
        ):

            x_off = (
                center_x
                + dx * tile_size
            )

            y_off = (
                center_y
                + dy * tile_size
            )

            # Align to the regular tile grid.
            x_off = (
                x_off // tile_size
            ) * tile_size

            y_off = (
                y_off // tile_size
            ) * tile_size

            if (
                x_off < 0
                or y_off < 0
            ):
                continue

            if (
                x_off + tile_size > width
                or y_off + tile_size > height
            ):
                continue

            candidate_windows.append(
                Window(
                    col_off=x_off,
                    row_off=y_off,
                    width=tile_size,
                    height=tile_size,
                )
            )

    # Remove duplicate windows while preserving order.
    unique_windows = {}

    for window in candidate_windows:

        key = (
            int(window.col_off),
            int(window.row_off),
        )

        if key not in unique_windows:
            unique_windows[key] = window

    return list(
        unique_windows.values()
    )[:max_tiles]


# ============================================================
# Raster window reading
# ============================================================

def read_rgb_window(
    src,
    window: Window,
    bands: Iterable[int] = (1, 2, 3),
    normalization: str = "dataset",
) -> np.ndarray:
    """
    Read selected raster bands from a rasterio dataset.

    Parameters
    ----------
    src
        Open rasterio dataset.

    window : Window
        Raster window.

    bands : iterable of int
        Raster bands to read.

    normalization : {"dataset", "inference"}
        Normalization method.

    Returns
    -------
    np.ndarray
        uint8 array with shape (height, width, bands).
    """

    bands = tuple(
        int(band)
        for band in bands
    )

    patch = src.read(
        bands,
        window=window,
        boundless=False,
    )

    if normalization == "dataset":

        return normalize_dataset_patch(
            patch
        )

    if normalization == "inference":

        return normalize_inference_tile(
            patch
        )

    raise ValueError(
        "normalization must be either "
        "'dataset' or 'inference'. "
        f"Received: {normalization}"
    )


def save_rgb_window(
    src,
    window: Window,
    output_path: str | Path,
    bands: Iterable[int] = (1, 2, 3),
) -> tuple[int, int, object]:
    """
    Read a raster window and save it as a PNG image.

    This helper is intended primarily for COCO dataset creation.

    Returns
    -------
    tuple
        (width, height, window_transform)
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    patch_rgb = read_rgb_window(
        src=src,
        window=window,
        bands=bands,
        normalization="dataset",
    )

    Image.fromarray(
        patch_rgb
    ).save(
        output_path
    )

    width = int(
        window.width
    )

    height = int(
        window.height
    )

    window_transform = (
        src.window_transform(
            window
        )
    )

    return (
        width,
        height,
        window_transform,
    )


# ============================================================
# Geospatial → pixel bounding box
# ============================================================

def geometry_to_pixel_bbox(
    label_geometry,
    patch_geometry,
    patch_transform,
    width: int,
    height: int,
):
    """
    Clip a geospatial label geometry to a raster patch and
    convert it to a COCO-format pixel bounding box.

    COCO output format:

        [x_min, y_min, width, height]

    Returns None when the geometry does not intersect the
    patch or produces a very small/invalid bounding box.
    """

    clipped = label_geometry.intersection(
        patch_geometry
    )

    if clipped.is_empty:
        return None

    minx, miny, maxx, maxy = (
        clipped.bounds
    )

    # rasterio.rowcol returns:
    #
    #     row, column
    #
    row_1, col_1 = rowcol(
        patch_transform,
        minx,
        maxy,
    )

    row_2, col_2 = rowcol(
        patch_transform,
        maxx,
        miny,
    )

    x_min = max(
        0,
        min(col_1, col_2),
    )

    y_min = max(
        0,
        min(row_1, row_2),
    )

    x_max = min(
        width,
        max(col_1, col_2),
    )

    y_max = min(
        height,
        max(row_1, row_2),
    )

    bbox_width = (
        x_max - x_min
    )

    bbox_height = (
        y_max - y_min
    )

    # Preserve the original project filtering behavior.
    if (
        bbox_width <= 1
        or bbox_height <= 1
    ):
        return None

    return [
        float(x_min),
        float(y_min),
        float(bbox_width),
        float(bbox_height),
    ]


# ============================================================
# Pixel bounding box → map geometry
# ============================================================

def pixel_bbox_to_map_polygon(
    transform,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
):
    """
    Convert a bounding box in full-raster pixel coordinates
    to a map-coordinate polygon.

    Parameters
    ----------
    transform
        Raster affine transform.

    x1, y1, x2, y2 : float
        Bounding-box coordinates in full-raster pixel space.

    Returns
    -------
    shapely Polygon
    """

    upper_left_x, upper_left_y = xy(
        transform,
        y1,
        x1,
        offset="center",
    )

    lower_right_x, lower_right_y = xy(
        transform,
        y2,
        x2,
        offset="center",
    )

    return box(
        upper_left_x,
        lower_right_y,
        lower_right_x,
        upper_left_y,
    )


# ============================================================
# Miscellaneous helpers
# ============================================================

def window_to_map_polygon(
    src,
    window: Window,
):
    """
    Convert a raster window to its map-coordinate polygon.
    """

    bounds = rasterio.windows.bounds(
        window,
        src.transform,
    )

    return box(
        *bounds
    )


def is_empty_raster_tile(
    tile: np.ndarray,
) -> bool:
    """
    Return True when every raster value in a tile is zero.

    This matches the skip behavior used by the original
    Miami full-AOI inference script.
    """

    return bool(
        np.all(tile == 0)
    )
