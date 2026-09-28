#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COCO dataset preparation for GeoAI-UrbanTree-RFDETR.

This module converts:

    GeoTIFF imagery
    +
    vector tree labels

into:

    COCO object-detection datasets

The implementation preserves the preparation logic used in the
original transfer-city workflow while making paths configurable.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import rasterio
from shapely.geometry import box
from tqdm import tqdm

from geoai_tree.data.geospatial import (
    build_full_tile_windows,
    geometry_to_pixel_bbox,
    save_rgb_window,
)
from geoai_tree.utils.paths import (
    resolve_project_path,
)


# ============================================================
# COCO template
# ============================================================

def make_coco_template(
    dataset_name: str,
) -> dict:
    """
    Create the one-class COCO template used by this project.
    """

    return {
        "info": {
            "description": (
                f"{dataset_name} NAIP Tree Detection Dataset"
            ),
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "tree",
                "supercategory": "tree",
            }
        ],
    }


# ============================================================
# Output directory handling
# ============================================================

def prepare_output_directory(
    output_dir: str | Path,
    clean: bool = False,
) -> Path:
    """
    Create an output directory.

    If clean=True, remove an existing directory first.
    """

    output_dir = resolve_project_path(
        output_dir
    )

    if clean and output_dir.exists():
        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


# ============================================================
# Label loading
# ============================================================

def load_labels_for_raster(
    labels_path: str | Path,
    raster_crs,
) -> gpd.GeoDataFrame:
    """
    Load vector labels and align them to a raster CRS.
    """

    labels_path = resolve_project_path(
        labels_path
    )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {labels_path}"
        )

    labels = gpd.read_file(
        labels_path
    )

    # Remove missing geometries.
    labels = labels[
        labels.geometry.notnull()
    ].copy()

    # Repair invalid geometries where possible.
    labels["geometry"] = (
        labels.geometry.make_valid()
    )

    # Remove geometries that became empty.
    labels = labels[
        ~labels.geometry.is_empty
    ].copy()

    if labels.crs is None:
        raise ValueError(
            f"Label file has no CRS: {labels_path}"
        )

    if raster_crs is None:
        raise ValueError(
            "Raster has no CRS."
        )

    if labels.crs != raster_crs:

        print(
            "Reprojecting labels to raster CRS..."
        )

        labels = labels.to_crs(
            raster_crs
        )

    # Build spatial index now so the first tile does not pay
    # the initialization cost.
    _ = labels.sindex

    return labels


# ============================================================
# Raster discovery
# ============================================================

def find_raster_files(
    raster_paths: Sequence[str | Path] | None = None,
    raster_dir: str | Path | None = None,
    raster_pattern: str = "*.tif",
) -> list[Path]:
    """
    Resolve raster inputs.

    Either provide:

        raster_paths

    or:

        raster_dir + raster_pattern
    """

    if raster_paths is not None:

        paths = [
            resolve_project_path(
                path
            )
            for path in raster_paths
        ]

    elif raster_dir is not None:

        raster_dir = resolve_project_path(
            raster_dir
        )

        if not raster_dir.exists():
            raise FileNotFoundError(
                f"Raster directory not found: "
                f"{raster_dir}"
            )

        paths = sorted(
            raster_dir.glob(
                raster_pattern
            )
        )

    else:

        raise ValueError(
            "Provide either raster_paths or raster_dir."
        )

    paths = [
        Path(path)
        for path in paths
    ]

    missing = [
        path
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing raster file(s):\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    if len(paths) == 0:
        raise ValueError(
            "No raster files were found."
        )

    return sorted(
        paths
    )


# ============================================================
# COCO dataset builder
# ============================================================

def build_coco_dataset(
    dataset_name: str,
    raster_paths: Sequence[str | Path],
    labels_path: str | Path,
    output_dir: str | Path,
    split_name: str = "test",
    tile_size: int = 256,
    bands: Iterable[int] = (1, 2, 3),
    min_bbox_area: float = 4.0,
    skip_empty: bool = False,
    keep_partial: bool = False,
    clean: bool = False,
) -> dict:
    """
    Convert GeoTIFF imagery + vector tree labels to COCO.

    Output
    ------
    output_dir/
        split_name/
            images/
            _annotations.coco.json
    """

    if tile_size <= 0:
        raise ValueError(
            "tile_size must be greater than zero."
        )

    if min_bbox_area < 0:
        raise ValueError(
            "min_bbox_area cannot be negative."
        )

    bands = tuple(
        int(band)
        for band in bands
    )

    if len(bands) != 3:
        raise ValueError(
            "Exactly three raster bands are required."
        )

    raster_paths = [
        resolve_project_path(
            path
        )
        for path in raster_paths
    ]

    if len(raster_paths) == 0:
        raise ValueError(
            "No raster files supplied."
        )

    for raster_path in raster_paths:

        if not raster_path.exists():
            raise FileNotFoundError(
                f"Raster not found: {raster_path}"
            )

    output_dir = prepare_output_directory(
        output_dir=output_dir,
        clean=clean,
    )

    split_dir = (
        output_dir
        / split_name
    )

    images_dir = (
        split_dir
        / "images"
    )

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    coco = make_coco_template(
        dataset_name
    )

    # --------------------------------------------------------
    # Load labels using first raster CRS
    # --------------------------------------------------------

    with rasterio.open(
        raster_paths[0]
    ) as first_src:

        labels = load_labels_for_raster(
            labels_path=labels_path,
            raster_crs=first_src.crs,
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "COCO DATASET PREPARATION"
    )

    print(
        "=" * 100
    )

    print(
        f"Dataset:        {dataset_name}"
    )

    print(
        f"Raster count:   {len(raster_paths)}"
    )

    print(
        f"Label count:    {len(labels)}"
    )

    print(
        f"Split:          {split_name}"
    )

    print(
        f"Tile size:      {tile_size}"
    )

    print(
        f"Bands:          {bands}"
    )

    print(
        f"Min bbox area:  {min_bbox_area}"
    )

    print(
        f"Skip empty:     {skip_empty}"
    )

    print(
        f"Keep partial:   {keep_partial}"
    )

    print(
        f"Output:         {split_dir}"
    )

    print(
        "=" * 100
    )

    image_id = 1
    annotation_id = 1
    global_patch_id = 1

    kept_patch_count = 0
    skipped_empty_count = 0

    # --------------------------------------------------------
    # Raster loop
    # --------------------------------------------------------

    for raster_index, raster_path in enumerate(
        raster_paths,
        start=1,
    ):

        raster_path = Path(
            raster_path
        )

        with rasterio.open(
            raster_path
        ) as src:

            # Mixed-CRS protection.
            if src.crs != labels.crs:

                labels_for_raster = (
                    labels.to_crs(
                        src.crs
                    )
                )

                _ = (
                    labels_for_raster.sindex
                )

            else:

                labels_for_raster = (
                    labels
                )

            windows = (
                build_full_tile_windows(
                    width=src.width,
                    height=src.height,
                    tile_size=tile_size,
                    keep_partial=keep_partial,
                )
            )

            for window in tqdm(
                windows,
                desc=(
                    f"{dataset_name}: "
                    f"{raster_path.name}"
                ),
            ):

                patch_bounds = (
                    rasterio.windows.bounds(
                        window,
                        src.transform,
                    )
                )

                patch_geometry = box(
                    *patch_bounds
                )

                # ------------------------------------------------
                # Spatial-index candidate search
                # ------------------------------------------------

                candidate_indices = list(
                    labels_for_raster.sindex.query(
                        patch_geometry,
                        predicate="intersects",
                    )
                )

                candidate_labels = (
                    labels_for_raster.iloc[
                        candidate_indices
                    ]
                )

                width = int(
                    window.width
                )

                height = int(
                    window.height
                )

                patch_transform = (
                    src.window_transform(
                        window
                    )
                )

                annotations_this_image = []

                # ------------------------------------------------
                # Labels → COCO bounding boxes
                # ------------------------------------------------

                for _, label_row in (
                    candidate_labels.iterrows()
                ):

                    bbox = (
                        geometry_to_pixel_bbox(
                            label_geometry=(
                                label_row.geometry
                            ),
                            patch_geometry=(
                                patch_geometry
                            ),
                            patch_transform=(
                                patch_transform
                            ),
                            width=width,
                            height=height,
                        )
                    )

                    if bbox is None:
                        continue

                    bbox_area = (
                        bbox[2]
                        * bbox[3]
                    )

                    if (
                        bbox_area
                        < min_bbox_area
                    ):
                        continue

                    annotation = {
                        "id": int(
                            annotation_id
                        ),
                        "image_id": int(
                            image_id
                        ),
                        "category_id": 1,
                        "bbox": bbox,
                        "area": float(
                            bbox_area
                        ),
                        "iscrowd": 0,
                        "segmentation": [],
                    }

                    annotations_this_image.append(
                        annotation
                    )

                    annotation_id += 1

                # ------------------------------------------------
                # Optional empty-patch removal
                # ------------------------------------------------

                if (
                    skip_empty
                    and len(
                        annotations_this_image
                    ) == 0
                ):

                    skipped_empty_count += 1
                    global_patch_id += 1

                    continue

                # ------------------------------------------------
                # Save RGB PNG
                # ------------------------------------------------

                image_filename = (
                    f"{dataset_name}_"
                    f"tif{raster_index:03d}_"
                    f"patch_{global_patch_id:06d}.png"
                )

                image_path = (
                    images_dir
                    / image_filename
                )

                save_rgb_window(
                    src=src,
                    window=window,
                    output_path=image_path,
                    bands=bands,
                )

                # ------------------------------------------------
                # COCO image record
                # ------------------------------------------------

                coco["images"].append(
                    {
                        "id": int(
                            image_id
                        ),
                        "file_name": (
                            f"images/"
                            f"{image_filename}"
                        ),
                        "width": width,
                        "height": height,
                    }
                )

                coco[
                    "annotations"
                ].extend(
                    annotations_this_image
                )

                image_id += 1
                global_patch_id += 1
                kept_patch_count += 1

    # ========================================================
    # Save COCO JSON
    # ========================================================

    annotation_path = (
        split_dir
        / "_annotations.coco.json"
    )

    annotation_path.write_text(
        json.dumps(
            coco,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "raster_count": int(
            len(raster_paths)
        ),
        "source_label_count": int(
            len(labels)
        ),
        "images": int(
            len(
                coco["images"]
            )
        ),
        "annotations": int(
            len(
                coco["annotations"]
            )
        ),
        "kept_patches": int(
            kept_patch_count
        ),
        "skipped_empty_patches": int(
            skipped_empty_count
        ),
        "tile_size": int(
            tile_size
        ),
        "bands": list(
            bands
        ),
        "min_bbox_area": float(
            min_bbox_area
        ),
        "skip_empty": bool(
            skip_empty
        ),
        "keep_partial": bool(
            keep_partial
        ),
        "annotation_file": str(
            annotation_path
        ),
    }

    summary_path = (
        split_dir
        / "preparation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "-" * 100
    )

    print(
        f"Finished:        {dataset_name}"
    )

    print(
        f"Images:          "
        f"{len(coco['images'])}"
    )

    print(
        f"Annotations:     "
        f"{len(coco['annotations'])}"
    )

    print(
        f"Skipped empty:   "
        f"{skipped_empty_count}"
    )

    print(
        f"COCO JSON:       "
        f"{annotation_path}"
    )

    print(
        f"Summary:         "
        f"{summary_path}"
    )

    print(
        "-" * 100
    )

    return summary
