#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full-AOI raster inference for GeoAI-UrbanTree-RFDETR.

This module runs RF-DETR on a GeoTIFF by:

1. Splitting the raster into non-overlapping tiles.
2. Reading RGB bands.
3. Normalizing each tile for inference.
4. Running RF-DETR prediction.
5. Converting tile detections to full-raster pixel coordinates.
6. Converting pixel bounding boxes to map-coordinate polygons.
7. Saving JSON, GeoJSON, and Shapefile outputs.

Test-mode outputs are written under:

    <output_root>/test/

Full-run outputs are written directly under:

    <output_root>/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

from geoai_tree.data.geospatial import (
    build_center_test_windows,
    build_full_tile_windows,
    is_empty_raster_tile,
    normalize_inference_tile,
    pixel_bbox_to_map_polygon,
)
from geoai_tree.inference.predictor import (
    load_rfdetr_model,
    predict_image,
)
from geoai_tree.utils.paths import (
    resolve_project_path,
)


# ============================================================
# Shapefile compatibility
# ============================================================

def prepare_shapefile_gdf(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Prepare output fields for ESRI Shapefile.

    Shapefile field names are limited to 10 characters.
    """

    rename_map = {
        "record_id": "rec_id",
        "source_tif": "src_tif",
        "model_name": "model",
        "checkpoint": "ckpt",
        "tile_id": "tile_id",
        "tile_x_off": "x_off",
        "tile_y_off": "y_off",
        "tile_size": "tile_size",
        "category_id": "cat_id",
        "score": "score",
        "x1_tile": "x1_tile",
        "y1_tile": "y1_tile",
        "x2_tile": "x2_tile",
        "y2_tile": "y2_tile",
        "x1_pixel": "x1_pix",
        "y1_pixel": "y1_pix",
        "x2_pixel": "x2_pix",
        "y2_pixel": "y2_pix",
    }

    keep_columns = [
        column
        for column in gdf.columns
        if column in rename_map
        or column == "geometry"
    ]

    output = gdf[
        keep_columns
    ].copy()

    output = output.rename(
        columns=rename_map
    )

    if "ckpt" in output.columns:
        output["ckpt"] = (
            output["ckpt"]
            .astype(str)
            .str[-80:]
        )

    if "src_tif" in output.columns:
        output["src_tif"] = (
            output["src_tif"]
            .astype(str)
            .str[:80]
        )

    return output


# ============================================================
# Output path handling
# ============================================================

def get_inference_output_paths(
    output_root: str | Path,
    test_mode: bool,
):
    """
    Build output paths for either test mode or full inference.

    Test mode:

        output_root/test/

    Full mode:

        output_root/
    """

    output_root = Path(
        output_root
    )

    if test_mode:
        run_root = (
            output_root
            / "test"
        )
    else:
        run_root = output_root

    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if test_mode:
        prefix = (
            "test_predictions"
        )
    else:
        prefix = (
            "miami_030_rfdetr_predictions"
        )

    return {
        "root": run_root,
        "json": (
            run_root
            / f"{prefix}.json"
        ),
        "geojson": (
            run_root
            / f"{prefix}.geojson"
        ),
        "shapefile": (
            run_root
            / f"{prefix}.shp"
        ),
        "test_tiles": (
            run_root
            / "test_tiles"
        ),
    }


# ============================================================
# Raster inference
# ============================================================

def run_raster_inference(
    raster_path: str | Path,
    checkpoint_path: str | Path,
    output_root: str | Path,
    architecture: str = "RFDETRBase",
    bands: Iterable[int] = (1, 2, 3),
    tile_size: int = 256,
    confidence_threshold: float = 0.30,
    max_detections: int = 200,
    test_mode: bool = False,
    max_test_tiles: int = 100,
    save_test_tiles: bool = False,
    save_json: bool = True,
    save_geojson: bool = True,
    save_shapefile: bool = True,
    model_name: str = "tvt",
):
    """
    Run RF-DETR inference on a GeoTIFF.

    Parameters
    ----------
    raster_path
        Input GeoTIFF.

    checkpoint_path
        RF-DETR checkpoint.

    output_root
        Root output directory.

    architecture
        RF-DETR architecture name.

    bands
        Raster bands used as RGB input.

    tile_size
        Non-overlapping tile size in pixels.

    confidence_threshold
        Prediction confidence threshold.

    max_detections
        Maximum detections retained per tile.

    test_mode
        If True, run only center test tiles.

    max_test_tiles
        Maximum number of center test tiles.

    save_test_tiles
        Save test tile PNGs when test_mode=True.

    save_json
        Save non-spatial JSON records.

    save_geojson
        Save GeoJSON polygons.

    save_shapefile
        Save ESRI Shapefile.

    model_name
        Name recorded in output attributes.

    Returns
    -------
    dict
        Summary information and output paths.
    """

    raster_path = resolve_project_path(
        raster_path
    )

    checkpoint_path = resolve_project_path(
        checkpoint_path
    )

    output_root = resolve_project_path(
        output_root
    )

    if not raster_path.exists():
        raise FileNotFoundError(
            f"Raster not found: {raster_path}"
        )

    if tile_size <= 0:
        raise ValueError(
            "tile_size must be greater than zero."
        )

    if max_detections <= 0:
        raise ValueError(
            "max_detections must be greater than zero."
        )

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError(
            "confidence_threshold must be between 0 and 1."
        )

    if test_mode and max_test_tiles <= 0:
        raise ValueError(
            "max_test_tiles must be greater than zero."
        )

    bands = tuple(
        int(band)
        for band in bands
    )

    output_paths = (
        get_inference_output_paths(
            output_root=output_root,
            test_mode=test_mode,
        )
    )

    if (
        test_mode
        and save_test_tiles
    ):
        output_paths[
            "test_tiles"
        ].mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_rfdetr_model(
        checkpoint_path=checkpoint_path,
        architecture=architecture,
    )

    json_records = []
    geo_records = []

    processed_tiles = 0
    skipped_empty_tiles = 0

    # --------------------------------------------------------
    # Open raster
    # --------------------------------------------------------

    with rasterio.open(
        raster_path
    ) as src:

        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs

        print(
            "\n"
            + "=" * 100
        )

        print(
            "RF-DETR FULL-AOI RASTER INFERENCE"
        )

        print(
            "=" * 100
        )

        print(
            f"Raster:      {raster_path}"
        )

        print(
            f"Raster size: {width} x {height}"
        )

        print(
            f"CRS:         {crs}"
        )

        print(
            f"Bands:       {src.count}"
        )

        print(
            f"RGB bands:   {bands}"
        )

        print(
            f"Tile size:   {tile_size}"
        )

        print(
            f"Threshold:   {confidence_threshold}"
        )

        print(
            f"MAXDET:      {max_detections}"
        )

        print(
            f"Test mode:   {test_mode}"
        )

        print(
            f"Output root: {output_paths['root']}"
        )

        print(
            "=" * 100
        )

        # ----------------------------------------------------
        # Build windows
        # ----------------------------------------------------

        if test_mode:

            windows = (
                build_center_test_windows(
                    width=width,
                    height=height,
                    tile_size=tile_size,
                    max_tiles=max_test_tiles,
                )
            )

        else:

            windows = (
                build_full_tile_windows(
                    width=width,
                    height=height,
                    tile_size=tile_size,
                    keep_partial=False,
                )
            )

        print(
            f"Total tiles to process: "
            f"{len(windows)}"
        )

        # ----------------------------------------------------
        # Tile loop
        # ----------------------------------------------------

        for tile_id, window in enumerate(
            tqdm(
                windows,
                desc="RF-DETR raster inference",
            )
        ):

            x_off = int(
                window.col_off
            )

            y_off = int(
                window.row_off
            )

            raster_tile = src.read(
                bands,
                window=window,
                boundless=False,
            )

            if is_empty_raster_tile(
                raster_tile
            ):
                skipped_empty_tiles += 1
                continue

            tile_rgb = (
                normalize_inference_tile(
                    raster_tile
                )
            )

            if (
                test_mode
                and save_test_tiles
            ):

                tile_path = (
                    output_paths[
                        "test_tiles"
                    ]
                    / (
                        f"tile_{tile_id:06d}_"
                        f"x{x_off}_"
                        f"y{y_off}.png"
                    )
                )

                Image.fromarray(
                    tile_rgb
                ).save(
                    tile_path
                )

            image = (
                Image.fromarray(
                    tile_rgb
                ).convert(
                    "RGB"
                )
            )

            predictions = (
                predict_image(
                    model=model,
                    image=image,
                    threshold=(
                        confidence_threshold
                    ),
                    max_detections=(
                        max_detections
                    ),
                )
            )

            processed_tiles += 1

            if len(
                predictions
            ) == 0:
                continue

            # ------------------------------------------------
            # Convert detections to full-raster coordinates
            # ------------------------------------------------

            for det_id, (
                box_xyxy,
                score,
                class_id,
            ) in enumerate(
                zip(
                    predictions.xyxy,
                    predictions.confidence,
                    predictions.class_id,
                )
            ):

                (
                    x1_tile,
                    y1_tile,
                    x2_tile,
                    y2_tile,
                ) = [
                    float(value)
                    for value
                    in box_xyxy
                ]

                if (
                    x2_tile <= x1_tile
                    or y2_tile <= y1_tile
                ):
                    continue

                x1_global = (
                    x1_tile
                    + x_off
                )

                y1_global = (
                    y1_tile
                    + y_off
                )

                x2_global = (
                    x2_tile
                    + x_off
                )

                y2_global = (
                    y2_tile
                    + y_off
                )

                geometry = (
                    pixel_bbox_to_map_polygon(
                        transform=transform,
                        x1=x1_global,
                        y1=y1_global,
                        x2=x2_global,
                        y2=y2_global,
                    )
                )

                record_id = (
                    f"{model_name}_"
                    f"tile_{tile_id}_"
                    f"det_{det_id}"
                )

                record = {
                    "record_id": (
                        record_id
                    ),
                    "source_tif": (
                        raster_path.name
                    ),
                    "model_name": (
                        model_name
                    ),
                    "checkpoint": (
                        str(
                            checkpoint_path
                        )
                    ),
                    "tile_id": (
                        int(
                            tile_id
                        )
                    ),
                    "tile_x_off": (
                        int(
                            x_off
                        )
                    ),
                    "tile_y_off": (
                        int(
                            y_off
                        )
                    ),
                    "tile_size": (
                        int(
                            tile_size
                        )
                    ),
                    "category_id": (
                        int(
                            class_id
                        )
                    ),
                    "score": (
                        float(
                            score
                        )
                    ),
                    "x1_tile": (
                        x1_tile
                    ),
                    "y1_tile": (
                        y1_tile
                    ),
                    "x2_tile": (
                        x2_tile
                    ),
                    "y2_tile": (
                        y2_tile
                    ),
                    "x1_pixel": (
                        float(
                            x1_global
                        )
                    ),
                    "y1_pixel": (
                        float(
                            y1_global
                        )
                    ),
                    "x2_pixel": (
                        float(
                            x2_global
                        )
                    ),
                    "y2_pixel": (
                        float(
                            y2_global
                        )
                    ),
                    "crs": (
                        str(
                            crs
                        )
                    ),
                }

                json_records.append(
                    record
                )

                geo_records.append(
                    {
                        **record,
                        "geometry": (
                            geometry
                        ),
                    }
                )

    # ========================================================
    # Save JSON
    # ========================================================

    if save_json:

        output_paths[
            "json"
        ].write_text(
            json.dumps(
                json_records,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"\nSaved JSON: "
            f"{output_paths['json']}"
        )

    # ========================================================
    # Save geospatial outputs
    # ========================================================

    if len(
        geo_records
    ) > 0:

        gdf = (
            gpd.GeoDataFrame(
                geo_records,
                geometry="geometry",
                crs=crs,
            )
        )

        if save_geojson:

            gdf.to_file(
                output_paths[
                    "geojson"
                ],
                driver="GeoJSON",
            )

            print(
                f"Saved GeoJSON: "
                f"{output_paths['geojson']}"
            )

        if save_shapefile:

            shp_gdf = (
                prepare_shapefile_gdf(
                    gdf
                )
            )

            shp_gdf.to_file(
                output_paths[
                    "shapefile"
                ],
                driver="ESRI Shapefile",
            )

            print(
                f"Saved Shapefile: "
                f"{output_paths['shapefile']}"
            )

    else:

        print(
            "\nNo detections were found."
        )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "raster": str(
            raster_path
        ),
        "checkpoint": str(
            checkpoint_path
        ),
        "output_root": str(
            output_paths[
                "root"
            ]
        ),
        "test_mode": bool(
            test_mode
        ),
        "total_windows": int(
            len(
                windows
            )
        ),
        "processed_tiles": int(
            processed_tiles
        ),
        "skipped_empty_tiles": int(
            skipped_empty_tiles
        ),
        "detections": int(
            len(
                json_records
            )
        ),
        "tile_size": int(
            tile_size
        ),
        "confidence_threshold": float(
            confidence_threshold
        ),
        "max_detections": int(
            max_detections
        ),
        "bands": list(
            bands
        ),
        "crs": str(
            crs
        ),
    }

    summary_path = (
        output_paths[
            "root"
        ]
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved summary: "
        f"{summary_path}"
    )

    print(
        "\n"
        + "-" * 100
    )

    print(
        f"Processed tiles:      "
        f"{processed_tiles}"
    )

    print(
        f"Skipped empty tiles:  "
        f"{skipped_empty_tiles}"
    )

    print(
        f"Total detections:     "
        f"{len(json_records)}"
    )

    print(
        f"Output root:          "
        f"{output_paths['root']}"
    )

    print(
        "-" * 100
    )

    return {
        "summary": summary,
        "paths": output_paths,
    }
