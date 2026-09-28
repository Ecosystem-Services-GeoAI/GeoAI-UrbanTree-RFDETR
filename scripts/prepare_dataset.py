#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare a GeoTIFF + vector-label dataset in COCO format.

Example
-------
PYTHONPATH=src python scripts/prepare_dataset.py \
    --config configs/dataset.yaml
"""

from __future__ import annotations

import argparse

from geoai_tree.data.preparation import (
    build_coco_dataset,
    find_raster_files,
)
from geoai_tree.utils.config import (
    load_config,
)


def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Prepare GeoTIFF imagery and vector tree labels "
            "as a COCO object-detection dataset."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/dataset.yaml",
        help=(
            "Dataset preparation YAML configuration file. "
            "Default: configs/dataset.yaml"
        ),
    )

    return parser.parse_args()


def main():
    """
    Main dataset-preparation CLI.
    """

    args = parse_args()

    config = load_config(
        args.config
    )

    dataset_config = config[
        "dataset"
    ]

    input_config = config[
        "input"
    ]

    processing_config = config[
        "processing"
    ]

    output_config = config[
        "output"
    ]

    # --------------------------------------------------------
    # Discover raster files
    # --------------------------------------------------------

    raster_paths = find_raster_files(
        raster_dir=(
            input_config[
                "raster_dir"
            ]
        ),
        raster_pattern=(
            input_config.get(
                "raster_pattern",
                "*.tif",
            )
        ),
    )

    print(
        f"Found {len(raster_paths)} raster file(s)."
    )

    for path in raster_paths:
        print(
            f"  - {path}"
        )

    # --------------------------------------------------------
    # Build COCO dataset
    # --------------------------------------------------------

    summary = build_coco_dataset(
        dataset_name=(
            dataset_config[
                "name"
            ]
        ),

        raster_paths=(
            raster_paths
        ),

        labels_path=(
            input_config[
                "labels"
            ]
        ),

        output_dir=(
            output_config[
                "root"
            ]
        ),

        split_name=(
            dataset_config.get(
                "split_name",
                "test",
            )
        ),

        tile_size=int(
            processing_config.get(
                "tile_size",
                256,
            )
        ),

        bands=(
            processing_config.get(
                "bands",
                [1, 2, 3],
            )
        ),

        min_bbox_area=float(
            processing_config.get(
                "min_bbox_area",
                4.0,
            )
        ),

        skip_empty=bool(
            processing_config.get(
                "skip_empty",
                False,
            )
        ),

        keep_partial=bool(
            processing_config.get(
                "keep_partial",
                False,
            )
        ),

        clean=bool(
            output_config.get(
                "clean",
                False,
            )
        ),
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print(
        "\nDataset preparation finished successfully."
    )

    print(
        f"Images:      "
        f"{summary['images']}"
    )

    print(
        f"Annotations: "
        f"{summary['annotations']}"
    )

    print(
        f"COCO JSON:   "
        f"{summary['annotation_file']}"
    )


if __name__ == "__main__":
    main()
