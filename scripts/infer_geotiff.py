#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command-line GeoTIFF inference entry point for
GeoAI-UrbanTree-RFDETR.

Example
-------
Run using settings from configs/inference.yaml:

    PYTHONPATH=src python scripts/infer_geotiff.py \
        --config configs/inference.yaml
"""

from __future__ import annotations

import argparse

from geoai_tree.inference.raster_inference import (
    run_raster_inference,
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
            "Run RF-DETR inference on a GeoTIFF."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/inference.yaml",
        help=(
            "Inference YAML configuration file. "
            "Default: configs/inference.yaml"
        ),
    )

    return parser.parse_args()


def main():
    """
    Main inference CLI.
    """

    args = parse_args()

    config = load_config(
        args.config
    )

    model_config = config.get(
        "model",
        {},
    )

    checkpoint_config = config[
        "checkpoint"
    ]

    input_config = config[
        "input"
    ]

    inference_config = config[
        "inference"
    ]

    output_config = config[
        "output"
    ]

    result = run_raster_inference(
        raster_path=(
            input_config[
                "raster"
            ]
        ),

        checkpoint_path=(
            checkpoint_config[
                "path"
            ]
        ),

        output_root=(
            output_config[
                "root"
            ]
        ),

        architecture=(
            model_config.get(
                "architecture",
                "RFDETRBase",
            )
        ),

        bands=(
            input_config.get(
                "bands",
                [1, 2, 3],
            )
        ),

        tile_size=int(
            inference_config.get(
                "tile_size",
                256,
            )
        ),

        confidence_threshold=float(
            inference_config.get(
                "confidence_threshold",
                0.30,
            )
        ),

        max_detections=int(
            inference_config.get(
                "max_detections",
                200,
            )
        ),

        test_mode=bool(
            inference_config.get(
                "test_mode",
                False,
            )
        ),

        max_test_tiles=int(
            inference_config.get(
                "max_test_tiles",
                100,
            )
        ),

        save_test_tiles=bool(
            inference_config.get(
                "save_test_tiles",
                False,
            )
        ),

        save_json=bool(
            output_config.get(
                "save_json",
                True,
            )
        ),

        save_geojson=bool(
            output_config.get(
                "save_geojson",
                True,
            )
        ),

        save_shapefile=bool(
            output_config.get(
                "save_shapefile",
                True,
            )
        ),

        model_name="tvt",
    )

    print(
        "\nInference finished successfully."
    )

    print(
        "Output root:",
        result[
            "summary"
        ][
            "output_root"
        ],
    )


if __name__ == "__main__":
    main()
