#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command-line evaluation entry point for GeoAI-UrbanTree-RFDETR.

Supported modes:

    tvt
    cv
    transfer

Examples
--------
TVT:

    PYTHONPATH=src python scripts/evaluate.py \
        --config configs/evaluate.yaml \
        --mode tvt

Cross-validation:

    PYTHONPATH=src python scripts/evaluate.py \
        --config configs/evaluate.yaml \
        --mode cv

Transfer cities:

    PYTHONPATH=src python scripts/evaluate.py \
        --config configs/evaluate.yaml \
        --mode transfer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from geoai_tree.evaluation.evaluator import (
    combine_evaluation_summaries,
    evaluate_split,
)
from geoai_tree.inference.predictor import (
    load_rfdetr_model,
)
from geoai_tree.utils.config import (
    load_config,
)
from geoai_tree.utils.paths import (
    resolve_project_path,
)


# ============================================================
# Shared configuration helpers
# ============================================================

def get_evaluation_settings(config: dict) -> dict:
    """
    Extract shared evaluation settings from YAML.
    """

    evaluation = config["evaluation"]

    visualization = config.get(
        "visualization",
        {},
    )

    return {
        "coco_threshold": float(
            evaluation.get(
                "coco_threshold",
                0.001,
            )
        ),
        "prediction_threshold": float(
            evaluation.get(
                "prediction_threshold",
                0.30,
            )
        ),
        "iou_threshold": float(
            evaluation.get(
                "iou_threshold",
                0.50,
            )
        ),
        "max_detections": int(
            evaluation.get(
                "max_detections",
                200,
            )
        ),
        "save_predictions": bool(
            evaluation.get(
                "save_predictions",
                True,
            )
        ),
        "save_visualizations": bool(
            evaluation.get(
                "save_visualizations",
                True,
            )
        ),
        "gt_color": visualization.get(
            "ground_truth_color",
            "lime",
        ),
        "pred_color": visualization.get(
            "prediction_color",
            "red",
        ),
        "box_width": int(
            visualization.get(
                "box_width",
                2,
            )
        ),
        "panel_gap": int(
            visualization.get(
                "panel_gap",
                20,
            )
        ),
    }


def save_combined_summary(
    summaries: list[dict],
    output_path: str | Path,
) -> Path:
    """
    Save multiple evaluation summaries as one CSV.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = (
        combine_evaluation_summaries(
            summaries
        )
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved combined summary: "
        f"{output_path}"
    )

    return output_path


# ============================================================
# TVT evaluation
# ============================================================

def evaluate_tvt(
    config: dict,
):
    """
    Evaluate TVT train / valid / test using the TVT EMA model.
    """

    print(
        "\n"
        + "=" * 100
    )
    print("TVT EVALUATION")
    print("=" * 100)

    architecture = config["model"].get(
        "architecture",
        "RFDETRBase",
    )

    checkpoint_path = (
        config["checkpoint"]["tvt"]
    )

    model = load_rfdetr_model(
        checkpoint_path=checkpoint_path,
        architecture=architecture,
    )

    dataset_config = (
        config["datasets"]["tvt"]
    )

    dataset_root = resolve_project_path(
        dataset_config["root"]
    )

    splits = dataset_config.get(
        "splits",
        [
            "train",
            "valid",
            "test",
        ],
    )

    output_root = resolve_project_path(
        config["output"]["tvt"]
    )

    settings = get_evaluation_settings(
        config
    )

    summaries = []

    for split in splits:

        split_dir = (
            dataset_root
            / split
        )

        split_output = (
            output_root
            / split
        )

        _, summary = evaluate_split(
            model=model,
            dataset_name="tvt",
            split_name=split,
            split_dir=split_dir,
            output_dir=split_output,
            **settings,
        )

        summaries.append(
            summary
        )

    save_combined_summary(
        summaries=summaries,
        output_path=(
            output_root
            / "tvt_summary.csv"
        ),
    )

    print(
        "\nTVT evaluation finished."
    )


# ============================================================
# Cross-validation evaluation
# ============================================================

def evaluate_cv(
    config: dict,
):
    """
    Evaluate every CV fold using its corresponding EMA model.
    """

    print(
        "\n"
        + "=" * 100
    )
    print("CROSS-VALIDATION EVALUATION")
    print("=" * 100)

    architecture = config["model"].get(
        "architecture",
        "RFDETRBase",
    )

    dataset_config = (
        config["datasets"]["cv"]
    )

    dataset_root = resolve_project_path(
        dataset_config["root"]
    )

    num_folds = int(
        dataset_config.get(
            "num_folds",
            5,
        )
    )

    splits = dataset_config.get(
        "splits",
        [
            "train",
            "valid",
        ],
    )

    checkpoint_config = (
        config["checkpoint"]
    )

    cv_root = resolve_project_path(
        checkpoint_config["cv_root"]
    )

    cv_pattern = checkpoint_config.get(
        "cv_pattern",
        "fold_{fold}/checkpoint_best_ema.pth",
    )

    output_root = resolve_project_path(
        config["output"]["cv"]
    )

    settings = get_evaluation_settings(
        config
    )

    summaries = []

    for fold in range(
        1,
        num_folds + 1,
    ):

        fold_name = (
            f"fold_{fold}"
        )

        checkpoint_path = (
            cv_root
            / cv_pattern.format(
                fold=fold
            )
        )

        model = load_rfdetr_model(
            checkpoint_path=checkpoint_path,
            architecture=architecture,
        )

        for split in splits:

            split_dir = (
                dataset_root
                / fold_name
                / split
            )

            split_output = (
                output_root
                / fold_name
                / split
            )

            _, summary = evaluate_split(
                model=model,
                dataset_name=fold_name,
                split_name=split,
                split_dir=split_dir,
                output_dir=split_output,
                **settings,
            )

            summaries.append(
                summary
            )

    save_combined_summary(
        summaries=summaries,
        output_path=(
            output_root
            / "cv_summary.csv"
        ),
    )

    print(
        "\nCross-validation evaluation finished."
    )


# ============================================================
# Transfer-city evaluation
# ============================================================

def evaluate_transfer(
    config: dict,
):
    """
    Evaluate the TVT-trained EMA model on transfer-city datasets.
    """

    print(
        "\n"
        + "=" * 100
    )
    print("TRANSFER-CITY EVALUATION")
    print("=" * 100)

    architecture = config["model"].get(
        "architecture",
        "RFDETRBase",
    )

    checkpoint_path = (
        config["checkpoint"]["tvt"]
    )

    model = load_rfdetr_model(
        checkpoint_path=checkpoint_path,
        architecture=architecture,
    )

    dataset_config = (
        config["datasets"]["transfer"]
    )

    dataset_root = resolve_project_path(
        dataset_config["root"]
    )

    cities = dataset_config.get(
        "cities",
        [],
    )

    split = dataset_config.get(
        "split",
        "test",
    )

    output_root = resolve_project_path(
        config["output"]["transfer"]
    )

    settings = get_evaluation_settings(
        config
    )

    summaries = []

    for city in cities:

        split_dir = (
            dataset_root
            / city
            / split
        )

        city_output = (
            output_root
            / city
        )

        _, summary = evaluate_split(
            model=model,
            dataset_name=city,
            split_name=split,
            split_dir=split_dir,
            output_dir=city_output,
            **settings,
        )

        summaries.append(
            summary
        )

    save_combined_summary(
        summaries=summaries,
        output_path=(
            output_root
            / "transfer_summary.csv"
        ),
    )

    print(
        "\nTransfer-city evaluation finished."
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RF-DETR urban-tree models "
            "on TVT, CV, or transfer datasets."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/evaluate.yaml",
        help=(
            "Evaluation YAML configuration file. "
            "Default: configs/evaluate.yaml"
        ),
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "tvt",
            "cv",
            "transfer",
        ],
        help=(
            "Evaluation mode: "
            "tvt, cv, or transfer"
        ),
    )

    return parser.parse_args()


def main():
    """
    Main CLI entry point.
    """

    args = parse_args()

    config = load_config(
        args.config
    )

    if args.mode == "tvt":

        evaluate_tvt(
            config
        )

    elif args.mode == "cv":

        evaluate_cv(
            config
        )

    elif args.mode == "transfer":

        evaluate_transfer(
            config
        )

    else:

        raise ValueError(
            f"Unsupported mode: "
            f"{args.mode}"
        )


if __name__ == "__main__":
    main()
