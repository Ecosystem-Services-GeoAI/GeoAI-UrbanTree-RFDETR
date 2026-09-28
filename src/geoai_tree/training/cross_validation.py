#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-validation training utilities for GeoAI-UrbanTree-RFDETR.

Each fold reuses the shared train_rfdetr() function from trainer.py.

Expected dataset structure:

datasets/coco/cv/
├── fold_1/
│   ├── train/
│   └── valid/
├── fold_2/
│   ├── train/
│   └── valid/
...
└── fold_5/
    ├── train/
    └── valid/
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from geoai_tree.training.trainer import (
    train_rfdetr,
    validate_training_dataset,
)
from geoai_tree.utils.paths import (
    resolve_project_path,
)


def validate_cv_dataset(
    cv_root: str | Path,
    num_folds: int = 5,
    fold_prefix: str = "fold_",
    annotation_file: str = "_annotations.coco.json",
) -> Path:
    """
    Validate all cross-validation fold directories.

    Each fold must contain:

        train/
        valid/

    with:

        images/
        _annotations.coco.json
    """

    cv_root = resolve_project_path(
        cv_root
    )

    if not cv_root.exists():
        raise FileNotFoundError(
            f"CV dataset root not found: {cv_root}"
        )

    if num_folds <= 0:
        raise ValueError(
            "num_folds must be greater than zero."
        )

    for fold in range(
        1,
        num_folds + 1,
    ):

        fold_dir = (
            cv_root
            / f"{fold_prefix}{fold}"
        )

        validate_training_dataset(
            dataset_dir=fold_dir,
            annotation_file=annotation_file,
            require_test=False,
        )

    return cv_root


def train_cv_fold(
    config: dict,
    fold: int,
):
    """
    Train one cross-validation fold.

    Parameters
    ----------
    config
        Loaded cross_validation.yaml configuration.

    fold
        Fold number, starting from 1.
    """

    dataset_config = config[
        "dataset"
    ]

    model_config = config.get(
        "model",
        {},
    )

    training_config = config[
        "training"
    ]

    output_config = config[
        "output"
    ]

    num_folds = int(
        dataset_config.get(
            "num_folds",
            5,
        )
    )

    if fold < 1 or fold > num_folds:
        raise ValueError(
            f"fold must be between 1 and {num_folds}. "
            f"Received: {fold}"
        )

    cv_root = resolve_project_path(
        dataset_config[
            "root"
        ]
    )

    fold_prefix = dataset_config.get(
        "fold_prefix",
        "fold_",
    )

    fold_name = (
        f"{fold_prefix}{fold}"
    )

    dataset_dir = (
        cv_root
        / fold_name
    )

    annotation_file = (
        dataset_config.get(
            "annotation_file",
            "_annotations.coco.json",
        )
    )

    output_root = resolve_project_path(
        output_config[
            "root"
        ]
    )

    output_fold_prefix = (
        output_config.get(
            "fold_prefix",
            "fold_",
        )
    )

    output_dir = (
        output_root
        / f"{output_fold_prefix}{fold}"
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        f"CROSS-VALIDATION TRAINING: "
        f"FOLD {fold}/{num_folds}"
    )

    print(
        "=" * 100
    )

    return train_rfdetr(
        dataset_dir=dataset_dir,
        output_dir=output_dir,

        architecture=(
            model_config.get(
                "architecture",
                "RFDETRBase",
            )
        ),

        annotation_file=(
            annotation_file
        ),

        epochs=int(
            training_config.get(
                "epochs",
                50,
            )
        ),

        batch_size=int(
            training_config.get(
                "batch_size",
                8,
            )
        ),

        grad_accum_steps=int(
            training_config.get(
                "grad_accum_steps",
                4,
            )
        ),

        learning_rate=float(
            training_config.get(
                "learning_rate",
                1e-4,
            )
        ),

        num_workers=int(
            training_config.get(
                "num_workers",
                8,
            )
        ),

        checkpoint_interval=int(
            training_config.get(
                "checkpoint_interval",
                10,
            )
        ),

        run_test=bool(
            training_config.get(
                "run_test_during_training",
                False,
            )
        ),

        require_test_split=False,
    )


def train_all_cv_folds(
    config: dict,
    folds: Iterable[int] | None = None,
):
    """
    Train multiple CV folds sequentially.

    By default, all configured folds are trained.
    """

    dataset_config = config[
        "dataset"
    ]

    num_folds = int(
        dataset_config.get(
            "num_folds",
            5,
        )
    )

    fold_prefix = dataset_config.get(
        "fold_prefix",
        "fold_",
    )

    annotation_file = dataset_config.get(
        "annotation_file",
        "_annotations.coco.json",
    )

    validate_cv_dataset(
        cv_root=dataset_config[
            "root"
        ],
        num_folds=num_folds,
        fold_prefix=fold_prefix,
        annotation_file=annotation_file,
    )

    if folds is None:

        folds = range(
            1,
            num_folds + 1,
        )

    folds = [
        int(fold)
        for fold in folds
    ]

    for fold in folds:

        if (
            fold < 1
            or fold > num_folds
        ):
            raise ValueError(
                f"Invalid fold: {fold}. "
                f"Expected 1–{num_folds}."
            )

    trained_models = {}

    for fold in folds:

        model = train_cv_fold(
            config=config,
            fold=fold,
        )

        trained_models[
            fold
        ] = model

    print(
        "\n"
        + "=" * 100
    )

    print(
        "CROSS-VALIDATION TRAINING FINISHED"
    )

    print(
        "=" * 100
    )

    return trained_models
