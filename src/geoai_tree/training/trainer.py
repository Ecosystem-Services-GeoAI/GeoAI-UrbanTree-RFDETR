#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared RF-DETR training utilities for GeoAI-UrbanTree-RFDETR.

Used by:

- standard train / valid / test training
- cross-validation fold training

The implementation preserves the settings used in the
original project training scripts while making paths and
hyperparameters configurable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rfdetr import RFDETRBase

from geoai_tree.utils.paths import (
    resolve_project_path,
)


# ============================================================
# Dataset validation
# ============================================================

def validate_training_dataset(
    dataset_dir: str | Path,
    annotation_file: str = "_annotations.coco.json",
    require_test: bool = False,
) -> Path:
    """
    Validate the COCO folder structure required by RF-DETR.

    Required:

        dataset_dir/
        ├── train/
        │   ├── images/
        │   └── _annotations.coco.json
        └── valid/
            ├── images/
            └── _annotations.coco.json

    Optionally:

        test/
        ├── images/
        └── _annotations.coco.json

    Parameters
    ----------
    dataset_dir
        Dataset root.

    annotation_file
        COCO annotation filename.

    require_test
        If True, require a test split too.

    Returns
    -------
    Path
        Resolved dataset directory.
    """

    dataset_dir = resolve_project_path(
        dataset_dir
    )

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: "
            f"{dataset_dir}"
        )

    required_splits = [
        "train",
        "valid",
    ]

    if require_test:
        required_splits.append(
            "test"
        )

    for split in required_splits:

        split_dir = (
            dataset_dir
            / split
        )

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Missing dataset split: "
                f"{split_dir}"
            )

        images_dir = (
            split_dir
            / "images"
        )

        if not images_dir.exists():
            raise FileNotFoundError(
                f"Missing images directory: "
                f"{images_dir}"
            )

        annotation_path = (
            split_dir
            / annotation_file
        )

        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Missing COCO annotation file: "
                f"{annotation_path}"
            )

    return dataset_dir


# ============================================================
# Model creation
# ============================================================

def create_training_model(
    architecture: str = "RFDETRBase",
):
    """
    Create a new RF-DETR model for training.

    RFDETRBase() without a custom checkpoint preserves the
    original training behavior used in this project.
    """

    if architecture != "RFDETRBase":
        raise ValueError(
            "Unsupported architecture: "
            f"{architecture}. "
            "Currently supported: RFDETRBase"
        )

    return RFDETRBase()


# ============================================================
# Shared training function
# ============================================================

def train_rfdetr(
    dataset_dir: str | Path,
    output_dir: str | Path,
    architecture: str = "RFDETRBase",
    annotation_file: str = "_annotations.coco.json",
    epochs: int = 50,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    learning_rate: float = 1e-4,
    num_workers: int = 8,
    checkpoint_interval: int = 10,
    run_test: bool = False,
    require_test_split: bool = False,
) -> Any:
    """
    Train RF-DETR on one COCO dataset.

    This is the shared training engine used by both TVT and
    cross-validation workflows.

    Parameters
    ----------
    dataset_dir
        RF-DETR COCO dataset root.

    output_dir
        Training output directory.

    architecture
        RF-DETR architecture.

    annotation_file
        COCO annotation filename.

    epochs
        Number of training epochs.

    batch_size
        Training batch size.

    grad_accum_steps
        Gradient accumulation steps.

    learning_rate
        Learning rate.

    num_workers
        DataLoader worker count.

    checkpoint_interval
        Save regular checkpoint every N epochs.

    run_test
        Whether RF-DETR should evaluate the test split during
        training.

    require_test_split
        Whether the dataset validator requires a test split.

    Returns
    -------
    Any
        RF-DETR model object after training.
    """

    # --------------------------------------------------------
    # Basic argument checks
    # --------------------------------------------------------

    if epochs <= 0:
        raise ValueError(
            "epochs must be greater than zero."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if grad_accum_steps <= 0:
        raise ValueError(
            "grad_accum_steps must be greater than zero."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers cannot be negative."
        )

    if checkpoint_interval <= 0:
        raise ValueError(
            "checkpoint_interval must be greater than zero."
        )

    # --------------------------------------------------------
    # Resolve + validate dataset
    # --------------------------------------------------------

    dataset_dir = (
        validate_training_dataset(
            dataset_dir=dataset_dir,
            annotation_file=annotation_file,
            require_test=require_test_split,
        )
    )

    output_dir = resolve_project_path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Print training configuration
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RF-DETR TRAINING"
    )

    print(
        "=" * 100
    )

    print(
        f"Dataset:              {dataset_dir}"
    )

    print(
        f"Output:               {output_dir}"
    )

    print(
        f"Architecture:         {architecture}"
    )

    print(
        f"Epochs:               {epochs}"
    )

    print(
        f"Batch size:           {batch_size}"
    )

    print(
        f"Gradient accumulation:{grad_accum_steps}"
    )

    print(
        f"Learning rate:        {learning_rate}"
    )

    print(
        f"Workers:              {num_workers}"
    )

    print(
        f"Checkpoint interval:  {checkpoint_interval}"
    )

    print(
        f"Run test:             {run_test}"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_training_model(
        architecture=architecture
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.train(
        dataset_dir=str(
            dataset_dir
        ),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=(
            grad_accum_steps
        ),
        lr=learning_rate,
        num_workers=num_workers,
        output_dir=str(
            output_dir
        ),
        run_test=run_test,
        checkpoint_interval=(
            checkpoint_interval
        ),
    )

    print(
        "\nTraining finished."
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )

    return model


# ============================================================
# YAML-driven wrapper
# ============================================================

def train_from_config(
    config: dict,
):
    """
    Train RF-DETR using a loaded train.yaml configuration.

    Expected configuration sections:

        dataset
        model
        training
        output
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

    dataset_root = dataset_config[
        "root"
    ]

    annotation_file = (
        dataset_config.get(
            "annotation_file",
            "_annotations.coco.json",
        )
    )

    output_root = (
        output_config[
            "root"
        ]
    )

    run_name = (
        output_config.get(
            "run_name",
            "rfdetr_run",
        )
    )

    output_dir = (
        Path(
            output_root
        )
        / run_name
    )

    # TVT config contains test, but the original project did
    # not run the test set during training.
    #
    # We validate the test split if dataset.type == "tvt",
    # while still keeping run_test=False unless explicitly
    # enabled in YAML.

    dataset_type = (
        dataset_config.get(
            "type",
            "tvt",
        )
    )

    require_test_split = (
        dataset_type == "tvt"
    )

    return train_rfdetr(
        dataset_dir=dataset_root,
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

        require_test_split=(
            require_test_split
        ),
    )