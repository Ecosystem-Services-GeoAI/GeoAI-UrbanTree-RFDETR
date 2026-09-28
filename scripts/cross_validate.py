#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train RF-DETR cross-validation folds.

Examples
--------
Train all folds:

PYTHONPATH=src python scripts/cross_validate.py \
    --config configs/cross_validation.yaml

Train one fold:

PYTHONPATH=src python scripts/cross_validate.py \
    --config configs/cross_validation.yaml \
    --fold 3

Train selected folds:

PYTHONPATH=src python scripts/cross_validate.py \
    --config configs/cross_validation.yaml \
    --folds 1 3 5
"""

from __future__ import annotations

import argparse

from geoai_tree.training.cross_validation import (
    train_all_cv_folds,
    train_cv_fold,
)
from geoai_tree.utils.config import (
    load_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train RF-DETR cross-validation folds."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/cross_validation.yaml",
        help=(
            "Cross-validation YAML configuration file. "
            "Default: configs/cross_validation.yaml"
        ),
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--fold",
        type=int,
        help=(
            "Train one fold only."
        ),
    )

    group.add_argument(
        "--folds",
        type=int,
        nargs="+",
        help=(
            "Train selected folds, e.g. "
            "--folds 1 3 5"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(
        args.config
    )

    if args.fold is not None:

        train_cv_fold(
            config=config,
            fold=args.fold,
        )

    elif args.folds is not None:

        train_all_cv_folds(
            config=config,
            folds=args.folds,
        )

    else:

        train_all_cv_folds(
            config=config
        )


if __name__ == "__main__":
    main()
