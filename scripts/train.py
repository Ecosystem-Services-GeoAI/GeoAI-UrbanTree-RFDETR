#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train the standard TVT RF-DETR model.

Example
-------
PYTHONPATH=src python scripts/train.py \
    --config configs/train.yaml
"""

from __future__ import annotations

import argparse

from geoai_tree.training.trainer import (
    train_from_config,
)
from geoai_tree.utils.config import (
    load_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train RF-DETR using a YAML configuration."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help=(
            "Training YAML configuration file. "
            "Default: configs/train.yaml"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(
        args.config
    )

    train_from_config(
        config
    )


if __name__ == "__main__":
    main()