#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Checkpoint utilities for GeoAI-UrbanTree-RFDETR.

This module centralizes checkpoint discovery and selection for:

- TVT evaluation
- CV evaluation
- transfer-city evaluation
- image prediction
- full-area GeoTIFF inference
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


DEFAULT_CHECKPOINT_CANDIDATES = (
    "checkpoint_best_ema.pth",
    "checkpoint_best_total.pth",
    "checkpoint_best_regular.pth",
    "checkpoint.pth",
)


def validate_checkpoint(checkpoint_path: str | Path) -> Path:
    """
    Validate that a user-specified checkpoint exists.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to a checkpoint file.

    Returns
    -------
    Path
        Resolved checkpoint path.

    Raises
    ------
    FileNotFoundError
        If the checkpoint does not exist.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint path is not a file: {checkpoint_path}"
        )

    return checkpoint_path


def find_checkpoint(
    run_dir: str | Path,
    candidates: Optional[Iterable[str]] = None,
) -> Path:
    """
    Find a checkpoint inside a training run directory.

    Checkpoints are searched in priority order.

    Default priority
    ----------------
    1. checkpoint_best_ema.pth
    2. checkpoint_best_total.pth
    3. checkpoint_best_regular.pth
    4. checkpoint.pth

    If none of the preferred names are found, the first .pth file
    in the directory is used as a fallback.

    Parameters
    ----------
    run_dir : str or Path
        Directory containing RF-DETR checkpoint files.

    candidates : iterable of str, optional
        Custom checkpoint filename priority.

    Returns
    -------
    Path
        Selected checkpoint path.

    Raises
    ------
    FileNotFoundError
        If no checkpoint can be found.
    """

    run_dir = Path(run_dir)

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {run_dir}"
        )

    if not run_dir.is_dir():
        raise NotADirectoryError(
            f"Checkpoint run path is not a directory: {run_dir}"
        )

    if candidates is None:
        candidates = DEFAULT_CHECKPOINT_CANDIDATES

    for name in candidates:
        checkpoint = run_dir / name

        if checkpoint.exists():
            print(f"Selected checkpoint: {checkpoint}")
            return checkpoint

    fallback = sorted(run_dir.glob("*.pth"))

    if fallback:
        checkpoint = fallback[0]

        print(
            "Warning: none of the preferred checkpoint names were found."
        )
        print(f"Using first available checkpoint: {checkpoint}")

        return checkpoint

    raise FileNotFoundError(
        f"No .pth checkpoint found in: {run_dir}"
    )


def resolve_checkpoint(
    checkpoint_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    candidates: Optional[Iterable[str]] = None,
) -> Path:
    """
    Resolve a checkpoint using either an explicit checkpoint path
    or a training run directory.

    Priority
    --------
    1. Explicit checkpoint_path
    2. Search run_dir

    Parameters
    ----------
    checkpoint_path : str or Path, optional
        Specific checkpoint supplied by the user.

    run_dir : str or Path, optional
        Directory to search if checkpoint_path is not provided.

    candidates : iterable of str, optional
        Custom checkpoint filename priority.

    Returns
    -------
    Path
        Selected checkpoint path.
    """

    if checkpoint_path is not None:
        checkpoint_path = str(checkpoint_path).strip()

        if checkpoint_path:
            return validate_checkpoint(checkpoint_path)

    if run_dir is not None:
        return find_checkpoint(
            run_dir=run_dir,
            candidates=candidates,
        )

    raise ValueError(
        "No checkpoint source provided. "
        "Provide either checkpoint_path or run_dir."
    )


def get_cv_checkpoint(
    cv_root: str | Path,
    fold: int,
    pattern: str = "fold_{fold}.pth",
) -> Path:
    """
    Resolve one cross-validation checkpoint.

    Example
    -------
    cv_root:
        weights/cv

    pattern:
        fold_{fold}.pth

    fold:
        3

    Result:
        weights/cv/fold_3.pth
    """

    if fold < 1:
        raise ValueError(
            f"Fold number must be >= 1, got: {fold}"
        )

    cv_root = Path(cv_root)

    checkpoint_name = pattern.format(
        fold=fold
    )

    checkpoint_path = cv_root / checkpoint_name

    return validate_checkpoint(
        checkpoint_path
    )
