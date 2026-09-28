from pathlib import Path


def validate_tvt_dataset(dataset_root, annotation_file="_annotations.coco.json"):
    """
    Validate the expected train/valid/test COCO dataset structure.

    Expected structure
    ------------------
    dataset_root/
        train/
            images/
            _annotations.coco.json
        valid/
            images/
            _annotations.coco.json
        test/
            images/
            _annotations.coco.json

    Parameters
    ----------
    dataset_root : str or Path
        Root directory of the TVT dataset.

    annotation_file : str
        COCO annotation filename.

    Returns
    -------
    bool
        True if all required folders/files exist.

    Raises
    ------
    FileNotFoundError
        If any required folder or annotation file is missing.
    """

    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}"
        )

    required_splits = ["train", "valid", "test"]

    for split in required_splits:
        split_dir = dataset_root / split
        images_dir = split_dir / "images"
        annotation_path = split_dir / annotation_file

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Missing split directory: {split_dir}"
            )

        if not images_dir.exists():
            raise FileNotFoundError(
                f"Missing images directory: {images_dir}"
            )

        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Missing COCO annotation file: {annotation_path}"
            )

    print("=" * 80)
    print("TVT dataset validation passed")
    print(f"Dataset root: {dataset_root}")
    print("=" * 80)

    return True
