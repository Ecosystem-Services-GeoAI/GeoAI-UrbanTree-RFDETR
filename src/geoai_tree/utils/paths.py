from pathlib import Path


def get_project_root():
    """
    Return the repository root directory.

    Assumes this file is located at:

        src/geoai_tree/utils/paths.py

    Therefore the repository root is three levels above this file.
    """
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path_value):
    """
    Resolve a path relative to the repository root.

    Absolute paths are returned unchanged.

    Parameters
    ----------
    path_value : str or Path
        Path from the YAML config or command line.

    Returns
    -------
    Path
        Resolved absolute path.
    """
    path_value = Path(path_value)

    if path_value.is_absolute():
        return path_value

    return get_project_root() / path_value
