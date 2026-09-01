"""Dataset loading helpers for small NPZ files and large memmap datasets."""

import json
from pathlib import Path

import numpy as np


def load_arrays(path):
    """Load a dataset without forcing large scale replays into duplicated RAM.

    Existing ``.npz`` datasets are materialized once, preserving the fast V2
    path.  A directory produced by ``merge_datasets.py --memmap`` is opened as
    read-only NPY memmaps, so million-transition replays remain practical on
    the 32 GB host.
    """
    path = Path(path)
    if path.is_dir():
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing memmap manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        names = manifest.get("arrays", {})
        return {
            name: np.load(path / info["file"], mmap_mode="r")
            for name, info in names.items()
        }
    source = np.load(path, allow_pickle=False)
    try:
        return {name: source[name] for name in source.files}
    finally:
        source.close()


def sampling_groups(data, indices):
    """Return scenario-balanced pools while retaining terrain-only fallback."""
    terrain = (data["terrain_kind"] if "terrain_kind" in data
               else np.full(len(data["env_id"]), "unknown"))
    if "difficulty" in data:
        labels = np.char.add(np.char.add(terrain.astype(str), "/"),
                             data["difficulty"].astype(str))
    else:
        labels = terrain.astype(str)
    return {
        str(label): indices[labels[indices] == label]
        for label in np.unique(labels[indices])
    }
