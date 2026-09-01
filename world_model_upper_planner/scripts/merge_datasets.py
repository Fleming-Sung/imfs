#!/usr/bin/env python3
"""Merge option datasets while preserving terrain and disjoint environment IDs."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memmap", action="store_true",
                        help="write a scalable directory of NPY memmaps")
    args = parser.parse_args()
    metadata = []
    candidates = None
    common = None
    first_files = None
    total_rows = 0
    for path in args.inputs:
        with np.load(path, allow_pickle=False) as data:
            current = data["candidates"]
            if candidates is None:
                candidates = current.copy()
                first_files = set(data.files)
            elif not np.allclose(current, candidates):
                raise ValueError("candidate grids differ")
            common = set(data.files) if common is None else common & set(data.files)
            rows = len(data["env_id"])
            metadata.append({
                "path": path, "rows": rows,
                "env_count": int(data["env_id"].max()) + 1,
            })
            total_rows += rows
    common.discard("candidates")
    required = first_files - {
        "candidates", "terrain_kind", "candidate_alignment", "difficulty"}
    if not required.issubset(common):
        raise ValueError(f"missing common arrays: {sorted(required - common)}")
    names = sorted(common | {"terrain_kind", "candidate_alignment", "difficulty"})
    arrays = {name: [] for name in names}
    env_offset = 0
    sources = []
    if args.memmap:
        args.output.mkdir(parents=True, exist_ok=True)
        writers = {}
        with np.load(args.inputs[0], allow_pickle=False) as first:
            for name in names:
                if name == "difficulty" and name not in first.files:
                    sample = np.full(1, "legacy", dtype="U24")
                elif name == "terrain_kind" and name not in first.files:
                    sample = np.full(1, "random_composite", dtype="U24")
                elif name == "candidate_alignment" and name not in first.files:
                    sample = np.zeros_like(first["candidate_progress"][:1], dtype=np.float32)
                else:
                    sample = first[name]
                writers[name] = np.lib.format.open_memmap(
                    args.output / f"{name}.npy", mode="w+", dtype=sample.dtype,
                    shape=(total_rows, *sample.shape[1:]))
        np.save(args.output / "candidates.npy", candidates)
    else:
        writers = None

    cursor = 0
    for item in metadata:
        path, rows, count = item["path"], item["rows"], item["env_count"]
        with np.load(path, allow_pickle=False) as data:
            for name in arrays:
                if name == "terrain_kind" and name not in data.files:
                    value = np.full(rows, "random_composite", dtype="U24")
                elif name == "difficulty" and name not in data.files:
                    value = np.full(rows, "legacy", dtype="U24")
                elif name == "candidate_alignment" and name not in data.files:
                    value = np.zeros_like(data["candidate_progress"], dtype=np.float32)
                elif name not in data.files:
                    continue
                else:
                    value = data[name]
                if name == "env_id":
                    value = value.astype(np.int64) + env_offset
                if args.memmap:
                    writers[name][cursor:cursor + rows] = value
                else:
                    arrays[name].append(value)
        sources.append({"path": str(path), "rows": rows,
                        "env_offset": env_offset, "env_count": count})
        env_offset += count
        cursor += rows
    if args.memmap:
        for writer in writers.values():
            writer.flush()
        merged = {name: np.load(args.output / f"{name}.npy", mmap_mode="r")
                  for name in writers}
    else:
        merged = {name: np.concatenate(parts) for name, parts in arrays.items() if parts}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, candidates=candidates, **merged)
    manifest = {"output": str(args.output), "format": "memmap" if args.memmap else "npz",
                "rows": len(merged["env_id"]),
                "environments": env_offset, "sources": sources,
                "terrain_counts": {str(k): int(v) for k, v in zip(
                    *np.unique(merged["terrain_kind"], return_counts=True))},
                "difficulty_counts": {str(k): int(v) for k, v in zip(
                    *np.unique(merged["difficulty"], return_counts=True))}}
    if args.memmap:
        manifest["arrays"] = {
            name: {"file": f"{name}.npy", "shape": list(value.shape),
                   "dtype": str(value.dtype)} for name, value in merged.items()}
        manifest["arrays"]["candidates"] = {
            "file": "candidates.npy", "shape": list(candidates.shape),
            "dtype": str(candidates.dtype)}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    else:
        args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
