#!/usr/bin/env python3
"""Dump a world_mocap.npz file to JSON next to the source file.

The default input is preprocessed/robot/hawor/world_mocap.npz and the default
output is preprocessed/robot/hawor/world_mocap.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float):
        return value
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _jsonable_array(array: np.ndarray) -> Any:
    if array.shape == ():
        return _json_scalar(array.item())
    if np.issubdtype(array.dtype, np.floating):
        return array.tolist()
    if np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        return array.tolist()
    return [_json_scalar(item) for item in array.tolist()]


def _array_stats(array: np.ndarray) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.shape == ():
        stats["finite"] = bool(np.isfinite(array)) if np.issubdtype(array.dtype, np.number) else True
        return stats
    if np.issubdtype(array.dtype, np.number):
        finite = np.isfinite(array)
        stats["finite_values"] = int(finite.sum())
        stats["total_values"] = int(array.size)
        stats["nan_values"] = int(np.isnan(array).sum()) if np.issubdtype(array.dtype, np.floating) else 0
        if np.any(finite):
            stats["min"] = float(np.min(array[finite]))
            stats["max"] = float(np.max(array[finite]))
    return stats


def dump_npz_to_json(input_path: Path, output_path: Path, include_stats: bool) -> None:
    data = np.load(input_path, allow_pickle=True)
    payload: dict[str, Any] = {}
    if include_stats:
        payload["_meta"] = {
            "source_npz": str(input_path),
            "nan_inf_policy": "non-finite numeric values are preserved as NaN/Infinity",
            "arrays": {key: _array_stats(data[key]) for key in data.files},
        }
    for key in data.files:
        print(f"dumping {key}: shape={data[key].shape} dtype={data[key].dtype}")
        payload[key] = _jsonable_array(data[key])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=True)
        f.write("\n")
    print(f"saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert world_mocap.npz arrays/scalars to a JSON file."
    )
    parser.add_argument(
        "--input",
        default="preprocessed/robot/hawor/world_mocap.npz",
        help="Path to world_mocap.npz.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <input_dir>/world_mocap.json.",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Do not include the _meta array shape/dtype/finite summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path = (
        Path(args.output).resolve()
        if args.output is not None
        else input_path.with_suffix(".json")
    )
    dump_npz_to_json(input_path, output_path, include_stats=not args.no_stats)


if __name__ == "__main__":
    main()
