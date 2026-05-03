"""Render PNG plots from pytest-benchmark JSON output.

Reads a JSON file produced by ``pytest --benchmark-json=<path>`` and emits
two figures matching the layout the original ``benchmark.py`` produced:

    vector_insertion_<N>_vectors.png   per-vector insert time vs. dim,
                                       grouped by (product, distance_type)
    vector_query_<N>_vectors.png       per-query search time vs. dim,
                                       grouped by (product, distance_type, ef_search)

Recall, when present, is included in a CSV companion file alongside each
PNG so it isn't lost.

Usage::

    python benchmark/plot.py bench.json
    python benchmark/plot.py bench.json --output-dir ./figures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _load(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    return doc["benchmarks"]


def _per_vector_us(stats_seconds: float, num_elements: int) -> float:
    """Median insert wall time divided by num_elements, in microseconds."""
    return (stats_seconds / num_elements) * 1_000_000


def _per_query_us(stats_seconds: float, num_queries: int) -> float:
    """Median search wall time divided by num_queries, in microseconds."""
    return (stats_seconds / num_queries) * 1_000_000


def _plot(name: str, df: pd.DataFrame, ylabel: str, out_dir: Path) -> Path:
    """Bar chart matching the original plot.py layout."""
    fig = df.plot.bar(
        x="dim",
        xlabel="vector dimension",
        ylabel=ylabel,
        title=f"{name} (lower is better)",
        rot=0,
        figsize=(10, 10),
    ).get_figure()
    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _build_insertion(records: List[dict]) -> Tuple[pd.DataFrame, int]:
    """Build a wide-format DataFrame of per-vector insert times.

    Rows = dim, columns = plot_label, values = microseconds per vector.
    Returns the DataFrame and the (assumed shared) num_elements.
    """
    if not records:
        return pd.DataFrame(), 0

    # dim -> {plot_label -> us}
    table: Dict[int, Dict[str, float]] = defaultdict(dict)
    num_elements = records[0]["extra_info"]["num_elements"]
    for r in records:
        info = r["extra_info"]
        median_seconds = r["stats"]["median"]
        table[info["dim"]][info["plot_label"]] = _per_vector_us(
            median_seconds, info["num_elements"])

    dims = sorted(table)
    columns = sorted({label for row in table.values() for label in row})
    df = pd.DataFrame(
        [[d] + [table[d].get(c, float("nan")) for c in columns] for d in dims],
        columns=["dim"] + columns,
    )
    return df, num_elements


def _build_query(records: List[dict]) -> Tuple[pd.DataFrame, int]:
    """Per-query search time table; honours include_in_query_plot."""
    if not records:
        return pd.DataFrame(), 0

    table: Dict[int, Dict[str, float]] = defaultdict(dict)
    num_elements = 0
    for r in records:
        info = r["extra_info"]
        if not info.get("include_in_query_plot", True):
            continue
        median_seconds = r["stats"]["median"]
        table[info["dim"]][info["plot_label"]] = _per_query_us(
            median_seconds, info["num_queries"])
        # num_elements isn't carried on query records by default; pull from
        # any companion insertion record via the surrounding benchmark file.
        # (The caller passes it in via _build_with_num_elements.)

    dims = sorted(table)
    columns = sorted({label for row in table.values() for label in row})
    df = pd.DataFrame(
        [[d] + [table[d].get(c, float("nan")) for c in columns] for d in dims],
        columns=["dim"] + columns,
    )
    return df, num_elements


def _write_recall_csv(records: List[dict], path: Path) -> None:
    rows = [
        (r["extra_info"].get("product"),
         r["extra_info"].get("distance_type"),
         r["extra_info"].get("dim"),
         r["extra_info"].get("ef_search"),
         r["extra_info"].get("recall"))
        for r in records if "recall" in r["extra_info"]
    ]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("product", "distance_type", "dim", "ef_search", "recall"))
        w.writerows(rows)


def _infer_num_elements(records: List[dict]) -> int:
    for r in records:
        n = r["extra_info"].get("num_elements")
        if n:
            return n
    return 0


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json_path", type=Path,
                        help="Path to pytest-benchmark JSON output")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Directory to write PNG/CSV files into "
                             "(default: current directory)")
    args = parser.parse_args(argv)

    records = _load(args.json_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    insertion = [r for r in records
                 if r["extra_info"].get("plot_kind") == "insertion"]
    query = [r for r in records
             if r["extra_info"].get("plot_kind") == "query"]

    # num_elements is always recorded on insertion records; query records
    # share the same dataset so we look it up from the insertion side.
    num_elements = _infer_num_elements(insertion) or _infer_num_elements(query)
    if num_elements == 0:
        print("error: could not infer num_elements from any benchmark record",
              file=sys.stderr)
        return 1

    if insertion:
        df, _ = _build_insertion(insertion)
        out = _plot(
            f"vector_insertion_{num_elements}_vectors",
            df, "time(us)/vector", args.output_dir)
        print(f"wrote {out}")

    if query:
        df, _ = _build_query(query)
        out = _plot(
            f"vector_query_{num_elements}_vectors",
            df, "time(us)/query", args.output_dir)
        print(f"wrote {out}")
        _write_recall_csv(
            query, args.output_dir / f"vector_query_{num_elements}_vectors_recall.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
