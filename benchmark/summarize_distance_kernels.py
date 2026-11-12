"""Validate complete matched kernel campaigns and print Markdown tables.

Usage: python benchmark/summarize_distance_kernels.py RUN.json [REPEAT.json ...]
Only matching case matrices/inputs may be combined. This validates descriptive
statistics, not a statistical confidence interval or ANN equivalence.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def key(case: dict) -> tuple:
    return (
        case["dimension"],
        case["metric"],
        case["alignment_offset_bytes"],
        case["corpus_count"],
    )


def load(path: Path) -> dict:
    report = json.loads(path.read_text())
    require(report["schema_version"] == 1, "unsupported schema")
    require(report["status"] == "complete", f"incomplete campaign: {path}")
    cases = report["cases"]
    require(bool(cases), "empty campaign")
    require(len({key(case) for case in cases}) == len(cases), "duplicate case")
    for case in cases:
        n = case["operations_per_sample"]
        require(isinstance(n, int) and n > 0, "invalid operation count")
        samples = case["samples"]
        require(len(samples) == report["pairs"], "pair count mismatch")
        hs, ds, ratios = [], [], []
        for i, sample in enumerate(samples):
            require(sample["pair"] == i, "pair sequence mismatch")
            expected = "highway,diskann" if i % 2 == 0 else "diskann,highway"
            require(sample["order"] == expected, "pair order mismatch")
            for name in ("highway", "diskann"):
                seconds = sample[f"{name}_seconds"]
                ns = sample[f"{name}_ns_per_distance"]
                checksum = sample[f"{name}_checksum"]
                require(math.isfinite(seconds) and seconds > 0, "invalid time")
                require(math.isfinite(checksum), "invalid checksum")
                require(math.isclose(ns, seconds * 1e9 / n), "time conversion mismatch")
            hs.append(sample["highway_ns_per_distance"])
            ds.append(sample["diskann_ns_per_distance"])
            ratios.append(ds[-1] / hs[-1])
        for name, actual in (
            ("highway_median_ns", statistics.median(hs)),
            ("diskann_median_ns", statistics.median(ds)),
            ("median_paired_native_over_highway", statistics.median(ratios)),
            ("paired_ratio_min", min(ratios)),
            ("paired_ratio_max", max(ratios)),
        ):
            require(math.isclose(case[name], actual), f"invalid summary: {name}")
        checks = case["numerical_checks"]
        require(checks["comparisons"] == case["corpus_count"] + 4, "check count mismatch")
        require(0 <= checks["bit_mismatches"] <= checks["comparisons"], "invalid bit count")
        for name in (
            "max_pair_abs_error",
            "max_highway_oracle_abs_error",
            "max_diskann_oracle_abs_error",
        ):
            require(math.isfinite(checks[name]) and checks[name] >= 0, "invalid error")
    return report


def summarize(paths: list[Path]) -> str:
    require(bool(paths), "no campaigns supplied")
    require(len({path.resolve() for path in paths}) == len(paths), "duplicate campaign path")
    reports = [load(path) for path in paths]
    first = reports[0]
    maps = [{key(case): case for case in report["cases"]} for report in reports]
    for report, cases in zip(reports, maps):
        for field in (
            "architecture", "os", "highway_target", "compiled_neon", "compiled_dotprod",
            "diskann_version", "native_api",
        ):
            require(report[field] == first[field], f"mixed build configurations: {field}")
        require(cases.keys() == maps[0].keys(), "case matrices differ")
        for k, case in cases.items():
            require(case["input_fnv1a64"] == maps[0][k]["input_fnv1a64"], "input mismatch")
            require(case["numerical_checks"] == maps[0][k]["numerical_checks"], "numerical checks differ")
    lines = [
        f"Campaigns: {len(reports)}; cases per campaign: {len(maps[0])}; "
        f"Highway target: {first['highway_target']}; "
        f"NEON={first['compiled_neon']}, dotprod={first['compiled_dotprod']}.",
        "",
        "Ratios >1 favor Highway; <1 favor DiskANN. Pooled paired samples; "
        "range is observed min–max, not a confidence interval.",
        "",
        "| Dimension | Metric | Offset B | Corpus | Highway median ns | DiskANN median ns | "
        "Median paired D/H (range) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for k in sorted(maps[0]):
        samples = [sample for cases in maps for sample in cases[k]["samples"]]
        hs = [sample["highway_ns_per_distance"] for sample in samples]
        ds = [sample["diskann_ns_per_distance"] for sample in samples]
        ratios = [d / h for h, d in zip(hs, ds)]
        dim, metric, offset, count = k
        lines.append(
            f"| {dim} | {metric} | {offset} | {count} | {statistics.median(hs):.3f} | "
            f"{statistics.median(ds):.3f} | {statistics.median(ratios):.4f} "
            f"({min(ratios):.4f}–{max(ratios):.4f}) |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    print(summarize(args.runs))


if __name__ == "__main__":
    main()
