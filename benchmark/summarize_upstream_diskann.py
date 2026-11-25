#!/usr/bin/env python3
"""Verify completed campaign artifacts and reproduce descriptive build tables.

The measurement runner enforces vector/edge and float64-oracle checks. This tool
verifies retained artifacts and recomputes summary/stage statistics; it is not a
second independent implementation of those numerical checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from upstream_diskann_benchmark import fingerprint, graph_snapshot, require, summarize


def analyze(path: Path) -> tuple[dict, str]:
    report = json.loads(path.read_text())
    require(report["schema_version"] == 1 and bool(report["cases"]) and bool(report["backends"]), "invalid campaign schema/matrix")
    summary = summarize(report)  # rejects incomplete/failed campaigns
    for artifact in report["artifacts"]:
        require(fingerprint(Path(artifact["path"])) == artifact, "frozen executable changed")
    for case, derived in zip(report["cases"], summary["cases"]):
        for pair in case["inputs"].values():
            for artifact in pair.values():
                require(fingerprint(Path(artifact["path"])) == artifact, "shared input changed")
        expected = {(backend, phase, i) for backend in report["backends"]
                    for phase, count in (("warmup", report["warmups"]), ("measured", report["repeats"]))
                    for i in range(count)}
        actual = {(run["backend"], run["phase"], run["repetition"]) for run in case["runs"]}
        require(actual == expected and len(case["runs"]) == len(expected), "campaign matrix mismatch")
        stages: dict[str, list[dict]] = {}
        for run in case["runs"]:
            require(run["status"] == "validated", "unvalidated run")
            worker_file = run["worker_result"]
            require(fingerprint(Path(worker_file["path"])) == worker_file, "worker evidence changed")
            for artifact in run["structure"].get("files", []):
                current = fingerprint(Path(artifact["path"]))
                require(all(current[key] == artifact[key] for key in current), "saved native index changed")
            worker = json.loads(Path(worker_file["path"]).read_text())
            if run["backend"].startswith("sqlite"):
                current_graph = graph_snapshot(Path(worker["snapshot"]), case["count"], case["dimension"], "l2")
                require(current_graph["sha256"] == run["structure"]["graph_sha256"], "SQLite snapshot changed")
                if run["backend"] == "sqlite-file":
                    original_graph = graph_snapshot(Path(worker["database"]), case["count"], case["dimension"], "l2")
                    require(original_graph["sha256"] == current_graph["sha256"], "SQLite database differs from its snapshot")
            if run["backend"] in ("memory", "ssd") and run["phase"] == "measured":
                stages.setdefault(run["backend"], []).append(worker["result"]["timings_seconds"])
        derived["stage_medians_seconds"] = {
            backend: {name: statistics.median(sample[name] for sample in samples) for name in samples[0]}
            for backend, samples in stages.items()
        }
        require(all(item["same_backend_predictions_exact"] and item["same_backend_structure_exact"]
                    and item["same_backend_reachability_exact"] for item in derived["backends"]),
                "same-backend repeat mismatch")
    measured = sum(run["phase"] == "measured" for case in report["cases"] for run in case["runs"])
    lines = [f"Validated {sum(len(case['runs']) for case in report['cases'])} builds; {measured} measured.",
             "Medians are seconds; RAM readiness, SSD pipeline, and SQLite commit have different boundaries.", "",
             "| Workload | Upstream RAM ready | Upstream SSD full build | SQLite :memory: | SQLite file |",
             "|---|---:|---:|---:|---:|"]
    for case in summary["cases"]:
        lookup = {item["backend"]: item for item in case["backends"]}
        cells = [f"{lookup[name]['median_seconds']:.6f}" if name in lookup else "—"
                 for name in ("memory", "ssd", "sqlite-memory", "sqlite-file")]
        lines.append(f"| {case['count']}×{case['dimension']}D | " + " | ".join(cells) + " |")
    lines += ["", "| Workload | Backend | Median seconds | Observed min–max | Recall@10 L10 / L50 / L100 |",
              "|---|---|---:|---:|---|"]
    for case in summary["cases"]:
        for item in case["backends"]:
            recall = " / ".join(f"{item['recall_at_10'][str(window)]:.3f}" for window in (10, 50, 100))
            lines.append(f"| {case['count']}×{case['dimension']}D | {item['backend']} | {item['median_seconds']:.6f} | "
                         f"{item['min_seconds']:.6f}–{item['max_seconds']:.6f} | {recall} |")
    return summary, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--json-output", type=Path, help="optional NEW summary destination")
    args = parser.parse_args()
    summary, markdown = analyze(args.results)
    if args.json_output:
        with args.json_output.open("x") as output:
            json.dump(summary, output, indent=2, allow_nan=False)
            output.write("\n")
    print(markdown, end="")


if __name__ == "__main__":
    main()
