"""Artifact-independent checks: python -m unittest discover -s benchmark -p test_distance_kernel_results.py."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import summarize_distance_kernels as summary


def fixture():
    return {
        "schema_version": 1,
        "status": "complete",
        "architecture": "aarch64",
        "os": "macos",
        "highway_target": "test-target",
        "compiled_neon": True,
        "compiled_dotprod": True,
        "diskann_version": "0.60.0",
        "native_api": "provider",
        "pairs": 2,
        "cases": [{
            "dimension": 128,
            "metric": "l2_squared",
            "alignment_offset_bytes": 0,
            "corpus_count": 16,
            "input_fnv1a64": "test-input",
            "operations_per_sample": 100,
            "highway_median_ns": 1_000_000.0,
            "diskann_median_ns": 2_000_000.0,
            "median_paired_native_over_highway": 2.0,
            "paired_ratio_min": 2.0,
            "paired_ratio_max": 2.0,
            "numerical_checks": {
                "comparisons": 20,
                "bit_mismatches": 2,
                "max_pair_abs_error": 1e-6,
                "max_highway_oracle_abs_error": 1e-6,
                "max_diskann_oracle_abs_error": 1e-6,
            },
            "samples": [{
                "pair": i,
                "order": "highway,diskann" if i == 0 else "diskann,highway",
                "highway_seconds": 0.1,
                "diskann_seconds": 0.2,
                "highway_ns_per_distance": 1_000_000.0,
                "diskann_ns_per_distance": 2_000_000.0,
                "highway_checksum": 3.0,
                "diskann_checksum": 3.0,
            } for i in range(2)],
        }],
    }


class KernelResultTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "run.json"

    def save(self, data, name="run.json"):
        path = self.path.with_name(name)
        path.write_text(json.dumps(data))
        return path

    def test_valid_summary_and_repeat(self):
        path = self.save(fixture())
        repeat = self.save(fixture(), "repeat.json")
        text = summary.summarize([path, repeat])
        self.assertIn("Campaigns: 2", text)
        self.assertIn("2.0000 (2.0000–2.0000)", text)
        with self.assertRaisesRegex(ValueError, "duplicate campaign path"):
            summary.summarize([path, path])

    def test_reject_incomplete(self):
        data = fixture()
        data["status"] = "running"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            summary.load(self.save(data))

    def test_reject_invalid_measurement_or_derived_statistic(self):
        for field, value in (("highway_seconds", 0), ("diskann_checksum", float("nan")),
                             ("order", "diskann,highway"), ("highway_ns_per_distance", 7)):
            with self.subTest(field=field):
                data = fixture()
                data["cases"][0]["samples"][0][field] = value
                with self.assertRaises(ValueError):
                    summary.load(self.save(data))
        data = fixture()
        data["cases"][0]["median_paired_native_over_highway"] = 1.5
        with self.assertRaisesRegex(ValueError, "invalid summary"):
            summary.load(self.save(data))

    def test_reject_duplicate_cases_or_missing_samples(self):
        for duplicate in (True, False):
            data = fixture()
            if duplicate:
                data["cases"].append(copy.deepcopy(data["cases"][0]))
            else:
                data["cases"][0]["samples"].pop()
            with self.assertRaises(ValueError):
                summary.load(self.save(data))

    def test_reject_mixed_inputs_or_features(self):
        a = self.save(fixture(), "a.json")
        for field in ("input", "feature", "numerics"):
            data = fixture()
            if field == "input":
                data["cases"][0]["input_fnv1a64"] = "different"
            elif field == "feature":
                data["compiled_dotprod"] = False
            else:
                data["cases"][0]["numerical_checks"]["bit_mismatches"] = 0
            with self.assertRaises(ValueError):
                summary.summarize([a, self.save(data, "b.json")])


if __name__ == "__main__":
    unittest.main()
