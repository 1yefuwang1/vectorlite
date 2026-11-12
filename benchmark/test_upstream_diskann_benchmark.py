"""Artifact-independent checks for the matched upstream build runner."""

import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

import upstream_diskann_benchmark as benchmark


class UpstreamBenchmarkTests(unittest.TestCase):
    def dataset(self):
        vectors = np.arange(24, dtype="<f4").reshape(12, 2)
        queries = np.zeros((1, 2), dtype="<f4")
        return {"vectors": vectors, "queries": queries, "truth": [list(range(10))]}

    def searches(self):
        dataset = self.dataset()
        values = np.square(dataset["vectors"][:10].astype(np.float64)).sum(axis=1)
        bits = [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]
        return [{"window": window, "ids": [list(range(10))], "bits": [bits.copy()]}
                for window in (10, 50, 100)]

    def test_quality_matches_exact_reference(self):
        quality = benchmark.validate_searches(self.searches(), self.dataset())
        self.assertEqual([item["recall_at_10"] for item in quality], [1.0, 1.0, 1.0])
        self.assertEqual([item["max_returned_distance_oracle_error"] for item in quality], [0, 0, 0])

    def test_invalid_queries_fail_closed(self):
        for variant in ("duplicate", "range", "nan", "wrong-distance", "window", "rows"):
            with self.subTest(variant=variant):
                searches = self.searches()
                first = searches[0]
                if variant == "duplicate":
                    first["ids"][0][1] = first["ids"][0][0]
                elif variant == "range":
                    first["ids"][0][0] = 12
                elif variant == "nan":
                    first["bits"][0][0] = 0x7FC00000
                elif variant == "wrong-distance":
                    first["bits"][0][0] = 0
                elif variant == "window":
                    first["window"] = 11
                else:
                    first["bits"].clear()
                with self.assertRaises(ValueError):
                    benchmark.validate_searches(searches, self.dataset())

    def test_input_file_encodings_have_identical_payloads(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            result = benchmark.inputs(out, 256, 4, 3)
            for key, shape in (("vectors", (256, 4)), ("queries", (3, 4))):
                files = result["files"][key]
                raw = Path(files["raw"]["path"]).read_bytes()
                headered = Path(files["bigann"]["path"]).read_bytes()
                self.assertEqual(struct.unpack("<II", headered[:8]), shape)
                self.assertEqual(raw, headered[8:])
            self.assertEqual(result["vectors"].shape, (256, 4))
            self.assertEqual(len(result["truth"]), 3)

    def test_native_search_normalization(self):
        source = {"result": {"searches": [{"search_l": 10, "results": [
            {"query_index": 0, "ids": [1], "distance_bits": [0]}]}]}}
        self.assertEqual(benchmark.normalized_searches(source, "memory"), [
            {"window": 10, "ids": [[1]], "bits": [[0]]}])
        source["result"]["searches"][0]["results"][0]["query_index"] = 1
        with self.assertRaises(ValueError):
            benchmark.normalized_searches(source, "ssd")

    def test_storage_completeness_is_not_complete_root_connectivity(self):
        result = {"vectors_inserted": 12, "validated_live_vectors": 12, "validated_nodes": 13,
                  "reachable_nodes": 11, "unreachable_nodes": 2, "unreachable_ids": [3, 5]}
        fields = ("validated_live_vectors", "validated_nodes", "reachable_nodes", "unreachable_nodes", "unreachable_ids")
        audit = {field: copy.deepcopy(result[field]) for field in fields}
        result["storage_validation"] = {"roundtrip_audit_matches": True, "post_build": audit,
                                        "post_load": copy.deepcopy(audit)}
        validated = benchmark.validate_memory_integrity(result, 12)
        self.assertEqual(validated["unreachable_live_nodes"], 2)
        for field, value in (("validated_live_vectors", 11), ("validated_nodes", 12),
                             ("reachable_nodes", 14), ("unreachable_nodes", 1),
                             ("unreachable_ids", [3, 3]), ("unreachable_ids", [3, 12])):
            with self.subTest(field=field, value=value):
                invalid = copy.deepcopy(result)
                invalid[field] = value
                with self.assertRaises(ValueError):
                    benchmark.validate_memory_integrity(invalid, 12)
        connected = copy.deepcopy(result)
        connected.update(reachable_nodes=13, unreachable_nodes=0, unreachable_ids=[])
        for stage in ("post_build", "post_load"):
            connected["storage_validation"][stage].update(reachable_nodes=13, unreachable_nodes=0, unreachable_ids=[])
        self.assertEqual(benchmark.validate_memory_integrity(connected, 12)["unreachable_live_nodes"], 0)
        invalid_roundtrip = copy.deepcopy(result)
        invalid_roundtrip["storage_validation"]["post_load"]["validated_nodes"] = 12
        with self.assertRaises(ValueError):
            benchmark.validate_memory_integrity(invalid_roundtrip, 12)

    def test_summary_rejects_partial_campaign_and_missing_runs(self):
        report = {"schema_version": 1, "status": "running", "cases": [], "backends": ["memory"], "repeats": 1}
        with self.assertRaises(ValueError):
            benchmark.summarize(report)
        report.update(status="complete", cases=[{"name": "fixture", "count": 12, "dimension": 2, "runs": []}])
        with self.assertRaises(ValueError):
            benchmark.summarize(report)

    def test_summary_uses_measured_runs_and_separate_quality(self):
        run = {"backend": "memory", "phase": "measured", "status": "validated", "build_seconds": 2.0,
               "quality": [{"window": 10, "recall_at_10": 0.9}], "output_bytes": 128,
               "predictions_sha256": "a", "structure_sha256": "b"}
        warmup = copy.deepcopy(run)
        warmup.update(phase="warmup", build_seconds=999)
        report = {"status": "complete", "repeats": 1, "backends": ["memory"], "cases": [
            {"name": "fixture", "count": 12, "dimension": 2, "runs": [warmup, run]}]}
        summary = benchmark.summarize(report)["cases"][0]["backends"][0]
        self.assertEqual(summary["median_seconds"], 2.0)
        self.assertEqual(summary["recall_at_10"], {"10": 0.9})
        self.assertTrue(summary["same_backend_structure_exact"])
        json.dumps(summary, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
