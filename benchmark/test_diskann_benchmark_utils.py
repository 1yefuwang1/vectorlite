"""Artifact-independent tests; run with unittest, without benchmark conftest."""
import contextlib
import io
import json
import math
import sqlite3
import struct
import unittest
from unittest import mock

import diskann_benchmark as bench


class StreamingUtilities(unittest.TestCase):
    def config(self, **changes):
        return {**vars(bench.parser().parse_args([])), **changes}

    def test_float32_little_endian_roundtrip(self):
        blob = bench.pack_vector([1, -2.5, 1 / 3])
        self.assertEqual(blob, struct.pack("<3f", 1, -2.5, 1 / 3))
        self.assertEqual(list(bench.unpack_vector(blob, 3)), list(struct.unpack("<3f", blob)))
        with self.assertRaises(ValueError):
            bench.unpack_vector(blob, 2)

    def test_seeded_vectors_are_repeatable_and_streamed(self):
        source = bench.random_vectors(3, 4, 42)
        self.assertIs(iter(source), source)
        first = list(source)
        self.assertEqual(first, list(bench.random_vectors(3, 4, 42)))
        self.assertNotEqual(first, list(bench.random_vectors(3, 4, 43)))
        self.assertTrue(all(len(blob) == 16 for blob in first))
        self.assertTrue(all(-1 <= value <= 1 for blob in first for value in bench.unpack_vector(blob, 4)))

    def test_batches_have_bounded_lookahead(self):
        seen = []
        def source():
            for item in range(257):
                seen.append(item)
                yield item
        stream = bench.batches(source(), 128)
        self.assertEqual(next(stream), list(range(128)))
        self.assertEqual(len(seen), 128)
        self.assertEqual([len(batch) for batch in stream], [128, 1])
        for invalid in (0, 129):
            with self.assertRaises(ValueError):
                list(bench.batches([], invalid))

    def test_percentile_interpolation_and_latency_units(self):
        self.assertEqual(bench.percentile([4, 1, 3, 2], 0.5), 2.5)
        self.assertAlmostEqual(bench.percentile([4, 1, 3, 2], 0.95), 3.85)
        self.assertIsNone(bench.percentile([], 0.5))
        result = bench.latency_summary([0.001, 0.003])
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["p50_ms"], 2)
        self.assertAlmostEqual(result["p95_ms"], 2.9)

    def test_peak_rss_platform_units(self):
        self.assertEqual(bench.rss_bytes(123, "Darwin"), 123)
        self.assertEqual(bench.rss_bytes(123, "Linux"), 123 * 1024)
        with self.assertRaises(ValueError):
            bench.rss_bytes(123, "Unverified")

    def test_heap_retains_nearest_and_stable_ties(self):
        heap = []
        for distance, rowid in [(9, 3), (1, 4), (1, 2), (0, 7), (3, 8), (1, 1)]:
            bench.offer_neighbor(heap, distance, rowid, 3)
        self.assertEqual(bench.truth_labels(heap), [7, 1, 2])
        self.assertEqual(len(heap), 3)

    def test_recall_uses_unique_labels_and_actual_truth_length(self):
        self.assertEqual(bench.recall([[1, 1, 8], [3]], [[1, 2], [3]]), 0.75)
        for found, expected in [([], []), ([[1]], [[]]), ([[1]], [[1], [2]])]:
            with self.assertRaises(ValueError):
                bench.recall(found, expected)

    def test_linear_vectors_and_adjacency(self):
        self.assertEqual(list(bench.unpack_vector(bench.linear_vector(7, 4), 4)), [7, 0, 0, 0])
        self.assertEqual(struct.unpack("<2Q", bench.linear_neighbors(1, 5)), (5, 1))
        self.assertEqual(struct.unpack("<Q", bench.linear_neighbors(5, 5)), (3,))
        self.assertEqual(bench.linear_truth(2.25, 65536, 4), [2, 3, 1, 4])
        self.assertEqual(bench.linear_truth(8.25, 1, 1), [1])

    def test_configuration_failures(self):
        bench.validate_config(self.config())
        for changes in ({"count": 0}, {"dim": 0}, {"batch_size": 129}, {"degree": 1},
                        {"alpha": math.nan}, {"updates": -1}, {"search_list_size": 3},
                        {"max_visits": 3}, {"queries": 1025}, {"query_memory_limit_mib": 32},
                        {"fixture": "storage-linear", "metric": "cosine"},
                        {"fixture": "storage-linear", "count": 2**24 + 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                bench.validate_config(self.config(**changes))

    def test_manifest_version_and_missing_row(self):
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute(f"CREATE TABLE {bench.MANIFEST}(singleton INTEGER,config TEXT)")
            with self.assertRaises(ValueError):
                bench.read_manifest(connection)
            config = {"manifest_version": 1, "fixture": "none"}
            connection.execute(f"INSERT INTO {bench.MANIFEST} VALUES(1,?)", (json.dumps(config),))
            self.assertEqual(bench.read_manifest(connection), config)
            connection.execute(f"UPDATE {bench.MANIFEST} SET config=?", ('{"manifest_version":2}',))
            with self.assertRaises(ValueError):
                bench.read_manifest(connection)
        finally:
            connection.close()

    def format4_descriptor(self, dimension):
        config = self.config()
        options = {key: config[key] for key in ("degree", "build_list_size", "search_list_size",
                                                "cache_bytes", "max_visits")}
        options["alpha"] = struct.unpack("<f", struct.pack("<f", config["alpha"]))[0]
        return {"backend": "diskann", "format_version": 4, "dimension": dimension,
                "metric": "l2_squared", "maintenance": "deferred-oneshot-v1", "options": options}

    def test_truth_worker_scans_only_published_current_versions_without_extension(self):
        description = self.format4_descriptor(2)
        config = self.config(count=4, dim=2, queries=2, k=2, batch_size=1,
                             storage_format_version=4, storage_descriptor=description)
        connection = sqlite3.connect(":memory:", isolation_level=None)
        connection.executescript(
            f"CREATE TABLE {bench.MANIFEST}(singleton INTEGER,config TEXT);"
            f"CREATE TABLE {bench.TABLE}_diskann_meta(singleton INTEGER,format_version INTEGER,descriptor BLOB,published_generation INTEGER);"
            f"CREATE TABLE {bench.TABLE}_diskann_current(public_rowid INTEGER,entry_id INTEGER);"
            f"CREATE TABLE {bench.TABLE}_diskann_entries(entry_id INTEGER,public_rowid INTEGER,vector BLOB);"
            f"CREATE TABLE {bench.TABLE}_diskann_graph_nodes(generation_id INTEGER,native_local_id INTEGER,entry_id INTEGER,state INTEGER);")
        connection.execute(f"INSERT INTO {bench.MANIFEST} VALUES(1,?)", (json.dumps({**config, "manifest_version": 1}),))
        connection.execute(f"INSERT INTO {bench.TABLE}_diskann_meta VALUES(1,4,?,1)", (json.dumps(description),))
        vectors = {1: [0.0, 0.0], 3: [1.0, 1.0]}
        # entry 2 is superseded, entry 4 is pending, entry 5 belongs to a candidate.
        for entry, rowid, vector in [(1, 1, vectors[1]), (2, 2, [0, 0]), (3, 3, vectors[3]),
                                      (4, 2, [99, 99]), (5, 4, [0, 0])]:
            connection.execute(f"INSERT INTO {bench.TABLE}_diskann_entries VALUES(?,?,?)", (entry, rowid, bench.pack_vector(vector)))
        connection.executemany(f"INSERT INTO {bench.TABLE}_diskann_current VALUES(?,?)", [(1, 1), (2, 4), (3, 3), (4, 5)])
        connection.executemany(f"INSERT INTO {bench.TABLE}_diskann_graph_nodes VALUES(?,?,?,?)", [(1, 0, 1, 0), (1, 1, 2, 0), (1, 2, 3, 0), (2, 0, 5, 0), (1, 3, None, 2)])
        with mock.patch.object(bench, "connect", return_value=connection):
            result = bench.truth_worker(config)
        expected = []
        for blob in bench.query_vectors(config):
            query = bench.unpack_vector(blob, 2)
            expected.append(sorted(vectors, key=lambda rowid: (
                sum((x - y) ** 2 for x, y in zip(vectors[rowid], query)), rowid)))
        self.assertEqual(result["labels"], expected)
        self.assertEqual(result["rows_scanned"], 2)
        self.assertFalse(result["numpy"])

    def test_storage_fixture_populates_format4_generation(self):
        config = self.config(fixture="storage-linear", count=5, dim=3, batch_size=2)
        connection = sqlite3.connect(":memory:", isolation_level=None)
        try:
            connection.executescript(
                f"CREATE TABLE {bench.TABLE}_diskann_current(public_rowid INTEGER,entry_id INTEGER);"
                f"CREATE TABLE {bench.TABLE}_diskann_meta(singleton INTEGER,format_version INTEGER,descriptor BLOB,revision INTEGER,next_generation_id INTEGER,next_job_id INTEGER,published_generation INTEGER);"
                f"CREATE TABLE {bench.TABLE}_diskann_generations(generation_id INTEGER,job_id INTEGER,state INTEGER,source_revision INTEGER,source_hash BLOB,config_hash BLOB,total_count INTEGER,eligible_count INTEGER,root_local_id INTEGER);"
                f"CREATE TABLE {bench.TABLE}_diskann_graph_nodes(generation_id INTEGER,native_local_id INTEGER,entry_id INTEGER,state INTEGER,frozen_vector BLOB,neighbors BLOB);"
                f"CREATE TABLE {bench.TABLE}_diskann_txn(singleton INTEGER,value INTEGER);"
                f"INSERT INTO {bench.TABLE}_diskann_txn VALUES(1,0),(2,0);")
            connection.execute(f"INSERT INTO {bench.TABLE}_diskann_meta VALUES(1,4,?,5,1,1,NULL)", (json.dumps(self.format4_descriptor(3)),))
            connection.executemany(f"INSERT INTO {bench.TABLE}_diskann_current VALUES(?,?)", [(i, i + 10) for i in range(1, 6)])
            bench.build_storage_fixture(connection, config)
            nodes = connection.execute(f"SELECT native_local_id,entry_id,state,frozen_vector,neighbors FROM {bench.TABLE}_diskann_graph_nodes ORDER BY native_local_id").fetchall()
            self.assertEqual(len(nodes), 6)
            for i, entry, state, vector, adjacency in nodes[:5]:
                self.assertEqual((entry, state, vector), (i + 11, 0, None))
                self.assertEqual(adjacency, bench.linear_neighbors(i + 1, 5))
            self.assertEqual(nodes[5], (5, None, 2, bench.linear_vector(0, 3), struct.pack("<Q", 0)))
            self.assertEqual(connection.execute(f"SELECT state,total_count,eligible_count,root_local_id FROM {bench.TABLE}_diskann_generations").fetchone(), (1, 5, 5, 5))
            self.assertEqual(connection.execute(f"SELECT next_generation_id,next_job_id,published_generation FROM {bench.TABLE}_diskann_meta").fetchone(), (2, 2, 1))
        finally:
            connection.close()

    def test_descriptor_rejects_mismatched_manifest(self):
        db = sqlite3.connect(":memory:")
        try:
            db.execute(f"CREATE TABLE {bench.TABLE}_diskann_meta(singleton INTEGER,format_version INTEGER,descriptor BLOB)")
            value = self.format4_descriptor(2)
            db.execute(f"INSERT INTO {bench.TABLE}_diskann_meta VALUES(1,4,?)", (json.dumps(value),))
            config = self.config(dim=2, storage_format_version=4, storage_descriptor=value)
            self.assertEqual(bench.descriptor(db, config), (4, value))
            with self.assertRaisesRegex(ValueError, "manifest"):
                bench.descriptor(db, {**config, "storage_format_version": 3})
            with self.assertRaisesRegex(ValueError, "option mismatch"):
                bench.descriptor(db, {**config, "degree": 32})
        finally:
            db.close()

    def test_worker_launch_executes_fresh_interpreter_and_checks_exit(self):
        with mock.patch.object(bench.subprocess, "run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = '{"ok":true}'
            self.assertEqual(bench.run_worker("query", {"sample": 1}), {"ok": True})
            command = run.call_args.args[0]
            self.assertEqual(command[0], bench.sys.executable)
            self.assertEqual(command[1], "-B")
            self.assertIn("--worker", command)
            run.return_value.returncode = 7
            run.return_value.stderr = "worker failure"
            with self.assertRaisesRegex(RuntimeError, "exited 7"):
                bench.run_worker("query", {})

    def test_help_does_not_load_extension(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as exit_status:
            bench.main(["--help"])
        self.assertEqual(exit_status.exception.code, 0)
        self.assertIn("--require-beyond-ram", output.getvalue())
        self.assertIn("--extension", output.getvalue())


if __name__ == "__main__":
    unittest.main()
