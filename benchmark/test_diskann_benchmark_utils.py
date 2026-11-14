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
        self.assertEqual(struct.unpack("<2Q", bench.linear_neighbors(1, 5)), (0, 2))
        self.assertEqual(struct.unpack("<Q", bench.linear_neighbors(5, 5)), (4,))
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

    def test_truth_worker_scans_live_rows_without_extension(self):
        config = self.config(count=3, dim=2, queries=2, k=2, batch_size=1)
        connection = sqlite3.connect(":memory:", isolation_level=None)
        connection.execute(f"CREATE TABLE {bench.MANIFEST}(singleton INTEGER,config TEXT)")
        connection.execute(f"INSERT INTO {bench.MANIFEST} VALUES(1,?)",
                           (json.dumps({**config, "manifest_version": 1}),))
        connection.execute(f"CREATE TABLE {bench.TABLE}_diskann_meta(live_count INTEGER)")
        connection.execute(f"INSERT INTO {bench.TABLE}_diskann_meta VALUES(2)")
        connection.execute(f"CREATE TABLE {bench.TABLE}_diskann_nodes"
                           "(node_id INTEGER,public_rowid INTEGER,state INTEGER,vector BLOB)")
        vectors = {1: [0.0, 0.0], 3: [1.0, 1.0]}
        for rowid, vector in vectors.items():
            connection.execute(f"INSERT INTO {bench.TABLE}_diskann_nodes VALUES(?,?,0,?)",
                               (rowid, rowid, bench.pack_vector(vector)))
        connection.execute(f"INSERT INTO {bench.TABLE}_diskann_nodes VALUES(2,NULL,1,?)",
                           (bench.pack_vector([99.0, 99.0]),))
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

    def test_storage_fixture_populates_only_bounded_linear_records(self):
        config = self.config(fixture="storage-linear", count=5, dim=3, batch_size=2)
        connection = sqlite3.connect(":memory:", isolation_level=None)
        try:
            connection.executescript(
                f"CREATE TABLE {bench.TABLE}(rowid INTEGER PRIMARY KEY,embedding BLOB);"
                f"CREATE TABLE {bench.TABLE}_diskann_nodes(node_id INTEGER PRIMARY KEY,"
                "public_rowid INTEGER,state INTEGER,vector BLOB,neighbors BLOB);"
                f"CREATE TABLE {bench.TABLE}_diskann_meta(singleton INTEGER,format_version INTEGER,descriptor BLOB,"
                "live_count INTEGER,next_node_id INTEGER,deleted_count INTEGER,entrypoints BLOB,revision INTEGER);"
                f"CREATE TABLE {bench.TABLE}_diskann_txn(singleton INTEGER,value INTEGER);"
                f"INSERT INTO {bench.TABLE}_diskann_txn VALUES(1,0),(2,0);")
            descriptor = json.dumps({"format_version": 3, "metric": "l2_squared",
                                     "atomic_write_protocol": "guard-two-row-v1"})
            connection.execute(f"INSERT INTO {bench.TABLE}_diskann_meta VALUES(1,3,?,1,2,0,?,1)",
                               (descriptor, struct.pack("<Q", 0)))
            for node in [0, 1]:
                connection.execute(f"INSERT INTO {bench.TABLE}_diskann_nodes VALUES(?,?,?, ?,?)",
                                   (node, 1 if node else None, 0 if node else 2,
                                    bench.linear_vector(1, 3), b""))
            bench.build_storage_fixture(connection, config)
            nodes = connection.execute(f"SELECT node_id,public_rowid,state,vector,neighbors "
                                       f"FROM {bench.TABLE}_diskann_nodes ORDER BY node_id").fetchall()
            self.assertEqual(len(nodes), 6)
            self.assertEqual(nodes[0][:3], (0, None, 2))
            self.assertEqual(nodes[0][4], struct.pack("<Q", 1))
            for rowid, public, state, blob, adjacency in nodes[1:]:
                self.assertEqual((public, state), (rowid, 0))
                self.assertEqual(list(bench.unpack_vector(blob, 3)), [rowid, 0, 0])
                self.assertEqual(adjacency, bench.linear_neighbors(rowid, 5))
            self.assertEqual(connection.execute(f"SELECT live_count,next_node_id,deleted_count "
                                                f"FROM {bench.TABLE}_diskann_meta").fetchone(), (5, 6, 0))
        finally:
            connection.close()

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
