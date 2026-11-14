const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const { DatabaseSync } = require('node:sqlite');
const vectorlite = require('../src/index.js');

function open(database) {
    const connection = new DatabaseSync(database, { allowExtension: true });
    try {
        connection.loadExtension(vectorlite.vectorlitePath());
        return connection;
    } catch (error) {
        connection.close();
        throw error;
    }
}

function snapshot(connection) {
    return ['meta', 'nodes', 'txn', 'rebuild'].map(suffix => connection.prepare(
        `SELECT * FROM vectors_diskann_${suffix} ORDER BY 1`,
    ).all());
}

test('built-in SQLite loads the packaged HNSW backend', () => {
    const db = open(':memory:');
    try {
        db.exec('CREATE VIRTUAL TABLE h USING vectorlite(e float32[2],hnsw(max_elements=8))');
        db.exec("INSERT INTO h(rowid,e) VALUES(1,vector_from_json('[1,0]')),(2,vector_from_json('[0,2]'))");
        const rows = db.prepare("SELECT rowid,distance FROM h WHERE knn_search(e,knn_param(vector_from_json('[1,0]'),1))").all();
        assert.equal(rows.length, 1);
        assert.equal(rows[0].rowid, 1);
        assert.equal(rows[0].distance, 0);
    } finally {
        db.close();
    }
});

test('built-in SQLite verifies packaged DiskANN rollback, rebuild and reopen', () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'vectorlite-node-sqlite-'));
    const database = path.join(directory, 'vectors.db');
    let db;
    try {
        db = open(database);
        db.exec('CREATE VIRTUAL TABLE vectors USING vectorlite(embedding float32[2],diskann())');
        db.exec('CREATE TABLE metadata(rowid INTEGER PRIMARY KEY,label TEXT)');
        db.exec("INSERT INTO vectors(rowid,embedding) VALUES(1,vector_from_json('[1,0]')),(7,vector_from_json('[0,2]'))");
        const before = snapshot(db);
        db.exec('BEGIN');
        db.exec("INSERT INTO metadata VALUES(42,'pending')");
        db.exec("INSERT INTO vectors(rowid,embedding) VALUES(42,vector_from_json('[4,2]'))");
        db.exec("UPDATE vectors SET embedding=vector_from_json('[8,1]') WHERE rowid=1");
        db.exec('DELETE FROM vectors WHERE rowid=7');
        db.exec('ROLLBACK');
        assert.deepEqual(snapshot(db), before);
        assert.equal(db.prepare('SELECT rowid FROM metadata WHERE rowid=42').get(), undefined);
        db.exec("CREATE TRIGGER fail_neighbors BEFORE UPDATE OF neighbors ON vectors_diskann_nodes BEGIN SELECT RAISE(ABORT,'late node smoke'); END");
        db.exec('BEGIN');
        assert.throws(() => db.exec("INSERT INTO vectors(rowid,embedding) VALUES(42,vector_from_json('[4,2]'))"), /late node smoke/);
        assert.deepEqual(snapshot(db), before);
        db.exec('ROLLBACK');
        db.exec('DROP TRIGGER fail_neighbors');
        db.exec("BEGIN; INSERT INTO metadata VALUES(42,'committed'); INSERT INTO vectors(rowid,embedding) VALUES(42,vector_from_json('[4,2]')); COMMIT");
        const sql = "SELECT rowid,distance FROM vectors WHERE knn_search(embedding,knn_param(vector_from_json('[1,0]'),3,?))";
        const options = JSON.stringify({ search_list_size: 64 });
        const neighbors = db.prepare(sql).all(options);
        assert.deepEqual(neighbors.map(row => row.rowid), [1, 7, 42]);
        assert.deepEqual(neighbors.map(row => row.distance), [0, 5, 13]);
        db.close();
        db = open(database);
        assert.deepEqual(db.prepare(sql).all(options), neighbors);
        assert.equal(db.prepare('SELECT label FROM metadata WHERE rowid=42').get().label, 'committed');
        db.exec("UPDATE vectors SET embedding=vector_from_json('[2,0]') WHERE rowid=1");
        db.exec('DELETE FROM vectors WHERE rowid=7');
        db.exec("INSERT INTO vectors(operation) VALUES('consolidate')");
        assert.equal(db.prepare('SELECT count(*) AS n FROM vectors_diskann_nodes WHERE state=1').get().n, 0);
        assert.equal(db.prepare('SELECT count(*) AS n FROM vectors_diskann_rebuild').get().n, 0);
        const nearest = db.prepare("SELECT rowid,distance FROM vectors WHERE knn_search(embedding,knn_param(vector_from_json('[2,0]'),1))").get();
        assert.equal(nearest.rowid, 1);
        assert.equal(nearest.distance, 0);
        assert.throws(() => db.prepare('SELECT vectorlite_atomic_write(NULL,1)').get(), /atomic/);
    } finally {
        if (db && db.isOpen) db.close();
        // Verify the exact created temporary directory before removing it.
        assert.equal(path.dirname(path.resolve(directory)), path.resolve(os.tmpdir()));
        assert.match(path.basename(directory), /^vectorlite-node-sqlite-/);
        fs.rmSync(directory, { recursive: true, force: true });
    }
});
