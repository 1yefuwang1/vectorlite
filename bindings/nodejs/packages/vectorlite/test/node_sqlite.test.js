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
        connection.loadExtension(process.env.VECTORLITE_RUST_EXTENSION || vectorlite.vectorlitePath());
        return connection;
    } catch (error) {
        connection.close();
        throw error;
    }
}

function snapshot(connection) {
    return ['meta', 'entries', 'current', 'txn'].map(suffix => connection.prepare(
        `SELECT * FROM vectors_diskann_${suffix} ORDER BY 1`,
    ).all());
}

function publish(db) {
    const control = (operation, options) => db.prepare('INSERT INTO vectors(operation,path) VALUES(?,?)').run(operation, JSON.stringify(options));
    const job = () => JSON.parse(db.prepare("SELECT vectorlite_index_status('main','vectors') AS value").get().value).current_job;
    control('index_prepare', {});
    let j = job();
    const ids = { job_id: j.job_id, fence: j.fence };
    for (let i = 0; j.phase === 0 && i < 100; ++i) {
        control('index_snapshot_step', { ...ids, max_records: 2 });
        j = job();
    }
    assert.equal(j.phase, 1);
    control('index_claim_build', ids);
    const candidate = JSON.parse(db.prepare("SELECT vectorlite_index_build('main','vectors',?) AS value").get(JSON.stringify({ ...ids, memory_budget_bytes: 268435456 })).value);
    const options = { ...ids, candidate_token: candidate.candidate_token, max_records: 1 };
    for (const [phase, operation] of [[2, 'index_import_step'], [3, 'index_validate_step']]) {
        for (let i = 0; job().phase === phase && i < 100; ++i) control(operation, options);
        assert.notEqual(job().phase, phase);
    }
    assert.equal(job().phase, 4);
    control('index_publish', { ...ids, candidate_token: candidate.candidate_token });
    assert.equal(job(), null);
}

test('built-in SQLite loads the selected HNSW extension', () => {
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

test('built-in SQLite verifies deferred DiskANN rollback, publication and reopen', () => {
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
        assert.deepEqual(JSON.parse(db.prepare('SELECT vector_to_json(embedding) AS v FROM vectors WHERE rowid=1').get().v), [8, 1]);
        assert.equal(db.prepare('SELECT rowid FROM vectors WHERE rowid=42').get().rowid, 42);
        db.exec('DELETE FROM vectors WHERE rowid=7');
        db.exec('ROLLBACK');
        assert.deepEqual(snapshot(db), before);
        assert.equal(db.prepare('SELECT rowid FROM metadata WHERE rowid=42').get(), undefined);
        db.exec("CREATE TRIGGER fail_entry BEFORE INSERT ON vectors_diskann_entries BEGIN SELECT RAISE(ABORT,'late entry smoke'); END");
        db.exec('BEGIN');
        assert.throws(() => db.exec("INSERT INTO vectors(rowid,embedding) VALUES(42,vector_from_json('[4,2]'))"), /late entry smoke/);
        assert.deepEqual(snapshot(db), before);
        db.exec('ROLLBACK');
        db.exec('DROP TRIGGER fail_entry');
        db.exec("BEGIN; INSERT INTO metadata VALUES(42,'committed'); INSERT INTO vectors(rowid,embedding) VALUES(42,vector_from_json('[4,2]')); COMMIT");
        const sql = "SELECT rowid,distance FROM vectors WHERE knn_search(embedding,knn_param(vector_from_json('[1,0]'),3,?))";
        const options = JSON.stringify({ search_list_size: 64 });
        assert.deepEqual(JSON.parse(db.prepare('SELECT vector_to_json(embedding) AS v FROM vectors WHERE rowid=42').get().v), [4, 2]);
        assert.deepEqual(db.prepare(sql).all(options), []);
        publish(db);
        const neighbors = db.prepare(sql).all(options);
        assert.deepEqual(neighbors.map(row => row.rowid), [1, 7, 42]);
        assert.deepEqual(neighbors.map(row => row.distance), [0, 5, 13]);
        db.close();
        db = open(database);
        assert.deepEqual(db.prepare(sql).all(options), neighbors);
        assert.equal(db.prepare('SELECT label FROM metadata WHERE rowid=42').get().label, 'committed');
        db.exec("UPDATE vectors SET embedding=vector_from_json('[2,0]') WHERE rowid=1");
        db.exec('DELETE FROM vectors WHERE rowid=7');
        assert.equal(db.prepare("SELECT rowid FROM vectors WHERE knn_search(embedding,knn_param(vector_from_json('[2,0]'),3)) AND rowid=1").get(), undefined);
        publish(db);
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
