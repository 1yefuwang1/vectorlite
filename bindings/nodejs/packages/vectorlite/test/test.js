const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const sqlite3 = require('better-sqlite3');
const vectorlite = require('../src/index.js');

const db = new sqlite3(':memory:');
db.loadExtension(process.env.VECTORLITE_RUST_EXTENSION || vectorlite.vectorlitePath());

console.log(db.prepare('select vectorlite_info()').all());

// Create a vectorlite virtual table hosting 10-dimensional float32 vectors with hnsw index
db.exec('create virtual table test using vectorlite(vec float32[10], hnsw(max_elements=100));')

// insert a json vector
db.prepare('insert into test(rowid, vec) values (?, vector_from_json(?))').run([0, JSON.stringify(Array.from({length: 10}, () => Math.random()))]);
// insert a raw vector
db.prepare('insert into test(rowid, vec) values (?, ?)').run([1, Buffer.from(Float32Array.from(Array.from({length: 10}, () => Math.random())).buffer)]);

// a normal vector query
let result = db.prepare('select rowid from test where knn_search(vec, knn_param(?, 2))')
    .all([Buffer.from(Float32Array.from(Array.from({length: 10}, () => Math.random())).buffer)]);

console.log(result);

// a vector query with rowid filter
result = db.prepare('select rowid from test where knn_search(vec, knn_param(?, 2)) and rowid in (1,2,3)')
    .all([Buffer.from(Float32Array.from(Array.from({length: 10}, () => Math.random())).buffer)]);

console.log(result);

// a vector query with rowid filter
result = db.prepare("select rowid, vector_distance(vec, ?, 'l2') from test where rowid in (0,1,2,3)")
    .all([Buffer.from(Float32Array.from(Array.from({length: 10}, () => Math.random())).buffer)]);

console.log(result);
db.close();

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

function snapshot(db) {
    return ['meta', 'entries', 'current', 'txn'].map(suffix => db.prepare(
        `SELECT * FROM vectors_diskann_${suffix} ORDER BY 1`,
    ).all());
}

// A deterministic, file-backed DiskANN smoke verifies this Node driver's host
// SQLite transaction and reopen behavior, not just extension loading.
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'vectorlite-diskann-'));
const databasePath = path.join(temporary, 'vectors.db');
let disk;
try {
    disk = new sqlite3(databasePath);
    disk.loadExtension(process.env.VECTORLITE_RUST_EXTENSION || vectorlite.vectorlitePath());
    disk.exec('CREATE VIRTUAL TABLE vectors USING vectorlite(embedding float32[2] l2,diskann())');
    disk.exec('CREATE TABLE metadata(rowid INTEGER PRIMARY KEY,label TEXT)');
    const insert = disk.prepare('INSERT INTO vectors(rowid,embedding) VALUES(?,vector_from_json(?))');
    insert.run(1, '[1,0]');
    insert.run(7, '[0,2]');
    const original = snapshot(disk);
    const failed = disk.transaction(() => {
        disk.prepare('INSERT INTO metadata VALUES(42,?)').run('pending');
        insert.run(42, '[4,2]');
        disk.exec("UPDATE vectors SET embedding=vector_from_json('[8,1]') WHERE rowid=1");
        assert.deepEqual(JSON.parse(disk.prepare('SELECT vector_to_json(embedding) AS v FROM vectors WHERE rowid=1').get().v), [8, 1]);
        assert.equal(disk.prepare('SELECT rowid FROM vectors WHERE rowid=42').get().rowid, 42);
        disk.exec('DELETE FROM vectors WHERE rowid=7');
        throw new Error('expected rollback smoke');
    });
    assert.throws(failed, /expected rollback smoke/);
    assert.deepEqual(snapshot(disk), original);
    assert.equal(disk.prepare('SELECT rowid FROM vectors WHERE rowid=42').get(), undefined);
    assert.equal(disk.prepare('SELECT rowid FROM metadata WHERE rowid=42').get(), undefined);
    disk.transaction(() => {
        disk.prepare('INSERT INTO metadata VALUES(42,?)').run('committed');
        insert.run(42, '[4,2]');
    })();
    const searchSql = "SELECT rowid,distance FROM vectors WHERE knn_search(embedding,knn_param(vector_from_json('[1,0]'),3,?))";
    const searchOptions = JSON.stringify({ search_list_size: 64 });
    assert.deepEqual(JSON.parse(disk.prepare('SELECT vector_to_json(embedding) AS v FROM vectors WHERE rowid=42').get().v), [4, 2]);
    assert.deepEqual(disk.prepare(searchSql).all(searchOptions), []);
    publish(disk);
    const neighbors = disk.prepare(searchSql).all(searchOptions);
    assert.deepEqual(neighbors.map(row => row.rowid), [1, 7, 42]);
    assert.deepEqual(neighbors.map(row => row.distance), [0, 5, 13]);
    disk.close();
    disk = new sqlite3(databasePath);
    disk.loadExtension(process.env.VECTORLITE_RUST_EXTENSION || vectorlite.vectorlitePath());
    assert.deepEqual(disk.prepare(searchSql).all(searchOptions), neighbors);
    assert.deepEqual(disk.prepare('SELECT label FROM metadata WHERE rowid=42').get(), { label: 'committed' });
    disk.exec("UPDATE vectors SET embedding=vector_from_json('[2,0]') WHERE rowid=1");
    disk.exec('DELETE FROM vectors WHERE rowid=7');
    const updated = disk.prepare('SELECT vector_to_json(embedding) AS vector FROM vectors WHERE rowid=1').get();
    assert.deepEqual(JSON.parse(updated.vector), [2, 0]);
    assert.equal(disk.prepare('SELECT rowid FROM vectors WHERE rowid=7').get(), undefined);
    assert.equal(disk.prepare("SELECT rowid FROM vectors WHERE knn_search(embedding,knn_param(vector_from_json('[2,0]'),3)) AND rowid=1").get(), undefined);
    publish(disk);
    const nearest = disk.prepare("SELECT rowid,distance FROM vectors WHERE knn_search(embedding,knn_param(vector_from_json('[2,0]'),1))").get();
    assert.deepEqual(nearest, { rowid: 1, distance: 0 });
    console.log('DiskANN Node transaction/reopen smoke passed');
} finally {
    if (disk && disk.open) disk.close();
    // Only remove the uniquely created direct child of the OS temporary root.
    assert.equal(path.dirname(path.resolve(temporary)), path.resolve(os.tmpdir()));
    assert.match(path.basename(temporary), /^vectorlite-diskann-/);
    fs.rmSync(temporary, { recursive: true, force: true });
}
