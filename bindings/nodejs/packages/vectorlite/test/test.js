const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const sqlite3 = require('better-sqlite3');
const vectorlite = require('../src/index.js');

const db = new sqlite3(':memory:');
db.loadExtension(vectorlite.vectorlitePath());

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

// A deterministic, file-backed DiskANN smoke verifies this Node driver's host
// SQLite transaction and reopen behavior, not just extension loading.
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'vectorlite-diskann-'));
const databasePath = path.join(temporary, 'vectors.db');
let disk;
try {
    disk = new sqlite3(databasePath);
    disk.loadExtension(vectorlite.vectorlitePath());
    disk.exec('CREATE VIRTUAL TABLE vectors USING vectorlite(embedding float32[2] l2,diskann())');
    disk.exec('CREATE TABLE metadata(rowid INTEGER PRIMARY KEY,label TEXT)');
    const insert = disk.prepare('INSERT INTO vectors(rowid,embedding) VALUES(?,vector_from_json(?))');
    insert.run(1, '[1,0]');
    insert.run(7, '[0,2]');
    const originalGraph = disk.prepare('SELECT node_id,public_rowid,state,hex(vector),hex(neighbors) FROM vectors_diskann_nodes ORDER BY node_id').all();
    const failed = disk.transaction(() => {
        disk.prepare('INSERT INTO metadata VALUES(42,?)').run('pending');
        insert.run(42, '[4,2]');
        disk.exec("UPDATE vectors SET embedding=vector_from_json('[8,1]') WHERE rowid=1");
        disk.exec('DELETE FROM vectors WHERE rowid=7');
        throw new Error('expected rollback smoke');
    });
    assert.throws(failed, /expected rollback smoke/);
    assert.deepEqual(disk.prepare('SELECT node_id,public_rowid,state,hex(vector),hex(neighbors) FROM vectors_diskann_nodes ORDER BY node_id').all(), originalGraph);
    assert.equal(disk.prepare('SELECT rowid FROM vectors WHERE rowid=42').get(), undefined);
    assert.equal(disk.prepare('SELECT rowid FROM metadata WHERE rowid=42').get(), undefined);
    disk.transaction(() => {
        disk.prepare('INSERT INTO metadata VALUES(42,?)').run('committed');
        insert.run(42, '[4,2]');
    })();
    const searchSql = "SELECT rowid,distance FROM vectors WHERE knn_search(embedding,knn_param(vector_from_json('[1,0]'),3,?))";
    const searchOptions = JSON.stringify({ search_list_size: 64 });
    const neighbors = disk.prepare(searchSql).all(searchOptions);
    assert.deepEqual(neighbors.map(row => row.rowid), [1, 7, 42]);
    assert.deepEqual(neighbors.map(row => row.distance), [0, 5, 13]);
    disk.close();
    disk = new sqlite3(databasePath);
    disk.loadExtension(vectorlite.vectorlitePath());
    assert.deepEqual(disk.prepare(searchSql).all(searchOptions), neighbors);
    assert.deepEqual(disk.prepare('SELECT label FROM metadata WHERE rowid=42').get(), { label: 'committed' });
    disk.exec("UPDATE vectors SET embedding=vector_from_json('[2,0]') WHERE rowid=1");
    disk.exec('DELETE FROM vectors WHERE rowid=7');
    const updated = disk.prepare('SELECT vector_to_json(embedding) AS vector FROM vectors WHERE rowid=1').get();
    assert.deepEqual(JSON.parse(updated.vector), [2, 0]);
    assert.equal(disk.prepare('SELECT rowid FROM vectors WHERE rowid=7').get(), undefined);
    console.log('DiskANN Node transaction/reopen smoke passed');
} finally {
    if (disk && disk.open) disk.close();
    // Only remove the uniquely created direct child of the OS temporary root.
    assert.equal(path.dirname(path.resolve(temporary)), path.resolve(os.tmpdir()));
    assert.match(path.basename(temporary), /^vectorlite-diskann-/);
    fs.rmSync(temporary, { recursive: true, force: true });
}
