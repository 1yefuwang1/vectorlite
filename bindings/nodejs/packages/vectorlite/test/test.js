const sqlite3 = require('better-sqlite3');
const vectorlite = require('../src/index.js');

const db = new sqlite3(':memory:');
db.loadExtension('/home/yefu/sqlite-hnsw/build/dev/vectorlite.so');

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
result = db.prepare('select rowid, vector_distance(vec, ?, \'l2\') from test where rowid in (0,1,2,3)')
    .all([Buffer.from(Float32Array.from(Array.from({length: 10}, () => Math.random())).buffer)]);

console.log(result);