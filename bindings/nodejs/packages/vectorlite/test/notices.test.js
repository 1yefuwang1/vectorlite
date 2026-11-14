const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { createHash } = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vectorlite = require('../src/index.js');

// Test the actual platform payload selected by this process, including npm's
// pack file list rather than assuming a manifest whitelist includes notices.
test('native npm payload includes unchanged project and pinned component notices', () => {
    const source = path.dirname(vectorlite.vectorlitePath());
    const packageRoot = path.dirname(source);
    const suffix = { darwin: 'dylib', win32: 'dll', linux: 'so' }[process.platform];
    const binary = `src/vectorlite.${suffix}`;
    const required = [
        'LICENSE', binary, 'src/licenses/LICENSE.txt',
        'src/licenses/diskann-0.60.0/LICENSE.txt',
        'src/licenses/diskann-0.60.0/NOTICE.txt',
        'src/licenses/diskann-0.60.0/PROVENANCE.json',
        'src/licenses/hnswlib/LICENSE.txt', 'src/licenses/highway/LICENSE.txt',
    ];
    const ownLicense = fs.readFileSync(path.join(packageRoot, 'LICENSE'), 'utf8');
    assert.match(ownLicense, /Apache License/);
    assert.equal(fs.readFileSync(path.join(source, 'licenses/LICENSE.txt'), 'utf8'), ownLicense);
    const upstream = path.join(source, 'licenses/diskann-0.60.0');
    assert.match(fs.readFileSync(path.join(upstream, 'LICENSE.txt'), 'utf8'), /Copyright \(c\) Microsoft Corporation\./);
    assert.match(fs.readFileSync(path.join(upstream, 'NOTICE.txt'), 'utf8'), /Cong Fu, Changxu Wang, Deng Cai/);
    const provenance = JSON.parse(fs.readFileSync(path.join(upstream, 'PROVENANCE.json'), 'utf8'));
    assert.equal(provenance.version, '0.60.0');
    assert.equal(provenance.revision, '97a828a500848018d8be28c3ea6d5a585b07362f');
    const expectedBlobs = {
        'LICENSE.txt': 'b2f52a2bad4e27e2d9c68a755abb74cb8943f2fa',
        'NOTICE.txt': 'faf70aa99f51f9bdd98c82b8949ba2826d408a23',
    };
    assert.deepEqual(provenance.source_git_blobs, expectedBlobs);
    for (const [name, expected] of Object.entries(expectedBlobs)) {
        const payload = fs.readFileSync(path.join(upstream, name));
        const digest = createHash('sha1').update(`blob ${payload.length}\0`, 'ascii').update(payload).digest('hex');
        assert.equal(digest, expected, `Bundled upstream notice changed: ${name}`);
    }
    const runtimeRoot = path.join(source, 'licenses/rust-runtime');
    const runtime = JSON.parse(fs.readFileSync(path.join(runtimeRoot, 'MANIFEST.json'), 'utf8'));
    assert.equal(runtime.format_version, 1);
    assert.deepEqual(runtime.unresolved_notice_gaps, []);
    assert.deepEqual(new Set(runtime.target_triples), new Set([
        'x86_64-unknown-linux-gnu', 'x86_64-pc-windows-msvc', 'aarch64-apple-darwin',
    ]));
    required.push('src/licenses/rust-runtime/MANIFEST.json', 'src/licenses/rust-runtime/README.md');
    for (const crate of runtime.packages) {
        assert.equal(typeof crate.declared_spdx, 'string');
        assert.ok(crate.files.length > 0);
        for (const file of crate.files) {
            assert.ok(!file.path.includes('\\\\') && !file.path.split('/').some(part => ['', '.', '..'].includes(part)));
            const payload = fs.readFileSync(path.join(runtimeRoot, file.path));
            assert.equal(createHash('sha256').update(payload).digest('hex'), file.sha256);
            required.push(`src/licenses/rust-runtime/${file.path}`);
        }
    }
    assert.ok(process.env.npm_execpath, 'Run this packaging check through npm test');
    const report = JSON.parse(execFileSync(process.execPath, [
        process.env.npm_execpath, 'pack', '--dry-run', '--json', '--ignore-scripts',
    ], { cwd: packageRoot, encoding: 'utf8' }));
    assert.equal(report.length, 1);
    const packed = new Set(report[0].files.map(file => file.path));
    for (const file of required) {
        assert.ok(packed.has(file), `npm pack omitted ${file}`);
    }
});
