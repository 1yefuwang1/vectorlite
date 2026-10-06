const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const nodejsRoot = path.resolve(__dirname, '../../..');
const packagesRoot = path.join(nodejsRoot, 'packages');
const mainManifest = require('../package.json');
const resolverPath = path.join(__dirname, '../src/index.js');
const resolverSource = fs.readFileSync(resolverPath, 'utf8');
const supportedPlatforms = [
    { platform: 'linux', arch: 'x64', packageName: 'vectorlite-linux-x64' },
    { platform: 'win32', arch: 'x64', packageName: 'vectorlite-win32-x64' },
    { platform: 'darwin', arch: 'arm64', packageName: 'vectorlite-darwin-arm64' },
];

function loadResolver(platform, arch) {
    const exports = {};
    const packageRequests = [];
    const libraryPath = `/mock/${platform}-${arch}/vectorlite`;
    let pathCalls = 0;
    vm.runInNewContext(resolverSource, {
        exports,
        require(name) {
            if (name === 'os') {
                return { platform: () => platform, arch: () => arch };
            }
            packageRequests.push(name);
            return {
                vectorlitePath() {
                    pathCalls += 1;
                    return libraryPath;
                },
            };
        },
    }, { filename: resolverPath });
    return { exports, packageRequests, libraryPath, pathCalls: () => pathCalls };
}

for (const { platform, arch, packageName } of supportedPlatforms) {
    test(`resolves and caches the ${platform}-${arch} platform package`, () => {
        const resolver = loadResolver(platform, arch);
        assert.equal(resolver.exports.vectorlitePath(), resolver.libraryPath);
        assert.equal(resolver.exports.vectorlitePath(), resolver.libraryPath);
        assert.deepEqual(resolver.packageRequests, [`@1yefuwang1/${packageName}`]);
        assert.equal(resolver.pathCalls(), 1);
    });
}

for (const [platform, arch] of [
    ['darwin', 'x64'],
    ['linux', 'arm64'],
    ['win32', 'arm64'],
]) {
    test(`rejects ${platform}-${arch} before requiring a platform package`, () => {
        const resolver = loadResolver(platform, arch);
        assert.throws(() => resolver.exports.vectorlitePath(), {
            message: `Platform ${platform}-${arch} is not supported`,
        });
        assert.deepEqual(resolver.packageRequests, []);
        assert.equal(resolver.pathCalls(), 0);
    });
}

test('optional dependencies contain exactly the supported platform packages', () => {
    const expected = Object.fromEntries(supportedPlatforms.map(({ packageName }) => [
        `@1yefuwang1/${packageName}`, mainManifest.version,
    ]));
    assert.deepEqual(mainManifest.optionalDependencies, expected);
});

test('publish workspaces contain only the main and supported platform packages', () => {
    const workspaceManifest = JSON.parse(fs.readFileSync(
        path.join(nodejsRoot, 'package.json.tpl'), 'utf8',
    ));
    const expected = [
        'packages/vectorlite',
        ...supportedPlatforms.map(({ packageName }) => `packages/${packageName}`),
    ];
    assert.deepEqual(workspaceManifest.workspaces.slice().sort(), expected.sort());
    assert.equal(fs.existsSync(path.join(packagesRoot, 'vectorlite-darwin-x64')), false);
});

test('supported platform package manifests retain their platform constraints', () => {
    for (const { platform, arch, packageName } of supportedPlatforms) {
        const manifest = JSON.parse(fs.readFileSync(
            path.join(packagesRoot, packageName, 'package.json'), 'utf8',
        ));
        assert.equal(manifest.name, `@1yefuwang1/${packageName}`);
        assert.equal(manifest.version, mainManifest.version);
        assert.deepEqual(manifest.os, [platform]);
        assert.deepEqual(manifest.cpu, [arch]);
    }
});
