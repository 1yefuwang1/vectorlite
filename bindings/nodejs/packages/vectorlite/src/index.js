const os = require('os');

const supportedPlatformsAndArchs = {
    'darwin-x64': '@1yefuwang1/vectorlite-darwin-x64',
    'darwin-arm64': '@1yefuwang1/vectorlite-darwin-arm64',
    'linux-x64': '@1yefuwang1/vectorlite-linux-x64',
    'win32-x64': '@1yefuwang1/vectorlite-win32-x64',
};

const platformAndArch = `${os.platform()}-${os.arch()}`;

let vectorlitePathCache = undefined;

// Returns path to the vectorlite shared library
function vectorlitePath() {
    if (vectorlitePathCache) {
        return vectorlitePathCache;
    }
    const packageName = supportedPlatformsAndArchs[platformAndArch];
    if (!packageName) {
        throw new Error(`Platform ${platformAndArch} is not supported`);
    }
    const package = require(packageName);
    vectorlitePathCache = package.vectorlitePath();
    return vectorlitePathCache;
}

exports.vectorlitePath = vectorlitePath;