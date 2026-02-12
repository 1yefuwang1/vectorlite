declare module 'vectorlite' {
    /**
     * Returns the path to the vectorlite shared library for the current platform
     */
    export function vectorlitePath(): string;
  
    /**
     * Supported platforms and architectures mapping
     */
    export const supportedPlatformsAndArchs: {
      'darwin-x64': string;
      'darwin-arm64': string;
      'linux-x64': string;
      'win32-x64': string;
    };
  }
  