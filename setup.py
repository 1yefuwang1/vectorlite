from distutils.command.build_ext import build_ext
from distutils.command.install_lib import install_lib
from wheel.bdist_wheel import bdist_wheel

import platform
import os
import shutil
from pathlib import Path

from setuptools import Extension, setup 
import cmake
import subprocess
import ninja

VERSION = '0.1.0'
PACKAGE_NAME = 'vectorlite_py'

system = platform.system()
machine = platform.machine()

print(f'Current platfrom: {system}, {machine}')
print(f'cmake bin dir: {cmake.CMAKE_BIN_DIR}. cwd: {os.getcwd()}')
cmake_path = os.path.join(cmake.CMAKE_BIN_DIR, 'cmake')
ctest_path = os.path.join(cmake.CMAKE_BIN_DIR, 'ctest')
ninja_path = os.path.join(ninja.BIN_DIR, 'ninja')
cmake_version = subprocess.run(['cmake', '--version'], check=True)
cmake_version.check_returncode()

class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])

def get_lib_name():
    if system.lower() == 'linux':
        return 'libvectorlite.so'
    if system.lower() == 'darwin':
        return 'libvectorlite.dylib'
    if system.lower() == 'windows':
        return 'libvectorlite.dll'
    raise ValueError(f'Unsupported platform: {system}')

class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        print(f'Building extension for {self.plat_name} {self.compiler.compiler_type}')
        extra_args = []
        if system.lower() == 'windows':
            extra_args = ['-DCMAKE_CXX_COMPILER=cl', '-DCMAKE_C_COMPILER=cl']
        configure = subprocess.run([cmake_path, '--preset', 'release', f'-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_path}', *extra_args])
        configure.check_returncode()

        subprocess.run([cmake_path, '--build', os.path.join('build', 'release'), '-j8'], check=True)
        print(f'Running unit tests')
        subprocess.run([ctest_path, '--test-dir', os.path.join('build', 'release'), '--rerun-failed', '--output-on-failure'], check=True)
        
class CMakeInstallLib(install_lib):
    def run(self):
        install_to = Path(self.build_dir, PACKAGE_NAME)
        print(f'Install lib to {install_to}')
        lib = Path('build', 'release', get_lib_name())
        if not lib.exists():
            raise FileNotFoundError(f'Build output not found: {lib}')

        shutil.copy(lib, install_to)
        super().run()

# modifed from https://github.com/joerick/python-ctypes-package-sample/blob/86d6abd4b0e2950fdeb4f69aaea3f88665af4405/setup.py#L29
class BuildAbiNoneWheel(bdist_wheel):
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = bdist_wheel.get_tag(self)
        return "py3", "none", plat

setup(
    name=PACKAGE_NAME,
    description='Fast vector search for sqlite3',
    long_description='Fast vector search for sqlite3',
    author='Yefu Wang',
    author_email='1yefuwang1@gmail.com',
    url='https://github.com/1yefuwang1/vectorlite',
    license='Apache License, Version 2.0',
    version=VERSION,
    packages=['vectorlite_py'],
    # package_data={"vectorlite_py": ['*.so', '*.dylib', '*.dll']},
    install_requires=[],
    ext_modules=[CMakeExtension('vectorlite')],
    cmdclass={
        'build_ext': CMakeBuild,
        'install_lib': CMakeInstallLib,
        'bdist_wheel': BuildAbiNoneWheel
    }
)