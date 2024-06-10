from setuptools import setup
import platform

VERSION = '0.1.0'

system = platform.system()
machine = platform.machine()

print(system, machine)

setup(
    name="vectorlite_py",
    description="Fast vector search for sqlite3",
    long_description="Fast vector search for sqlite3",
    author="Yefu Wang",
    url="https://github.com/1yefuwang1/vectorlite",
    license="Apache License, Version 2.0",
    version=VERSION,
    packages=['vectorlite_py'],
    package_data={"vectorlite_py": ['*.so', '*.dylib', '*.dll']},
    install_requires=[],
    has_ext_modules=lambda: True,
)