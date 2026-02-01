"""
Stub setup.py for backward compatibility.

This project now uses scikit-build-core and pyproject.toml for configuration.
All metadata and build configuration is in pyproject.toml.

For modern installation, use:
    pip install .

For development installation:
    pip install -e .

For building wheels:
    pip wheel .
    # or use cibuildwheel for multi-platform wheels

Note: Direct `python setup.py` commands are deprecated in favor of PEP 517 builds.
"""

# This allows `pip install -e .` and other pip commands to work correctly
# by delegating to the PEP 517 build backend (scikit-build-core)
from setuptools import setup

setup()
