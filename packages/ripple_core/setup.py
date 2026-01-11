#!/usr/bin/env python3
"""
Setup script for ripple_core package.

Install in development mode:
    pip install -e packages/ripple_core/

Or install normally:
    pip install packages/ripple_core/
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README if it exists
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text()

setup(
    name="ripple_core",
    version="0.1.0",
    description="Core library for ripple detection and analysis in neural data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Pesaran Lab",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
