#!/usr/bin/env python3
"""
Setup script for DISCOS (DISrete COllinear Skeletonization)
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Dependencies are now managed in pyproject.toml
requirements = [
    "numpy>=1.20.0",
    "trimesh>=3.15.0",
    "networkx>=2.6",
    "matplotlib>=3.5.0",
    "plotly>=5.0.0",
]

setup(
    name="discos",
    version="0.1.0",
    author="Jordan Fox",
    author_email="jmrfox@example.com",
    description="DISrete COllinear Skeletonization for 3D mesh analysis and neuronal morphology processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jmrfox/discos",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Biology",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        # Note: CLI disabled since it depends on modules moved to dev_storage
        # "console_scripts": [
        #     "discos=discos.cli:main",
        # ],
    },
)
