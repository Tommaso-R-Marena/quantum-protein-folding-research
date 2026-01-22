"""Setup configuration for quantum_protein_folding package."""

from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quantum_protein_folding",
    version="0.1.0",
    author="Tommaso R. Marena",
    author_email="tommaso.marena@example.com",
    description="NISQ-compatible quantum algorithms for protein structure prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tommaso-R-Marena/quantum-protein-folding-research",
    project_urls={
        "Bug Tracker": "https://github.com/Tommaso-R-Marena/quantum-protein-folding-research/issues",
        "Documentation": "https://quantum-protein-folding-research.readthedocs.io",
        "Source Code": "https://github.com/Tommaso-R-Marena/quantum-protein-folding-research",
    },
    package_dir={"":  "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.2.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=6.2.0",
            "sphinx-rtd-theme>=1.2.0",
            "nbsphinx>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qpf-benchmark=quantum_protein_folding.cli:benchmark",
            "qpf-fold=quantum_protein_folding.cli:fold",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
