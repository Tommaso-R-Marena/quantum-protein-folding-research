from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantum-protein-folding",
    version="0.1.0",
    author="Tommaso R. Marena",
    author_email="marena@cua.edu",
    description="Quantum algorithms for lattice protein folding using VQE and QAOA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tommaso-R-Marena/quantum-protein-folding-research",
    project_urls={
        "Bug Tracker": "https://github.com/Tommaso-R-Marena/quantum-protein-folding-research/issues",
        "Documentation": "https://github.com/Tommaso-R-Marena/quantum-protein-folding-research/blob/main/README.md",
        "Source Code": "https://github.com/Tommaso-R-Marena/quantum-protein-folding-research",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "qiskit>=1.0.0",
        "qiskit-aer>=0.13.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "biopython>=1.81",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "jupyter>=1.0.0",
            "ipython>=8.12.0",
        ],
        "viz": [
            "seaborn>=0.12.0",
            "pandas>=2.0.0",
        ],
    },
)
