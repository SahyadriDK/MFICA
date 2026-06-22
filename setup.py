#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:13:04 2026

@author: sk322
"""

from setuptools import setup, find_packages

# Load the requirements.txt file
with open("requirements.txt") as f:
    requirements = [x.strip() for x in f if x.strip() and not x.startswith("#")]
    
# Load the current version
version = {}
with open("mfica_decomp/version.py") as f:
    exec(f.read(), version)
    
# Load the README file
with open("README.md") as f:
    long_descript = f.read()

setup(
    name="mfica_decomp",
    version=version["__version__"],
    packages=find_packages(),
    package_data={
        "mfica_decomp": ["Components/*"],  # includes all .txt files in Folder/
    },
    install_requires=requirements,
    include_package_data=True,
    author="Sahyadri D. Krishna",
    author_email="sk322@st-andrews.ac.uk",
    description="Code for decomposition of quasar and galaxy spectra with MFICA components.",
    long_description=long_descript,
    url="https://github.com/SahyadriDK/MFICA",
    python_requires=">=3.9",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Citation":"https://doi.org/10.1093/mnras/staf1557"
    },
)