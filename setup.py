"""
HeCoReS-DR: Hetnet Connectivity Search for Drug Repositioning
Setup configuration for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="hecores-dr",
    version="1.0.0",
    author="Donato Aveiro",
    author_email="dgoncalo.ba@gmail.com",
    description="Hetnet Connectivity Search in Recommender Systems for Drug Repositioning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matpato/hecores-dr",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    python_requires=">=3.8,<3.12",
    
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "nltk>=3.8",
        "spacy>=3.5.0",
        "hunflair>=2.0.0",
        "networkx>=3.0",
        "py2neo>=2021.2.3",
        "neo4j>=5.0.0",
        "lxml>=4.9.0",
        "scikit-surprise>=1.1.3",
        "implicit>=0.7.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "tqdm>=4.65.0",
        "requests>=2.28.0",
        "python-dotenv>=1.0.0",
    ],
    
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "deep-learning": [
            "torch>=2.0.0",
            "tensorflow>=2.12.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "hecores-train=training_testing:main",
            "hecores-preprocess=preprocessing.preprocessing:main",
            "hecores-evaluate=evaluation.validation:main",
        ],
    },
    
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.csv"],
    },
    
    keywords=[
        "drug repositioning",
        "knowledge graph",
        "recommender systems",
        "bioinformatics",
        "hetnet connectivity search",
        "machine learning",
    ],
    
    project_urls={
        "Bug Reports": "https://github.com/matpato/hecores-dr/issues",
        "Source": "https://github.com/matpato/hecores-dr",
    },
)
