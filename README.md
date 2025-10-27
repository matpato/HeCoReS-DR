<div id="top"></div>

<!-- INSTITUTIONS LOGO -->
<div style="display: flex; align-items: center; justify-content: space-around; flex-wrap: wrap; margin: 20px 0;">
 <a href="https://ciencias.ulisboa.pt" target="_blank">
<img src="./img/ciencias_ul_azul_h.png" alt="FCUL logo" style="width: 150px; height: auto;">
</a>
<a href="https://lasige.pt" target="_blank">
<img src="./img/lasige_h_logo.png" alt="Lasige logo" style="width: 150px; height: auto;">
</a>
<a href="https://www.ipl.pt" target="_blank">
<img src="./img/IPL Horizontal MainPng.png" alt="IPL logo" style="width: 200px; height: auto;">
</a>
<a href="https://isel.pt" target="_blank">
<img src="./img/01_ISEL-Logotipo-RGB_Horizontal.png" alt="ISEL logo" style="width: 200px; height: auto;">
</a>
</div>
<div style="flex: 3; text-align: left; padding-left: 20px;">
<h3>Hetnet Connectivity Search in Recommender Systems for Drug Repositioning (HeCoReS-DR).
</h3>
</div>


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](https://pandas.pydata.org)

## Overview

HeCoReS-DR: reproducible code to study hetnet connectivity search (HCS) for drug repositioning, addressing data sparsity and explainability. Includes dataset processing, knowledge graph (KG) construction, collaborative filtering models, and evaluation scripts.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Key Features](#key-features)
3. [Research Applications](#research-applications)
4. [Analyses](#analyses)
5. [Prerequisites](#prerequisites)
6. [License](#license)
7. [How to Cite](#how-to-cite)

## Project Structure

The project is organized into several files, each responsible for specific aspects of the data processing pipeline:

- `utils.py`: common helper functions for the project;
- `drugbank_processing.py`: downloads all relevant data from DrugBank's full database and stores it in specific JSON files;
- `mesh.py`, `go_target_processing.py`, `ctd_processing.py`, `condition_associations.py`: process (in this order) MeSH drug categories, Gene Ontology (GO) terms and their association to protein targets, diseases in the Comparative Toxicogenomics Database (CTD), and associations between diseases and protein targets;
- `extract_conditions.py`: applies the HunFlair2 framework to extract diseases from each drug's textual descriptions in DrugBank;
- `atc_codes_processing.py`: processes Anatomical Therapeutic Chemical (ATC) data and their associations to drugs;
- `processing_for_neo4j.py`: prepares all data for the construction of a knowledge graph (KG) in Neo4j;
- `neo4j.py`: constructs the KG, including functions to retrieve specific nodes and edges;
- `hpy.py`: transforms the KG into a hetnet structure compatible with the hetnetpy package, and generates pruned versions for ablation studies;
- `hmp.py`: constructs metapaths and applies Hetnet Connectivity Search (HCS) to the original KG and its pruned versions according to the hetmatpy package;
- `sims.py`: generates the datasets (HCS_0, HCS_1, HCS_RANDOM(1,2,3)) and the drug-drug and disease-disease similarity matrices;
- `clinical_trials.py`: extracts clinical trial studies from ClinicalTrials.gov;
- `AlternatingLeastSquares.py`, `BayesianPairwiseRanking.py`, `LogisticMatrixFactorization.py`: classes for matrix factorization (MF) models;
- `models.py`: implementations of MF models + Bounded Nuclear Norm Regularization (BNNR);
- `datasets.py`: dataset-specific functions;
- `preprocessing.py`: preprocesses data for training and testing;
- `training_testing.py`: training and testing pipeline;
- `validation.py`: ranking accuracy functions;
- `utils_stanscofi.py`: helper functions from the stanscofi package;
- `scofi.py`: applies the final step of the pipeline, generating all the results.
  
## Key Features

- **Entity Extraction:** HunFlair2 framework applied to DrugBank Indication fields
- **Knowledge Graph Construction:** Integration of multiple biomedical databases
- **Metapath Analysis:** HCS applied to a knowledge graph for identification of biologically plausible associations
- **Similarity Computation:** Jaccard coefficient calculation across feature vectors
- **Matrix Construction:** Sparse binary matrices with standardized identifiers

## Research Applications
- **Drug Repositioning:** Identifying new therapeutic applications for existing drugs
- **Recommender Systems:** Developing and evaluating drug-disease prediction algorithms
- **Network Analysis:** Studying connectivity patterns in biomedical knowledge graphs
- **Sparsity Mitigation:** Comparing augmentation strategies in sparse datasets

## Analyses
- **Baseline Comparison:** HCS_0 as ground truth for evaluation
- **Augmentation Evaluation:** Compare performance between HCS_1 and random controls
- **Similarity Integration:** Leverage drug/disease similarities for enhanced predictions
- **Ablation Studies:** Compare performance under reduced KG sparsity
- **Clinical Trials:** Use clinical trial studies for hypothetical drug-disease associations

## Prerequisites

- pronto
- flair
- torch
- neo4j
- hetnetpy
- hetmatpy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## How to Cite

...

---

<p align="center">
Developed by <b>Donato Aveiro</b> as part of the <i>Hetnet Connectivity Search in Recommender Systems for Drug Repositioning: Addressing Data Sparsity and Explainability</i> master's thesis.
</p>
