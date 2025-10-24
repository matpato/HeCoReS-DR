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
[![Docker](https://img.shields.io/badge/Docker-Available-blue.svg)](https://www.docker.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](https://pandas.pydata.org)
[![Website](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://shields.io/)

## Overview

HeCoReS-DR: a toolkit and reproducible code to study hetnet-based connectivity search for drug repositioning, addressing data sparsity and explainability. Includes dataset processing, knowledge-graph construction, recommender baselines, KG-enhanced models, and evaluation scripts.

<div align="center">
  <img src="./img/airflow_orchestration.png" width="600" alt="Airflow Orchestration Diagram"/>
</div>

## Table of Contents

1. [Project Structure](#project-structure)
2. [Key Features](#key-features)
3. [Technologies](#technologies)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Pipeline Workflow](#pipeline-workflow)
7. [License](#license)
8. [How to Cite](#how-to-cite)

## Project Structure

The project is organized into several key modules, each responsible for specific aspects of the data processing pipeline:

```
hecores-dr/
├── README.md
├── LICENSE      # e.g., MIT
├── data/        # small example datasets (or scripts to download the data)
├── notebooks/   # Google Colab-compatible notebooks
├── src/
│   ├── preprocessing/
│   ├── kg/
│   ├── models/
│   └── evaluation/
├── experiments/ # configs and results
└── requirements.txt

```

## Key Features

- **Entity Extraction:** HunFlair2 framework applied to DrugBank Indication fields
- **Knowledge Graph Construction:** Integration of multiple biomedical databases
- **Metapath Analysis:** HCS applied to knowledge graph for identification of biologically plausible associations
- **Similarity Computation:** Jaccard coefficient calculation across feature vectors
- **Matrix Construction:** Sparse binary matrices with standardized identifiers

## Quality Assurance
- **Identifier Validation:** All DrugBank and MeSH IDs verified against current databases
- **Consistency Checks:** Matrix dimensions and sparsity levels validated

## Research Applications
- **Drug Repositioning:** Identify new therapeutic applications for existing drugs
- **Recommender Systems:** Develop and evaluate drug-disease prediction algorithms
- **Network Analysis:** Study connectivity patterns in biomedical knowledge graphs
- **Sparsity Mitigation:** Compare augmentation strategies in sparse datasets

## Analyses
- **Baseline Comparison:** Use HCS_0 as ground truth for evaluation
- **Augmentation Evaluation:** Compare performance between HCS_1 and random controls
- **Similarity Integration:** Leverage drug/disease similarities for enhanced predictions
- **Cross-validation:** Employ multiple random controls for robust statistical analysis

## Technologies

- **Python**: Core programming language
- **Neo4j**: Graph database for knowledge representation
- **Docker**: Containerization for deployment
- **NLTK**: Natural Language Processing for text preprocessing
- **pandas**: Data manipulation and transformation
- **lxml**: XML processing and XSLT transformations
- **HunFlair2**: Named entity recognition and linking

## Installation

### Prerequisites

- Docker and Docker Compose
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/MedJsonify.git
```

2. Navigate to the project directory:
```bash
cd MedJsonify
```

3. Grant execution permissions to the Docker script:
```bash
chmod +x docker.sh
```

4. Ensure MER is installed and grant execution permissions to the entity extraction script:
```bash
chmod +x get_entities.sh
```

## Usage

1. Build and run the Docker containers:
```bash
./docker.sh
```

2. Access the Apache Airflow web interface:
```bash
http://localhost:8080
```

3. Log in with the default credentials:
   - Username: admin
   - Password: admin

4. From the Airflow UI, activate and trigger the desired DAG:
   - `converter_dag`: Only converts files to JSON
   - `ner_dag`: Processes JSON files with NER
   - `medjsonify_dag`: Runs the complete pipeline

5. After processing, the Neo4j database will contain the knowledge graph. Access the Neo4j Browser:
```bash
http://localhost:7474
```

## Pipeline Workflow

The complete data processing pipeline consists of the following steps:

1. **Data Acquisition**:
   - Download files from configured URLs
   - Extract ZIP archives
   - Extract specific files based on type

2. **Data Conversion**:
   - Convert XML files using either Python-based parsing or XSLT
   - Convert CSV files with appropriate delimiters and headers
   - Convert TXT files with specified delimiters
   - Standardize to JSON format

3. **Named Entity Recognition**:
   - Download and update biomedical vocabularies and ontologies
   - Preprocess JSON text fields for NER
   - Extract drug and disease entities
   - Normalize entity identifiers to standard ontologies

4. **Knowledge Graph Construction**:
   - Create nodes for drugs, diseases, administration routes, and approval years
   - Establish relationships between entities (TREATS, CONTRAINDICATED_FOR, etc.)
   - Apply constraints to ensure data integrity

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## How to Cite

...

```

---

<div align="center">
  <p>Developed by Donato Aveiro as part of the Hetnet Connectivity Search in Recommender Systems for Drug Repositioning: Addressing Data Sparsity and Explainability master's thesis.</p>
</div>
