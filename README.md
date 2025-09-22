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

(*Adaptar à situação*)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Available-blue.svg)](https://www.docker.com/)
[![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-017CEE.svg?logo=Apache%20Airflow&logoColor=white)](https://airflow.apache.org)
[![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![NLTK](https://img.shields.io/badge/NLTK-3776AB?logo=python&logoColor=fff)](https://www.nltk.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](https://pandas.pydata.org)
[![Website](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://shields.io/)

## Overview

HeCoReS-DR a toolkit and reproducible code to study hetnet-based connectivity search for drug repositioning, addressing data sparsity and explainability. Includes dataset processing, knowledge-graph construction, recommender baselines, KG-enhanced models, and evaluation scripts.


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

- **Multi-format Data Processing**: Converts XML, CSV, and TXT files to standardized JSON format
- **Named Entity Recognition (NER)**: Extracts biomedical entities like drugs, diseases, and chemical compounds
- **Knowledge Graph Construction**: Creates a structured graph in Neo4j representing relationships between entities
- **Workflow Orchestration**: Uses Apache Airflow to manage and schedule the complete data pipeline
- **Containerized Deployment**: Packaged with Docker for easy deployment and environment consistency
- **Ontology Integration**: Leverages biomedical ontologies like ChEBI, Disease Ontology, and Orphanet

## Technologies

- **Python**: Core programming language
- **Apache Airflow**: Workflow orchestration and scheduling
- **Neo4j**: Graph database for knowledge representation
- **Docker**: Containerization for deployment
- **MER (Minimal Entity Recognition)**: Biomedical entity extraction
- **NLTK**: Natural Language Processing for text preprocessing
- **Pandas**: Data manipulation and transformation
- **lxml**: XML processing and XSLT transformations

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

If you use MedJsonify in your research, please cite it as follows:

```
@conference{Pereira2025,
    author = Carolina Pereira, Matilde Pato and Nuno Datia,
    booktitle = 12th ACM Celebration of Women in Computing: womENcourage™ 2025,
    title = Knowledge Graphs as Educational Tools in Biomedical Education,
    year = 2025
}
```

---

<div align="center">
  <p>Developed by Carolina Pereira as part of the Workflow System for Data Integration's Project.</p>
</div>
