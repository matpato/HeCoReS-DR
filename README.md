<div style="display: flex; align-items: center; justify-content: space-around; flex-wrap: wrap; margin: 20px 0;">
  <a href="https://ciencias.ulisboa.pt" target="_blank">
    <img src="/img/ciencias_ul_azul_h.png" alt="FCUL logo" style="width: 300px; height: auto;">
  </a>
  <a href="https://lasige.pt" target="_blank">
    <img src="/img/lasige_h_logo.png" alt="Lasige logo" style="width: 300px; height: auto;">
  </a>
  <a href="https://www.ipl.pt" target="_blank">
    <img src="./img/IPL Horizontal MainPng.png" alt="IPL logo" style="width: 400px; height: auto;">
  </a>
  <a href="https://isel.pt" target="_blank">
    <img src="./img/01_ISEL-Logotipo-RGB_Horizontal.png" alt="ISEL logo" style="width: 400px; height: auto;">
  </a>
</div>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Available-blue.svg)](https://www.docker.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](https://pandas.pydata.org)
[![Scipy](https://img.shields.io/badge/-Scipy-blue?style=flat&logo=Scipy&logoColor=white)](https://scipy.org)
[![Scikit Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/#)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Website](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://shields.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

# HeCoReS-DR: Hetnet Connectivity Search for Drug Repositioning

**Version:** 1.0 (2025-09-19)

## Overview

**HeCoReS-DR** is a comprehensive toolkit and reproducible codebase for studying *hetnet-based connectivity search in drug repositioning*. This project addresses critical challenges in computational drug repurposing, including data sparsity and explainability, through knowledge graph-based approaches. The toolkit provides end-to-end capabilities from dataset processing and knowledge graph construction to recommender system implementation and evaluation.


## Table of Contents
1. [Project Structure](#project-structure)
2. [Key Features](#key-features)
3. [Dataset Statistics](#dataset-statistics)
4. [Technologies](#technologies)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Pipeline Workflow](#pipeline-workflow)
8. [Research Applications](#research-applications)
9. [Quality Assurance](#quality-assurance)
10. [License](#license)
10. [Authors & Affiliations](#authors--affiliations)
11. [How to Cite](#how-to-cite)

## Project Structure

The project is organized into modular components for reproducibility and extensibility:
```
hecores-dr/
├── README.md
├── CONTRIBUTING.md
├── LICENSE                      # MIT License
├── data/                        # Datasets and data processing scripts
│   ├── raw/                    # Raw source data
│   └── processed/              # Processed matrices and similarity files
├── src/
│   ├── models.py
│   ├── training_testing.py
│   ├── preprocessing/          # Data extraction and transformation
│   │   ├── datasets.py
│   │   ├── preprocessing.py
│   │   ├── ...
│   │   └── mesh.py
│   ├── kg/                     # Knowledge graph construction
│   │   ├── condition_associations.py
│   │   ├── neo4j.py
│   │   └── processing_for_neo4j.py
│   ├── models/                 # Recommender system implementations
│   │   ├── baselines/         # Traditional collaborative filtering
│   │   ├── kg_enhanced/       # KG-enhanced models
│   │   └── similarity_based/  # Similarity-based approaches
│   └── evaluation/             # Metrics and validation
│       ├── scofi.py
│       └── validation.py      
├── results/                # Output logs, metrics, and visualizations
├── docker/                     # Docker configurations
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt            # Python dependencies
└── setup.py                    # Package installation script
```

## Key Features

### Data Processing & Knowledge Graph Construction
- **Entity Extraction:** HunFlair2 framework applied to DrugBank Indication fields for automated biomedical entity recognition
- **Knowledge Graph Construction:** Integration of multiple biomedical databases (DrugBank, MeSH, MEDIC ontology) into a unified heterogeneous network
- **Metapath Analysis:** Hetnet Connectivity Search (HCS) algorithm for discovering biologically plausible drug-disease associations
- **Similarity Computation:** Jaccard coefficient calculation across multiple feature dimensions (pharmacological, therapeutic, chemical, protein targets)

### Recommender Systems
- **Baseline Models:** Traditional collaborative filtering approaches for drug repositioning
- **KG-Enhanced Models:** Knowledge graph-augmented recommender systems leveraging hetnet connectivity
- **Similarity-Based Models:** Drug and disease similarity integration for enhanced predictions
- **Ensemble Methods:** Combined approaches for improved performance

### Evaluation & Analysis
- **Comprehensive Metrics:** Precision, recall, F1-score, NDCG, and domain-specific measures
- **Cross-Validation:** Robust evaluation with multiple random controls
- **Statistical Testing:** Significance testing for model comparisons
- **Explainability:** Metapath-based explanations for predictions

## Dataset Statistics

| Metric | Count |
|--------|-------|
| **Drugs** | 1,179 |
| **Diseases** | 722 |
| **Total Possible Associations** | 851,238 |
| **Baseline Sparsity (hcs_0.csv)** | 99.47% |
| **Augmented Sparsity (hcs_1.csv)** | 96.18% |

**Identifiers:**
- Drugs: DrugBank IDs (columns)
- Diseases: MeSH IDs (rows)

### Dataset Files

#### Core Association Matrices
- **`hcs_0.csv`**: Baseline drug-disease associations from DrugBank (binary matrix)
- **`hcs_1.csv`**: HCS-augmented associations with biologically plausible connections
- **`hcs_random1-3.csv`**: Random control matrices for comparative analysis

#### Similarity Matrices
- **`drug_sims.csv`**: Drug-drug Jaccard similarity (pharmacological, therapeutic, chemical features)
- **`disease_sims.csv`**: Disease-disease Jaccard similarity (protein targets, MEDIC Slim terms)

## Technologies

### Core Technologies
- **Python 3.8+**: Core programming language
- **Neo4j**: Graph database for knowledge representation and HCS queries
- **Docker & Docker Compose**: Containerization for reproducible deployment

### Python Libraries
- **Data Processing**: pandas, NumPy, scikit-learn
- **NLP & Entity Recognition**: NLTK, HunFlair2
- **Graph Processing**: NetworkX, py2neo
- **XML Processing**: lxml, ElementTree
- **Machine Learning**: scikit-surprise, implicit, TensorFlow/PyTorch
- **Evaluation**: scikit-learn, SciPy
- **Visualization**: matplotlib, seaborn, Plotly

## Installation

### Prerequisites
- Docker Engine 20.10+ ([Install Docker](https://docs.docker.com/engine/install/))
- Docker Compose 2.0+ ([Install Docker Compose](https://docs.docker.com/compose/install/))- Python 3.8 or higher
- Git
- At least 8GB RAM (16GB recommended for large-scale experiments)
- At least 10GB free disk space

### Option 1: Docker Installation (Recommended)
```bash
# Clone the repository
git clone https://github.com/matpato/hecores-dr.git
cd hecores-dr

# Build and start containers
docker-compose up -d

# Stop containers
docker-compose down

# Stop and Remove Volumes (WARNING: Deletes data)
docker-compose down -v

# Restart Services
docker-compose restart
```

### Option 2: Local Installation
```bash
# Clone the repository
git clone https://github.com/matpato/hecores-dr.git
cd hecores-dr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Access Neo4j Browser
Open your browser and navigate to: http://localhost:7474

- **Username:** neo4j
- **Password:** hecores_password

Port Conflicts
If ports 7474 or 7687 are in use:
```bash
# Check what's using the port
lsof -i :7474
lsof -i :7687
```

## Usage

### Quick Start
```python
from src.preprocessing import load_datasets
from src.models.baselines import CollaborativeFiltering
from src.evaluation.validation import compute_metrics

# Load datasets
hcs_0 = load_datasets('hcs_0.csv')
hcs_1 = load_datasets('hcs_1.csv')

# Train baseline model
model = CollaborativeFiltering()
model.fit(hcs_0)

# Evaluate
results = compute_metrics(model, test_data)
print(results)
```

## Pipeline Workflow

The HeCoReS-DR pipeline consists of the following stages:

1. **Data Acquisition**: Download and parse DrugBank, MeSH, and other biomedical databases
2. **Entity Extraction**: Apply HunFlair2 to extract and normalize drug-disease associations
3. **Knowledge Graph Construction**: Build heterogeneous network in Neo4j
4. **HCS Application**: Execute metapath-based connectivity search
5. **Similarity Computation**: Calculate drug and disease similarity matrices
6. **Model Training**: Train recommender system models with various configurations
7. **Evaluation**: Assess model performance using comprehensive metrics
8. **Analysis**: Generate visualizations and statistical comparisons

## Research Applications

### Drug Repositioning
Identify new therapeutic applications for existing drugs by leveraging knowledge graph connectivity patterns and similarity-based reasoning.

### Recommender Systems
Develop and benchmark drug-disease prediction algorithms that address data sparsity through knowledge graph augmentation.

### Network Analysis
Study connectivity patterns in biomedical knowledge graphs to understand drug-disease relationship mechanisms.

### Sparsity Mitigation
Compare different strategies for handling sparse data in drug repositioning, including random baselines vs. biologically-informed augmentation.

### Explainability Research
Generate interpretable predictions using metapath-based explanations derived from knowledge graph traversals.

## Recommended Analyses

- **Baseline Comparison**: Use `hcs_0.csv` as ground truth for evaluating model performance
- **Augmentation Evaluation**: Compare performance gains between HCS-augmented (`hcs_1.csv`) and random control matrices
- **Similarity Integration**: Assess the contribution of drug/disease similarities to prediction accuracy
- **Cross-Validation**: Employ multiple random controls for robust statistical validation
- **Metapath Analysis**: Investigate which biological pathways contribute most to successful predictions

## Quality Assurance

### Validation Procedures
- **Identifier Validation**: All DrugBank and MeSH IDs verified against current database versions
- **Consistency Checks**: Matrix dimensions, sparsity levels, and data types validated
- **Biological Plausibility**: HCS-discovered associations filtered using domain knowledge
- **Reproducibility**: Fixed random seeds and version-controlled dependencies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors & Affiliations

- **Donato Aveiro** - fc46269@alunos.fc.ul.pt (dgoncalo.ba@gmail.com)
- **Prof. Dr. Francisco Couto** - fjcouto@ciencias.ulisboa.pt
- **Prof. Dr. Matilde Pato** - matilde.pato@isel.pt

**Institutions:**
- Faculty of Sciences of the University of Lisbon (FCUL)
- LASIGE - Large-Scale Informatics Systems Laboratory
- Lisbon School of Engineering of the Polytechnic University of Lisbon (ISEL-IPL)

## How to Cite

If you use HeCoReS-DR in your research, please cite:

```bibtex
@software{aveiro2025hecoresdr,
  author = {Aveiro, Donato and Pato, Matilde and Couto, Francisco},
  title = {HeCoReS-DR: Hetnet Connectivity Search for Drug Repositioning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/matpato/hecores-dr},
  version = {1.0}
}

@mastersthesis{aveiro2025thesis,
  author = {Aveiro, Donato},
  title = {Hetnet Connectivity Search in Recommender Systems for Drug Repositioning: Addressing Data Sparsity and Explainability},
  school = {Faculty of Sciences, University of Lisbon},
  year = {2025}
}
```

### Dataset Citation
```bibtex
@dataset{aveiro2025datasets,
  author = {Aveiro, Donato and Pato, Matilde and Couto, Francisco},
  title = {Drug Repositioning Datasets for Hetnet Connectivity Search},
  year = {2025},
  publisher = {Zenodo},
  doi = {[DOI_TO_BE_ASSIGNED]}
}
```

## References

- Sänger, M., et al. (2024). HunFlair2 in a cross-corpus evaluation of biomedical named entity recognition and normalization tools.
- Himmelstein, D. S., et al. (2023). Hetnet connectivity search provides rapid insights into how biomedical entities are related. *GigaScience*, 12:giad047.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Contact

For questions, suggestions, or collaboration inquiries:
- **Donato Aveiro**: fc46269@alunos.fc.ul.pt
- **Issues**: Please use the GitHub issue tracker

## Acknowledgments

We acknowledge the contributors to DrugBank, MeSH, MEDIC ontology, and the developers of HunFlair2 and Hetnet Connectivity Search frameworks, whose work made this research possible. This work was supported by LASIGE and the Faculty of Sciences of the University of Lisbon.

---

**Version:** 1.0  
**Release Date:** September 19, 2025  
**Last Updated:** October 27, 2025