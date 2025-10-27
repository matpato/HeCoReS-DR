# Contributing to Drug Repositioning Datasets

Thank you for your interest in contributing to this drug repositioning research project! This document provides guidelines for various types of contributions.

## Table of Contents

- [Types of Contributions](#types-of-contributions)
- [Reporting Issues](#reporting-issues)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Data Contributions](#data-contributions)
- [Code Contributions](#code-contributions)
- [Documentation Improvements](#documentation-improvements)
- [Development Guidelines](#development-guidelines)

## Types of Contributions

We welcome several types of contributions:

- **Bug Reports:** Issues with data quality, matrix inconsistencies, or identifier errors
- **Feature Requests:** Suggestions for additional datasets or similarity metrics
- **Data Enhancements:** Proposals for new data sources or augmentation methods
- **Code Contributions:** Analysis scripts, validation tools, or processing pipelines
- **Documentation:** Improvements to README, methodology descriptions, or usage examples
- **Validation Studies:** Independent validation of the datasets or methodology

## Reporting Issues

### Data Quality Issues

If you identify problems with the datasets:

1. **Check existing issues** to avoid duplicates
2. **Provide specific details:**
   - File name and version
   - Row/column identifiers (DrugBank ID, MeSH ID)
   - Description of the issue
   - Expected vs. actual values
3. **Include evidence** when possible (e.g., references to DrugBank entries)

**Example:**
```
Issue: Incorrect association in hcs_0.csv
File: hcs_0.csv (v1.0)
Location: Drug DB00001, Disease D012559
Problem: Association marked as 1, but DrugBank indicates no indication for this disease
Reference: [DrugBank URL]
```

### Technical Issues

For technical problems (file format, parsing errors, etc.):

1. Describe your environment (OS, software versions)
2. Provide error messages or unexpected behavior
3. Include steps to reproduce the issue
4. Attach relevant code snippets if applicable

## Suggesting Enhancements

We welcome suggestions for improving the datasets:

### New Data Sources

Propose additional biomedical databases for integration:
- Explain the data source and its relevance
- Describe how it addresses current limitations
- Discuss integration challenges
- Provide access information (if publicly available)

### Additional Similarity Metrics

Suggest alternative similarity calculations:
- Justify why the new metric would be beneficial
- Describe the computational approach
- Compare with existing Jaccard similarity
- Provide references to methodology

### Metapath Variations

Propose new metapaths for HCS analysis:
- Describe the metapath structure
- Explain biological rationale
- Discuss expected impact on data sparsity
- Consider computational feasibility

## Data Contributions

### Validation Data

If you have validated drug-disease associations:

1. **Format requirements:**
   - DrugBank IDs for drugs
   - MeSH IDs for diseases
   - Evidence type (clinical trials, case reports, etc.)
   - Reference sources

2. **Quality criteria:**
   - Peer-reviewed sources preferred
   - Clear indication of association type (treatment, contraindication, etc.)
   - Recent data (within last 5 years preferred)

3. **Submission process:**
   - Open an issue describing the validation data
   - Attach data in CSV format
   - Include methodology for validation
   - Provide all relevant references

### Additional Annotations

Contribute supplementary annotations:
- Drug mechanisms of action
- Disease phenotype information
- Adverse event data
- Pharmacokinetic properties

## Code Contributions

### Analysis Scripts

Share analysis code that uses these datasets:

1. **Code requirements:**
   - Well-commented and readable
   - Include dependencies and versions
   - Provide usage examples
   - Document expected inputs/outputs

2. **Useful contributions:**
   - Benchmark algorithms for drug repositioning
   - Visualization tools for association matrices
   - Statistical analysis scripts
   - Performance evaluation frameworks

### Data Processing Tools

Contribute tools for:
- Data validation and quality checks
- Format conversion utilities
- Similarity matrix computation
- Identifier mapping between databases

### Submission Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/analysis-tool`)
3. Commit your changes with clear messages
4. Ensure code follows style guidelines
5. Submit a pull request with detailed description

## Documentation Improvements

### README Enhancements

Suggest improvements to:
- Clarity of dataset descriptions
- Usage examples
- Technical details
- File format specifications

### Methodology Documentation

Help clarify:
- HCS algorithm details
- Similarity computation methods
- Data processing pipeline
- Quality assurance procedures

## Development Guidelines

### Code Style

- **Python:** Follow PEP 8 style guide
- **R:** Follow tidyverse style guide
- **Comments:** Explain complex logic and methodology choices
- **Documentation:** Include docstrings for all functions

### Testing

- Validate against known drug-disease associations
- Test edge cases (missing values, boundary conditions)
- Ensure reproducibility with fixed random seeds
- Document test datasets and expected outcomes

### Version Control

- Write clear, descriptive commit messages
- Keep commits focused on single changes
- Reference issue numbers in commits when applicable
- Maintain clean commit history

---

Thank you for contributing to advancing drug repositioning research!