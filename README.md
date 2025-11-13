# cs4775-final-project

CS4775 Final Project: A Maximum Expected Accuracy Approach to Pairwise Alignment

## Overview

This project implements a pair Hidden Markov Model (HMM) for RNA sequence alignment using maximum likelihood estimation (MLE) from multiple sequence alignments. The implementation includes parameter estimation, forward-backward algorithms, and utilities for working with FASTA and Stockholm alignment formats.

## Setup

### Prerequisites

- Python 3.10 or higher
- [Poetry](https://python-poetry.org/) for dependency management

### Installation

1. **Install Poetry** (if not already installed):

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd cs4775-final-project
   ```

3. **Install dependencies**:

   ```bash
   poetry install
   ```

   This will create a virtual environment and install all required packages.

4. **Activate the virtual environment**:

   ```bash
   poetry shell
   ```

   Alternatively, prefix commands with `poetry run` to run them in the virtual environment without activating it.

## Project Structure

```
cs4775-final-project/
├── data/
│   ├── alignments/     # Stockholm alignment files (.sto)
│   └── fasta/          # FASTA sequence files (.fa)
├── results/
│   └── parameters/     # Output HMM parameters (YAML)
├── src/
│   ├── algorithms/     # HMM and forward-backward algorithms
│   ├── types/          # Data structures (sequences, alignments, parameters)
│   └── utils/          # File I/O and serialization utilities
├── scripts/            # Executable scripts
└── tests/              # Unit tests
```

## Usage

### Running Scripts

All scripts should be run as modules using `python -m` to ensure proper import paths:

#### 1. Read FASTA and Stockholm Files

Preview sequences and alignments from the data directory:

```bash
python -m scripts.read
```

This will display:

- All Stockholm alignments in `data/alignments/`

#### 2. Estimate HMM Parameters

Estimate HMM parameters from Stockholm alignments using maximum likelihood estimation with pseudocount smoothing:

```bash
python -m scripts.estimate_parameters
```

This will:

- Read all `.sto` files from `data/alignments/`
- Match them with corresponding `.fa` files in `data/fasta/`
- Estimate emission and transition probabilities (in log-space)
- Apply additive smoothing (pseudocount = 0.5)
- Save parameters to `results/parameters/hmm.yaml`

The output YAML file contains:

- Metadata (number of alignments, source directory)
- Log-space emission probabilities (match, insert_x, insert_y)
- Log-space transition probabilities
- Gap parameters (delta, epsilon)

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Tests with Verbose Output

```bash
pytest -v
```

### Run Specific Test Files

```bash
pytest tests/test_hmm.py
pytest tests/test_forward_backward_skeleton.py
```

### Run Tests with Coverage

```bash
pytest --cov=src --cov-report=term-missing
```

## Development

### Code Formatting

Format code using Black:

```bash
black src/ tests/ scripts/
```

## Data Files

### Input Format

- **FASTA files** (`.fa`): Unaligned RNA sequences

  - Located in `data/fasta/`
  - Format: standard FASTA with `>identifier description` headers

- **Stockholm files** (`.sto`): Multiple sequence alignments
  - Located in `data/alignments/`
  - Format: Stockholm 1.0 with metadata annotations
  - Must have matching FASTA file with same basename

## Authors

- Lawrence Granda
- Joyce Shen
- Soham Goswami
- Joshua Ochalek

## License

This project is part of coursework for CS4775.
