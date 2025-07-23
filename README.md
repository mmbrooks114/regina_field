# Regina Field Toolkit

_A modular Python toolkit for structural analysis, scoring, and prediction of prime numbers using motif decomposition and harmonic evaluation._

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2F4AQZV-blue)](https://doi.org/10.17605/OSF.IO/4AQZV)

---

## Overview

The **Regina Field Toolkit** implements a full discovery and enrichment pipeline for the _Regina Field Hypothesis_ — a theory that prime numbers exhibit structure in motif space and harmonic signal profiles.

This system supports:

- 🔍 Motif-based scoring and anti-phase signal analysis  
- 📊 Tier ranking and prime-like candidate classification  
- 🌌 PCA and UMAP projection of prime distributions  
- 🔁 Iterative refinement of extrapolated prime candidates  
- 📈 Evolution tracking of candidate scores over refinement cycles  
- 🧠 Integration of special prime types and their resonance features

---

## Directory Structure

```
REGINA_FIELD_TOOLKIT/
├── analyze_score_evolution.py
├── data_refinement_engine.py
├── generate_projection_animation.py
├── regina_projection_pipeline.py
├── run_enrichment_cycle.py
├── run_extrapolation_cycle.py
│
├── data/
│   └── init/
│       ├── Calibration_Dataset.csv
│       ├── false_elites.csv
│       └── known_primes_up_to_10_million.csv
│
├── engine/
│   ├── extrapolation.py
│   ├── io_utils.py
│   ├── model.py
│   ├── ranking.py
│   ├── scoring.py
│   ├── scoring_by_prime_type.py
│   └── validation.py
│
└── output/
    ├── projection/
    └── refinement/
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/regina-field-toolkit.git
cd regina-field-toolkit
pip install -r requirements.txt
```

---

## Dataset Description

The dataset `output/refinement/regina_field_dataset_v1.csv` includes:

- `Candidate`: the number being evaluated  
- `PrimeStatus`: one of `Prime`, `False`, or `Candidate`  
- Additional features:
  - Motif frequency and structure  
  - Entropy and harmonic resonance  
  - Elite tier indicators  
  - Classification metrics for projection and clustering  

See [`README_DATA.md`](README_DATA.md) for column documentation.

---

## Usage

Run full enrichment:

```bash
python data_refinement_engine.py
```

Extrapolate new candidates:

```bash
python run_extrapolation_cycle.py
```

Generate projections:

```bash
python regina_projection_pipeline.py
```

Create animation:

```bash
python generate_projection_animation.py
```

---

## Citation

This toolkit is published on the Open Science Framework:

**DOI:** [10.17605/OSF.IO/4AQZV](https://doi.org/10.17605/OSF.IO/4AQZV)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

Created in honor of Regina J. Middlebrooks, whose name and memory inspire the pursuit of knowledge.
