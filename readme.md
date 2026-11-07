# DSA4212 Assignment: Bayesian Optimisation vs CMA-ES

Topic: Bayesian Optimisation vs Evolutionary Strategies

Assignment page: https://alexxthiery.github.io/teaching/bo-vs-es/bo-vs-es.html

## Research question

How does rotation affect the relative performance of Gaussian-process Bayesian optimisation (GP-BO) and CMA-ES across different effective dimensions under a fixed evaluation budget?

## Experimental design

### Objective function

We minimise synthetic weighted-quadratic functions with the following settings:

- Search domain: [-1, 1]^20.
- Ambient dimension: D = 20.
- Effective dimensions: d in {2, 5, 10, 15, 20}.
- Weights: geometrically spaced from 1 to 100, then normalised to sum to one (`code/objective.py`).
- Shift: a random vector in the d-dimensional active space, scaled to Euclidean norm 0.5.
- Minimum objective value: 0, attainable within the search domain.

The effective dimension is the number of directions that affect the objective value. All methods search over the full 20-dimensional domain without being given these directions.

### Orientation

For each effective dimension, we compare:

- Axis-aligned objectives: the important directions correspond to the first d input coordinates.
- Rotated objectives: a random rotation (QR decomposition of a Gaussian matrix, determinant +1) changes the orientation of the important directions.

Each aligned/rotated pair uses the same effective dimension and problem seed, so both versions share the same weights and shift. The search domain stays fixed at [-1, 1]^20, so rotation can change how the objective's level sets meet the domain boundaries.

### Methods

We compare four methods:

1. GP-BO with an ARD kernel, which learns a separate length scale for each input coordinate.
2. GP-BO with an isotropic kernel, which uses one shared length scale.
3. CMA-ES with full covariance adaptation.
4. Uniform random search as a baseline.

### Evaluation budget and performance

- Problem seeds 0–4, aligned and rotated, method seeds 100 and 101: 100 runs per method.
- Each run has a budget of 400 objective evaluations, including initial evaluations.
- Performance is the best objective value found so far. The rotation penalty compares matched aligned and rotated runs.

### Supplementary experiment: equal curvature

To test why ARD-BO keeps an advantage over isotropic BO after rotation, both GP-BO variants were rerun on an equal-curvature objective (`code/objective_equal_curvature.py`). It uses equal weights and a Hadamard-based rotation, so that after rotation every input coordinate has the same curvature. Seeds, budget and BO settings are unchanged.

## Repository layout

- `authors.md`: names and student numbers.
- `requirements.txt`: Python dependencies.
- `report/`: report source and figures.
- `code/`:
  - `objective.py`, `objective_equal_curvature.py`: objective functions.
  - `shared_bo.py`, `shared_bo_equal_curvature.py`: shared BO procedure and settings.
  - `bo_ard.py`, `iso_bo.py`, `cmaes.py`, `random_search.py`: main experiment runners.
  - `bo_ard_equal_curvature.py`, `iso_bo_equal_curvature.py`: supplementary runners.
  - `summarise_*.py`: summary scripts.
  - `analysis.ipynb`: comparisons and figures for the main experiment.
- `results/<method>/`: main run CSVs and summaries.
- `results/supplementary_equal_curvature/<method>/`: supplementary run CSVs and summaries.


## How to reproduce

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the experiments (BO runs take several hours)
python code/cmaes.py
python code/random_search.py
python code/iso_bo.py
python code/bo_ard.py
python code/iso_bo_equal_curvature.py
python code/bo_ard_equal_curvature.py

# Summarise results and make figures (no rerun needed)
python code/summarise_cmaes.py cmaes
python code/summarise_cmaes.py bo_ard
python code/summarise_iso_bo.py
python code/summarise_random_search.py
python code/summarise_iso_bo_equal_curvature.py
python code/summarise_iso_bo_equal_curvature.py bo_ard_equal_curvature
jupyter notebook code/analysis.ipynb
```

Notes:
- Main ARD-BO used scikit-learn 1.7.2 / NumPy 2.2.6; all other runs used scikit-learn 1.9.1 / NumPy 2.4.6. Each CSV records its versions. Small numerical differences across machines can change individual runs, so compare aggregates.
