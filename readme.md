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

1. GP-BO with an ARD kernel: learns a separate length scale for each input coordinate. 
2. GP-BO with an isotropic kernel: uses one shared length scale across all input coordinates. 
3. CMA-ES: uses full covariance adaptation.
4. Uniform random search: serves as a baseline.

### Evaluation budget and performance

- Problem seeds 0–4
- Method seeds 100 and 101
- Both aligned and rotated versions are evaluated
- Total: 100 runs per method (5 effective dimensions x 5 problem seeds x 2 orientations x 2 method seeds)
- Each run has a budget of 400 objective evaluations, including initial evaluations 

Performance is measured using the best objective value found so far.
The rotation penalty compares matched aligned and rotated runs.

### Supplementary experiment: equal curvature

To investigate why ARD-BO retains an advantage over isotropic BO after rotation, both GP-BO variants were rerun on an equal-curvature objective (code/objective_equal_curvature.py).

This objective uses equal weights and a Hadamard-based rotation, such that there is equal diagonal curvature across input coordinates after rotation. Seeds, evaluation budget and BO settings are unchanged.

**Note:** The supplementary equal-curvature experiment compares only ARD-BO and isotropic BO; CMA-ES and random search are not included.

## Repository layout

```text
.
├── readme.md
├── authors.md
├── requirements.txt
├── report/
│   ├── report.tex (complete machine-readable report source) 
│   └── figures/
├── code/
│   ├── objective.py
│   ├── objective_equal_curvature.py
│   ├── shared_bo.py
│   ├── shared_bo_equal_curvature.py
│   ├── bo_ard.py
│   ├── iso_bo.py
│   ├── cmaes.py
│   ├── random_search.py
│   ├── bo_ard_equal_curvature.py
│   ├── iso_bo_equal_curvature.py
│   ├── summarise_*.py
│   └── analysis.ipynb
└── results/
    ├── cmaes/
    ├── bo_ard/
    ├── iso_bo/
    ├── random_search/
    └── supplementary_equal_curvature/
        ├── isotropic_bo_equal_curvature/
        └── bo_ard_equal_curvature/
```
- authors.md: names and student numbers of group members.
- requirements.txt: Python package dependencies.
- report/: complete report source (including bibliography) and figures.
- code/: objective functions, optimisation algorithms, experiment runners, summary scripts and analysis notebook.
- results/: saved numerical results, including individual run CSVs and summary files.

# Environment and Dependencies 
The project was developed using Python 3.12.1.

The required Python packages are listed in `requirements.txt.` The main dependencies include:
- NumPy 
- SciPy 
- scikit-learn
- CMA-ES (cma)
- Matplotlib
- Pandas 
- Jupyter Notebook

**Data preparation:** No external datasets or manual data preparation are required. The synthetic objective functions are generated programmatically by the experiment scripts using specified random seeds. 

To create and activate a virtual environment, install the dependencies and run the experiments, use the instructions below.

## Environment setup 

### From the repository root:
```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```
### Check the installed versions using:
```bash 
python --version
pip list 
```

## How to reproduce

All commands below should be executed from the repository root with the virtual environment activated.

### 1. Run the experiments (BO runs take more than an hour)
```bash
python code/cmaes.py
python code/random_search.py
python code/iso_bo.py
python code/bo_ard.py

# Supplementary equal-curvature experiments 
python code/iso_bo_equal_curvature.py
python code/bo_ard_equal_curvature.py
```
### 2. Summarise results 
```bash
python code/summarise_cmaes.py cmaes
python code/summarise_cmaes.py bo_ard
python code/summarise_iso_bo.py
python code/summarise_random_search.py

# Supplementary equal-curvature summaries
python code/summarise_iso_bo_equal_curvature.py
python code/summarise_iso_bo_equal_curvature.py bo_ard_equal_curvature
```

### 3. Generate and inspect figures 
```bash
jupyter notebook code/analysis.ipynb
```
- The notebook loads the saved main and supplementary experiment results and generates the performance comparisons and figures used in the analysis.
- Existing CSVs and summaries are included in results/, so the saved results can be inspected without rerunning the experiments.

## Reproducibility notes:
- The main ARD-BO experiment was run using scikit-learn 1.7.2 and NumPy 2.2.6.
- All other experiments were run using scikit-learn 1.9.1 and NumPy 2.4.6.
- Individual CSV files record the package versions used for each run.
- The current requirements.txt specifies the newer package versions used for the other experiments.
- Small numerical differences across machines or package versions may affect individual runs. Therefore, comparisons should focus on aggregate results rather than exact reproduction of individual numerical values.
- The experiments use fixed problem and method seeds to support reproducibility.
