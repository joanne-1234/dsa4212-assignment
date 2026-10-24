# DSA4212 Assignment: Bayesian Optimisation vs CMA-ES

Topic: Bayesian Optimisation vs Evolutionary Strategies

Assignment page: https://alexxthiery.github.io/teaching/bo-vs-es/bo-vs-es.html

## Overview 
This project investigates how effective dimension and rotation affect the relative performance of Gaussian-process Bayesian optimisation (BO) and the covariance matrix adaptation evolution strategy (CMA-ES) under a fixed evaluation budget.

We use synthetic weighted-quadratic functions to control the effective dimension and orientation of important search directions while keeping the ambient dimension fixed. Random search is included as a baseline.

## Research question

How does rotation affect the relative performance of Gaussian-process Bayesian optimisation (GP-BO) and CMA-ES across different effective dimensions under a fixed evaluation budget?

## Experimental design

### Objective function

We minimise synthetic weighted-quadratic functions with the following settings:

- Search domain: [-1, 1]^20.
- Ambient dimension: D = 20.
- Effective dimensions: d in {2, 5, 10, 15, 20}.
- Weights: geometrically spaced from 1 to 100, then normalised as implemented in `code/objective.py`.
- Shift: a randomly generated vector in the d-dimensional active space, scaled to have Euclidean norm 0.5.
- Minimum objective value: 0, attainable within the search domain.

The effective dimension is the number of directions that affect the objective value. All methods search over the full 20-dimensional domain without being given these important directions.

### Orientation

For each effective dimension, we compare:

- Axis-aligned objectives: the important directions correspond to the first d input coordinates.
- Rotated objectives: a random orthogonal matrix, generated using QR decomposition, changes the orientation of the important directions.

Each aligned/rotated pair is generated using the same effective dimension and objective seed, so both versions share the same weights and shift in the active space.

The search domain remains fixed at [-1, 1]^20 in both cases. Consequently, rotation can change how the objective's level sets interact with the domain boundaries.

### Methods

We compare four methods:

1. GP-BO with an ARD kernel, which learns a separate length scale for each input coordinate.
2. GP-BO with an isotropic kernel, which uses one shared length scale.
3. CMA-ES with full covariance adaptation.
4. Uniform random search as a baseline.

### Evaluation budget and performance

- Each run has a total budget of 400 objective evaluations.
- Any initial objective evaluations count towards this budget.
- Performance is measured using the best objective value found so far as a function of the number of evaluations.
- Since the known minimum is 0, the best objective value also equals the simple regret.
- Comparisons are repeated across multiple objective instances and optimiser runs to assess variability.

## Repository layout

- `authors.md`: names and student numbers.
- `requirements.txt`: required Python libraries.
- `report/`: LaTeX report source, bibliography and figures.
- `code/`: shared objective function, experiment and plotting code.
- `configs/`: experiment configurations.
- `results/`: saved numerical results, organised by experiment.


## How to reproduce the results
