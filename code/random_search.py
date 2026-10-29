# importing packages and function 

import csv
import time 
from pathlib import Path 
import numpy as np
from objective import D,  make_weighted_quadratic # import the D which is the ambient dimension too 
import matplotlib.pyplot as plt
import pandas as pd

# Shared experimental settings.
EFFECTIVE_DIMS = [2, 5, 10, 15, 20]
PROBLEM_SEEDS = [0, 1, 2, 3, 4]
METHOD_SEEDS = [100, 101] # the random choice made by the algorithm 
BUDGET = 400

RESULTS_DIR = Path("results/random_search")

# Creating the random search function
def run_random_search(d, problem_seed, rotated, method_seed):
  
    """
    Run uniform random search on one shared weighted quadratic.
    """
    # construct the weighted objective 
    f = make_weighted_quadratic(
        d=d, # the effective dimension
        seed=problem_seed,
        rotated=rotated, # determine whether function axis aligned or rotated
    )

    rng = np.random.default_rng(method_seed) # generates random candidate points 

    rows = [] # store result from every eval
    best = np.inf 

    # start the timer 
    start = time.perf_counter()

    for evaluation in range(1, BUDGET + 1):
        # Sample uniformly from the full 20D domain to evaluate 400 points/candidates 
        
        x = rng.uniform(-1.0, 1.0, size=D) # main part of random search alg 
    
        value = float(f(x)) # get the loss_eval for this point x that was sampled 

        if not np.isfinite(value):
            raise RuntimeError("Objective returned a non-finite value.")

        best = min(best, value) # is current obj val smaller than the best found so far? 

        row = {
            "evaluation": evaluation, # eval number 
            "generation": evaluation, # set equal to the evaluation number for compatibility with CMA-ES output format; no info gained from this column
            "value": value, # obj value of current point
            "best_so_far": best, # best obj value encountered so far 
        }

        row.update({
            f"x{i}": float(coordinate)
            for i, coordinate in enumerate(x) # to see which points x0,..,x19 were evaluated 
        }) 

        rows.append(row) # you should have 400 rows at the end  

    runtime = time.perf_counter() - start

    # add experiment metadata to every record
    for row in rows:
        row.update(
            method="random_search",
            ambient_dim=D,
            d=d,
            problem_seed=problem_seed,
            rotated=rotated,
            method_seed=method_seed,
            runtime_s=runtime,
            budget=BUDGET,
        )

    # Verify the evaluation budget and best-so-far trajectory -- validation purposes 
    if len(rows) != BUDGET:
        raise RuntimeError(
            f"Expected {BUDGET} evaluations, got {len(rows)}."
        ) 
        # checks that run contains exactly 400 runs 
    
    values = np.array([row["value"] for row in rows]) # extract current value into numpy array
    best_values = np.array([row["best_so_far"] for row in rows]) # extract best value into numpy array

    # best-so-far history should naturally be non-increasing so this makes sure it is --> a sanity check 
    np.testing.assert_allclose(
        best_values,
        np.minimum.accumulate(values),
        rtol=0,
        atol=0,
    )

    # since the weighted quad has a min of zero: we should not be getting negative values 
    if np.any(values < 0):
        raise RuntimeError("Objective returned a negative value.")

    return rows


## Saving each run as a CSV 
def save_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True) # saves each run as a csv 

    with path.open("w", newline="", encoding="utf-8") as file: # opens file; writes column headers and all 400 eval records 
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

# resulting csv can be used for plotting/analysis  


## Run every experimental condition
def run_all():
    completed = 0

    total = (
        len(EFFECTIVE_DIMS) # 5
        * len(PROBLEM_SEEDS) # 5 random problems
        * 2 # the 2 orientations 
        * len(METHOD_SEEDS) # both method seeds 
    )

    for d in EFFECTIVE_DIMS:
        for problem_seed in PROBLEM_SEEDS:
            for rotated in [False, True]:
                for method_seed in METHOD_SEEDS:

                    rows = run_random_search(  # run random search on each condition
                                               # d5_problem0_rotated_method100.csv e.g of a file name 
                        d=d,
                        problem_seed=problem_seed,
                        rotated=rotated,
                        method_seed=method_seed,
                    )

                    orientation = (
                        "rotated" if rotated else "aligned"
                    )

                    filename = ( # constructing filename 
                        f"d{d}_problem{problem_seed}_"
                        f"{orientation}_method{method_seed}.csv"
                    )

                    save_csv(rows, RESULTS_DIR / filename)

                    completed += 1

                    print(
                        f"[{completed}/{total}] {filename} | "
                        f"best={rows[-1]['best_so_far']:.6g}"
                    )


if __name__ == "__main__":
    run_all()
