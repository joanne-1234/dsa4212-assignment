import csv
import time
from pathlib import Path

import cma
import numpy as np

from objective import D, make_weighted_quadratic


# Shared experimental settings.
EFFECTIVE_DIMS = [2, 5, 10, 15, 20]
PROBLEM_SEEDS = [0, 1, 2, 3, 4]
METHOD_SEEDS = [100, 101]
BUDGET = 400

# CMA-ES-specific settings: keep these fixed across all conditions.
POPSIZE = 12
SIGMA0 = 0.3

RESULTS_DIR = Path("results/cmaes")


def run_cmaes(d, problem_seed, rotated, method_seed):
    """
    Run CMA-ES on one shared weighted quadratic.

    problem_seed determines the objective's shift and rotation.
    method_seed determines CMA-ES's random sampling.

    Returns one result row per objective evaluation.
    """
    f = make_weighted_quadratic(
        d=d,
        seed=problem_seed,
        rotated=rotated,
    )

    # Start at the centre with an initially spherical distribution.
    # The objective's shift ensures the origin is not already optimal.
    es = cma.CMAEvolutionStrategy(
        np.zeros(D),
        SIGMA0,
        {
            "bounds": [-1.0, 1.0],
            "BoundaryHandler": cma.BoundTransform,
            "popsize": POPSIZE,
            "seed": method_seed,
            "CMA_diagonal": False,  # Full covariance from the start.
            "CMA_active": True,     # Retain pycma's active covariance update.
            "verbose": -9,
            "verb_log": 0,
        },
    )

    rows = []
    best = np.inf
    generation = 0

    # Measure search and recording time, excluding problem generation
    # and CMA-ES initialisation.
    start = time.perf_counter()

    # Use the fixed evaluation budget rather than convergence-based
    # stopping criteria. Numerical errors are raised rather than hidden.
    while len(rows) < BUDGET:
        generation += 1

        # Ask for a population, then evaluate only the remaining budget.
        candidates = es.ask()
        remaining = BUDGET - len(rows)
        evaluated_candidates = candidates[:remaining]
        values = []

        for x in evaluated_candidates:
            x = np.asarray(x, dtype=float)

            # Boundary handling must keep every evaluated point feasible.
            if not np.all(np.isfinite(x)):
                raise RuntimeError("CMA-ES generated a non-finite candidate.")

            if np.any(x < -1.0) or np.any(x > 1.0):
                raise RuntimeError("CMA-ES generated an out-of-bounds candidate.")

            value = f(x)

            if not np.isfinite(value):
                raise RuntimeError("Objective returned a non-finite value.")

            best = min(best, value)
            values.append(value)

            row = {
                "evaluation": len(rows) + 1,
                "generation": generation,
                "value": value,
                "best_so_far": best,
            }

            # Save the actual input for checking and recovering solutions.
            row.update({
                f"x{i}": float(coordinate)
                for i, coordinate in enumerate(x)
            })
            rows.append(row)

        if len(evaluated_candidates) == len(candidates):
            # Update the search distribution after a complete population.
            es.tell(candidates, values)
        else:
            # At budget 400 and population 12, the final four candidates
            # contribute to the results but do not trigger a partial update.
            break

    runtime = time.perf_counter() - start

    # Record settings so each CSV can be identified independently.
    for row in rows:
        row.update(
            method="cmaes",
            ambient_dim=D,
            d=d,
            problem_seed=problem_seed,
            rotated=rotated,
            method_seed=method_seed,
            runtime_s=runtime,
            budget=BUDGET,
            popsize=POPSIZE,
            sigma0=SIGMA0,
            cma_active=True,
            boundary_handler="BoundTransform",
            cma_version=cma.__version__,
            numpy_version=np.__version__,
        )

    # Check evaluation accounting and the recorded best-value trajectory.
    if len(rows) != BUDGET:
        raise RuntimeError(f"Expected {BUDGET} evaluations, got {len(rows)}.")

    values = np.array([row["value"] for row in rows])
    best_values = np.array([row["best_so_far"] for row in rows])

    if np.any(values < 0):
        raise RuntimeError("Weighted quadratic returned a negative value.")

    np.testing.assert_allclose(
        best_values,
        np.minimum.accumulate(values),
        rtol=0,
        atol=0,
    )

    return rows


def save_csv(rows, path):
    """Save one run's settings, evaluated points and progress."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_all():
    """Run every agreed dimension, problem, orientation and method seed."""
    completed = 0
    total = (
        len(EFFECTIVE_DIMS)
        * len(PROBLEM_SEEDS)
        * 2
        * len(METHOD_SEEDS)
    )

    for d in EFFECTIVE_DIMS:
        for problem_seed in PROBLEM_SEEDS:
            for rotated in [False, True]:
                for method_seed in METHOD_SEEDS:
                    rows = run_cmaes(
                        d=d,
                        problem_seed=problem_seed,
                        rotated=rotated,
                        method_seed=method_seed,
                    )

                    orientation = "rotated" if rotated else "aligned"
                    filename = (
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
    # Start with one test run. Change to False after checking its output.
    TEST_RUN = False

    if TEST_RUN:
        rows = run_cmaes(
            d=2,
            problem_seed=0,
            rotated=True,
            method_seed=100,
        )
        save_csv(rows, RESULTS_DIR / "test_run.csv")

        print(
            f"Test completed | evaluations={len(rows)} | "
            f"best={rows[-1]['best_so_far']:.6g}"
        )
    else:
        run_all()