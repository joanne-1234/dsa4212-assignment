import sys
from pathlib import Path

import numpy as np
from sklearn.gaussian_process.kernels import RBF

from objective_equal_curvature import D
from shared_bo_equal_curvature import (
    EFFECTIVE_DIMS,
    LENGTH_SCALE_BOUNDS,
    METHOD_SEEDS,
    PROBLEM_SEEDS,
    run_gp_bo,
    save_csv,
)

METHOD = "bo_ard_equal_curvature"
RESULTS_DIR = Path("results/supplementary_equal_curvature") / METHOD


def make_ard_kernel():
    """RBF kernel with one length scale per input coordinate."""
    return RBF(length_scale=np.ones(D), length_scale_bounds=LENGTH_SCALE_BOUNDS)


def run_all(dims):
    for d in dims:
        for problem_seed in PROBLEM_SEEDS:
            for rotated in [False, True]:
                for method_seed in METHOD_SEEDS:
                    orientation = "rotated" if rotated else "aligned"
                    filename = f"d{d}_problem{problem_seed}_{orientation}_method{method_seed}.csv"
                    path = RESULTS_DIR / filename

                    # Slow runs: skip any already saved.
                    if path.exists():
                        print(f"{filename} | skipped", flush=True)
                        continue

                    rows = run_gp_bo(
                        d=d,
                        problem_seed=problem_seed,
                        rotated=rotated,
                        method_seed=method_seed,
                        base_kernel=make_ard_kernel(),
                        method=METHOD,
                    )
                    save_csv(rows, path)
                    print(
                        f"{filename} | best={rows[-1]['best_so_far']:.6g} | "
                        f"runtime={rows[-1]['runtime_s']:.0f}s",
                        flush=True,
                    )


if __name__ == "__main__":
    # Optional: choose which d to run, e.g. `python code/bo_ard_equal_curvature.py 2 5`
    dims = [int(a) for a in sys.argv[1:]] or EFFECTIVE_DIMS
    run_all(dims)