from pathlib import Path

import numpy as np
from sklearn.gaussian_process.kernels import RBF

from objective import D
from shared_bo import (
    EFFECTIVE_DIMS,
    LENGTH_SCALE_BOUNDS,
    METHOD_SEEDS,
    PROBLEM_SEEDS,
    run_gp_bo,
    save_csv,
)


RESULTS_DIR = Path("results/bo_ard")


def make_ard_kernel():
    """
    RBF kernel with one length scale per input coordinate.

    Each of the D length scales is fitted separately, so the GP can learn
    that some coordinates barely affect the objective (long length scale).
    This suits axis-aligned objectives; after rotation, every coordinate
    matters and the advantage is expected to shrink.
    """
    return RBF(
        length_scale=np.ones(D),
        length_scale_bounds=LENGTH_SCALE_BOUNDS,
    )


def run_bo_ard(d, problem_seed, rotated, method_seed):
    """Run ARD GP-BO on one shared weighted quadratic."""
    return run_gp_bo(
        d=d,
        problem_seed=problem_seed,
        rotated=rotated,
        method_seed=method_seed,
        base_kernel=make_ard_kernel(),
        method="bo_ard",
    )


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
                    orientation = "rotated" if rotated else "aligned"
                    filename = (
                        f"d{d}_problem{problem_seed}_"
                        f"{orientation}_method{method_seed}.csv"
                    )
                    path = RESULTS_DIR / filename
                    completed += 1

                    # GP-BO runs are slow, so skip runs already saved.
                    if path.exists():
                        print(f"[{completed}/{total}] {filename} | skipped")
                        continue

                    rows = run_bo_ard(
                        d=d,
                        problem_seed=problem_seed,
                        rotated=rotated,
                        method_seed=method_seed,
                    )
                    save_csv(rows, path)

                    print(
                        f"[{completed}/{total}] {filename} | "
                        f"best={rows[-1]['best_so_far']:.6g} | "
                        f"runtime={rows[-1]['runtime_s']:.0f}s"
                    )


if __name__ == "__main__":
    # Start with one test run. Change to False after checking its output.
    TEST_RUN = False

    if TEST_RUN:
        rows = run_bo_ard(
            d=2,
            problem_seed=0,
            rotated=False,
            method_seed=100,
        )
        save_csv(rows, RESULTS_DIR / "test_run.csv")

        scales = [rows[-1][f"length_scale{i}"] for i in range(D)]
        print(
            f"Test completed | evaluations={len(rows)} | "
            f"best={rows[-1]['best_so_far']:.6g} | "
            f"runtime={rows[-1]['runtime_s']:.0f}s"
        )
        print("Final length scales:", np.round(scales, 3))
    else:
        run_all()
