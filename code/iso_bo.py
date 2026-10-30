from pathlib import Path

from sklearn.gaussian_process.kernels import RBF

from shared_bo import (
    EFFECTIVE_DIMS,
    PROBLEM_SEEDS,
    METHOD_SEEDS,
    BOUNDS,
    LENGTH_SCALE_BOUNDS,
    run_gp_bo,
    save_csv,
)

# Method name is also used as the results folder.
METHOD = "isotropic_bo"

# Isotropic RBF: one shared length scale across all D coordinates.
# The bounds are kept identical to the shared BO settings.
def make_isotropic_kernel():
    return RBF(
        length_scale=1.0,
        length_scale_bounds=LENGTH_SCALE_BOUNDS,
    )


def main():
    output_dir = Path("results") / METHOD
    output_dir.mkdir(parents=True, exist_ok=True)

    total_runs = (
        len(EFFECTIVE_DIMS)
        * len(PROBLEM_SEEDS)
        * 2
        * len(METHOD_SEEDS)
    )

    run_number = 0

    for d in EFFECTIVE_DIMS:
        for problem_seed in PROBLEM_SEEDS:
            for rotated in [False, True]:
                for method_seed in METHOD_SEEDS:

                    run_number += 1

                    print(
                        f"[{run_number}/{total_runs}] "
                        f"d={d}, "
                        f"problem_seed={problem_seed}, "
                        f"rotated={rotated}, "
                        f"method_seed={method_seed}",
                        flush=True,
                    )

                    base_kernel = make_isotropic_kernel()

                    rows = run_gp_bo(
                        d=d,
                        problem_seed=problem_seed,
                        rotated=rotated,
                        method_seed=method_seed,
                        base_kernel=base_kernel,
                        method=METHOD,
                    )

                    filename = (
                        f"d{d}_problem{problem_seed}_"
                        f"rotated{rotated}_method{method_seed}.csv"
                    )

                    save_csv(
                        rows,
                        output_dir / filename,
                    )

    print(f"\nCompleted all {total_runs} isotropic BO runs.")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
