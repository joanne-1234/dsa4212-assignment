"""Summarise the random search results.

Usage (from the repo folder):
    python code/random_search.py 

Reads results/random_search/d*_problem*.csv (one file per run, written by
code/random_search.py) and writes three files to results/random_search/:
    random_search_runs.csv              one row per run (best at 100, 200 and 400 evaluations)
    random_search_summary.csv           one row per effective dimension (medians, rotation penalty)
    random_search_rotation_effects.csv  log rotation penalty for each problem pair
and prints the summary table. Stops with an error if any run is missing,
duplicated or incomplete.

The rotation penalty comes from matched runs, so it need not equal
rotated_median / aligned_median.

Other methods can reuse this script by passing their folder name instead
of random_search, if their run files use the same name pattern and columns.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

method = sys.argv[1] if len(sys.argv) > 1 else "random_search"
results = Path("results")
files = sorted((results / method).glob("d*_problem*.csv"))
if not files:
    sys.exit(f"No result files found in {results / method}")

# 1. One row per run.
runs = []
for path in files:
    df = pd.read_csv(path)
    if len(df) != 400:
        raise ValueError(f"{path.name}: expected 400 evaluations.")
    if df["evaluation"].tolist() != list(range(1, 401)):
        raise ValueError(f"{path.name}: evaluations must be ordered 1-400.")
    last = df.iloc[-1]
    runs.append({
        "method": method,
        "d": int(last["d"]),
        "problem_seed": int(last["problem_seed"]),
        "rotated": str(last["rotated"]) == "True",
        "method_seed": int(last["method_seed"]),
        "n_evals": len(df),
        "best_at_100": df["best_so_far"].iloc[99],
        "best_at_200": df["best_so_far"].iloc[199],
        "final_best": last["best_so_far"],
    })
runs = pd.DataFrame(runs)

# 2. Check that every agreed setting appears exactly once, with a full budget.
keys = ["d", "problem_seed", "rotated", "method_seed"]
if runs.duplicated(keys).any():
    raise ValueError("Duplicate runs found.")
expected = {
    (d, p, rotated, s)
    for d in [2, 5, 10, 15, 20]
    for p in [0, 1, 2, 3, 4]
    for rotated in [False, True]
    for s in [100, 101]
}
actual = set(runs[keys].itertuples(index=False, name=None))
if actual != expected:
    raise ValueError(f"Missing: {expected - actual}\nExtra: {actual - expected}")
if not runs["n_evals"].eq(400).all():
    raise ValueError("Every run must contain 400 evaluations.")
if not np.isfinite(runs["final_best"]).all() or runs["final_best"].lt(0).any():
    raise ValueError("Final values must be finite and non-negative.")

runs.to_csv(results / method / f"{method}_runs.csv", index=False)

# 3. Final performance: median over the 10 runs per orientation.
# The median describes typical performance and limits the influence
# of unusually large values.
summary = runs.groupby(["d", "rotated"])["final_best"].median().unstack()
summary.columns = ["aligned_median", "rotated_median"]

# 4. Rotation penalty from matched runs (same d, problem seed and method seed).
# Calculate matched log-ratios, then average over method seeds within each
# problem pair. Log-ratio 0 means no rotation effect. Exponentiate the
# summaries to report multiplicative factors:
# 1 = no effect, >1 = worse after rotation, <1 = better.
EPSILON = 1e-12  # numerical floor so a value of exactly 0 cannot break the log
wide = runs.pivot_table(index=["d", "problem_seed", "method_seed"],
                        columns="rotated", values="final_best")
log_ratio = (np.log(wide[True].clip(lower=EPSILON))
             - np.log(wide[False].clip(lower=EPSILON)))
pair_effect = log_ratio.groupby(["d", "problem_seed"]).mean()
pair_effect.rename("log_rotation_penalty").to_csv(results / method / f"{method}_rotation_effects.csv")
summary["penalty_geomean"] = np.exp(pair_effect.groupby("d").mean())
summary["penalty_min"] = np.exp(pair_effect.groupby("d").min())
summary["penalty_max"] = np.exp(pair_effect.groupby("d").max())
summary["n_runs"] = runs.groupby("d").size()

summary.to_csv(results / method / f"{method}_summary.csv")
print(f"{method}: {len(runs)} runs, all checks passed\n")
print(summary.to_string(float_format="{:.3g}".format))
