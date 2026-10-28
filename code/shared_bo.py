import csv
import time
import warnings
from pathlib import Path

import numpy as np
import sklearn
from scipy.special import ndtr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel

from objective import D, make_weighted_quadratic


# Shared experimental settings (match code/cmaes.py).
EFFECTIVE_DIMS = [2, 5, 10, 15, 20]
PROBLEM_SEEDS = [0, 1, 2, 3, 4]
METHOD_SEEDS = [100, 101]
BUDGET = 400

# Shared GP-BO settings (from shared_bo_design.ipynb).
# Keep these identical for ARD and isotropic so only the kernel differs.
BOUNDS = (-1.0, 1.0)
N_INITIAL = 40
N_CANDIDATES = 2048
JITTER = 1e-8
N_RESTARTS = 1
EI_XI = 0.01
LENGTH_SCALE_BOUNDS = (1e-3, 1e2)
MATERN_NU = 2.5

# Signal variance of the GP prior, on the standardised y scale.
CONSTANT_VALUE_BOUNDS = (1e-3, 1e3)


def make_gp_kernel(base_kernel):
    """Scale a Matérn base kernel by a learned signal variance."""
    return ConstantKernel(1.0, CONSTANT_VALUE_BOUNDS) * base_kernel


def length_scales(kernel):
    """Return the fitted Matérn length scale(s) as a 1D array."""
    # Kernel is ConstantKernel * Matern, so the Matérn part is k2.
    return np.atleast_1d(kernel.k2.length_scale).astype(float)


def expected_improvement(mu, sigma, best, xi=EI_XI):
    """
    Expected improvement for MINIMISATION.

    mu, sigma and best are on the standardised y scale, so xi is a
    fraction of the observed objective spread rather than an absolute
    value. This keeps the exploration bonus meaningful as values shrink.
    """
    sigma = np.maximum(sigma, 1e-12)
    improvement = best - mu - xi
    z = improvement / sigma
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return improvement * ndtr(z) + sigma * pdf


def run_gp_bo(d, problem_seed, rotated, method_seed, base_kernel, method):
    """
    Run GP-BO with expected improvement on one shared weighted quadratic.

    problem_seed determines the objective's shift and rotation.
    method_seed determines the initial design, candidate sets and
    GP hyperparameter restarts.

    base_kernel is the only method-specific input: a Matérn kernel with
    either one length scale (isotropic) or D length scales (ARD).

    Returns one result row per objective evaluation.
    """
    f = make_weighted_quadratic(
        d=d,
        seed=problem_seed,
        rotated=rotated,
    )
    rng = np.random.default_rng(method_seed)
    low, high = BOUNDS

    X = np.empty((0, D))
    y = np.empty(0)
    rows = []
    best = np.inf

    # Start from the untrained kernel; later fits warm-start from the
    # previous iteration's hyperparameters.
    kernel = make_gp_kernel(base_kernel)
    fitted_scales = np.full(len(length_scales(kernel)), np.nan)

    def evaluate(x, phase, scales):
        nonlocal X, y, best

        # Every evaluated point must be feasible.
        if not np.all(np.isfinite(x)):
            raise RuntimeError("GP-BO generated a non-finite candidate.")

        if np.any(x < low) or np.any(x > high):
            raise RuntimeError("GP-BO generated an out-of-bounds candidate.")

        value = f(x)

        if not np.isfinite(value):
            raise RuntimeError("Objective returned a non-finite value.")

        X = np.vstack([X, x])
        y = np.append(y, value)
        best = min(best, value)

        row = {
            "evaluation": len(rows) + 1,
            "phase": phase,
            "value": value,
            "best_so_far": best,
        }
        row.update({
            f"x{i}": float(coordinate)
            for i, coordinate in enumerate(x)
        })

        # Length scales of the GP that proposed this point (NaN for the
        # initial design). Shows which directions the model treats as active.
        row.update({
            f"length_scale{i}": float(scale)
            for i, scale in enumerate(scales)
        })
        rows.append(row)

    # Measure search and recording time, excluding problem generation.
    start = time.perf_counter()

    # Initial design: uniform samples, counted towards the budget.
    for x in rng.uniform(low, high, size=(N_INITIAL, D)):
        evaluate(x, "initial", fitted_scales)

    while len(rows) < BUDGET:
        # Standardise y so the GP prior and xi have a consistent scale.
        y_mean, y_std = y.mean(), y.std()
        y_std = y_std if y_std > 0 else 1.0
        y_scaled = (y - y_mean) / y_std

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=JITTER,
            normalize_y=False,
            n_restarts_optimizer=N_RESTARTS,
            random_state=int(rng.integers(2**31 - 1)),
        )

        # Hyperparameters hitting their bounds is expected (e.g. long
        # length scales on inactive directions), so silence that warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(X, y_scaled)

        kernel = gp.kernel_
        fitted_scales = length_scales(kernel)

        # Maximise EI over a fresh set of uniform candidates.
        candidates = rng.uniform(low, high, size=(N_CANDIDATES, D))
        mu, sigma = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu, sigma, best=y_scaled.min())
        x_next = candidates[np.argmax(ei)]

        evaluate(x_next, "bo", fitted_scales)

    runtime = time.perf_counter() - start

    # Record settings so each CSV can be identified independently.
    for row in rows:
        row.update(
            method=method,
            ambient_dim=D,
            d=d,
            problem_seed=problem_seed,
            rotated=rotated,
            method_seed=method_seed,
            runtime_s=runtime,
            budget=BUDGET,
            n_initial=N_INITIAL,
            n_candidates=N_CANDIDATES,
            ei_xi=EI_XI,
            matern_nu=MATERN_NU,
            jitter=JITTER,
            n_restarts=N_RESTARTS,
            sklearn_version=sklearn.__version__,
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
