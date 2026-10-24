import numpy as np

# Ambient dimension: every candidate solution has 20 input coordinates.
D = 20


def make_weighted_quadratic(d, seed, rotated=False):
    """
    Create a weighted quadratic function to MINIMISE.

    Parameters
    ----------
    d : int
        Effective dimension: how many directions affect the objective.
        Allowed values: 2, 5, 10, 15, 20.

    seed : int
        Controls the random shift and rotation.
        Must be supplied explicitly.
        Use the same d and seed across methods and for a matched
        aligned/rotated pair.

    rotated : bool
        False: important directions follow the original coordinate axes.
        True: important directions are rotated across all 20 coordinates.

    Returns
    -------
    objective : callable
        Accepts one candidate of shape (20,) and returns a float.

    Notes
    -----
    Search domain: [-1, 1]^20.
    Minimum objective value: 0.

    Each optimiser must enforce the search bounds and evaluation budget.
    This seed defines the problem, not the optimiser's search randomness.
    """
    if d not in (2, 5, 10, 15, 20):
        raise ValueError("d must be one of 2, 5, 10, 15, 20.")

    # Local generator makes the problem reproducible without changing
    # NumPy's global random state.
    rng = np.random.default_rng(seed)

    # Unequal weights create different curvature along active directions.
    # Logarithmic spacing gives a largest-to-smallest weight ratio of 100.
    weights = np.geomspace(1.0, 100.0, d)

    # Normalisation helps control objective scale while preserving
    # the weight ratio. It does not make all dimensions equally difficult.
    weights = weights / weights.sum()

    # Generate a random direction in the d active coordinates.
    shift = rng.normal(size=d)
    shift_norm = np.linalg.norm(shift)

    # Guard against the extremely unlikely all-zero draw.
    while shift_norm == 0:
        shift = rng.normal(size=d)
        shift_norm = np.linalg.norm(shift)

    # Place one minimiser at distance 0.5 from the origin.
    # It remains inside the box after rotation, and the origin is not optimal.
    shift = 0.5 * shift / shift_norm

    # QR decomposition of a Gaussian matrix gives an orthogonal matrix.
    # The sign correction removes the QR convention's sign bias.
    Q, R = np.linalg.qr(rng.normal(size=(D, D)))
    Q = Q * np.where(np.diag(R) < 0, -1.0, 1.0)

    # Ensure determinant +1: a proper rotation rather than a reflection.
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1

    def objective(x):
        x = np.asarray(x, dtype=float)

        # Each evaluation accepts exactly one 20-dimensional candidate.
        if x.shape != (D,):
            raise ValueError(f"x must have shape ({D},), got {x.shape}.")

        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite values.")

        # Express x in rotated coordinates when requested.
        # This changes orientation without changing effective dimension.
        z = Q.T @ x if rotated else x

        # Only the first d transformed coordinates affect the value.
        # Compute the weighted sum of squared deviations from the shift.
        return float(np.sum(weights * (z[:d] - shift) ** 2))

    # Metadata for checking and recording the problem.
    # Optimisers must not use these attributes to guide their search.
    objective.Q = Q
    objective.shift = shift
    objective.weights = weights
    objective.d = d
    objective.seed = seed
    objective.rotated = rotated

    return objective