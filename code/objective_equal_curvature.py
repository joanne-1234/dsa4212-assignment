import numpy as np

# Ambient dimension: every candidate solution has 20 input coordinates.
D = 20

#Using a Hadamard matrix instead of standard random rotation ensures that every coordinate has the same curvature after rotation.
def _hadamard_20():
    """
    Build a 20 x 20 Hadamard matrix H (entries +-1, H @ H.T = 20 I).

    Uses the Paley construction with q = 19, a prime with q % 4 == 3.
    Produces an orthogonal matrix with entries +-1, then scales it to have unit length columns.
    """
    q = D - 1
    squares = {(i * i) % q for i in range(1, q)}

    chi = np.array([0] + [1 if a in squares else -1 for a in range(1, q)])

    jacobsthal = np.array([[chi[(j - i) % q] for j in range(q)] for i in range(q)])

    S = np.zeros((D, D), dtype=int)
    S[0, 1:] = 1
    S[1:, 0] = -1
    S[1:, 1:] = jacobsthal

    H = np.eye(D, dtype=int) + S

    if not np.array_equal(H @ H.T, D * np.eye(D, dtype=int)):
        raise RuntimeError("Paley construction did not give a Hadamard matrix.")

    return H.astype(float)


HADAMARD = _hadamard_20()


def make_equal_curvature_quadratic(d, seed, rotated=False):
    """
    Create an equal-curvature quadratic function to MINIMISE.

    Same parameters, domain, shift and minimum as make_weighted_quadratic
    in objective.py. Only the function changes: every input coordinate
    that matters has the same curvature.

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
        The d active coordinates have equal curvature; the rest are flat.
        True: important directions are rotated across all 20 coordinates,
        and every one of the 20 coordinates has the same curvature.

    Returns
    -------
    objective : callable
        Accepts one candidate of shape (20,) and returns a float.

    Notes
    -----
    Search domain: [-1, 1]^20.
    Minimum objective value: 0.

    Curvature along every input coordinate is exactly
    1/20 after rotation. Any remaining ARD advantage cannot come from
    uneven per-coordinate curvature.

    Each optimiser must enforce the search bounds and evaluation budget.
    This seed defines the problem, not the optimiser's search randomness.
    """
    if d not in (2, 5, 10, 15, 20):
        raise ValueError("d must be one of 2, 5, 10, 15, 20.")

    # Local generator makes the problem reproducible without changing
    # NumPy's global random state.
    rng = np.random.default_rng(seed)

    # Equal weights: no direction is steeper than another.
    # Normalised to sum to 1, matching the scale used in objective.py.
    weights = np.full(d, 1.0 / d)

    # Generate a random direction in the d active coordinates.
    shift = rng.normal(size=d)
    shift_norm = np.linalg.norm(shift)

    # This shift determines the location of the minimum.
    while shift_norm == 0:
        shift = rng.normal(size=d)
        shift_norm = np.linalg.norm(shift)

    shift = 0.5 * shift / shift_norm

    # Randomise the Hadamard matrix without changing its entry magnitudes
    signs = rng.choice([-1.0, 1.0], size=D)
    H = (signs[:, None] * HADAMARD)[rng.permutation(D)][:, rng.permutation(D)]
    Q = H / np.sqrt(D)

    # Random rotation within the active subspace. This varies the problem
    # across seeds but keeps every coordinate's curvature at 1/20.
    R, upper = np.linalg.qr(rng.normal(size=(d, d)))
    R = R * np.where(np.diag(upper) < 0, -1.0, 1.0)
    Q[:, :d] = Q[:, :d] @ R

    # Ensure determinant +1: a proper rotation rather than a reflection.
    # Flipping one column's sign leaves the curvature unchanged.
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
