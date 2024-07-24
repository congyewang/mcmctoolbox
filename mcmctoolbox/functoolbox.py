import re

import numpy as np
from scipy.optimize import minimize
from scipy.stats._multivariate import _PSD
from stein_thinning.kernel import vfk0_imq


def cartesian_cross_product(x, y):
    """
    Cartesian Product
    """
    cross_product = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    return cross_product


def k0xx(sx, linv):
    """
    Stein IMQ kernel, k_p(x,x)
    """
    return np.trace(linv) + np.sum(sx**2, axis=1)


def k_mat(x, grad_log_p, linv):
    """
    KSD Matrix
    """
    x1 = np.tile(x, len(x)).reshape(-1, 1)
    x2 = np.repeat(x, len(x)).reshape(-1, 1)
    sx1 = grad_log_p(x1)
    sx2 = grad_log_p(x2)
    res_array = vfk0_imq(x1, x2, sx1, sx2, linv)
    res_mat = res_array.reshape(x.size, x.size)
    return res_mat


def strat_sample(x_grid, P_grid, n_max):
    """
    Stratified Sampling
    """
    # Ensure P_grid is normalised
    P_grid = P_grid / np.sum(P_grid)

    u_grid = np.linspace(0, 1, n_max + 2)[1:-1]

    c_P = np.cumsum(P_grid)

    X_P = np.zeros(n_max)

    for i in range(n_max):
        for j in range(len(x_grid) - 1):
            if (u_grid[i] > c_P[j]) and (u_grid[i] <= c_P[j + 1]):
                X_P[i] = x_grid[j]

    return X_P


def comp_wksd(X, grad_log_p, Sigma):
    """
    Computing Weighted Kernel Stein Discrepancy
    """
    # remove duplicates
    X = np.unique(X)

    # dimensions
    n = len(X)

    # Stein kernel matrix
    K = k_mat(X, grad_log_p, Sigma)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, None) for _ in range(n)]
    res = minimize(
        lambda w: np.sqrt(np.dot(w.T, np.dot(K, w))),
        np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"disp": False},
    )
    wksd = res.fun

    return wksd


def discretesample(p, n):
    """
    Samples from a discrete distribution
    """
    # Parse and verify input arguments
    assert np.issubdtype(
        p.dtype, np.floating
    ), "p should be an numpy array with floating-point value type."
    assert (
        np.isscalar(n) and isinstance(n, int) and n >= 0
    ), "n should be a non-negative integer scalar."

    # Process p if necessary
    p = p.ravel()

    # Construct the bins
    edges = np.concatenate(([0], np.cumsum(p)))
    s = edges[-1]
    if abs(s - 1) > np.finfo(p.dtype).eps:
        edges = edges * (1 / s)

    # Draw bins
    rv = np.random.rand(n)
    c = np.histogram(rv, edges)[0]
    ce = c[-1]
    c = c[:-1]
    c[-1] += ce

    # Extract samples
    xv = np.nonzero(c)[0]
    if xv.size == n:  # each value is sampled at most once
        x = xv
    else:  # some values are sampled more than once
        xc = c[xv]
        dv = np.diff(xv, prepend=xv[0])
        dp = np.concatenate(([0], np.cumsum(xc[:-1])))
        d = np.zeros(n, dtype=int)
        d[dp] = dv
        x = np.cumsum(d)

    # Randomly permute the sample's order
    x = np.random.permutation(x)
    return x


def flat(nested_list):
    """
    Expand nested list
    """
    res = []
    for i in nested_list:
        if isinstance(i, list):
            res.extend(flat(i))
        else:
            res.append(i)
    return res


def nearestPD(A):
    """
    Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """
    Returns true when input is positive-definite, via Cholesky, det, and _PSD from scipy.
    """
    try:
        _ = np.linalg.cholesky(B)
        res_cholesky = True
    except np.linalg.LinAlgError:
        res_cholesky = False

    try:
        assert np.linalg.det(B) > 0, "Determinant is negative"
        res_det = True
    except AssertionError:
        res_det = False

    try:
        _PSD(B, allow_singular=False)
        res_PSD = True
    except Exception as e:
        if re.search("[Pp]ositive", str(e)):
            return False
        else:
            raise

    res = res_cholesky and res_det and res_PSD

    return res
