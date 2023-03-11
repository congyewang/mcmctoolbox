import numpy as np
from stein_thinning.kernel import vfk0_imq
from scipy.optimize import minimize


def cartesian_cross_product(x,y):
    """
    Cartesian Product
    """
    cross_product = np.transpose([np.tile(x, len(y)),np.repeat(y,len(x))])
    return cross_product

def k0xx(sx, linv):
    """
    Stein IMQ kernel, k_p(x,x)
    """
    return np.trace(linv) + np.sum(sx ** 2, axis=1)

def k_mat(x, grad_log_p, linv):
    """
    KSD Matrix
    """
    x1 = np.tile(x, len(x)).reshape(-1, 1)
    x2 = np.repeat(x,len(x)).reshape(-1, 1)
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

    u_grid = np.linspace(0, 1, n_max+2)[1:-1]

    c_P = np.cumsum(P_grid)

    X_P = np.zeros(n_max)

    for i in range(n_max):
        for j in range(len(x_grid)-1):
            if (u_grid[i] > c_P[j]) and (u_grid[i] <= c_P[j+1]):
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

    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, None) for _ in range(n)]
    res = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(K, w))), np.ones(n)/n, method='SLSQP', bounds=bounds, constraints=cons, options={'disp': False})
    wksd = res.fun

    return wksd

def discretesample(p, n):
    """
    Samples from a discrete distribution
    """
    # Parse and verify input arguments
    assert np.issubdtype(p.dtype, np.floating), \
        'p should be an numpy array with floating-point value type.'
    assert np.isscalar(n) and isinstance(n, int) and n >= 0, \
        'n should be a non-negative integer scalar.'

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
    else:             # some values are sampled more than once
        xc = c[xv]
        dv = np.diff(xv, prepend=xv[0])
        dp = np.concatenate(([0], np.cumsum(xc[:-1])))
        d = np.zeros(n, dtype=int)
        d[dp] = dv
        x = np.cumsum(d)

    # Randomly permute the sample's order
    x = np.random.permutation(x)
    return x
