from itertools import product

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jit, random, vmap
from scipy.optimize import minimize


def make_kp(k, p):
    """
    Make Kernel Stein Discrepancy
    """
    d_log_p = jacfwd(lambda x: jnp.log(p(x)))
    dx_k = jacfwd(k, argnums=0)
    dy_k = jacfwd(k, argnums=1)
    dxdy_k = jacfwd(dy_k, argnums=0)
    k_p = (
        lambda x, y: dxdy_k(x, y)
        + dx_k(x, y) * d_log_p(y)
        + dy_k(x, y) * d_log_p(x)
        + k(x, y) * d_log_p(x) * d_log_p(y)
    )
    return k_p


def vectorized_kp(**kwargs):
    """
    Vectorized Kernel Stein Discrepancy
    """
    k_p = make_kp(k=kwargs["k"], p=kwargs["p"])
    k_p_v = lambda x, y: vmap(k_p, in_axes=0, out_axes=0)(x, y)
    return k_p_v


def cartesian_product(a, b):
    """
    Cartesian Cross Product in 1D
    """
    # Reshape input matrices to vectors
    a_vec = jnp.reshape(a, (-1,))
    b_vec = jnp.reshape(b, (-1,))

    # Compute Cartesian product
    aa, bb = jnp.meshgrid(a_vec, b_vec, indexing="ij")
    result = jnp.stack([aa, bb], axis=-1)
    return result.reshape(-1, 2)


def k_mat(x, **kwargs):
    """
    KSD Matrix
    """
    kp_v = jit(vectorized_kp(k=kwargs["k"], p=kwargs["p"]))
    xx = jnp.array(list(product(x, x)))
    res = kp_v(xx[:, 0], xx[:, 1])
    return res.reshape(x.size, x.size)


def strat_sample(x_grid, P_grid, n_max):
    """
    Stratified Sampling
    """
    # Ensure P_grid is normalised
    P_grid = P_grid / jnp.sum(P_grid)

    u_grid = jnp.linspace(0, 1, n_max + 2)[1:-1]

    c_P = jnp.cumsum(P_grid)

    sample = lambda u: x_grid[jnp.argmax(u <= c_P)]

    X_P = vmap(sample)(u_grid)

    return X_P


def discretesample(p, n, key):
    """
    Samples from a discrete distribution
    """
    uniform_key, permutation_key = random.split(key)
    # Parse and verify input arguments
    assert jnp.issubdtype(
        p.dtype, jnp.floating
    ), "p should be an jax array with floating-point value type."
    assert (
        jnp.isscalar(n) and isinstance(n, int) and n >= 0
    ), "n should be a non-negative integer scalar."

    # Process p if necessary
    p = p.ravel()

    # Construct the bins
    edges = jnp.concatenate((jnp.array([0]), jnp.cumsum(p)))
    s = edges[-1]
    if abs(s - 1) > jnp.finfo(p.dtype).eps:
        edges = edges * (1 / s)
    # Draw bins
    rv = random.uniform(uniform_key, shape=(n,))
    c = jnp.histogram(rv, edges)[0]
    ce = c[-1]
    c = c[:-1]
    c = c.at[-1].add(ce)

    # Extract samples
    xv = jnp.nonzero(c)[0]
    if xv.size == n:  # each value is sampled at most once
        x = xv
    else:  # some values are sampled more than once
        xc = c[xv]
        d = jnp.zeros(n, dtype=int)
        dv = jnp.diff(xv, prepend=jnp.atleast_1d(xv[0]))
        dp = jnp.concatenate((jnp.array([0]), jnp.cumsum(xc[:-1])))
        d = d.at[dp.astype(jnp.int8)].set(dv)
        x = jnp.cumsum(d)

    # Randomly permute the sample's order
    x = random.permutation(permutation_key, x)
    return x


def comp_wksd(X, **kwargs):
    """
    Computing Weighted Kernel Stein Discrepancy
    """
    # remove duplicates
    X = np.unique(X)

    # dimensions
    n = len(X)

    # Stein kernel matrix
    K = k_mat(X, k=kwargs["k"], p=kwargs["p"])

    K = np.asarray(K, dtype=np.float64)
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

    return jnp.array(wksd)
