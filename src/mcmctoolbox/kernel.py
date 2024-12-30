from itertools import product
from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jit, random, vmap
from jaxtyping import Array
from scipy.optimize import minimize


def make_kp(
    k: Callable[[Array], Array],
    p: Callable[[Array], Array],
) -> Callable[[Array, Array], Array]:
    """
    Make Kernel Stein Discrepancy Derivative Function.

    Args:
        k (Callable[[Array], Array]): Kernel function.
        p (Callable[[Array], Array]): Target distribution.

    Returns:
        Callable[[Array, Array], Array]: Kernel Stein Discrepancy Derivative Function.
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


def vectorized_kp(
    k: Callable[[Array], Array],
    p: Callable[[Array], Array],
) -> Callable[[Array, Array], Array]:
    """
    Vectorized Kernel Stein Discrepancy Derivative Function.

    Args:
        k (Callable[[Array], Array]): Kernel function.
        p (Callable[[Array], Array]): Target distribution.

    Returns:
        Callable[[Array, Array], Array]: Vectorized Kernel Stein Discrepancy Derivative Function.
    """
    k_p = make_kp(k=k, p=p)
    k_p_v = lambda x, y: vmap(k_p, in_axes=0, out_axes=0)(x, y)

    return k_p_v


def cartesian_product(a: Array, b: Array) -> Array:
    """
    Cartesian Cross Product in 1D arrays.

    Args:
        a (Array): 1D array.
        b (Array): 1D array.

    Returns:
        Array: Cartesian Cross Product.
    """
    # Reshape input matrices to vectors
    a_vec = jnp.reshape(a, (-1,))
    b_vec = jnp.reshape(b, (-1,))

    # Compute Cartesian product
    aa, bb = jnp.meshgrid(a_vec, b_vec, indexing="ij")
    result = jnp.stack([aa, bb], axis=-1)
    return result.reshape(-1, 2)


def k_mat(
    x: Array,
    k: Callable[[Array], Array],
    p: Callable[[Array], Array],
) -> Array:
    """
    KSD Matrix for a given kernel and target distribution function p.

    Args:
        x (Array): 1D array.
        k (Callable[[Array], Array]): Kernel function.
        p (Callable[[Array], Array]): Target distribution.

    Returns:
        Array: KSD Matrix.
    """
    kp_v = jit(vectorized_kp(k=k, p=p))
    xx = jnp.array(list(product(x, x)))
    res = kp_v(xx[:, 0], xx[:, 1])

    return res.reshape(x.size, x.size)


def strat_sample(
    x_grid: Array,
    P_grid: Array,
    n_max: int,
) -> Array:
    """
    Stratified Sampling from a Discrete Distribution P_grid.

    Args:
        x_grid (Array): Grid of values.
        P_grid (Array): Discrete distribution.
        n_max (int): Number of samples.

    Returns:
        Array: Stratified Sample.
    """
    # Ensure P_grid is normalised
    P_grid = P_grid / jnp.sum(P_grid)

    u_grid = jnp.linspace(0, 1, n_max + 2)[1:-1]

    c_P = jnp.cumsum(P_grid)

    sample = lambda u: x_grid[jnp.argmax(u <= c_P)]

    X_P = vmap(sample)(u_grid)

    return X_P


def discretesample(
    p: Array,
    n: int,
    key: Array,
) -> Array:
    """
    Samples from a discrete distribution with probabilities p.

    Args:
        p (Array): Probability mass function.
        n (int): Number of samples.
        key (Array): Random key.

    Returns:
        Array: Samples from a discrete distribution.
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


def comp_wksd(
    X: Array,
    k: Callable[[Array], Array],
    p: Callable[[Array], Array],
) -> Array:
    """
    Computing Weighted Kernel Stein Discrepancy (WKSD) for a given kernel and target distribution function p.

    Args:
        X (Array): 1D array.
        k (Callable[[Array], Array]): Kernel function.
        p (Callable[[Array], Array]): Target distribution.

    Returns:
        Array: Weighted Kernel Stein Discrepancy.
    """
    # remove duplicates
    X = np.unique(X)

    # dimensions
    n = len(X)

    # Stein kernel matrix
    K = k_mat(X, k=k, p=p)

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
