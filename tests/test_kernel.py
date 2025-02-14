import jax
import jax.numpy as jnp
from jaxtyping import Array

from mcmctoolbox.kernel import (
    cartesian_product,
    comp_wksd,
    discrete_sample,
    ksd_matrix,
    make_ksd_derivative,
    strat_sample,
    vectorized_make_ksd_derivative,
)


def mock_kernel(x: Array, y: Array) -> Array:
    return jnp.exp(-jnp.sum((x - y) ** 2))


def mock_target_distribution(x: Array) -> Array:
    return jnp.exp(-0.5 * jnp.sum(x**2))


def test_make_ksd_derivative() -> None:
    ksd_derivative = make_ksd_derivative(mock_kernel, mock_target_distribution)
    x = jnp.array([1.0, 2.0])
    y = jnp.array([2.0, 3.0])
    result = ksd_derivative(x, y)
    assert result.shape == (2, 2)


def test_vectorized_make_ksd_derivative() -> None:
    vectorized_ksd_derivative = vectorized_make_ksd_derivative(
        mock_kernel, mock_target_distribution
    )
    x = jnp.array([[1.0, 2.0], [2.0, 3.0]])
    y = jnp.array([[2.0, 3.0], [3.0, 4.0]])
    result = vectorized_ksd_derivative(x, y)
    assert result.shape == (2, 2, 2)


def test_cartesian_product() -> None:
    a = jnp.array([1, 2])
    b = jnp.array([3, 4])
    result = cartesian_product(a, b)
    assert result.shape == (4, 2)
    assert jnp.array_equal(result, jnp.array([[1, 3], [1, 4], [2, 3], [2, 4]]))


def test_ksd_matrix() -> None:
    x = jnp.array([1.0, 2.0, 3.0])
    result = ksd_matrix(x, mock_kernel, mock_target_distribution)
    assert result.shape == (3, 3)


def test_strat_sample() -> None:
    x_grid = jnp.array([1.0, 2.0, 3.0])
    P_grid = jnp.array([0.2, 0.5, 0.3])
    n_max = 5
    result = strat_sample(x_grid, P_grid, n_max)
    assert result.shape == (n_max,)
    assert jnp.all(result >= x_grid.min()) and jnp.all(result <= x_grid.max())


def test_discrete_sample() -> None:
    p = jnp.array([0.2, 0.5, 0.3])
    n = 5
    key = jax.random.PRNGKey(0)
    result = discrete_sample(p, n, key)
    assert result.shape == (n,)
    assert jnp.all(result >= 0) and jnp.all(result < len(p))


def test_comp_wksd() -> None:
    X = jnp.array([[1.0, 2.0], [2.0, 3.0]])
    result = comp_wksd(X, mock_kernel, mock_target_distribution)
    assert result.shape == ()
