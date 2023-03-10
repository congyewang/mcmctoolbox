import jax.numpy as jnp
from jax import jacfwd, jit
import jax


def make_kp(k, p):
    """
    Make Kernel Stein Discrepancy
    """
    d_log_p = jacfwd(lambda x: jnp.log(p(x)))
    dx_k = jacfwd(k, argnums=0)
    dy_k = jacfwd(k, argnums=1)
    dxdy_k = jacfwd(dy_k, argnums=0)
    k_p = lambda x, y: dxdy_k(x, y) + dx_k(x, y) * d_log_p(y) + dy_k(x, y) * d_log_p(x) + k(x, y) * d_log_p(x) * d_log_p(y)
    return k_p

def vectorized_kp(**kwargs):
    """
    Vectorized Kernel Stein Discrepancy
    """
    k_p = make_kp(k=kwargs['k'], p=kwargs['p'])
    k_p_v = lambda x,y : jax.vmap(k_p, in_axes=0, out_axes=0)(x, y)
    return k_p_v
