import numpy as np
from stein_thinning.kernel import vfk0_imq


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
