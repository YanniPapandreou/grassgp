import jax.numpy as np
from jax import jit
from functools import partial
from mlkernels.jax import EQ, Delta, Matern12, Matern32, Matern52

@partial(jit, static_argnums=(5, 6))
def rbf_covariance(x, xp, var, length, noise,  jitter=1.0e-6,
                   include_noise=True):
    """Computes the kernel matrix associated with a RBF + DiracDelta kernel function

    Parameters
    ----------
    x : array
        grid of points
    xp : array
        grid of points
    var : float
        variance for RBF
    length : float
        length-scale for RBF
    noise : float
        noise level for DiracDelta
    jitter : float, optional
        additional jitter to add in DiracDelta part, by default 1.0e-6
    include_noise : bool, optional
        boolean controlling whether to add DiracDelta or not, by default True

    Returns
    -------
    array
        kernel matrix K, with ij-th entry k(x_i,xp_j) where k(x,y) := (RBF(var,length) + (noise+jitter)DiracDelta)(x,y)
    """
    diff = np.expand_dims(x / length, 1) - np.expand_dims(xp / length, 0)
    Z = var * np.exp(-0.5 * np.sum(diff**2, axis=2))  # ! axis = 2 ??

    if include_noise:
        return Z + (noise + jitter) * np.eye(x.shape[0])
        # return Z + (noise + jitter) * np.eye(x.shape[0],xp.shape[0])
    else:
        return Z


def rbf(x, xp, params, jitter = 1.0e-6, include_noise = True):
    var = params['var']
    length = params['length']
    noise = params['noise']

    k = var * EQ().stretch(length)

    if include_noise:
        k += (jitter + noise) * Delta(0.0)

    return k(x, xp).mat


def spatio_temporal_matern_52(xt, zt, params, jitter = 1.0e-6, include_noise = True):
    var_x = params['var_x']
    var_t = params['var_t']
    length_x = params['length_x']
    length_t = params['length_t']
    noise = params['noise']

    k_x = var_x * Matern52().stretch(length_x)
    k_t = var_t * Matern52().stretch(length_t)
    
    k = k_x.select([0]) * k_t.select([1])
    
    if include_noise:
        k += (jitter + noise) * Delta(0.0)
    
    return k(xt, zt).mat