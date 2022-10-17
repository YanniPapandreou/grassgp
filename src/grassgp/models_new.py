import jax.numpy as np
from jax import vmap

import numpyro
import numpyro.distributions as dist

from grassgp.utils import unvec, kron_chol
from grassgp.kernels import rbf
from grassgp.grassmann import convert_to_projs


def grassmann_process(s, anchor_point, jitter=1e-06, proj_jitter=1e-4, L_jitter=1e-8, require_noise = False, reorthonormalize = True):
    D, n = anchor_point.shape
    n_s = s.shape[0]
    proj_dim = D * n

    N_projs = n_s * proj_dim
    
    # # ! LogNormal params?
    sigmas = numpyro.sample("sigmas", dist.LogNormal(0.0, 1.0).expand([proj_dim]))
    L_factor = numpyro.sample("L_factor", dist.LKJ(proj_dim, 1.0)) 
    L = numpyro.deterministic("L", L_factor + L_jitter * np.eye(proj_dim))
    Omega = numpyro.deterministic("Omega", np.outer(sigmas, sigmas) * L)
    # Omega = np.eye(proj_dim)

    proj_mean = numpyro.sample("proj_mean", dist.MultivariateNormal(scale_tril=np.eye(proj_dim)))
    proj_locs = np.tile(proj_mean, n_s)

    proj_params = numpyro.sample("standard_proj_params",
        dist.MultivariateNormal(covariance_matrix=np.eye(N_projs))
    )

    # mu = numpyro.sample("proj_params_mean",
    #     dist.Uniform(-10,10).expand([N_projs])
    # )
    # mu = numpyro.sample("proj_params_mean", dist.Uniform(-10,10))
    # mu = 1.0

    if n_s > 1:
        # parameters for the kernel of the Grassmann Process
        var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 1.0))
        length = numpyro.sample("kernel_length", dist.LogNormal(0.0, 1.0))
        
        if require_noise:
            noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 1.0))
            kernel_params = {'var': var, 'length': length, 'noise': noise}
            K = rbf(s.reshape(n_s,-1), s.reshape(n_s,-1), kernel_params, jitter=jitter)
        else:
            kernel_params = {'var': var, 'length': length, 'noise': 0.0}
            K = rbf(s.reshape(n_s,-1), s.reshape(n_s,-1), kernel_params, jitter=jitter)

        M_chol = kron_chol(K, Omega)
    else:
        M_chol = Omega
    
    # projection_parameters = numpyro.deterministic("proj_params", mu * np.ones(N_projs) + (M_chol+proj_jitter) @ proj_params)
    projection_parameters = numpyro.deterministic("proj_params", proj_locs + (M_chol+proj_jitter) @ proj_params)
    # projection_parameters = numpyro.deterministic("proj_params", (M_chol+proj_jitter) @ proj_params)

    # split projection_parameters up into params for each time
    projection_parameters_split = np.array(projection_parameters.split(n_s))

    # unvec each chunk
    unvec_Vs = vmap(lambda params: unvec(params, D, n))(projection_parameters_split)

    # form projector
    I_UUT = (np.eye(D) - anchor_point @ anchor_point.T)

    # apply this to each unvec_V
    Deltas = np.einsum('ij,ljk->lik', I_UUT, unvec_Vs)

    # convert to projections
    Ps = numpyro.deterministic("Ps", convert_to_projs(Deltas, anchor_point, reorthonormalize=reorthonormalize))

    return Ps


def univariate_gp_model(x, y, jitter=1e-06, learn_kernel_params: bool = True, kernel_params: dict = {}):
    if learn_kernel_params:
        # set uninformative log-normal priors on our three kernel hyperparameters
        # # ! log-normal params?
        var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
        noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
        length = numpyro.sample("kernel_length", dist.LogNormal(0.0, 10.0))
        kernel_params = {'var': var, 'length': length, 'noise': noise}

    # compute kernel
    K = rbf(x.reshape(-1,1), x.reshape(-1,1), kernel_params, jitter=jitter)

    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "obs_y",
        dist.MultivariateNormal(loc=np.zeros(K.shape[0]), covariance_matrix=K),
        obs=y,
    )




