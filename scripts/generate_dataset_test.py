# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: grassgp
#     language: python
#     name: grassgp
# ---

# +
# # %load_ext autoreload
# # %autoreload 2

# +
from typing import Callable
import jax.numpy as np
import jax.numpy.linalg as lin
from jax import random, vmap

import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

from grassgp.utils import vec, unvec
from grassgp.kernels import rbf
from grassgp.grassmann import convert_to_projs

# from hydra_zen import builds, instantiate, to_yaml
from hydra_zen import make_custom_builds_fn, instantiate, to_yaml, builds, make_config
# +
# def grassmann_process(s, anchor_point, model_params: dict = {}, sample_Omega: bool = True, sample_proj_locs: bool = True, sample_var: bool = True, sample_length: bool = True, sample_noise: bool = True, require_noise: bool = False, jitter: float = 1e-06, proj_jitter: float = 1e-4, L_jitter: float = 1e-8, reorthonormalize: bool = True, b: float = 1.0):
#     D, n = anchor_point.shape
#     n_s = s.shape[0]
#     proj_dim = D * n

#     N_projs = n_s * proj_dim
    
#     if sample_Omega:
#         # sample Omega
#         sigmas = numpyro.sample("sigmas", dist.LogNormal(0.0, 1.0).expand([proj_dim]))
#         L_factor = numpyro.sample("L_factor", dist.LKJ(proj_dim, 1.0)) 
#         L = numpyro.deterministic("L", L_factor + L_jitter * np.eye(proj_dim))
#         Omega = numpyro.deterministic("Omega", np.outer(sigmas, sigmas) * L)
#     else:
#         Omega = model_params['Omega']
        
    
#     if sample_proj_locs:
#         # sample proj_locs
#         proj_mean = numpyro.sample("proj_mean", dist.MultivariateNormal(scale_tril=np.eye(proj_dim)))
#         proj_locs = np.tile(proj_mean, n_s)
#     else:
#         proj_locs = model_params['proj_locs']
        
#     proj_params = numpyro.sample("standard_proj_params",
#         dist.MultivariateNormal(covariance_matrix=np.eye(N_projs))
#     )
    
#     if n_s > 1:
#         # parameters for the kernel of the Grassmann Process
#         if sample_var:
#             # sample var
#             var = numpyro.sample("kernel_var", dist.LogNormal(0.0, b))
#         else:
#             var = model_params['var']
        
#         if sample_length:
#             # sample length
#             length = numpyro.sample("kernel_length", dist.LogNormal(0.0, b))
#         else:
#             length = model_params['length']
        
#         if require_noise:
#             if sample_noise:
#                 # sample noise
#                 noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, b))
#             else:
#                 noise = model_params['noise']
            
#             params = {'var': var, 'length': length, 'noise': noise}
#             K = rbf(s, s, params, jitter=jitter)
#         else:
#             params = {'var': var, 'length': length, 'noise': 0.0}
#             K = rbf(s, s, params, jitter=jitter)
        
#         M = np.kron(K, Omega) + proj_jitter * np.eye(N_projs)
#         M_chol = lin.cholesky(M)
#     else:
#         M_chol = lin.cholesky(Omega)
    
    
#     projection_parameters = numpyro.deterministic("proj_params", proj_locs + M_chol @ proj_params)
    
#     # split projection_parameters up into params for each time
#     projection_parameters_split = np.array(projection_parameters.split(n_s))

#     # unvec each chunk
#     unvec_Vs = vmap(lambda params: unvec(params, D, n))(projection_parameters_split)

#     # form projector
#     I_UUT = (np.eye(D) - anchor_point @ anchor_point.T)

#     # apply this to each unvec_V
#     Deltas = np.einsum('ij,ljk->lik', I_UUT, unvec_Vs)

#     # convert to projections
#     Ps = numpyro.deterministic("Ps", convert_to_projs(Deltas, anchor_point, reorthonormalize=reorthonormalize))
    
#     return Ps
# -

def grassmann_process(s, anchor_point, Omega = None, proj_locs = None, var = None, length = None, noise = None, require_noise: bool = False, jitter: float = 1e-06, proj_jitter: float = 1e-4, L_jitter: float = 1e-8, reorthonormalize: bool = True, b: float = 1.0):
    D, n = anchor_point.shape
    n_s = s.shape[0]
    proj_dim = D * n

    N_projs = n_s * proj_dim
    
    if Omega is None:
        # sample Omega
        sigmas = numpyro.sample("sigmas", dist.LogNormal(0.0, 1.0).expand([proj_dim]))
        L_factor = numpyro.sample("L_factor", dist.LKJ(proj_dim, 1.0)) 
        L = numpyro.deterministic("L", L_factor + L_jitter * np.eye(proj_dim))
        Omega = numpyro.deterministic("Omega", np.outer(sigmas, sigmas) * L)
    else:
        Omega = np.array(Omega)
        
    
    if proj_locs is None:
        # sample proj_locs
        proj_mean = numpyro.sample("proj_mean", dist.MultivariateNormal(scale_tril=np.eye(proj_dim)))
        proj_locs = np.tile(proj_mean, n_s)
    else:
        proj_locs = np.array(proj_locs)
        
    proj_params = numpyro.sample("standard_proj_params",
        dist.MultivariateNormal(covariance_matrix=np.eye(N_projs))
    )
    
    if n_s > 1:
        # parameters for the kernel of the Grassmann Process
        if var is None:
            # sample var
            var = numpyro.sample("kernel_var", dist.LogNormal(0.0, b))
        
        if length is None:
            # sample length
            length = numpyro.sample("kernel_length", dist.LogNormal(0.0, b))
        
        if require_noise:
            if noise is None:
                # sample noise
                noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, b))
            
            params = {'var': var, 'length': length, 'noise': noise}
            K = rbf(s, s, params, jitter=jitter)
        else:
            params = {'var': var, 'length': length, 'noise': 0.0}
            K = rbf(s, s, params, jitter=jitter)
        
        M = np.kron(K, Omega) + proj_jitter * np.eye(N_projs)
        M_chol = lin.cholesky(M)
    else:
        M_chol = lin.cholesky(Omega)
    
    
    projection_parameters = numpyro.deterministic("proj_params", proj_locs + M_chol @ proj_params)
    
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

n_s = 5
s = np.linspace(0,1,n_s)
D = 2
active_dimension = 1
anchor_point = np.eye(D, active_dimension)

# GrassConf = builds(grassmann_process, sample_Omega=True, sample_proj_locs=True, sample_var=True,sample_length=True, sample_noise=True, require_noise=False, jitter=1e-06, proj_jitter=1e-4, L_jitter=1e-8, reorthonormalize=True, b=5.0, zen_partial=True)
# Omega = np.eye(D).tolist()
# proj_locs = np.tile(np.zeros(D*active_dimension),n_s).tolist()
GrassConf = builds(grassmann_process, Omega=None, proj_locs=None, var=None,length=None, noise=None, require_noise=False, jitter=1e-06, proj_jitter=1e-4, L_jitter=1e-8, reorthonormalize=True, b=5.0, zen_partial=True)
full_grass_model_reortho = GrassConf()
full_grass_model_no_reortho = GrassConf(reorthonormalize=False)

# +
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()

cs.store(group="model/grassmann", name="full_reortho", node=full_grass_model_reortho)
cs.store(group="model/grassmann", name="full_no_reortho", node=full_grass_model_no_reortho)
# -

print(to_yaml(Config))


def gen_from_gp(key, x: np.ndarray, m: Callable, k: Callable, **kwargs):
    """Function to generation from a GP with mean m and kernel k."""
    # create numpyro gp model
    def model(x, **kwargs):
        # compute mean vector
        mu = vmap(m)(x).flatten()
        
        # compute kernel mat
        K = k(x, x, **kwargs)

        # sample
        numpyro.sample("y", dist.MultivariateNormal(loc=mu, covariance_matrix=K))
    
    # form prior from model
    prior = Predictive(model ,num_samples=1)
    pred = prior(key, x, **kwargs)
    return pred["y"][0]


my_builds = make_custom_builds_fn(populate_full_signature=True)

from itertools import product
input_key = random.PRNGKey(4537658)
projs_key = random.PRNGKey(658769)
outer_key = random.PRNGKey(4357)
D = 2
active_dimension = 1
n_s = 10
s_lims = [0.0, 1.0]
s = np.linspace(s_lims[0],s_lims[1],n_s)
x_lims = [-2,2]
N_sqrt = 5
x_range = np.linspace(x_lims[0], x_lims[1], N_sqrt)
X = np.array([v for v in product(x_range, repeat=D)])


def generate_input_data(n_s: int = 10, s_lims: list = [0.0, 1.0], D: int = 2, x_lims: list = [-2.0, 2.0], N_sqrt: int = 10):
    # create s grid
    s = np.linspace(s_lims[0],s_lims[1],n_s)
    
    # create X grid
    x_range = np.linspace(x_lims[0], x_lims[1], N_sqrt)
    X = np.array([v for v in product(x_range, repeat=D)])
    
    return X, s


def generate_dataset(n_s: int = 10, s_lims: list = [0.0, 1.0], D: int = 2, x_lims: list = [-2.0, 2.0], N_sqrt: int = 10):
    # get input data
    s = np.linspace(s_lims[0],s_lims[1],n_s)
    
    # create X grid
    x_range = np.linspace(x_lims[0], x_lims[1], N_sqrt)
    X = np.array([v for v in product(x_range, repeat=D)])
    #


generate_data()


def univariate_gp(m: Callable, k: Callable, **kwargs):
    """Function to generation from a GP with mean m and kernel k."""
    # create numpyro gp model
    def model(x, **kwargs):
        # compute mean vector
        mu = vmap(m)(x).flatten()
        
        # compute kernel mat
        K = k(x, x, **kwargs)

        # sample
        numpyro.sample("y", dist.MultivariateNormal(loc=mu, covariance_matrix=K))
    
    return model


def sample_from_gp(key, x):
    # form prior from model
    prior = Predictive(model ,num_samples=1)
    pred = prior(key, x, **kwargs)
    return pred["y"][0]


print(to_yaml(my_builds(univariate_gp)))


def mu_f(x):
    # y = x + 0.2 * (x ** 3) + 0.5 * ((0.5 + x) ** 2) * np.sin(4.0 * x)
    # return y
    return 0.0


kernel_inputs = {'params': {'var': 1.0, 'length': 0.5, 'noise': 0.1}, 'jitter': 1e-8}
outer_key = random.PRNGKey(235)
Y_fine = gen_from_gp(outer_key, np.linspace(0,1,10), mu_f, rbf, **kernel_inputs)

print(to_yaml(full_grass_model_no_reortho))

print(to_yaml(GrassModelConf))

grass_model = instantiate(GrassConf)

prior = Predictive(grass_model, num_samples=1)

key = random.PRNGKey(235)

grass_inputs = {'s': s, 'anchor_point': anchor_point}

pred = prior(key, **grass_inputs)

pred

cs
