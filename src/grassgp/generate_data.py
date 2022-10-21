# from typing import Callable
from itertools import product
import jax.numpy as np
from jax import random, vmap

# from grassgp.utils import vec, unvec
from grassgp.means import zero_mean
from grassgp.kernels import rbf
# from grassgp.grassmann import rand_grass_point, rand_grass_tangent, grass_exp
# from grassgp.models import grassmann_process

import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

def generate_input_data(D: int = 2, active_dimension: int = 1, n_s: int = 10, s_lims: tuple = (0.0, 1.0), x_lims: tuple = (-2, 2), n_x_sqrt: int = 50):
    s = np.linspace(s_lims[0], s_lims[1], n_s)
    x_range = np.linspace(x_lims[0], x_lims[1], n_x_sqrt)
    X = np.array([v for v in product(x_range, repeat=D)])
    return X, s


def gen_from_univariate_gp(x, gp_config: dict = {'seed': 4357, 'm': zero_mean, 'k': rbf, 'params': {'var': 1.0, 'length': 1.0, 'noise': 0.01}, 'jitter': 1e-08, 'include_noise': True}):
    key = random.PRNGKey(gp_config['seed'])
    # create numpyro gp model
    def model(x, **kwargs):
        # compute mean vector
        mu = vmap(kwargs['m'])(x).flatten()
        
        # compute kernel mat
        K = kwargs['k'](x, x, kwargs['params'], jitter=kwargs['jitter'], include_noise=kwargs['include_noise'])
        
        # sample
        numpyro.sample("y", dist.MultivariateNormal(loc=mu, covariance_matrix=K))
    
    prior = Predictive(model, num_samples=1)
    pred = prior(key, x, **gp_config)
    return pred["y"][0]    


# def generate_inputs(key, D: int, N: int, n_s: int):
#     """function to generate input data pairs (x,s) for dataset generation.
#
#     Parameters
#     ----------
#     key : array
#         random.PRNGkey
#     D : int
#         Dimension of ambient spatial argument
#     N : int
#         number of spatial points to sample
#     n_s : int
#         number of time points in [0,1] to create
#
#     Returns
#     -------
#     tuple
#         tuple of spatial and time locations
#     """
#     # generate X grid
#     X = random.normal(key, (N, D))
#     s = np.linspace(0,1,n_s)
#     return X, s
#
# def generate_fine_coarse_inputs(key, D:int, N_fine_sqrt: int, N: int, n_s_fine: int, s_gap: int, x_lims: tuple = (-1, 1), s_lims: tuple = (0, 1)):
#     x_range = np.linspace(x_lims[0], x_lims[1], N_fine_sqrt)
#     X_fine = np.array([v for v in product(x_range, repeat=D)])
#     s_fine = np.linspace(s_lims[0], s_lims[1], n_s_fine)
#     inds = random.choice(key, N_fine_sqrt ** 2, shape=(N,), replace=False)
#     X = X_fine[inds,:].copy()
#     s = s_fine[::s_gap].copy()
#     return X_fine, X, s_fine, s
#
# def generate_input_grid(key, D: int, N_sqrt: int, n_s: int, x_lims: tuple = (-2, 2), s_lims: tuple = (0, 1)):
#     x_range = np.linspace(x_lims[0], x_lims[1], N_sqrt)
#     X = np.array([v for v in product(x_range, repeat=D)])
#     s = np.linspace(s_lims[0], s_lims[1], n_s)
#     return X, s
#
# def gen_data_from_prior(key, model, **kwargs):
#     """Function to generate data from prior of a numpyro model"""
#     prior = Predictive(model, num_samples=1)
#     pred = prior(key, **kwargs)
#     return pred
#
#
# def gen_data_from_model(key, model, n_samples = 1, **kwargs):
#     """Function to generate data from prior of a numpyro model"""
#     prior = Predictive(model, num_samples=n_samples)
#     pred = prior(key, **kwargs)
#     return pred
#
#
# def gen_proj_from_grass_process(key, **kwargs):
#     """Function to generate projections from Grassmann Process prior"""
#     pred = gen_data_from_prior(key, grassmann_process, **kwargs)
#     return pred['Ps'][0], pred
#
# def gen_data_from_grass_process(key, **kwargs):
#     """Function to generate initial parameters from prior for each site in Grassmann Process"""
#     pred = gen_data_from_prior(key, grassmann_process, **kwargs)
#     return pred
#
# def gen_from_gp(key, x: np.ndarray, m: Callable, k: Callable, input_dim = (1,), **kwargs):
#     """Function to generation from a GP with mean m and kernel k."""
#     # create numpyro gp model
#     def model(x, **kwargs):
#         assert x.shape[1:] == input_dim
#         # compute mean vector
#         mu = vmap(m)(x).flatten()
#         
#         # compute kernel mat
#         K = k(x, x, **kwargs)
#
#         # sample
#         numpyro.sample("y", dist.MultivariateNormal(loc=mu, covariance_matrix=K))
#     
#     # form prior from model
#     prior = Predictive(model ,num_samples=1)
#     pred = prior(key, x, **kwargs)
#     return pred["y"][0]
#
# def gp_with_projs_from_grass_exp(key, params: dict):
#     D = params['D']
#     N = params['N']
#     n_s = params['n_s']
#     d = params['active_dimension']
#     s_noise = params['s_noise']
#     var = params['var']
#     length = params['length']
#     noise = params['noise']
#     
#     sk1, sk2 = random.split(key, 2)
#     
#     # generate X grid
#     X = random.normal(sk1, (N, D))
#     
#     # generate projection on grassmann
#     proj = rand_grass_point(sk2, D, d)
#     
#     s = np.linspace(0,1,n_s)
#     if n_s > 1:
#         sk3, sk4, sk5 = random.split(sk2, 3)
#         tangent = rand_grass_tangent(sk3, proj)
#         new_projs = [proj]
#         for i in range(n_s-1):
#             proj_old = new_projs[-1]
#             new_proj = grass_exp(proj_old, tangent)
#             new_projs.append(new_proj)
#         
#         Ps = np.hstack(new_projs)
#         X_proj = X @ Ps
#         
#         # compute true cov
#         # first collect all X_proj samples across time into a long vector
#         X_proj_samples = vec(X_proj).reshape(-1,1)
#         cov_truth = rbf_covariance(var, length, noise, X_proj_samples, X_proj_samples)
#
#         # generate y's
#         y = random.multivariate_normal(sk4, np.zeros(N*n_s), cov_truth)
#
#         # add iid noise across times
#         y += s_noise * random.normal(sk5, y.shape)
#         
#         Ys = unvec(y, N, n_s)
#         data = {'X': X, 's': s, 'Ys': Ys, 'Ps': Ps}
#         return data
#     else:
#         sk3, sk4 = random.split(sk2, 2)
#         X_proj = X @ proj
#         X_proj_samples = vec(X_proj).reshape(-1,1)
#         cov_truth = rbf_covariance(var, length, noise, X_proj_samples, X_proj_samples)
#
#         # generate y's
#         y = random.multivariate_normal(sk3, np.zeros(N*n_s), cov_truth)
#
#         # add iid noise across times
#         y += s_noise * random.normal(sk4, y.shape)
#         
#         Ys = unvec(y, N, n_s)
#         data = {'X': X, 's': s, 'Ys': Ys, 'Ps': proj}
#         return data
#
# def univariate_gp(x, y, mu, var, noise, length, jitter=1e-06):
#     # compute kernel
#     K = rbf_covariance(var, length, noise, x.reshape(-1,1), x.reshape(-1,1), jitter=jitter)
#
#     numpyro.sample(
#         "obs_y",
#         dist.MultivariateNormal(loc=mu, covariance_matrix=K),
#         obs=y,
#     )
#
# def gen_outer_gp(key, proj_data, mu, var, noise, length, jitter=1e-06):
#     predictive = Predictive(univariate_gp, num_samples=1)
#     pred = predictive(key, proj_data, None, mu, var, noise, length, jitter=jitter)
#     return pred['obs_y'].flatten()
