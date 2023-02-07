# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os

# %%
# import jax
import time
from hydra_zen import instantiate, make_config, builds 
import sys

# %%
from itertools import product
import jax.numpy as np
from jax import random 
from jax import random, vmap

import chex
from typing import Tuple

# %%
from grassgp.utils import get_save_path, subspace_angle, to_dictconf
from grassgp.utils import vec
# from grassgp.utils import safe_save_jax_array_dict as safe_save
# from grassgp.utils import load_and_convert_to_samples_dict as load_data
from grassgp.grassmann import valid_grass_point, convert_to_projs, grass_log, grass_exp, compute_barycenter
from grassgp.kernels import rbf
from grassgp.models_optimised import GrassGP
from grassgp.means import zero_mean
# from grassgp.plot_utils import flatten_samples, plot_grass_dists, plot_AS_dir_preds

# %%
import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_value
import numpyro.distributions as dist

# %%
from grassgp.inference import run_inference


# %%
# import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (10,6)

# %%
def run_svi_for_map(rng_key, model, maxiter, step_size, *args):
    start = time.time()
    guide = autoguide.AutoDelta(model)
    optimzer = numpyro.optim.Adam(step_size)
    svi = SVI(model, guide, optimzer, Trace_ELBO())
    svi_results = svi.run(rng_key, maxiter, *args)
    print('\nSVI elapsed time:', time.time() - start)
    return svi_results

# %%
# generate dataset
N = 40
s_test = np.linspace(0, 1, N)
k = 2 * np.pi
x = np.cos(k * s_test).reshape(-1, 1)
y = np.sin(k * s_test).reshape(-1, 1)
Ws_test = np.hstack((x,y))[:,:,None]
assert vmap(valid_grass_point)(Ws_test).all()
d, n = Ws_test.shape[1:]

# # plot dataset
# for i in range(d):
#     plt.plot(s_test, Ws_test[:,i,0])
#     plt.title(f'{i+1}th component of projection')
#     plt.grid()
#     plt.xlabel(r'$s$')
#     plt.show()

# %%
n_x_sqrt = 50
x_range = np.linspace(-1, 1, n_x_sqrt)
X_test = np.array([v for v in product(x_range, repeat=d)])

# %%
X_projs_test = np.einsum('ij,ljk->lik', X_test, Ws_test)
X_projs_all_test = np.vstack([X_projs_test[i,:,:] for i in range(N)])

# %%
Ys_test = X_projs_test + 0.2 * np.power(X_projs_test, 3.0) + 0.5 * np.power(0.5 + X_projs_test, 2.0) * np.sin(4.0 * X_projs_test)
Ys_test += 0.15 * random.normal(random.PRNGKey(547), shape=Ys_test.shape)
Ys_test = vmap(lambda Y: Y - Y.mean())(Ys_test)
Ys_test = vmap(lambda Y: Y/Y.std())(Ys_test)

# %%
# for i in range(40):
#     plt.scatter(X_projs_test[i,:,0], Y_test[i,:,0])
#     plt.show()

# %%
s_gap = 8
s_train = s_test[::s_gap].copy()
print(f"{s_train.shape = }")

# %%
Ws_train = Ws_test[::s_gap,:,:].copy()
print(f"{Ws_train.shape = }")

# %%
x_gap = 45
X_train = X_test[::x_gap,:].copy()
print(f"{X_train.shape = }")
# plt.scatter(X_test[:,0], X_test[:,1], label='test', alpha=0.2)
# plt.scatter(X_train[:,0], X_train[:,1], label='train', alpha=0.5)
# plt.legend()
# plt.show()

# %%
X_projs_train = np.einsum('ij,ljk->lik', X_train, Ws_train)
print(f"{X_projs_train.shape = }")

# %%
Ys_train = Ys_test[::s_gap,::x_gap,:].copy()
print(f"{Ys_train.shape = }")

# %%
# plt.plot(np.linspace(0,5,100),np.exp(dist.LogNormal(0,10).log_prob(np.linspace(0,10,100))))
# plt.show()

# %%
# for i in range(5):
#     plt.scatter(X_projs_train[i,:,0], Ys_train[i,:,0],alpha=0.5)
#     plt.show()

# %%
anchor_point = np.eye(d,n)
assert valid_grass_point(anchor_point)
d_in = 1
U = anchor_point


# %%
def model(s, X, Ys):
    d, n = U.shape
    N = s.shape[0]
    d_n = d * n
    Omega_diag_chol = numpyro.sample('Omega_diag_chol', dist.LogNormal(0.0, 1.0).expand([d_n]))
    grass_length = numpyro.sample("grass_length", dist.LogNormal(0.0, 10.0))
    grass_kernel_params = {'var': 1.0, 'length': grass_length, 'noise': 0.0}
    k_grass = lambda t, s: rbf(t, s, grass_kernel_params, jitter=1e-06)
    mu_grass = lambda s: zero_mean(s, d, n)
    grass_gp = GrassGP(d_in=1, d_out=(d,n), mu=mu_grass, k=k_grass, Omega_diag_chol=Omega_diag_chol, U=U, cov_jitter=1e-4)
    Deltas = grass_gp.tangent_model(s)
    Ws = numpyro.deterministic("Ws", convert_to_projs(Deltas, U, reorthonormalize=False))
    X_projs = np.einsum('ij,ljk->lik', X, Ws)
    X_projs_all = np.vstack([X_projs[i,:,:] for i in range(N)])
    vec_Ys = vec(Ys)
    
    outer_var = numpyro.sample("outer_var", dist.LogNormal(0.0, 10.0))
    outer_length = numpyro.sample("outer_length", dist.LogNormal(0.0, 10.0))
    outer_noise = numpyro.sample("outer_noise", dist.LogNormal(0.0, 10.0))
    outer_kernel_params = {'var': outer_var, 'length': outer_length, 'noise': outer_noise}
    K_outer = rbf(X_projs_all, X_projs_all, outer_kernel_params, jitter=1e-06)
    
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=np.zeros(K_outer.shape[0]), covariance_matrix=K_outer),
        obs=vec_Ys,
    )


# %%
# numpyro.render_model(model, model_args=(s_train,X_train,Ys_train))

# %%
# run SVI to get MAP estimate to initialise MCMC
svi_key = random.PRNGKey(68569)
maxiter = 10000
step_size = 0.001
print("Running SVI for MAP estimate to initialise MCMC")
svi_results = run_svi_for_map(svi_key, model, maxiter, step_size, s_train, X_train, Ys_train)

# %%
# plt.plot(svi_results.losses)
# plt.show()

# %%
# get initialisation from SVI results
map_est = svi_results.params
strip_val = len('_auto_loc')
init_values = {key[:-strip_val]:value for (key, value) in map_est.items()}

# %%
train_key = random.PRNGKey(234265)
mcmc_config = {'num_warmup' : 1000, 'num_samples' : 1000, 'num_chains' : 1, 'thinning' : 2, 'init_strategy' : init_to_value(values=init_values)}
print("HMC starting.")
mcmc = run_inference(train_key, mcmc_config, model, s_train, X_train, Ys_train)
original_stdout = sys.stdout
with open('hmc_log.txt', 'w') as f:
    sys.stdout = f
    mcmc.print_summary()
    sys.stdout = original_stdout

# %%
samples = mcmc.get_samples()
inference_data = samples.copy()
for param, initial_val in init_values.items():
    inference_data[f"{param}-initial_value"] = initial_val

# %%
head = os.getcwd()
inference_main_name = "inference_data"
inference_path = get_save_path(head, inference_main_name)
try:
    safe_save(inference_path, inference_data)
except FileExistsError:
    print("File exists so not saving.")

# %%
