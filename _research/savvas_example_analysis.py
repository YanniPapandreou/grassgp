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
import time
from hydra_zen import instantiate, load_from_yaml, to_yaml
from pathlib import Path
import os

import chex
from typing import Tuple

import jax.numpy as np
from jax import random, vmap, jit

from grassgp.grassmann import valid_grass_point, convert_to_projs, compute_barycenter, grass_dist, rand_grass_point
from grassgp.utils import to_dictconf
from grassgp.utils import load_and_convert_to_samples_dict as load_data
from grassgp.kernels import rbf
from grassgp.plot_utils import flatten_samples, plot_grass_dists, plot_AS_dir_preds
from grassgp.models import GrassGP
from grassgp.means import zero_mean

import numpyro

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.rcParams["figure.figsize"] = (10,6)


# %%
def print_file(x: Path):
    with x.open("r") as f:
        print(f.read())


# %% tags=["parameters"]
base_path = Path(os.getcwd())
output_folder = "outputs"
date = "2023-01-09"
time = "17-45-42"
# tol = 1e-05

# %%
# job path
job_path = base_path / output_folder / date / time
assert job_path.exists()

# %%
# load training and test data
dataset_path = job_path  / "training_test_data.npz"
assert dataset_path.exists()

training_test_data = load_data(str(dataset_path))

s_train = np.array(training_test_data['s_train'])
s_test = np.array(training_test_data['s_test'])

Ws_train = np.array(training_test_data['Ws_train'])
Ws_test = np.array(training_test_data['Ws_test'])

log_Ws_train = np.array(training_test_data['log_Ws_train'])
log_Ws_test = np.array(training_test_data['log_Ws_test'])

anchor_point = np.array(training_test_data['anchor_point'])

d, n = anchor_point.shape

# %%
plt.scatter(s_train[:,0],s_train[:,1], label='training locs')
plt.scatter(s_test[:,0],s_test[:,1], label='test locs')
plt.legend()
plt.show()

# %%
# load config in
config = load_from_yaml(job_path / ".hydra" / "config.yaml")
print("Config used for training:")
print(to_yaml(config))

# %%
# get inference data
inference_path = job_path / "inference_data.npz"
assert inference_path.exists()
inference_data = load_data(str(inference_path))

inference_data = {k:np.array(v) for k, v in inference_data.items()}

# %%
samples = dict(filter(lambda elem: 'initial_value' not in elem[0], inference_data.items()))
initial_values = dict(filter(lambda elem: 'initial_value' in elem[0], inference_data.items()))
assert set(samples.keys()).union(initial_values.keys()) == set(inference_data.keys())

# %% tags=[]
print_file(job_path / "hmc_log.txt")

# %%
my_samples = flatten_samples(samples, ignore=[])

# %%
trace_plot_vars = ['kernel_length']
for key in my_samples.keys():
    if 'sigmas' in key:
        trace_plot_vars.append(key)

my_samples[trace_plot_vars].plot(subplots=True, figsize=(10,40), sharey=False)
plt.show()

# %%
tol=1e-5

samples_Ws_train = vmap(lambda Deltas: convert_to_projs(Deltas, anchor_point, reorthonormalize=False))(samples['Deltas'])

for ws in samples_Ws_train:
    assert vmap(lambda w: valid_grass_point(w, tol=tol))(ws).all()

# %% tags=[]
# plot_grass_dists(samples_Ws_train, Ws_train, s_train)

# %%
# from rieoptax.core import rgrad, ManifoldArray
# from rieoptax.geometry.grassmann import GrassmannCanonical
# from rieoptax.optimizers.first_order import rsgd
# from rieoptax.optimizers.update import apply_updates
# from timeit import default_timer as timer

# def compute_barycenter_jax(Ws, seed=326234, epochs=200, lr=1e-3):
#     def fit(params, data, optimizer, epochs):
#         @jit
#         def step(params, opt_state, data):

#             def cost(params, data):
#                 def _cost(params, data):
#                     return grass_dist(data, params.value)**2
#                 return vmap(_cost, in_axes=(None,0))(params, data).sum()

#             rgrads = rgrad(cost)(params, data) # calculates Riemannian gradients
#             updates, opt_state = optimizer.update(rgrads, opt_state, params)
#             params = apply_updates(params, updates) # Update using Riemannian Exp
#             loss_value = cost(params, data)
#             return params, opt_state, loss_value

#         opt_state = optimizer.init(params)
#         rieoptax_loss = []
#         rieoptax_time = []
#         start = timer()
#         for i in range(epochs):
#             params, opt_state, loss_value = step(params, opt_state, data)
#             # print(f"{loss_value=}")
#             rieoptax_loss.append(loss_value)

#         return params, rieoptax_loss
    
#     d, n = Ws[0].shape
#     key_init = random.PRNGKey(seed)
#     jnp_init = rand_grass_point(key_init, d, n)
#     U_init = ManifoldArray(value=jnp_init, manifold=GrassmannCanonical(d,n))
#     # print(f"{U_init=}")
#     optimizer = rsgd(lr)
#     params, rieoptax_loss = fit(U_init, data, optimizer, epochs)
#     return params.value

# data = samples_Ws_train[:,0,:,:]
# center = compute_barycenter_jax(data)

# barycenter = compute_barycenter(data)

# grass_dist(center, barycenter)

# %%
mcmc_centers = vmap(compute_barycenter_jax,in_axes=1)(samples_Ws_train)

# %%
mcmc_barycenters = []
for i in tqdm(range(s_train.shape[0])):
    barycenter = compute_barycenter(samples_Ws_train[:,i,:,:])
    mcmc_barycenters.append(barycenter)

# %%
mcmc_barycenters = np.array(mcmc_barycenters)

# %%
barycenter_save_path = str(job_path) + "/mcmc_barycenters.npz"
if os.path.exists(barycenter_save_path):
    print("File exists so not saving")
else:
    np.savez(barycenter_save_path, mcmc_barycenters)

# %%
in_sample_errors = vmap(grass_dist)(Ws_train, mcmc_barycenters)

# %%
plt.plot(in_sample_errors)
plt.show()

# %%
pd_data = {'x': s_train[:,0], 'y': s_train[:,1], 'errors': in_sample_errors}
in_sample_errors_df = pd.DataFrame(data=pd_data)
in_sample_errors_df.head()

# %%
sns.scatterplot(data=in_sample_errors_df, x="x", y="y", size="errors", legend=False, alpha=0.75)
plt.title("In sample errors")
plt.show()


# %%
def predict_tangents(
    key: chex.ArrayDevice,
    s_test: chex.ArrayDevice,
    s_train: chex.ArrayDevice,
    Vs_train: chex.ArrayDevice,
    dict_cfg,
    samples: dict,
    jitter: float = 1e-8
) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
    
    d_in = dict_cfg.model.grass_config.d_in
    U = np.array(dict_cfg.model.grass_config.anchor_point)
    d, n = U.shape
    cov_jitter = dict_cfg.model.grass_config.cov_jitter
    k_include_noise = dict_cfg.model.grass_config.k_include_noise
    kern_jitter = dict_cfg.model.grass_config.jitter
    n_samples = dict_cfg.train.n_samples // dict_cfg.train.n_thinning
    assert n_samples == samples['Deltas'].shape[0]
    
    def predict(
        key: chex.ArrayDevice,
        Omega: chex.ArrayDevice,
        var: float,
        length: float,
        noise: float,
    ) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        # iniatilize GrassGP
        kernel_params = {'var': var, 'length': length, 'noise': noise}
        k = lambda t, s: rbf(t, s, kernel_params, jitter=kern_jitter, include_noise=k_include_noise)
        mu = lambda s: zero_mean(s, d, n)
        grass_gp = GrassGP(d_in=d_in, d_out=(d, n), mu=mu, k=k, Omega=Omega, U=U, cov_jitter=cov_jitter)

        # predict
        Deltas_mean, Deltas_pred = grass_gp.predict_tangents(key, s_test, s_train, Vs_train, jitter=jitter)
        return Deltas_mean, Deltas_pred

    # initialize vmap args
    vmap_args = (random.split(key, n_samples),)
    
    cfg_Omega = dict_cfg.model.grass_config.Omega
    cfg_var = dict_cfg.model.grass_config.var
    cfg_length = dict_cfg.model.grass_config.length
    cfg_noise = dict_cfg.model.grass_config.noise
    cfg_require_noise = dict_cfg.model.grass_config.require_noise
    
    if cfg_Omega is None:
        vmap_args += (samples['Omega'],)
    else:
        cfg_Omega = np.array(cfg_Omega)
        vmap_args += (np.repeat(cfg_Omega[None,:,:], n_samples, axis=0),)
    
    if cfg_var is None:
        vmap_args += (samples['kernel_var'],)
    else:
        vmap_args += (cfg_var * np.ones(n_samples),)
        
    if cfg_length is None:
        vmap_args += (samples['kernel_length'],)
    else:
        vmap_args += (cfg_length * np.ones(n_samples),)
        
    if cfg_require_noise:
        if cfg_noise is None:
            vmap_args += (samples['kernel_noise'],)
        else:
            vmap_args += (cfg_noise * np.ones(n_samples),)
    else:
        vmap_args += (np.zeros(n_samples),)
    
    assert len(vmap_args) == 5
    Deltas_means, Deltas_preds = vmap(predict)(*vmap_args)
    return Deltas_means, Deltas_preds


# %%
pred_key = random.PRNGKey(6578)
Deltas_means, Deltas_preds = predict_tangents(pred_key, s_test, s_train, log_Ws_train, config, samples)
assert np.isnan(Deltas_means).sum() == 0
assert np.isnan(Deltas_preds).sum() == 0

# %%
Ws_test_means = vmap(lambda Deltas: convert_to_projs(Deltas, anchor_point, reorthonormalize=False))(Deltas_means)

for ws in Ws_test_means:
    assert vmap(lambda w: valid_grass_point(w, tol=tol))(ws).all()

# %%
Ws_test_means.shape

# %%
test_means_mcmc_barycenters = []
for i in tqdm(range(s_test.shape[0])):
    barycenter = compute_barycenter(Ws_test_means[:,i,:,:])
    test_means_mcmc_barycenters.append(barycenter)

# %%
test_means_mcmc_barycenters = np.array(test_means_mcmc_barycenters)

# %%
test_means_mcmc_barycenter_save_path = str(job_path) + "/test_means_mcmc_barycenters.npz"
if os.path.exists(test_means_mcmc_barycenter_save_path):
    print("File exists so not saving")
else:
    np.savez(test_means_mcmc_barycenter_save_path, test_means_mcmc_barycenters)

# %%
out_sample_errors = vmap(grass_dist)(Ws_test, test_means_mcmc_barycenters)

# %%
plt.plot(out_sample_errors)
plt.show()

# %%
test_pd_data = {'x': s_test[:,0], 'y': s_test[:,1], 'errors': out_sample_errors}
out_sample_errors_df = pd.DataFrame(data=test_pd_data)
out_sample_errors_df.head()

# %%
sns.scatterplot(data=out_sample_errors_df, x="x", y="y", size="errors", legend=False, alpha=0.75)
plt.title("out sample errors")
plt.show()

# %%
sns.scatterplot(data=in_sample_errors_df, x="x", y="y", size="errors", legend=False, label='in sample', alpha=0.75, c='blue')
sns.scatterplot(data=out_sample_errors_df, x="x", y="y", size="errors", legend=False, label='out sample', alpha=0.75, c='red')
plt.title("in and out of sample errors")
plt.legend()
plt.show()

# %%
