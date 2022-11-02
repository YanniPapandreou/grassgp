# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: grassgp
#     language: python
#     name: grassgp
# ---

# %%
# # %load_ext autoreload
# # %autoreload 2

# %%
import time
from hydra_zen import instantiate, make_config, to_yaml, load_from_yaml
from pathlib import Path
import os
import jax.numpy as np
from jax import random, vmap

from grassgp.utils import load_and_convert_to_samples_dict as load_data
from grassgp.plot_utils import flatten_samples, traceplots, plot_projected_data
from grassgp.prediction import run_prediction_at_train_times

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,6)


# %%
def print_file(x: Path):
    with x.open("r") as f:
        print(f.read())


# %% tags=["parameters"]
# Job Parameters
base_path = Path(os.getcwd())
output_folder = "multirun"
date = "2022-10-31"
time = "17-36-39"
id_num = "0"

# %%
job_path = base_path / output_folder / date / time / id_num
assert job_path.exists()

# load config in
config = load_from_yaml(job_path / ".hydra" / "config.yaml")
overrides = load_from_yaml(job_path / ".hydra" / "overrides.yaml")
# print overrides and config
print("Config used for training:")
print(to_yaml(config))
print("Overrides used for training:")
print(to_yaml(overrides))

# %%
# get data and training data
dataset_path = Path(config.dataset_path) / "dataset.npz"
# print config and overrides used to generate full dataset

dataset_config_path = Path(config.dataset_path) / ".hydra" / "config.yaml"
assert dataset_config_path.exists()
print("Config used to generate dataset:")
print(to_yaml(load_from_yaml(dataset_config_path)))
dataset_overrides_path = Path(config.dataset_path) / ".hydra" / "overrides.yaml"
assert dataset_overrides_path.exists()
print("Overrides used to generate dataset:")
print(to_yaml(load_from_yaml(dataset_overrides_path)))
training_dataset_path = job_path / "training_data.npz"
assert dataset_path.exists()
assert training_dataset_path.exists()
data = load_data(dataset_path)
training_data = load_data(training_dataset_path)

# %%
# process data and training data
X_fine = np.array(data['X'])
s_fine = np.array(data['s'])
Ps_fine = np.array(data['Ps'])
Ys_fine = np.array(data['Ys'])
anchor_point = np.array(data['anchor_point'])
X_fine_projs = np.einsum('ij,ljk->lik', X_fine, Ps_fine)

X = np.array(training_data['X'])
s = np.array(training_data['s'])
Ps = np.array(training_data['Ps'])
Ys = np.array(training_data['Ys'])
X_projs = np.einsum('ij,ljk->lik', X, Ps)

# %%
# get inference data
inference_path = job_path / "inference_data.npz"
assert inference_path.exists()
inference_data = load_data(inference_path)

# %%
samples = dict(filter(lambda elem: 'initial_value' not in elem[0], inference_data.items()))
initial_values = dict(filter(lambda elem: 'initial_value' in elem[0], inference_data.items()))
assert set(samples.keys()).union(initial_values.keys()) == set(inference_data.keys())

# %%
my_samples = flatten_samples(samples)

# %% tags=[]
# traceplots(my_samples, a=0.75)

# %% tags=[]
# plot_projected_data(X_fine_projs, s_fine, Ys_fine)

# %% tags=[]
# plot_projected_data(X_projs, s, Ys)

# %% tags=[]
# plt.scatter(X_fine[:,0],X_fine[:,1],c='b',alpha=0.25,label='fine')
# plt.scatter(X[:,0],X[:,1],c='r',label='coarse')
# plt.legend()
# plt.show()

# %%
pred_key = random.PRNGKey(3276359)
means, predictions = run_prediction_at_train_times(pred_key, X, X, s, Ys, config, samples)

# %%
from grassgp.plot_utils import plot_preds_train_locs

# %%
plot_preds_train_locs(means, predictions, X, X, s, Ys, Ps, [2.5,97.5], fig_size=(12,8))

# %%
from grassgp.kernels import rbf
from grassgp.grassmann import convert_to_projs
from grassgp.utils import kron_solve, unvec
import numpyro.distributions as dist


# %%
def grass_predict(key, s_test, s, anchor_point, Omega, var, length, noise, proj_params, reortho, jitter=5e-4):
    D, n = anchor_point.shape
    n_train = s.shape[0]
    n_test = s_test.shape[0]
    
    # compute (temporal) kernels between train and test locs
    grass_kernel_params = {'var': var, 'length': length, 'noise': noise}
    T_K_pp = rbf(s_test, s_test, grass_kernel_params, include_noise=False)
    T_K_pt = rbf(s_test, s, grass_kernel_params, include_noise=False)
    T_K_tt = rbf(s, s, grass_kernel_params)
    
    # form M_covs between train and test locs
    M_cov_pp = np.kron(T_K_pp, Omega)
    M_cov_pt = np.kron(T_K_pt, Omega)
    
    # add jitter to M_cov_tt, M_cov_pp
    M_cov_pp += jitter * np.eye(M_cov_pp.shape[0])
    
    # Get posterior cov for grass part    
    T_K = M_cov_pp - np.matmul(M_cov_pt, vmap(lambda v: kron_solve(T_K_tt, Omega, v), in_axes=1, out_axes=1)(M_cov_pt.T))
    
    # Get posterior mean for grass part
    T_mean = np.matmul(M_cov_pt, kron_solve(T_K_tt, Omega, proj_params))
    
    # sample projection params for test locs
    T_sample = dist.MultivariateNormal(loc=T_mean, covariance_matrix=T_K).sample(key)
    
    
    # split each up into params for each time
    T_mean_split = np.array(T_mean.split(n_test))
    T_sample_split = np.array(T_sample.split(n_test))
    
    # unvec each
    unvec_T_mean = vmap(lambda params: unvec(params, D, n))(T_mean_split)
    unvec_T_sample = vmap(lambda params: unvec(params, D, n))(T_sample_split)
    
    # form projector
    I_UUT = (np.eye(D) - anchor_point @ anchor_point.T)
    
    # apply this to each
    Deltas_mean = np.einsum('ij,ljk->lik', I_UUT, unvec_T_mean)
    Deltas_sample = np.einsum('ij,ljk->lik', I_UUT, unvec_T_sample)
    
    # convert posterior means to projections
    Ps_mean = convert_to_projs(Deltas_mean, anchor_point, reorthonormalize=reortho)
    Ps_sample = convert_to_projs(Deltas_sample, anchor_point, reorthonormalize=reortho)
    
    # return Ps_mean and Ps_sample
    return Ps_mean, Ps_sample


# %%
def run_grass_predict(pred_key, s_test, s, cfg, samples:dict, jitter=5e-4):
    anchor_point = np.array(cfg.inner_model.grass_config.anchor_point)
    n_samples = cfg.train.n_samples // cfg.train.n_thinning
    proj_params_samples = samples['grass-proj_params']
    assert n_samples == proj_params_samples.shape[0]
    
    # initialize vmap_args
    vmap_args = (random.split(pred_key, n_samples), proj_params_samples)

    cfg_Omega = cfg.inner_model.grass_config.Omega
    cfg_var = cfg.inner_model.grass_config.var
    cfg_length = cfg.inner_model.grass_config.length
    cfg_noise = cfg.inner_model.grass_config.noise
    require_noise = cfg.inner_model.grass_config.require_noise
    reortho = cfg.inner_model.grass_config.reorthonormalize
    
    if cfg_Omega is None:
        vmap_args += (samples['grass-Omega'],)
    else:
        cfg_Omega = np.array(cfg_Omega)
        vmap_args += (np.repeat(cfg_Omega[None,:,:], n_samples, axis=0),)
        
    if cfg_var is None:
        vmap_args += (samples['grass-kernel_var'],)
    else:
        vmap_args += (cfg_var * np.ones(n_samples),)
        
    if cfg_length is None:
        vmap_args += (samples['grass-kernel_length'],)
    else:
        vmap_args += (cfg_length * np.ones(n_samples),)
    
    if require_noise:
        if cfg_noise is None:
            vmap_args += (samples['grass-kernel_noise'],)
        else:
            vmap_args += (cfg_noise * np.ones(n_samples),)
    else:
        vmap_args += (np.zeros(n_samples),)
        
    Ps_means, Ps_preds = vmap(lambda key, proj_params, Omega, var, length, noise: grass_predict(key, s_test, s, anchor_point, Omega, var, length, noise, proj_params, reortho, jitter=jitter))(*vmap_args)
    return Ps_means, Ps_preds


# %%
pred_key = random.PRNGKey(457657)
Ps_means, Ps_preds = run_grass_predict(pred_key, s_fine, s, config, samples)

# %%
from grassgp.plot_utils import plot_grass_preds, plot_grass_dists
plot_grass_preds(s, s_fine, Ps_means, Ps_preds, Ps_fine, [2.5, 97.5])

# %%
plot_grass_dists(samples['grass-Ps'], Ps, s)

# %%
