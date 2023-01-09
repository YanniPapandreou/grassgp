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
from jax import random, vmap

from grassgp.grassmann import valid_grass_point, convert_to_projs
from grassgp.utils import subspace_angle
from grassgp.utils import load_and_convert_to_samples_dict as load_data
from grassgp.kernels import rbf
from grassgp.plot_utils import flatten_samples, plot_grass_dists, plot_AS_dir_preds
from grassgp.models import GrassGP
from grassgp.means import zero_mean

import numpyro

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
date = "2022-12-12"
time = "16-44-29"
tol = 1e-05
id_num = "1"

# %%
# job path
job_path = base_path / output_folder / date / time / id_num
assert job_path.exists()

# %%
# load training and test data
dataset_path = base_path / "datasets" / "training_test_data_gpsr_example.npz"
assert dataset_path.exists()

training_test_data = load_data(str(dataset_path))

s_train = np.array(training_test_data['s_train'])
s_test = np.array(training_test_data['s_test'])

Ws_train = np.array(training_test_data['Ws_train'])
Ws_test = np.array(training_test_data['Ws_test'])

log_Ws_train = np.array(training_test_data['log_Ws_train'])
log_Ws_test = np.array(training_test_data['log_Ws_test'])

anchor_point = np.array(training_test_data['anchor_point'])

# %%
d, n = anchor_point.shape

# plot dataset
for i in range(d):
    plt.plot(s_test, Ws_test[:,i,0])
    plt.scatter(s_train, Ws_train[:,i,0], c='r')
    plt.title(f'{i+1}th component of projection')
    plt.grid()
    plt.xlabel(r'$s$')
    plt.show()

# %%
# load config in
config = load_from_yaml(job_path / ".hydra" / "config.yaml")
overrides = load_from_yaml(job_path / ".hydra" / "overrides.yaml")

print("Overrides used for training:")
print(to_yaml(overrides))

# %%
# get inference data
inference_path = job_path / "inference_data.npz"
assert inference_path.exists()
inference_data = load_data(str(inference_path))

# %%
samples = dict(filter(lambda elem: 'initial_value' not in elem[0], inference_data.items()))
initial_values = dict(filter(lambda elem: 'initial_value' in elem[0], inference_data.items()))
assert set(samples.keys()).union(initial_values.keys()) == set(inference_data.keys())

# %%
# numpyro.render_model(instantiate(config.model), model_args=(s_train,log_Ws_train))

# %%
print_file(job_path / "hmc_log.txt")

# %%
my_samples = flatten_samples(samples, ignore=[])

# %%
trace_plot_vars = ['kernel_var']
for key in my_samples.keys():
    if 'Omega' in key:
        trace_plot_vars.append(key)

my_samples[trace_plot_vars].plot(subplots=True, figsize=(10,40), sharey=False)
plt.show()

# %%
samples_Ws_train = vmap(lambda Deltas: convert_to_projs(Deltas, anchor_point, reorthonormalize=False))(samples['Deltas'])

for ws in samples_Ws_train:
    assert vmap(lambda w: valid_grass_point(w, tol=tol))(ws).all()

# %% tags=[]
plot_grass_dists(samples_Ws_train, Ws_train, s_train)

# %%
alphas = np.array([subspace_angle(w) for w in Ws_test])
alphas_train = np.array([subspace_angle(w) for w in Ws_train])
samples_alphas_train = np.array([[subspace_angle(w)for w in Ws_sample] for Ws_sample in samples_Ws_train])

# %%
percentile_levels = [2.5, 97.5]
conf_level = percentile_levels[-1] - percentile_levels[0]
percentiles = np.percentile(samples_alphas_train, np.array(percentile_levels), axis=0)
lower = percentiles[0,:]
upper = percentiles[1,:]

# %%
plt.plot(s_test, alphas, c='black', alpha=0.5, label='full data')
plt.scatter(s_train, alphas_train, label='training data', c='g')
plt.scatter(s_train, samples_alphas_train.mean(axis=0), label='mean samples', c='r')
plt.fill_between(s_train, lower, upper,  color='lightblue', alpha=0.75,label=f'{conf_level}% credible interval')
plt.xlabel(r"$s$")
plt.ylabel("subspace angle")
plt.legend()
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
plt.rcParams["figure.figsize"] = (12,6)
percentile_levels = [2.5, 97.5]
conf_level = percentile_levels[-1] - percentile_levels[0]
for i in range(d):
    obs = log_Ws_train[:,i,0]
    means = Deltas_means[:,:,i,0]
    means_avg = np.mean(means, axis=0)
    preds = Deltas_preds[:,:,i,0]
    percentiles = np.percentile(preds, np.array(percentile_levels), axis=0)
    lower = percentiles[0,:]
    upper = percentiles[1,:]
    plt.plot(s_test, log_Ws_test[:,i,0], label='full data',c='black', alpha=0.75, linestyle='dashed')
    plt.scatter(s_train, log_Ws_train[:,i,0], label='training data', c='g')
    plt.plot(s_test, means_avg, label='averaged mean prediction', c='r', alpha=0.75)
    plt.fill_between(s_test, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
    plt.xlabel(r"$s$")
    plt.legend()
    plt.vlines(s_train, -1, 1, colors='green', linestyles='dashed')
    plt.title(f"{i+1}th component of tangents")
    plt.show()


# %%
def predict_grass(
    key: chex.ArrayDevice,
    s_test: chex.ArrayDevice,
    s_train: chex.ArrayDevice,
    Vs_train: chex.ArrayDevice,
    dict_cfg,
    samples: dict,
    jitter: float = 1e-8,
    reortho: bool = False
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
        Ws_mean, Ws_pred = grass_gp.predict_grass(key, s_test, s_train, Vs_train, jitter=jitter, reortho=reortho)
        return Ws_mean, Ws_pred

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
    Ws_means, Ws_preds = vmap(predict)(*vmap_args)
    return Ws_means, Ws_preds


# %%
pred_key_grass = random.PRNGKey(7695)
Ws_means, Ws_preds = predict_grass(pred_key, s_test, s_train, log_Ws_train, config, samples)
assert np.isnan(Ws_means).sum() == 0
assert np.isnan(Ws_preds).sum() == 0

# %%
plt.rcParams["figure.figsize"] = (12,6)
percentile_levels = [2.5, 97.5]
conf_level = percentile_levels[-1] - percentile_levels[0]
for i in range(d):
    obs = Ws_train[:,i,0]
    means = Ws_means[:,:,i,0]
    means_avg = np.mean(means, axis=0)
    preds = Ws_preds[:,:,i,0]
    percentiles = np.percentile(preds, np.array(percentile_levels), axis=0)
    lower = percentiles[0,:]
    upper = percentiles[1,:]
    plt.plot(s_test, Ws_test[:,i,0], label='full data',c='black', alpha=0.75, linestyle='dashed')
    plt.scatter(s_train, Ws_train[:,i,0], label='training data', c='g')
    plt.plot(s_test, means_avg, label='averaged mean prediction', c='r', alpha=0.75)
    plt.fill_between(s_test, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
    plt.xlabel(r"$s$")
    plt.legend()
    plt.vlines(s_train, -1, 1, colors='green', linestyles='dashed')
    plt.title(f"{i+1}th component of projections")
    plt.show()

# %%
alphas_means = np.array([[subspace_angle(w) for w in mean] for mean in Ws_means])
alphas_preds = np.array([[subspace_angle(w) for w in pred] for pred in Ws_preds])

# %%
plt.rcParams["figure.figsize"] = (12,6)
percentile_levels = [2.5, 97.5]
conf_level = percentile_levels[-1] - percentile_levels[0]
alphas_means_avg = np.mean(alphas_means, axis=0)
percentiles = np.percentile(alphas_preds, np.array(percentile_levels), axis=0)
lower = percentiles[0,:]
upper = percentiles[1,:]
plt.plot(s_test, alphas, label='full data',c='black', alpha=0.75, linestyle='dashed')
plt.scatter(s_train, alphas_train, label='training data', c='g')
plt.plot(s_test, alphas_means_avg, label='averaged mean prediction', c='r', alpha=0.75)
plt.fill_between(s_test, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
plt.xlabel(r"$s$")
plt.ylabel("subspace angle")
plt.legend()
plt.vlines(s_train, 0, np.pi, colors='green', linestyles='dashed')
plt.title(f"predictions for subspace angles")
plt.show()

# %%
plot_AS_dir_preds(Ws_preds, Ws_test, s_test, s_train)

# %%
plot_grass_dists(Ws_preds, Ws_test, s_test)

# %%
