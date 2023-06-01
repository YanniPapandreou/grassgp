# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import time
from hydra_zen import instantiate, make_config, builds, to_yaml, load_from_yaml, launch
import sys
import pickle

from pathlib import Path

import jax.numpy as np
from jax import vmap, random, grad, jit
import jax.numpy.linalg as lin
from functools import partial
from tqdm import tqdm

from grassgp.grassmann import valid_grass_point, sample_karcher_mean, grass_dist, grass_exp, valid_grass_tangent, grass_log, convert_to_projs, compute_barycenter
from grassgp.means import zero_mean
from grassgp.kernels import rbf
from grassgp.models_optimised import GrassGP
from grassgp.plot_utils import flatten_samples, plot_grass_dists, mat_heatmap
from grassgp.utils import to_dictconf, get_save_path, vec, unvec
from grassgp.utils import safe_save_jax_array_dict as safe_save
from grassgp.utils import load_and_convert_to_samples_dict as load_data

import chex
from chex import assert_shape, assert_rank
from dataclasses import dataclass, field
from typing import Tuple, Union, Callable, Tuple

import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_value
import numpyro.distributions as dist

from grassgp.inference import run_inference

import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,6)


# %%
def print_file(x: Path):
    with x.open("r") as f:
        print(f.read())


# %%
def pickle_save(obj, name: str):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


# %%
def pickle_load(name: str):
    with open(name, 'rb') as f:
        obj = pickle.load(f)
    return obj


# %%
base_path = Path(os.getcwd())
output_folder = "multirun"
date = "2023-05-24"
time = "10-50-48"
id_num = "3"

# %%
job_path = base_path / output_folder / date / time / id_num
assert job_path.exists()

# %%
config = load_from_yaml(job_path / ".hydra" / "config.yaml")
overrides = load_from_yaml(job_path / ".hydra" / "overrides.yaml")

# %%
print("Overrides used for training:")
print(to_yaml(overrides))

# %%
dataset_path = job_path / "training_test_data.pickle"
assert dataset_path.exists()
training_test_data = pickle_load(dataset_path)

s_train = np.array(training_test_data['s_train'])
s_test = np.array(training_test_data['s_test'])

Ws_train = np.array(training_test_data['Ws_train'])
Ws_test = np.array(training_test_data['Ws_test'])

log_Ws_train = np.array(training_test_data['log_Ws_train'])
log_Ws_test = np.array(training_test_data['log_Ws_test'])

anchor_point = np.array(training_test_data['anchor_point'])

d, n = anchor_point.shape

# %%
# print_file(job_path / "hmc_log.txt")

# %%
inference_data_path = job_path / "inference_data.pickle"
assert inference_data_path.exists()
inference_data = pickle_load(inference_data_path)
samples = dict(filter(lambda elem: 'initial_value' not in elem[0], inference_data.items()))
initial_values = dict(filter(lambda elem: 'initial_value' in elem[0], inference_data.items()))
assert set(samples.keys()).union(initial_values.keys()) == set(inference_data.keys())

# %%
Deltas_means = pickle_load(job_path / "Deltas_means.pickle")
Deltas_preds = pickle_load(job_path / "Deltas_preds.pickle")

# %%
# i=0
# percentile_levels = [2.5, 97.5]
# conf_level = percentile_levels[-1] - percentile_levels[0]
# in_preds = samples['Deltas'][:,:,i,0]
# percentiles = np.percentile(in_preds, np.array(percentile_levels), axis=0)
# lower = percentiles[0,:]
# upper = percentiles[1,:]
# plt.plot(s_test, log_Ws_test[:,i,0], label='full data',c='black', alpha=0.75, linestyle='dashed')
# plt.scatter(s_train, log_Ws_train[:,i,0], label='training data', c='g')
# plt.plot(s_train, samples['Deltas'].mean(axis=0)[:,i,0],c='red')
# plt.fill_between(s_train, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
# plt.show()

# %%
# i=0
# percentile_levels = [2.5, 97.5]
# conf_level = percentile_levels[-1] - percentile_levels[0]
# in_preds = samples['Deltas'][:,:,i,0]
# percentiles = np.percentile(in_preds, np.array(percentile_levels), axis=0)
# lower = percentiles[0,:]
# upper = percentiles[1,:]
# plt.plot(s_test, log_Ws_test[:,i,0], label='full data',c='black', alpha=0.75, linestyle='dashed')
# plt.scatter(s_train, log_Ws_train[:,i,0], label='training data', c='g')
# plt.plot(s_train, samples['Deltas'].mean(axis=0)[:,i,0],c='red',label='in-mean')
# plt.fill_between(s_train, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')

# means = Deltas_means[:,:,i,0]
# means_avg = np.mean(means, axis=0)
# preds = Deltas_preds[:,:,i,0]
# out_percentiles = np.percentile(preds, np.array(percentile_levels), axis=0)
# out_lower = out_percentiles[0,:]
# out_upper = out_percentiles[1,:]
# plt.plot(s_test, means_avg, label='averaged mean prediction (out)', c='purple', alpha=0.75)
# plt.fill_between(s_test, out_lower, out_upper, color='coral', alpha=0.5, label=f'{conf_level}% credible interval')
# plt.xlabel(r"$s$")
# plt.legend()
# plt.vlines(s_train, 0.99*lower.min(), 1.01*upper.max(), colors='green', linestyles='dashed')
# # plt.title(f"{i+1}th component of tangents")
# plt.show()

# %%
# plt.rcParams["figure.figsize"] = (12,6)
# percentile_levels = [2.5, 97.5]
# conf_level = percentile_levels[-1] - percentile_levels[0]
# for i in range(5):
#     obs = log_Ws_train[:,i,0]
#     means = Deltas_means[:,:,i,0]
#     means_avg = np.mean(means, axis=0)
#     preds = Deltas_preds[:,:,i,0]
#     percentiles = np.percentile(preds, np.array(percentile_levels), axis=0)
#     lower = percentiles[0,:]
#     upper = percentiles[1,:]
#     plt.plot(s_test, log_Ws_test[:,i,0], label='full data',c='black', alpha=0.75, linestyle='dashed')
#     plt.scatter(s_train, log_Ws_train[:,i,0], label='training data', c='g')
#     plt.plot(s_test, means_avg, label='averaged mean prediction', c='r', alpha=0.75)
#     plt.fill_between(s_test, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
#     plt.xlabel(r"$s$")
#     plt.legend()
#     plt.vlines(s_train, 0.99*lower.min(), 1.01*upper.max(), colors='green', linestyles='dashed')
#     plt.title(f"{i+1}th component of tangents")
#     # plt.ylim((-0.5,0.5))
#     plt.show()

# %%
# N_test = len(s_test)
# N_train = len(s_train)

# %%
# Omega_diag_chol = samples['Omega_diag_chol'][0]
# var = 1.0
# noise = 0.0
# length = float(samples['kernel_length'][0])
# # length = 10
# print(length)
# Omega = np.diag(Omega_diag_chol**2)
# U = anchor_point
# I_UUT = np.eye(d) - U@U.T
# Omega_X = I_UUT @ Omega @ I_UUT.T

# kernel_params = {'var': var, 'length': length, 'noise': noise}

# k = lambda t,s: rbf(t, s, kernel_params)

# %%
# K_test_train = k(s_test,s_train)
# assert_shape(K_test_train, (N_test, N_train))

# K_train_train = k(s_train, s_train)
# assert_shape(K_train_train, (N_train, N_train))

# K_train_test = k(s_train, s_test)
# assert_shape(K_train_test, (N_train, N_test))

# %%
# sigma = config.model.ell

# %%
# B = np.kron(K_train_train, Omega_X) + ((sigma**2) * np.eye(N_train*n*d))

# %%
# B_inv = lin.inv(B)

# %%
# B_pinv = lin.pinv(B)

# lin.norm(B @ B_inv - np.eye(N_train*n*d)) / lin.norm(np.eye(N_train*d*n))

# lin.norm(B_inv @ B - np.eye(N_train*n*d)) / lin.norm(np.eye(N_train*d*n))

# lin.norm(B @ B_pinv @ B - B) / lin.norm(B)

# lin.norm(B_pinv @ B @ B_pinv - B_pinv) / lin.norm(B_pinv)

# A = np.kron(K_test_train, Omega_X)

# v = vec(log_Ws_train)

# M = A @ B_inv @ v

# B.shape

# post_mean = vmap(lambda params: unvec(params, d, n))(np.array(M.split(N_test)))

# %%
# post_mean.shape

# %%
# plt.rcParams["figure.figsize"] = (12,6)
# for i in range(d):
#     plt.plot(s_test, log_Ws_test[:,i,0], label='full data',c='black', alpha=0.25)
#     plt.scatter(s_train, log_Ws_train[:,i,0], label='training data', c='g')
#     # plt.fill_between(s_test, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
#     plt.xlabel(r"$s$")
#     # plt.legend()
#     # plt.vlines(s_train, 0.99*lower.min(), 1.01*upper.max(), colors='green', linestyles='dashed')
#     # plt.title(f"{i+1}th component of tangents")
#     # plt.ylim((-0.5,0.5))
# plt.show()

# %%
N_test = len(s_test)
N_train = len(s_train)
v = vec(log_Ws_train)


# %%
@jit
def predict(key, Omega_diag_chol, var, noise, length, U, ell):
    Omega = np.diag(Omega_diag_chol**2)
    U = anchor_point
    I_UUT = np.eye(d) - U@U.T
    Omega_X = I_UUT @ Omega @ I_UUT.T

    kernel_params = {'var': var, 'length': length, 'noise': noise}

    k = lambda t,s: rbf(t, s, kernel_params, jitter=1e-10)
    
    K_test_train = k(s_test,s_train)
    # assert_shape(K_test_train, (N_test, N_train))

    K_train_train = k(s_train, s_train)
    # assert_shape(K_train_train, (N_train, N_train))

    K_train_test = k(s_train, s_test)
    # assert_shape(K_train_test, (N_train, N_test))
    
    K_test_test = k(s_test, s_test)
    # assert_shape(K_test_test, (N_test, N_test))
    
    B = np.kron(K_train_train, Omega_X) + ((ell**2) * np.eye(N_train*n*d))
    
    B_inv = lin.inv(B)
    
    A = np.kron(K_test_train, Omega_X)
    
    M = A @ B_inv @ v
    
    S = np.kron(K_test_test, Omega_X) - (A @ B_inv @ (A.T))
    
    chol_S = lin.cholesky(S)
    
    post_mean = vmap(lambda params: unvec(params, d, n))(np.array(M.split(N_test)))
    
    Z = dist.MultivariateNormal(loc=np.zeros_like(M), covariance_matrix=np.eye(len(M))).sample(key)
    
    vec_pred = M + chol_S @ Z
    
    post_pred = vmap(lambda params: unvec(params, d, n))(np.array(vec_pred.split(N_test)))
    
    return post_mean, post_pred


# %%
key = random.PRNGKey(275437)
Omega_diag_chol = samples['Omega_diag_chol'][0]
var = 1.0
noise = 0.0
length = float(samples['kernel_length'][0])
U = anchor_point.copy()
ell = config.model.ell
post_mean, post_pred = predict(key, Omega_diag_chol, var, noise, length, U, ell)

# %%
var = 1.0
noise = 0.0
U = anchor_point.copy()
ell = config.model.ell

post_means = []
post_preds = []
key = random.PRNGKey(34563)
keys = random.split(key,3500)
for i in tqdm(range(3500)):
    Omega_diag_chol = samples['Omega_diag_chol'][i]
    length = float(samples['kernel_length'][i])
    post_mean, post_pred = predict(key, Omega_diag_chol, var, noise, length, U, ell)
    post_means.append(post_mean)
    post_preds.append(post_pred)

post_means = np.array(post_means)
post_preds = np.array(post_preds)

# %%
pickle_save(post_means, "debugging_preds_means.pickle")
pickle_save(post_preds, "debugging_preds_preds.pickle")
