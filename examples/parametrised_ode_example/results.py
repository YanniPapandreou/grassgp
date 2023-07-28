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

# %% tags=[]
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
from grassgp.plot_utils import flatten_samples, plot_grass_dists
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


# %% tags=[]
def print_file(x: Path):
    with x.open("r") as f:
        print(f.read())


# %% tags=[]
def pickle_save(obj, name: str):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


# %% tags=[]
def pickle_load(name: str):
    with open(name, 'rb') as f:
        obj = pickle.load(f)
    return obj


# %% [markdown]
# # Load results and analyse

# %% tags=["parameters"]
base_path = Path(os.getcwd())
output_folder = "multirun"
date = "2023-06-02"
time = "10-55-33"
id_num = "0"

# %% tags=[]
job_path = base_path / output_folder / date / time / id_num
assert job_path.exists()

# %% tags=[]
config = load_from_yaml(job_path / ".hydra" / "config.yaml")
overrides = load_from_yaml(job_path / ".hydra" / "overrides.yaml")

# %% tags=[]
print("Overrides used for training:")
print(to_yaml(overrides))

# %% tags=[]
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
try:
    os.makedirs(job_path / "images")
    print("Creating image directory in job path")
except FileExistsError:
    print("Directory already exists; skipping.")


# %% tags=[]
def dataset_plot(s_train, s_test, Ws_train, Ws_test):
    W0 = Ws_test[0]
    # W0 = np.eye(100)[:,0][:,None]
    # W0 = rand_grass_point(random.PRNGKey(365),100,1)
    dists_test = vmap(lambda W: grass_dist(W, W0))(Ws_test)
    dists_train = vmap(lambda W: grass_dist(W, W0))(Ws_train)
    plt.plot(s_test,dists_test, label='test data')
    plt.scatter(s_train, dists_train, label='train data', c='r')
    plt.title(r"Grassmann distantance from $P(x_{0}^{*})$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$d_{\operatorname{GR}}(P(x),P(x_{0}^{*}))$")
    plt.legend()
    plt.grid()
    plt.savefig(job_path / 'images/parametrised-ode-dataset-plot.pdf',dpi=300,bbox_inches='tight',facecolor="w")
    plt.show()


# %% tags=[]
dataset_plot(s_train, s_test, Ws_train, Ws_test)

# %% tags=[]
print_file(job_path / "hmc_log.txt")

# %% tags=[]
inference_data_path = job_path / "inference_data.pickle"
assert inference_data_path.exists()
inference_data = pickle_load(inference_data_path)
samples = dict(filter(lambda elem: 'initial_value' not in elem[0], inference_data.items()))
initial_values = dict(filter(lambda elem: 'initial_value' in elem[0], inference_data.items()))
assert set(samples.keys()).union(initial_values.keys()) == set(inference_data.keys())


# %%
def diagnostic_plots(cfg):
    my_samples = flatten_samples(samples, ignore=[])
    trace_plot_vars = [f"Omega_diag_chol[{i}]" for i in range(4)]
    trace_plot_vars.insert(0,'kernel_length')
    
    fig, axs = plt.subplots(5,2,figsize=(20,30))
    samples_filtered = my_samples[trace_plot_vars]
    name_map = {
        "kernel_length": r"$\ell$",
        "Omega_diag_chol[0]": r"$\omega_{1}$",
        "Omega_diag_chol[1]": r"$\omega_{2}$",
        "Omega_diag_chol[2]": r"$\omega_{3}$",
        "Omega_diag_chol[3]": r"$\omega_{4}$",
    }
    for (i,var) in enumerate(trace_plot_vars):
        var_name = name_map[var]
        axs[i,0].plot(samples_filtered[var],c='k')
        axs[i,0].set_xlabel(r"$n$")
        axs[i,0].grid()
        axs[i,0].set_title(f"Traceplot for {var_name}")
        sm.graphics.tsa.plot_acf(samples_filtered[var], lags=cfg.plots.acf_lags, ax=axs[i,1], title=f"Autocorrelation for {var_name}")
        axs[i,1].set_xlabel("lag")
        axs[i,1].grid()

    fig.savefig(job_path / 'images/parametrised-ode-diagnostic-plots.pdf',dpi=300,bbox_inches='tight',facecolor="w")
    plt.show()    


# %% tags=[]
diagnostic_plots(config)

# %% tags=[]
in_sample_errors_df = pickle_load(job_path / "in_sample_errors_df.pickle")
in_sample_errors_df.describe()

# %% tags=[]
out_sample_errors_df = pickle_load(job_path / "out_sample_errors_df.pickle")
out_sample_errors_df.describe()

# %% tags=[]
out_sample_mean_errors = out_sample_errors_df["errors_mean"]
out_sample_pred_errors = out_sample_errors_df["errors_pred"]
sd_s_test_means = out_sample_errors_df["sd_mean"]
sd_s_test_preds = out_sample_errors_df["sd_pred"]
upper_mean = out_sample_mean_errors + sd_s_test_means
upper_pred = out_sample_pred_errors + sd_s_test_preds


# %%
def error_plot(cfg):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    ax.plot(s_test,out_sample_mean_errors, c='k', alpha=0.75, label='error using means')
    ax.plot(s_test,out_sample_pred_errors, c='b', alpha=0.75, label='error using preds')
    # ax.plot(s_train,in_sample_errors_df['errors'], c='purple', alpha=0.75, label='in sample errors')
    ax.vlines(s_train, 0, 0.1, colors="green", linestyles="dashed")
    ax.fill_between(s_test, np.array(out_sample_pred_errors), np.array(upper_pred), color='lightblue', alpha=0.75, label=f'error + 1 std using means')
    ax.fill_between(s_test, np.array(out_sample_mean_errors), np.array(upper_mean), color='coral', alpha=0.75, label=f'error + 1 std using preds')
    ax.set_xlabel(r"$s$")
    ax.legend()
    ax.grid()
    ax.set_ylim((0.0,0.08))
    fig.savefig(job_path / 'images/out-sample-error-plot.png',dpi=300,bbox_inches='tight',facecolor="w")
    plt.show()


# %% tags=[]
error_plot(config)

# %%
