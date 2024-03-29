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
def pickle_load(name: str):
    with open(name, 'rb') as f:
        obj = pickle.load(f)
    return obj


# %% tags=[]
def pickle_save(obj, name: str):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


# %% [markdown]
# # Load results and analyse

# %%
locs = [{'date': "2023-06-29", 'time': "09-49-38"}, {'date': "2023-06-28", 'time': "09-46-00"}]

# %% tags=["parameters"]
loc_id = 1

# %% tags=[]
base_path = Path(os.getcwd())
output_folder = "multirun"
date = locs[loc_id]['date']
time = locs[loc_id]['time']
# date = "2023-06-29"
# time = "09-49-38"
# date = "2023-06-28"
# time = "09-46-00"
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
print(f"Number of training points = {s_train.shape[0]}")

# %% tags=[]
assert vmap(lambda W: valid_grass_point(W))(Ws_test).all()

# %%
try:
    os.makedirs(job_path / "images")
    print("Creating image directory in job path")
except FileExistsError:
    print("Directory already exists; skipping.")

# %%
fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(s_test[:,0],s_test[:,1],c='b',label='test locations',alpha=1,marker='x')
ax.scatter(s_train[:,0],s_train[:,1],c='r',label='train locations',alpha=0.75)
ax.grid()
ax.legend()
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_ylim((0,1))
ax.set_xlim((0,1))
fig.savefig(job_path / 'images/train-test-locations.pdf',dpi=300,bbox_inches='tight',facecolor="w")
plt.show()

# %% tags=[]
i=0
W0 = Ws_test[i]
# W0 = np.eye(100)[:,0][:,None]

dists = vmap(lambda W: grass_dist(W, W0))(Ws_test)

fig, ax = plt.subplots()
tcf = ax.tricontourf(s_test[:,0],s_test[:,1],dists)
# ax.scatter(s_train[:,0],s_train[:,1],c='r')
# ax.scatter(s_test[:,0],s_test[:,1],c='b',alpha=0.5)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
fig.colorbar(tcf)
ax.set_title(r"Grassmann distantance from $P(\mathbf{x}_{0}^{*})$")
# fig.savefig(job_path / 'images/parametrised-pde-dataset-plot.pdf',dpi=300,bbox_inches='tight',facecolor="w")
plt.show()

# %% tags=[]
print_file(job_path / "hmc_log.txt")

# %% tags=[]
inference_data_path = job_path / "inference_data.pickle"
assert inference_data_path.exists()
inference_data = pickle_load(inference_data_path)
samples = dict(filter(lambda elem: 'initial_value' not in elem[0], inference_data.items()))
initial_values = dict(filter(lambda elem: 'initial_value' in elem[0], inference_data.items()))
assert set(samples.keys()).union(initial_values.keys()) == set(inference_data.keys())

# %% tags=[]
my_samples = flatten_samples(samples, ignore=[])
trace_plot_vars = ['kernel_length']
# for key in my_samples.keys():
#     if 'Omega' in key:
#         trace_plot_vars.append(key)
#     if 'sigmas' in key:
#         trace_plot_vars.append(key)

my_samples[trace_plot_vars].plot(subplots=True, figsize=(10,6), sharey=False)
plt.show()

# %% tags=[]
for var in trace_plot_vars:
    sm.graphics.tsa.plot_acf(my_samples[var], lags=config.plots.acf_lags)
    plt.title(f"acf for {var}")
    plt.show()

trace_plot_vars = []
for name in my_samples.columns:
    if "Omega" in name:
        trace_plot_vars.append(name)

my_samples.plot(y=trace_plot_vars,legend=False,alpha=0.75)
plt.show()

# %% tags=[] jupyter={"outputs_hidden": true}
trace_plot_vars = []
for name in my_samples.columns:
    if "Omega" in name:
        plt.plot(my_samples[name])
        plt.title(name)
        plt.show()

# %% tags=[]
trace_plot_vars = []
for name in my_samples.columns:
    if "Omega" in name:
        sm.graphics.tsa.plot_acf(my_samples[name], lags=config.plots.acf_lags)
        plt.title(name)
        plt.show()

# %% tags=[]
in_sample_errors_df = pickle_load(job_path / "in_sample_errors_df.pickle")
in_sample_errors_df.head()

# %% tags=[]
in_sample_errors_df.describe()

# %% tags=[]
fig, ax = plt.subplots()
tcf = ax.tricontourf(in_sample_errors_df['x'],in_sample_errors_df['y'],in_sample_errors_df['errors'])
ax.scatter(in_sample_errors_df['x'],in_sample_errors_df['y'],c='r')
fig.colorbar(tcf)
plt.title("In sample errors")
plt.show()

# %% tags=[]
fig, ax = plt.subplots()
tcf = ax.tricontourf(in_sample_errors_df['x'],in_sample_errors_df['y'],in_sample_errors_df['sd'])
ax.scatter(in_sample_errors_df['x'],in_sample_errors_df['y'],c='r')
fig.colorbar(tcf)
plt.title("In sample errors sd")
plt.show()

# %% tags=[]
out_sample_errors_df = pickle_load(job_path / "out_sample_errors_df.pickle")
out_sample_errors_df.head()

# %% tags=[]
out_sample_errors_df.describe()

# %% tags=[]
fig, ax = plt.subplots()
tcf = ax.tricontourf(out_sample_errors_df['x'],out_sample_errors_df['y'],out_sample_errors_df['errors_mean'])
ax.scatter(in_sample_errors_df['x'],in_sample_errors_df['y'],c='r',marker='x')
ax.scatter(out_sample_errors_df['x'],out_sample_errors_df['y'],c='black',alpha=0.25)
fig.colorbar(tcf)
plt.title("Out sample errors (using mean)")
plt.show()

# %% tags=[]
fig, ax = plt.subplots()
tcf = ax.tricontourf(out_sample_errors_df['x'],out_sample_errors_df['y'],out_sample_errors_df['errors_pred'])
ax.scatter(in_sample_errors_df['x'],in_sample_errors_df['y'],c='r',marker='x')
ax.scatter(out_sample_errors_df['x'],out_sample_errors_df['y'],c='black',alpha=0.25)
fig.colorbar(tcf)
plt.title("Out sample errors (using pred)")
plt.show()

# %% tags=[]
fig, ax = plt.subplots()
tcf = ax.tricontourf(out_sample_errors_df['x'],out_sample_errors_df['y'],out_sample_errors_df['sd_mean'])
ax.scatter(in_sample_errors_df['x'],in_sample_errors_df['y'],c='r',marker='x')
ax.scatter(out_sample_errors_df['x'],out_sample_errors_df['y'],c='black',alpha=0.25)
fig.colorbar(tcf)
plt.title("Out sample errors sd (using mean)")
plt.show()

# %% tags=[]
fig, ax = plt.subplots()
tcf = ax.tricontourf(out_sample_errors_df['x'],out_sample_errors_df['y'],out_sample_errors_df['sd_pred'])
ax.scatter(in_sample_errors_df['x'],in_sample_errors_df['y'],c='r',marker='x')
ax.scatter(out_sample_errors_df['x'],out_sample_errors_df['y'],c='black',alpha=0.25)
fig.colorbar(tcf)
plt.title("Out sample errors sd (using pred)")
plt.show()


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

    fig.savefig(job_path / 'images/parametrised-pde-diagnostic-plots.pdf',dpi=300,bbox_inches='tight',facecolor="w")
    plt.show()    


# %% tags=[]
diagnostic_plots(config)

# %%
config.keys()

# %%
print(to_yaml(config.svi))

# %%
print(to_yaml(config.train))

# %%
