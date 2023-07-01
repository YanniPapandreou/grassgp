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
from hydra_zen import instantiate, make_config, builds, launch, to_yaml, load_from_yaml, launch
import os
import sys
from pathlib import Path

# from jax.config import config
# config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import vmap, random
import jax.numpy.linalg as lin
from tqdm import tqdm

from grassgp.utils import subspace_angle, unvec, vec, kron_chol
from grassgp.grassmann import valid_grass_point, grass_dist, grass_log, convert_to_projs, grass_exp, grass_dist
from grassgp.means import zero_mean
from grassgp.kernels import rbf
from grassgp.plot_utils import flatten_samples

import chex
from chex import assert_shape, assert_rank
from dataclasses import dataclass, field
from typing import Tuple, Union, Callable, Tuple

import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_value
import numpyro.distributions as dist

from grassgp.inference import run_inference

import pickle
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
# plt.rc('text', usetex=True)
# plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
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
date = "2023-06-30"
time = "15-59-36"
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
# plot dataset
for i in range(d):
    plt.plot(s_test, Ws_test[:,i,0])
    plt.title(f'{i+1}th component of projection')
    plt.grid()
    plt.xlabel(r'$s$')
    plt.show()

# %%
print(f"Number of training points: {s_train.shape[0]}")
for i in range(d):
    plt.plot(s_test, Ws_test[:,i,0])
    plt.scatter(s_train, Ws_train[:,i,0], c='r')
    plt.title(f'{i+1}th component of projection')
    plt.grid()
    plt.xlabel(r'$s$')
    plt.show()


# %%
def subspace_angle_to_grass_pt(theta):
    x = np.cos(theta).reshape(-1,1)
    y = np.sin(theta).reshape(-1,1)
    W = np.hstack((x,y))[:,:,None]
    W = W[0]
    return W

def loss(theta, Ws):
    W = subspace_angle_to_grass_pt(theta)
    return vmap(lambda x: grass_dist(W, x)**2)(Ws).sum()


# %%
try:
    os.makedirs(job_path / "images")
    print("Creating image directory in job path")
except FileExistsError:
    print("Directory already exists; skipping.")

# %%
thetas = np.linspace(0, np.pi, 1000)
losses = vmap(lambda theta: loss(theta, Ws_train))(thetas)

theta_argmin = thetas[losses.argmin()]
anchor_point = subspace_angle_to_grass_pt(theta_argmin)
assert valid_grass_point(anchor_point)

plt.plot(thetas,losses)
plt.scatter(theta_argmin, losses.min(), color="red", label=r'value of $S(\theta)$ for anchor point')
plt.grid()
plt.xlabel(r"$\theta$")
plt.legend()
plt.savefig(job_path / 'images/karcher-mean-brute-force-loss-plot.png',dpi=300,bbox_inches='tight',facecolor="w")
# plt.title(r"Plot of $\theta$ vs $S(\theta)$")
plt.show()

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
ax.set_ylim((-1.25,1.25))
ax.set_xlim((-1.25,1.25))
ax.scatter(Ws_train[:,0,0], Ws_train[:,1,0], color="blue", alpha=0.25, label='training points')
ax.scatter(anchor_point[0,0], anchor_point[1,0], color="red", marker="*", label="anchor point")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.grid()
ax.legend()
plt.tight_layout()
fig.savefig(job_path / 'images/anchor-point-circle-rep.png',dpi=300,bbox_inches='tight',facecolor="w")
# ax.set_title("Plot of observed points with anchor point on circle representa of Gr(2,1)")

plt.show()

# %%
alphas = np.array([subspace_angle(w) for w in Ws_test])
alphas_train = np.array([subspace_angle(w) for w in Ws_train])
alphas_test = np.array([subspace_angle(w) for w in Ws_test])

# %%
ordinals_dict = {1: 'st', 2: 'nd', 3: 'rd'}
fig, axs = plt.subplots(2,2, figsize=(16,12))
for i in range(d):
    axs[0,i].plot(s_test, Ws_test[:,i,0], label="test data")
    axs[0,i].scatter(s_train, Ws_train[:,i,0], c='r', label="train data")
    axs[0,i].set_title(rf'{i+1}{ordinals_dict[i+1]} component of $P(s)$')
    axs[0,i].set_xlabel(r'$s$')
    axs[0,i].legend()
    axs[0,i].grid()

axs[1,0].plot(s_test, alphas_test, label="test data")
axs[1,0].scatter(s_train, alphas_train,c='r', label="train data")
axs[1,0].set_title(r'Plot of $s$ vs $\alpha(s)$')
axs[1,0].set_xlabel(r'$s$')
axs[1,0].legend()
axs[1,0].grid()

axs[1,1].set_ylim((-1.25,1.25))
axs[1,1].set_xlim((-1.25,1.25))
axs[1,1].scatter(Ws_test[:,0,0], Ws_test[:,1,0], alpha=0.5, label='test data')
axs[1,1].scatter(Ws_train[:,0,0], Ws_train[:,1,0], color='r', marker='x', alpha=0.9, label='train data')
axs[1,1].scatter(anchor_point[0,0], anchor_point[1,0], color="green", marker="*", label="anchor point")
axs[1,1].set_title(r'Datasets plotted on $S^{1}$ representation of $\operatorname{Gr}(2,1)$')
axs[1,1].set_xlabel(r"$x$")
axs[1,1].set_ylabel(r"$y$")
axs[1,1].legend()
axs[1,1].grid()

fig.savefig(job_path / 'images/dataset-plot.png',dpi=300,bbox_inches='tight',facecolor="w")
plt.show()

# %%
print_file(job_path / "hmc_log.txt")


# %%
def diagnostic_plots(cfg):
    # instantiate grass model
    svi_results = pickle_load(job_path / 'svi_results.pickle')
    
    plt.plot(svi_results.losses)
    plt.show()
    
    inference_data = pickle_load(job_path / 'inference_data.pickle')
    # print(inference_data.keys())
    
    # get initialisation from SVI results
    map_est = svi_results.params
    strip_val = len('_auto_loc')
    init_values = {key[:-strip_val]:value for (key, value) in map_est.items()}
        
    samples = dict(filter(lambda elem: 'initial_value' not in elem[0], inference_data.items()))
    initial_values = dict(filter(lambda elem: 'initial_value' in elem[0], inference_data.items()))
    assert set(samples.keys()).union(initial_values.keys()) == set(inference_data.keys())
    
    my_samples = flatten_samples(samples, ignore=[])
    trace_plot_vars = ['kernel_length']
    for key in my_samples.keys():
        if 'L_factor[0,1]' in key:
            trace_plot_vars.append(key)
        if 'L_factor[1,0]' in key:
            trace_plot_vars.append(key)
        if 'sigmas' in key:
            trace_plot_vars.append(key)

    fig, axs = plt.subplots(5,2,figsize=(20,30))
    samples_filtered = my_samples[trace_plot_vars]
    name_map = {"kernel_length": r"$\ell$", "L_factor[0,1]": r"$L_{12}$", "L_factor[1,0]": r"$L_{21}$", "sigmas[0]": r"$\sigma_{1}$", "sigmas[1]": r"$\sigma_{2}$"}
    for (i,var) in enumerate(trace_plot_vars):
        var_name = name_map[var]
        axs[i,0].plot(samples_filtered[var],c='k')
        axs[i,0].set_xlabel(r"$n$")
        axs[i,0].grid()
        axs[i,0].set_title(f"Traceplot for {var_name}")
        sm.graphics.tsa.plot_acf(samples_filtered[var], lags=config.plots.acf_lags, ax=axs[i,1], title=f"Autocorrelation for {var_name}")
        axs[i,1].set_xlabel("lag")
        axs[i,1].grid()

    fig.savefig(job_path / 'images/diagnostic-plots.png',dpi=300,bbox_inches='tight',facecolor="w")
    plt.show()    
    


# %%
diagnostic_plots(config)


# %%
def in_sample_plots(cfg):        
    training_test_data = pickle_load(job_path / 'training_test_data.pickle')
    anchor_point = training_test_data['anchor_point']
    s_train = training_test_data['s_train']
    Ws_train = training_test_data['Ws_train']
    s_test = training_test_data['s_test']
    Ws_test = training_test_data['Ws_test']
    log_Ws_train = training_test_data['log_Ws_train']
    log_Ws_test = training_test_data['log_Ws_test']
    
    svi_key = random.PRNGKey(cfg.svi.seed)
    maxiter = cfg.svi.maxiter
    step_size = cfg.svi.step_size
    svi_results = pickle_load(job_path / 'svi_results.pickle')
    
    inference_data = pickle_load(job_path / 'inference_data.pickle')
    
    # get initialisation from SVI results
    map_est = svi_results.params
    strip_val = len('_auto_loc')
    init_values = {key[:-strip_val]:value for (key, value) in map_est.items()}
        
    samples = dict(filter(lambda elem: 'initial_value' not in elem[0], inference_data.items()))
    initial_values = dict(filter(lambda elem: 'initial_value' in elem[0], inference_data.items()))
    assert set(samples.keys()).union(initial_values.keys()) == set(inference_data.keys())
    
    tol=1e-5
    samples_Ws_train = pickle_load(job_path / "samples_Ws_train.pickle")
    for ws in samples_Ws_train:
        assert vmap(lambda w: valid_grass_point(w, tol=tol))(ws).all()
        
        
    mcmc_barycenters = pickle_load(job_path / "mcmc_barycenters.pickle")
    
    in_sample_errors_df = pickle_load(job_path / "in_sample_errors_df.pickle")
    # plt.plot(s_train,in_sample_errors_df['errors'])
    # plt.show()

    samples_alphas_train = pickle_load(job_path / "samples_alphas_train.pickle")
    # convert data for kde plot
    tuples = []
    for i in tqdm(range(samples_alphas_train.shape[0])):
        angles = samples_alphas_train[i]
        for j,angle in enumerate(angles):
            tup = (angle, s_train[j])
            tuples.append(tup)
    
    data = np.array(tuples)
    df = pd.DataFrame(data, columns=['alpha','s'])
        
    # percentile_levels = [2.5, 97.5]
    # conf_level = percentile_levels[-1] - percentile_levels[0]
    # percentiles = np.percentile(samples_alphas_train, np.array(percentile_levels), axis=0)
    # lower = percentiles[0,:]
    # upper = percentiles[1,:]
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111)
    # sns.kdeplot(
    #     data=df, x="s", y="alpha",
    #     fill=True, thresh=0, levels=100, ax=ax, cbar=True
    # )
    sns.kdeplot(
        data=df, x="s", y="alpha",
        fill=True, ax=ax, cbar=True
    )
    ax.plot(s_test, alphas, c='black', alpha=0.5, label='test data',linestyle='dashed')
    ax.scatter(s_train, alphas_train, label='train data', c='r')
    # ax.scatter(s_train, samples_alphas_train.mean(axis=0), label='mean of HMC samples', c='r')
    # ax.fill_between(s_train, lower, upper,  color='lightblue', alpha=0.75,label=f'{conf_level}% credible interval')
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$\alpha(s)$")
    # ax.legend()
    # ax.grid()
    fig.savefig(job_path / 'images/in-sample-subspace-angle-plot.png',dpi=1200,bbox_inches='tight')
    plt.show()

# %%
in_sample_errors_df = pickle_load("in_sample_errors_df.pickle")
plt.plot(s_train,in_sample_errors_df['errors'])
plt.show()

# %%
in_sample_plots(config)

# %%
in_sample_errors_df = pickle_load(job_path / "in_sample_errors_df.pickle")
plt.plot(s_train,in_sample_errors_df['errors'])
plt.show()

# %%
in_sample_errors_df.describe()


# %%
def out_of_sample_pred_plots(cfg):
    # plt.rc('text', usetex=True)
    # plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    Deltas_means = pickle_load(job_path / "Deltas_means.pickle")
    Deltas_preds = pickle_load(job_path / "Deltas_preds.pickle")
    assert np.isnan(Deltas_means).sum() == 0
    assert np.isnan(Deltas_preds).sum() == 0
    
    fig, axs = plt.subplots(2,1,figsize=(12,12),sharey=False)
    ordinals_dict = {1: 'st', 2: 'nd', 3: 'rd'}
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
        axs[i].plot(s_test, log_Ws_test[:,i,0], label='test data',c='black', alpha=0.75, linestyle='dashed')
        axs[i].scatter(s_train, log_Ws_train[:,i,0], label='train data', c='g')
        axs[i].plot(s_test, means_avg, label='averaged mean prediction', c='r', alpha=0.75)
        axs[i].fill_between(s_test, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
        axs[i].set_xlabel(r"$s$")
        axs[i].grid()
        axs[i].legend()
        # axs[i].vlines(s_train, 0.99*lower.min(), 1.01*upper.max(), colors='green', linestyles='dashed')
        axs[i].set_title(f'{i+1}{ordinals_dict[i+1]} component of ' + r'$\mathbf{U}(s)$')
    
    axs[0].set_ylim([-0.01,0.01])
    fig.savefig(job_path / 'images/out-sample-tangent-predictions-plot.png',dpi=300,bbox_inches='tight',facecolor="w")
    plt.show()

    Ws_means = pickle_load(job_path / "Ws_means.pickle")
    Ws_preds = pickle_load(job_path / "Ws_preds.pickle")
    assert np.isnan(Ws_means).sum() == 0
    assert np.isnan(Ws_preds).sum() == 0
    
    alphas_means = pickle_load(job_path / "alpha_means.pickle")
    alphas_preds = pickle_load(job_path / "alpha_preds.pickle")
    
    # percentile_levels = [2.5, 97.5]
    # conf_level = percentile_levels[-1] - percentile_levels[0]
    # alphas_means_avg = np.mean(alphas_means, axis=0)
    # percentiles = np.percentile(alphas_preds, np.array(percentile_levels), axis=0)
    # lower = percentiles[0,:]
    # upper = percentiles[1,:]
    
    tuples = []
    for i in tqdm(range(alphas_preds.shape[0])):
        angles = alphas_preds[i]
        for j,angle in enumerate(angles):
            tup = (angle, s_test[j])
            tuples.append(tup)
        
    data = np.array(tuples)
    df = pd.DataFrame(data, columns=['alpha','s'])
    
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    # sns.kdeplot(
    #     data=df, x="s", y="alpha",
    #     fill=True, thresh=0, levels=100, cmap="viridis", ax=ax, cbar=True
    # )
    sns.kdeplot(
        data=df, x="s", y="alpha",
        fill=True, ax=ax, cbar=True
    )
    ax.plot(s_test, alphas, label='test data',c='black', alpha=0.75, linestyle='dashed')
    ax.scatter(s_train, alphas_train, label='train data', c='r')
    # ax.plot(s_test, np.mean(alphas_preds, axis=0), label='averaged mean prediction', c='r', alpha=0.75)
    # ax.fill_between(s_test, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$\alpha(s)$")
    # ax.legend()
    # ax.grid()
    # ax.vlines(s_train, 0, np.pi, colors='green', linestyles='dashed')
    # ax.set_title(f"predictions for subspace angles")
    fig.savefig(job_path / 'images/out-sample-subspace-angle-plot.png',dpi=1200,bbox_inches='tight')
    plt.show()


# %%
out_of_sample_pred_plots(config)

# %%
out_sample_errors_df = pickle_load(job_path / "out_sample_errors_df.pickle")
    
out_sample_mean_errors = out_sample_errors_df["errors_mean"]
out_sample_pred_errors = out_sample_errors_df["errors_pred"]
sd_s_test_means = out_sample_errors_df["sd_mean"]
sd_s_test_preds = out_sample_errors_df["sd_pred"]
upper_mean = out_sample_mean_errors + sd_s_test_means
upper_pred = out_sample_pred_errors + sd_s_test_preds

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
ax.plot(s_test,out_sample_mean_errors, c='k', alpha=0.75, label='error using means')
ax.plot(s_test,out_sample_pred_errors, c='b', alpha=0.75, label='error using preds')
ax.vlines(s_train, 0, 1.2, colors="green", linestyles="dashed")
ax.fill_between(s_test, np.array(out_sample_pred_errors), np.array(upper_pred), color='lightblue', alpha=0.75, label=f'error + 1 std using means')
ax.fill_between(s_test, np.array(out_sample_mean_errors), np.array(upper_mean), color='coral', alpha=0.75, label=f'error + 1 std using preds')
ax.set_xlabel(r"$s$")
ax.legend()
ax.grid()
fig.savefig(job_path / 'images/out-sample-error-plot.png',dpi=300,bbox_inches='tight',facecolor="w")
plt.show()

# %%
out_sample_errors_df.describe()

# %%
# Deltas_means = pickle_load(job_path / "Deltas_means.pickle")
# Deltas_preds = pickle_load(job_path / "Deltas_preds.pickle")

# plt.rcParams["figure.figsize"] = (12,6)
# percentile_levels = [2.5, 97.5]
# conf_level = percentile_levels[-1] - percentile_levels[0]
# for i in range(d):
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
