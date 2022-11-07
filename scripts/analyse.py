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
from hydra_zen import to_yaml, load_from_yaml
from pathlib import Path
import os
import jax.numpy as np
from jax import random 

from grassgp.utils import load_and_convert_to_samples_dict as load_data
from grassgp.plot_utils import flatten_samples, plot_projected_data, traceplots, plot_grids, plot_preds_train_locs, plot_grass_preds, plot_grass_dists, plot_preds, plot_AS_dir_preds, plot_fixed_x_preds_vs_time
from grassgp.prediction import run_prediction_at_train_times, run_grass_predict, run_predict
from grassgp.grassmann import compute_karcher_mean

from train import sub_grid_inds

import matplotlib.pyplot as ax
ax.rcParams["figure.figsize"] = (10,6)


# %%
def print_file(x: Path):
    with x.open("r") as f:
        print(f.read())


# %% tags=["parameters"]
# Job Parameters
base_path = Path(os.getcwd())
output_folder = "multirun"
date = "2022-11-02"
time = "16-34-21"
id_num = "0"

# %%
job_path = base_path / output_folder / date / time / id_num
assert job_path.exists()

# load config in
config = load_from_yaml(job_path / ".hydra" / "config.yaml")
overrides = load_from_yaml(job_path / ".hydra" / "overrides.yaml")
# print overrides and config
# print("Config used for training:")
# print(to_yaml(config))
print("Overrides used for training:")
print(to_yaml(overrides))

# %%
# get data and training data
dataset_path = Path(config.dataset_path) / "dataset.npz"
# print config and overrides used to generate full dataset

dataset_config_path = Path(config.dataset_path) / ".hydra" / "config.yaml"
assert dataset_config_path.exists()
# print("Config used to generate dataset:")
# print(to_yaml(load_from_yaml(dataset_config_path)))
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

# %%
print_file(job_path / "hmc_log.txt")

# %% tags=[] jupyter={"source_hidden": true}
# ignore_list = ['grass-L', 'grass-L_factor', 'grass-proj_mean', 'grass-proj_params', 'grass-sigmas', 'grass-standard_proj_params', 'grass-Ps']
# for key, value in samples.items():
#     if key in ignore_list:
#         continue
#     else:
#         print(f"Current parameter is {key}")
#         print("True value of parameter is:")
#         if 'grass' in key:
#             if 'kernel' in key:
#                 cfg_param = config.inner_model.grass_config[key[len('grass-kernel_'):]]
#             else:
#                 cfg_param = config.inner_model.grass_config[key[len('grass-'):]]
#             if cfg_param is None:
#                 print(np.round(data[key],decimals=3))
#             else:
#                 print(np.round(np.array(cfg_param),decimals=3))
#         elif 'reg' in key:
#             cfg_param = config.outer_model.gp_config.params[key[len('reg-kernel_'):]]
#             if cfg_param is None:
#                 print(np.round(data[key], decimals=3))
#             else:
#                 print(np.round(np.array(cfg_param),decimals=3))
        
#         print("Mean value of parameter from HMC is:")
#         print(np.round(value.mean(axis=0), decimals=3))
#         print("Std. of parameter from HMC is:")
#         print(np.round(value.std(axis=0), decimals=3))

# %% tags=[]
traceplots(my_samples, a=1.0)

# %% tags=[]
plot_projected_data(X_fine_projs, s_fine, Ys_fine, cols=3, ex=0.95, fontsize=20)

# %% tags=[]
plot_projected_data(X_projs, s, Ys)

# %% tags=[]
plot_grids(X_fine, X)

# %%
pred_key = random.PRNGKey(3276359)
means_train, predictions_train = run_prediction_at_train_times(pred_key, X, X, s, Ys, config, samples)

# %% tags=[]
plot_preds_train_locs(means_train, predictions_train, X, X, s, Ys, Ps, [2.5,97.5], base_fig_size=(12,8))


# %%
pred_key = random.PRNGKey(457657)
Ps_means, Ps_preds = run_grass_predict(pred_key, s_fine, s, config, samples)

# %% tags=[]
plot_grass_preds(s, s_fine, Ps_means, Ps_preds, Ps_fine, [2.5, 97.5])

# %% tags=[]
plot_grass_dists(samples['grass-Ps'], Ps, s, base_fig_size=(6,4))


# %%
pred_key_test = random.PRNGKey(2352345)
means_test, predictions_test, projs_test = run_predict(pred_key_test, X, s_fine, X, s, Ys, config, samples)


# %%
N_gap = config.subsample_conf.x_gap
N_fine_sqrt = config.subsample_conf.n_x_samples
inds = sub_grid_inds(N_gap,N_gap,N_fine_sqrt)
assert (X_fine[inds,:] == X).all()

# %% tags=[]
plot_preds(means_test, predictions_test, X, X, s_fine, Ys_fine[inds,:], Ps_fine, s, [2.5, 97.5], base_fig_size=(12,8))

# %%
Ps_fixed = compute_karcher_mean(projs_test)

# %% tags=[]
plot_preds(means_test, predictions_test, X, X, s_fine, Ys_fine[inds,:], Ps_fixed, s, [2.5, 97.5], base_fig_size=(12,8))

# %%
plot_AS_dir_preds(projs_test, Ps_fine, s_fine, s)

# %%
plot_grass_dists(projs_test, Ps_fine, s_fine, base_fig_size=(6,4))

# %%
i = 10
# i= 35
x_fixed = X[i,:].reshape(1,-1)
Ys_fixed = Ys[i,:].reshape(1,-1)
print(x_fixed)
print(Ys_fixed)

pred_key_fixed_x= random.PRNGKey(423657658)
n_s_grid = 40
s_grid = np.linspace(0.,1.,n_s_grid)
means_fixed_x, predictions_fixed_x, projs_fixed_x = run_predict(pred_key_fixed_x, x_fixed, s_grid, x_fixed, s, Ys_fixed, config, samples)
plot_fixed_x_preds_vs_time(means_fixed_x, predictions_fixed_x, s_grid, s, x_fixed, Ys_fixed)

# %%
i= 35
x_fixed = X[i,:].reshape(1,-1)
Ys_fixed = Ys[i,:].reshape(1,-1)
print(x_fixed)
print(Ys_fixed)

pred_key_fixed_x= random.PRNGKey(423657658)
n_s_grid = 40
s_grid = np.linspace(0.,1.,n_s_grid)
means_fixed_x, predictions_fixed_x, projs_fixed_x = run_predict(pred_key_fixed_x, x_fixed, s_grid, x_fixed, s, Ys_fixed, config, samples)
plot_fixed_x_preds_vs_time(means_fixed_x, predictions_fixed_x, s_grid, s, x_fixed, Ys_fixed)

# %%
i= 22
x_fixed = X[i,:].reshape(1,-1)
Ys_fixed = Ys[i,:].reshape(1,-1)
print(x_fixed)
print(Ys_fixed)

pred_key_fixed_x= random.PRNGKey(423657658)
n_s_grid = 40
s_grid = np.linspace(0.,1.,n_s_grid)
means_fixed_x, predictions_fixed_x, projs_fixed_x = run_predict(pred_key_fixed_x, x_fixed, s_grid, x_fixed, s, Ys_fixed, config, samples)
plot_fixed_x_preds_vs_time(means_fixed_x, predictions_fixed_x, s_grid, s, x_fixed, Ys_fixed)
