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
# %load_ext autoreload
# %autoreload 2

# %%
import time
from hydra_zen import to_yaml, load_from_yaml, instantiate, builds, make_config
from pathlib import Path
import os
import jax.numpy as np
from jax import random

import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_value
from numpyro.handlers import scope

from grassgp.utils import get_config_and_data, vec
from grassgp.grassmann import grass_dist
from grassgp.inference import run_inference

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,8)
from grassgp.plot_utils import plot_projected_data


# %%
def print_file(x: Path):
    with x.open("r") as f:
        print(f.read())


# %%
################################
### Choose dataset here ###
base_path = Path(os.getcwd()) / "multirun" / "2022-10-21" / "15-35-19"
key_path = "0"
path = base_path / key_path
config_and_dataset = get_config_and_data(path)
for override in config_and_dataset['overrides']:
    print(override)

data = config_and_dataset['data']
data = {k:np.array(array) for k, array in data.items()}
X_fine = data['X']
s_fine = data['s']
Ps_fine = data['Ps']
anchor_point = data['anchor_point']
Ys_fine = data['Ys']
X_projs_fine = np.einsum('ij,ljk->lik', X_fine, Ps_fine)
################################

# %%
## subsample data to get training set
def sub_grid_inds(h_gap, v_gap, N_sqrt):
    inds = []
    for i in range(0,N_sqrt,h_gap):
        v_inds = [50 * i + j for j in range(0, N_sqrt, v_gap)]
        inds.extend(v_inds)
    return inds


# %%
subsample_key = random.PRNGKey(64879)
N_fine_sqrt = int(np.sqrt(X_fine.shape[0]))
N_gap = 8
inds = sub_grid_inds(N_gap,N_gap,N_fine_sqrt)
X = X_fine[inds,:].copy()
print(f"X.shape = {X.shape}")
s_gap = 2
s = s_fine[::s_gap].copy()
n_s = s.shape[0]
Ps = Ps_fine[::s_gap,:,:].copy()
Ys = Ys_fine[inds, ::s_gap].copy()
X_projs = np.einsum('ij,ljk->lik', X, Ps)

# %% tags=[] jupyter={"outputs_hidden": true}
plot_projected_data(X_projs_fine, s_fine, Ys_fine)

# %% tags=[] jupyter={"outputs_hidden": true}
plot_projected_data(X_projs, s, Ys)

# %% tags=[] jupyter={"outputs_hidden": true}
plt.scatter(X_fine[:,0],X_fine[:,1],c='b',alpha=0.25,label='fine')
plt.scatter(X[:,0],X[:,1],c='r',label='coarse')
plt.legend()
plt.show()


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
from grassgp.models import grassmann_process

# %%
grass_model_config = {
    'anchor_point': [[1.0], [0.0]],
    'Omega' : None, 
    'proj_locs' : None,
    'var' : None,
    'length' : None,
    'noise' : None,
    'require_noise' : False,
    'jitter' : 1e-06,
    'proj_jitter' : 1e-4,
    'L_jitter' : 1e-8,
    'reorthonormalize' : True,
    'b' : 5.0
}

# %%
GrassConf = builds(grassmann_process, grass_config = grass_model_config, zen_partial=True)

# %%
print(to_yaml(GrassConf))

# %%
# ?grassmann_process

# %%
from grassgp.kernels import rbf
import numpyro.distributions as dist


# %%
def univariate_gp_model(x, y, gp_config: dict = {'params': {'var': None, 'length': None, 'noise': None}, 'jitter': 1e-06, 'b': 10.0}):
    params = gp_config['params']
    
    # # loop over params and sample any missing
    # for param, value in params.items():
    #     if value is None:
    #         params[param] = numpyro.sample(f"kernel_{param}", dist.LogNormal(0.0, gp_config['b']))
    # numpyro.sample(f"kernel_{param}", dist.LogNormal(0.0, gp_config['b']))
    
    if params['var'] is None:
        var = numpyro.sample("kernel_var", dist.LogNormal(0.0, gp_config['b']))
    else:
        var = params['var']
        
    if params['length'] is None:
        length = numpyro.sample("kernel_length", dist.LogNormal(0.0, gp_config['b']))
    else:
        length = params['length']
        
    if params['noise'] is None:
        noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, gp_config['b']))
    else:
        noise = params['noise']

    kernel_params = {'var': var, 'length': length, 'noise': noise}
    K = rbf(x, x, kernel_params, jitter = gp_config['jitter'])
    
    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "obs_y",
        dist.MultivariateNormal(loc=np.zeros(K.shape[0]), covariance_matrix=K),
        obs=y,
    )   


# %%
grass_model = instantiate(GrassConf)

# %%
gp_model_config_full = {
    'params': {'var': None, 'length': None, 'noise': None},
    'jitter': 1e-06,
    'b': 10.0
}

GPConf = builds(univariate_gp_model, gp_config = gp_model_config_full, zen_partial=True)

# %%
gp_model = instantiate(GPConf)
gp_model


# %%
def model(X, s, Ys, anchor_point, n_subsample_gap = 1):
    # get num of aux params
    n_s = s.shape[0]
    
    # get projections using Grassmann Process
    Ps = scope(grass_model, prefix="grass", divider="-")(s)
    
    # project the data
    projected_data = np.einsum('ij,ljk->lik', X, Ps)
    
    # collect the projected data over aux axis and subsample
    projected_data_all = np.vstack([projected_data[i,:,:] for i in range(n_s)])
    projected_data_subsampled = projected_data_all[::n_subsample_gap, :]
    
    # subsample vec(Ys)
    vec_Ys_subsampled = vec(Ys)[::n_subsample_gap]
    
    # fit outer univariate gp
    # scope(gp_model, prefix="reg", divider="-")(projected_data_all, vec(Ys))
    scope(gp_model, prefix="reg", divider="-")(projected_data_subsampled, vec_Ys_subsampled)


# %%
key = random.PRNGKey(123514354575)
maxiter=10000
step_size=0.001
anchor_point_guess = anchor_point
print(f'Grass-dist btw true anchor point and anchor point used for model = {grass_dist(anchor_point, anchor_point_guess)}')
# svi_results = run_svi_for_map(key, model, maxiter, step_size, X, s, Ys_centred, anchor_point_guess, model_options['grass_jitter'], model_options['reg_jitter'], model_options['proj_jitter'], model_options['grassmann_noise'], model_options['reorthonormalize'], model_options['params'], model_options['know_reg_kernel_params'])
svi_results = run_svi_for_map(key, model, maxiter, step_size, X, s, Ys, anchor_point_guess)

# %%
plt.plot(svi_results.losses)

# %%
# def model_non_hydra(X, s, Ys, anchor_point, grass_config, gp_config):
#     # get num of aux params
#     n_s = s.shape[0]
    
#     # get projections using Grassmann Process
#     Ps = scope(grassmann_process, prefix="grass", divider="-")(s, grass_config)
    
#     # project the data
#     projected_data = np.einsum('ij,ljk->lik', X, Ps)
    
#     # collect the projected data over aux axis
#     projected_data_all = np.vstack([projected_data[i,:,:] for i in range(n_s)])
    
#     # fit outer univariate gp
#     scope(univariate_gp_model, prefix="reg", divider="-")(projected_data_all, vec(Ys), gp_config)

# %%
# key = random.PRNGKey(123514354575)
# maxiter=10000
# step_size=0.001
# anchor_point_guess = anchor_point
# print(f'Grass-dist btw true anchor point and anchor point used for model = {grass_dist(anchor_point, anchor_point_guess)}')
# # svi_results = run_svi_for_map(key, model, maxiter, step_size, X, s, Ys_centred, anchor_point_guess, model_options['grass_jitter'], model_options['reg_jitter'], model_options['proj_jitter'], model_options['grassmann_noise'], model_options['reorthonormalize'], model_options['params'], model_options['know_reg_kernel_params'])
# svi_results_non_hydra = run_svi_for_map(key, model_non_hydra, maxiter, step_size, X, s, Ys, anchor_point_guess, grass_model_config, gp_model_config_full)

# %%
# plt.plot(svi_results_non_hydra.losses)

# %%
# (svi_results_non_hydra.losses == svi_results.losses).all()

# %%
map_est = svi_results.params

# %%
map_est.keys()

# %%
strip_val = len('_auto_loc')

# %%
strip_val

# %%
init_values = {key[:-9]:value for (key, value) in map_est.items()}

# %%
init_values.keys()

# %%
seed = 9870687
train_key = random.PRNGKey(seed)
mcmc_config = {'num_warmup' : 1000, 'num_samples' : 1000, 'num_chains' : 1, 'thinning' : 2, 'init_strategy' : init_to_value(values=init_values)}
print("Inference starting.")
n_subsample_gap = 4
mcmc = run_inference(train_key, mcmc_config, model, X, s, Ys, anchor_point, n_subsample_gap)

# %%
