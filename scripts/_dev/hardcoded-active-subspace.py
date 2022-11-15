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
import sys
import os
import time
from pathlib import Path
from hydra_zen import launch, instantiate, builds, make_config, to_yaml

import jax.numpy as np
from jax import random

import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_value
from numpyro.handlers import scope

from grassgp.utils import unvec, vec, get_save_path
from grassgp.grassmann import rand_grass_point, valid_grass_point
from grassgp.generate_data import generate_input_data
from grassgp.configs.outer_config import OuterGPConf, GPConfFull
from grassgp.plot_utils import plot_projected_data
from grassgp.inference import run_inference
from grassgp.utils import safe_save_jax_array_dict as safe_save

# %%
proj_key = random.PRNGKey(2568745)
D = 2
active_dimension = 1
P = rand_grass_point(proj_key, D, active_dimension)
print(np.round(P,decimals=2))

# %%
n_x_sqrt = 10
X_fine, s_fine = generate_input_data(D, active_dimension, n_x_sqrt=n_x_sqrt)
n_s_fine = s_fine.shape[0]

# %%
Ps_fine = np.array([P for i in range(n_s_fine)])
print(np.round(Ps_fine,decimals=2))

# %%
X_projs_fine = np.einsum('ij,ljk->lik', X_fine, Ps_fine)
X_projs_fine_all = np.vstack([X_projs_fine[i,:,:] for i in range(n_s_fine)])

# %%
Data_config = make_config(
    outer_model = OuterGPConf
)

# %%
print(to_yaml(Data_config))

# %%
gen_from_gp_model = instantiate(Data_config.outer_model)

# %%
Y_fine = gen_from_gp_model(X_projs_fine_all)
Ys_fine = unvec(Y_fine, n_x_sqrt**2, n_s_fine)

# %%
data = {'X': X_fine, 's': s_fine, 'Ps': Ps_fine, 'Ys': Ys_fine}
head = os.getcwd()
main_name = "dataset-hardcoded-proj"
path = get_save_path(head, main_name)
try:
    safe_save(path, data)
except FileExistsError:
    print("File exists so not saving.")


# %%
def subsample(key, X_fine, s_fine, Ys_fine,Ps_fine, n_samples, s_gap):
    n = X_fine.shape[0]
    inds = random.choice(key, n, shape=(n_samples,),replace=False)
    X = X_fine[inds,:].copy()
    Ys = Ys_fine[inds,::s_gap].copy()
    Ps = Ps_fine[::s_gap,:,:].copy()
    s = s_fine[::s_gap].copy()
    return X, s, Ys, Ps


# %%
subsample_key = random.PRNGKey(3463457)
n_x_samples = 49
s_gap = 2
X, s, Ys, Ps = subsample(subsample_key, X_fine, s_fine, Ys_fine, Ps_fine, n_x_samples, s_gap)
X_projs = np.einsum('ij,ljk->lik', X, Ps)

training_data = {'X': X, 's': s, 'Ys': Ys, 'Ps': Ps}

head = os.getcwd()
main_name_training = "training-data-hardcoded-proj"
path_training = get_save_path(head, main_name_training)
try:
    safe_save(path_training, training_data)
except FileExistsError:
    print("File exists so not saving.")


# %%
# plot_projected_data(X_projs_fine,s_fine,Ys_fine)
# plot_projected_data(X_projs,s,Ys)

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
SVIConfig = make_config(
    seed = 123514354575,
    maxiter = 10000,
    step_size = 0.001
)

TrainConfig = make_config(
    seed = 9870687,
    n_warmup = 1000,
    n_samples = 1000,
    n_chains = 1,
    n_thinning = 2,
    n_subsample_gap = 1,
)

Config = make_config(
    proj_angle = 0.0,
    outer_model = GPConfFull,
    svi = SVIConfig,
    train = TrainConfig
)

# %%
print(to_yaml(Config))


# %%
def train(cfg):
    # instantiate outer GP model
    gp_model = instantiate(cfg.outer_model)
    
    # hardcoded projection
    P_x = np.cos(cfg.proj_angle)
    P_y = np.sin(cfg.proj_angle)
    P = np.array([[P_x],[P_y]])
    assert valid_grass_point(P)
    Ps = np.array([P for i in range(s.shape[0])])
    
    # create model
    def model(X, s, Ps, Ys, n_subsample_gap = 1):
        # get num of aux params
        n_s = s.shape[0]
        
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
    
    # run SVI to get MAP estimate to initialise MCMC
    svi_key = random.PRNGKey(cfg.svi.seed)
    maxiter = cfg.svi.maxiter
    step_size = cfg.svi.step_size
    print("Running SVI for MAP estimate to initialise MCMC")
    svi_results = run_svi_for_map(svi_key, model, maxiter, step_size, X, s, Ps, Ys)
    
    # print("Plotting SVI losses")
    # plt.plot(svi_results.losses)
    # plt.show()
    
    # get initialisation from SVI results
    map_est = svi_results.params
    strip_val = len('_auto_loc')
    init_values = {key[:-strip_val]:value for (key, value) in map_est.items()}
    
    # # run HMC
    train_key = random.PRNGKey(cfg.train.seed)
    mcmc_config = {'num_warmup' : cfg.train.n_warmup, 'num_samples' : cfg.train.n_samples, 'num_chains' : cfg.train.n_chains, 'thinning' : cfg.train.n_thinning, 'init_strategy' : init_to_value(values=init_values)}
    n_subsample_gap = cfg.train.n_subsample_gap
    print("HMC starting.")
    mcmc = run_inference(train_key, mcmc_config, model, X, s, Ps, Ys, n_subsample_gap)
    original_stdout = sys.stdout
    with open('hmc_log.txt', 'w') as f:
        sys.stdout = f
        mcmc.print_summary()
        sys.stdout = original_stdout
    
    samples = mcmc.get_samples()
    inference_data = samples.copy()
    for param, initial_val in init_values.items():
        inference_data[f"{param}-initial_value"] = initial_val
    
    head = os.getcwd()
    main_name = "inference-data-hardcoded-proj"
    path = get_save_path(head, main_name)
    try:
        safe_save(path, inference_data)
    except FileExistsError:
        print("File exists so not saving.")


# %%
print(to_yaml(Config))

# %%
angles = np.linspace(0,2*np.pi,20)
angle_overrides = "proj_angle="
for theta in angles:
    angle_overrides += f"{theta},"
# print(angles)
angle_overrides = angle_overrides[:-1]
# print(angle_overrides)
assert (angles == np.array([float(s) for s in angle_overrides[len('proj_angle='):].split(',')])).all()

# %%
(jobs,) = launch(
    Config,
    train,
    overrides=[
        angle_overrides,
        "outer_model.gp_config.b=1.0,5.0"
    ],
    multirun=True,
    version_base="1.1"
)

# %%
