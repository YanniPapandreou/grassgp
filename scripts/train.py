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
from hydra_zen import instantiate, make_config
from pathlib import Path
import os
import jax.numpy as np
from jax import random

import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_value
from numpyro.handlers import scope

from grassgp.utils import get_config_and_data, vec, get_save_path
from grassgp.inference import run_inference
from grassgp.configs.grass_config import GrassConfFullReortho_b_5
from grassgp.configs.outer_config import GPConfFull
from grassgp.utils import safe_save_jax_array_dict as safe_save


# %%
## subsample data to get training set
def sub_grid_inds(h_gap, v_gap, N_sqrt):
    inds = []
    for i in range(0,N_sqrt,h_gap):
        v_inds = [50 * i + j for j in range(0, N_sqrt, v_gap)]
        inds.extend(v_inds)
    return inds


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
def load_dataset(cfg):
    config_and_dataset = get_config_and_data(cfg.dataset_path)
    print("Dataset loaded; overrides used in generation:")
    for override in config_and_dataset['overrides']:
        print(f"-{override}")
    return config_and_dataset


# %%
def get_training_data(cfg, X_fine, s_fine, Ys_fine, Ps_fine):
    if cfg.random:
        assert cfg.seed is not None, "random subsampling chosen, so need to specify a seed"
        assert cfg.n_x_samples is not None, "random subsampling chosen, so need to specify number of samples"
        key = random.PRNGKey(cfg.seed)
        X, s, Ys, Ps = subsample(key, X_fine, s_fine, Ys_fine, Ps_fine, cfg.n_x_samples, cfg.s_gap)
    else:
        assert cfg.x_gap is not None, "deterministic subsampling chosen, so need to specify x_gap"
        N_fine_sqrt = int(np.sqrt(X_fine.shape[0]))
        inds = sub_grid_inds(cfg.x_gap, cfg.x_gap,N_fine_sqrt)
        X = X_fine[inds,:].copy()
        Ys = Ys_fine[inds,::cfg.s_gap].copy()
        Ps = Ps_fine[::cfg.s_gap,:,:].copy()
        s = s_fine[::cfg.s_gap].copy()
    
    return X, s, Ys, Ps


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
########################################
### Choose default dataset here ########
base_path = Path(os.getcwd()) / "multirun" / "2022-10-22" / "23-06-22"
key_path = "0"
path = base_path / key_path
########################################

SubsampleConfig = make_config(
    random = False,
    n_x_samples = None,
    x_gap = 8,
    s_gap = 2,
    seed = None
)

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
    n_subsample_gap = 4,
)

Config = make_config(
    dataset_path = path,
    subsample_conf = SubsampleConfig,
    inner_model = GrassConfFullReortho_b_5,
    outer_model = GPConfFull,
    svi = SVIConfig,
    train = TrainConfig
)


# %%
def train(cfg):
    # load in dataset and config which generated it
    config_and_dataset = load_dataset(cfg)
    
    # extract data from dataset
    data = config_and_dataset['data']
    data = {k:np.array(array) for k, array in data.items()}
    X_fine = data['X']
    s_fine = data['s']
    Ps_fine = data['Ps']
    Ys_fine = data['Ys']
    
    # subsample to get training data
    X, s, Ys, Ps = get_training_data(cfg.subsample_conf, X_fine, s_fine, Ys_fine, Ps_fine)

    # save training data
    training_data = {'X': X, 's': s, 'Ys': Ys, 'Ps': Ps}
    head = os.getcwd()
    main_name_training = "training_data"
    path_training = get_save_path(head, main_name_training)
    try:
        safe_save(path_training, training_data)
    except FileExistsError:
        print("File exists so not saving.")

    # instantiate inner grass_model
    grass_model = instantiate(cfg.inner_model)
    # instantiate outer GP model
    gp_model = instantiate(cfg.outer_model)
    
    # create model
    def model(X, s, Ys, n_subsample_gap = 1):
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

    # run SVI to get MAP estimate to initialise MCMC
    svi_key = random.PRNGKey(cfg.svi.seed)
    maxiter = cfg.svi.maxiter
    step_size = cfg.svi.step_size
    print("Running SVI for MAP estimate to initialise MCMC")
    svi_results = run_svi_for_map(svi_key, model, maxiter, step_size, X, s, Ys)
    
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
    mcmc = run_inference(train_key, mcmc_config, model, X, s, Ys, n_subsample_gap)
    
    samples = mcmc.get_samples()
    inference_data = samples.copy()
    for param, initial_val in init_values.items():
        inference_data[f"{param}-initial_value"] = initial_val
    
    head = os.getcwd()
    main_name = "inference_data"
    path = get_save_path(head, main_name)
    try:
        safe_save(path, data)
    except FileExistsError:
        print("File exists so not saving.")
