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
from hydra_zen import instantiate, make_config, builds, launch
import sys

from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import vmap, random
from tqdm import tqdm

from grassgp.grassmann import valid_grass_point, sample_karcher_mean, grass_dist, grass_log, convert_to_projs
from grassgp.means import zero_mean
from grassgp.kernels import rbf
from grassgp.models_optimised import GrassGP
from grassgp.plot_utils import flatten_samples

import chex
from dataclasses import dataclass
from typing import Tuple, Union

import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_value
import numpyro.distributions as dist

from grassgp.inference import run_inference

import pickle
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,6)

# %% [markdown]
# # Load dataset

# %%
with open('/home/yanni/projects/grassgp/examples/wing_function_example/experiments/parametrised_AS_datasets.pickle', 'rb') as f:
    datasets = pickle.load(f)

# %%
i = 2
s_test = datasets[i]['s']
Ws_test = datasets[i]['Ws']
sort_inds = s_test.argsort()
s_test = s_test[sort_inds].copy()
Ws_test = Ws_test[sort_inds].copy()

# %%
# dists_from_start = vmap(lambda w: grass_dist(w,Ws_test[0]))(Ws_test)
# plt.plot(s_test, dists_from_start)
# plt.title(f"Grass distance from starting AS")
# plt.show()

# %%
# for j in range(9):
#     plt.plot(s_test, Ws_test[:,j,0])
#     plt.show()

# %%
s_gap = 100
s_train = s_test[::s_gap].copy()
Ws_train = Ws_test[::s_gap].copy()
n_train = s_train.shape[0]

# %%
# plt.scatter(range(n_train),s_train)
# plt.show()

# %%
# compute barycenter of training data
anchor_point, _, _ = sample_karcher_mean(Ws_train)
assert valid_grass_point(anchor_point)

# %%
# compute log of training data
log_Ws_train = vmap(lambda W: grass_log(anchor_point, W))(Ws_train)


# %% [markdown]
# # Model

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
@dataclass
class Model:
    name: str
    anchor_point: list
    d_in: int
    Omega_diag_chol: Union[list, None]
    k_include_noise: bool
    var: Union[float, None]
    length: Union[float, None]
    noise: Union[float, None]
    require_noise: bool
    jitter: float
    cov_jitter: float
    L_jitter: float
    reorthonormalize: bool
    b: float
    savvas_param: bool
    
    def model(self, s, log_Ws):
        U = np.array(self.anchor_point)
        d, n = U.shape
        N = s.shape[0]
        d_n = d * n
        # N_params = N * d_n
        if log_Ws is not None:
            assert log_Ws.shape == (N, d, n), f"log_Ws.shape=({log_Ws.shape}) instead of {(N, d, n)} as expected"

        # get/sample Omega
        if self.Omega_diag_chol is None:
            # full example
            # sigmas = numpyro.sample('sigmas', dist.LogNormal(0.0, 1.0).expand([d_n]))
            # L_factor = numpyro.sample('L_factor', dist.LKJ(d_n, 1.0))
            # L = numpyro.deterministic('L', L_factor + self.L_jitter * np.eye(d_n))
            # Omega = numpyro.deterministic('Omega', np.outer(sigmas, sigmas) * L)
            # simpler diagonal structure
            # sigmas = numpyro.sample('sigmas', dist.LogNormal(0.0, 1.0).expand([d_n]))
            # Omega_diag = numpyro.deterministic('Omega_diag', sigmas**2)
            Omega_diag_chol = numpyro.sample('Omega_diag_chol', dist.LogNormal(0.0, 1.0).expand([d_n]))
        else:
            Omega_diag_chol = np.array(self.Omega_diag_chol)

        # get/sample kernel params
        if self.var is None:
            # sample var
            var = numpyro.sample("kernel_var", dist.LogNormal(0.0, self.b))
        else:
            var = self.var

        if self.length is None:
            # sample length
            # # # ! my parametrisation
            # length = numpyro.sample("kernel_length", dist.LogNormal(0.0, self.b))
            # # # ! savvas parametrisation
            if self.savvas_param:
                length = numpyro.sample("kernel_length", dist.LogNormal(-0.7, self.b))
            else:
                length = numpyro.sample("kernel_length", dist.LogNormal(0.0, self.b))

        else:
            length = self.length

        if self.require_noise:
            if self.noise is None:
                # sample noise
                noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, self.b))
            else:
                noise = self.noise
        else:
            noise = 0.0


        if self.savvas_param:
            kernel_params = {'var': var, 'length': np.sqrt(1 / length), 'noise': noise} # savvas
        else:
            kernel_params = {'var': var, 'length': length, 'noise': noise} # mine

        # create kernel function
        k = lambda t, s: rbf(t, s, kernel_params, jitter=self.jitter, include_noise=self.k_include_noise)
        # create mean function
        mu = lambda s: zero_mean(s, d, n)

        # initialize GrassGp
        grass_gp = GrassGP(d_in=self.d_in, d_out=(d,n), mu=mu, k=k, Omega_diag_chol=Omega_diag_chol, U=U, cov_jitter=self.cov_jitter)

        # sample Deltas
        Deltas = grass_gp.tangent_model(s)

        # # # # # # ! check what power this should be
        # likelihood
        # ell = self.ell # mine
        ell = numpyro.sample("ell", dist.LogNormal(-6, 0.0015))
        with numpyro.plate("N", N):
            numpyro.sample("log_W", dist.continuous.MatrixNormal(loc=Deltas, scale_tril_row=ell * np.eye(d), scale_tril_column=np.eye(n)), obs=log_Ws)

# %% tags=[]
# my_model = Model(
#     name = "default",
#     anchor_point = anchor_point.tolist(),
#     d_in = 1,
#     Omega_diag_chol = None,
#     k_include_noise= True,
#     var = 1.0,
#     length = None,
#     noise = None,
#     require_noise = False,
#     jitter = 1e-06,
#     cov_jitter = 1e-4,
#     L_jitter = 1e-8,
#     reorthonormalize = False,
#     b = 0.001,
# );

# auto_name_store = store(name=lambda cfg: cfg.name)

# model_store = auto_name_store(group="Model")

# model_store(my_model)

# %%
TangentSpaceModelConf = builds(Model, populate_full_signature=True)

my_model_conf = TangentSpaceModelConf(
    name = "My Model",
    anchor_point = anchor_point.tolist(),
    d_in = 1,
    Omega_diag_chol = None,
    k_include_noise= True,
    var = 1.0,
    length = None,
    noise = None,
    require_noise = False,
    jitter = 1e-06,
    cov_jitter = 1e-4,
    L_jitter = 1e-8,
    reorthonormalize = False,
    b = 0.001,
    savvas_param = True,
)

SVIConfig = make_config(
    seed = 123514354575,
    maxiter = 15000,
    step_size = 0.001
)

TrainConfig = make_config(
    seed = 9870687,
    n_warmup = 2000,
    n_samples = 7000,
    n_chains = 1,
    n_thinning = 2
)

PredictConfig = make_config(
    seed = 6578,
    splits = 25
)

PlotsConfig = make_config(
    acf_lags = 100,
    plot = True,
)

Config = make_config(
    model = my_model_conf,
    svi = SVIConfig,
    train = TrainConfig,
    predict = PredictConfig,
    plots = PlotsConfig,
    save_results = True,
    save_stdout = True,
)

# %%
# numpyro.render_model(instantiate(Config.model).model, model_args=(s_train, log_Ws_train))


# %%
def pickle_save(obj, name: str):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


# %%
def predict_tangents(
    key: chex.ArrayDevice,
    s_test: chex.ArrayDevice,
    s_train: chex.ArrayDevice,
    Vs_train: chex.ArrayDevice,
    cfg,
    samples: dict,
    jitter: float = 1e-8
) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
    
    d_in = cfg.model.d_in
    U = np.array(cfg.model.anchor_point)
    d, n = U.shape
    cov_jitter = cfg.model.cov_jitter
    k_include_noise = cfg.model.k_include_noise
    kern_jitter = cfg.model.jitter
    n_samples = cfg.train.n_samples // cfg.train.n_thinning
    assert n_samples == samples['Deltas'].shape[0]
    
    def predict(
        key: chex.ArrayDevice,
        Omega_diag_chol: chex.ArrayDevice,
        var: float,
        length: float,
        noise: float,
    ) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        # iniatilize GrassGP
        kernel_params = {'var': var, 'length': length, 'noise': noise}
        k = lambda t, s: rbf(t, s, kernel_params, jitter=kern_jitter, include_noise=k_include_noise)
        mu = lambda s: zero_mean(s, d, n)
        grass_gp = GrassGP(d_in=d_in, d_out=(d, n), mu=mu, k=k, Omega_diag_chol=Omega_diag_chol, U=U, cov_jitter=cov_jitter)

        # predict
        Deltas_mean, Deltas_pred = grass_gp.predict_tangents(key, s_test, s_train, Vs_train, jitter=jitter)
        return Deltas_mean, Deltas_pred

    # initialize vmap args
    vmap_args = (random.split(key, n_samples),)
    
    cfg_Omega_diag_chol = cfg.model.Omega_diag_chol
    cfg_var = cfg.model.var
    cfg_length = cfg.model.length
    cfg_noise = cfg.model.noise
    cfg_require_noise = cfg.model.require_noise
    
    if cfg_Omega_diag_chol is None:
        vmap_args += (samples['Omega_diag_chol'],)
    else:
        cfg_Omega_diag_chol = np.array(cfg_Omega_diag_chol)
        vmap_args += (np.repeat(cfg_Omega_diag_chol[None,:,:], n_samples, axis=0),)
    
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
def train_analyse(cfg):
    # instantiate grass model
    model = instantiate(cfg.model).model
    
    save_results = cfg.save_results
    plot_figs = cfg.plots.plot
    save_stdout = cfg.save_stdout
    
    log_Ws_test = vmap(lambda W: grass_log(anchor_point, W))(Ws_test)
    if save_results:
        training_test_data = {'s_train': s_train, 's_test': s_test, 'Ws_train': Ws_train, 'Ws_test': Ws_test, 'log_Ws_train': log_Ws_train, 'log_Ws_test': log_Ws_test, 'anchor_point': anchor_point}
        pickle_save(training_test_data, 'training_test_data.pickle')
    
    # run SVI to get MAP esimtate to initialise MCMC
    svi_key = random.PRNGKey(cfg.svi.seed)
    maxiter = cfg.svi.maxiter
    step_size = cfg.svi.step_size
    print("Running SVI for MAP estimate to initialise MCMC")
    svi_results = run_svi_for_map(svi_key, model, maxiter, step_size, s_train, log_Ws_train)
    
    if save_results:
        pickle_save(svi_results, 'svi_results.pickle')
    
    if plot_figs:
        # plot svi losses
        plt.plot(svi_results.losses)
        plt.show()
    
    # get initialisation from SVI results
    map_est = svi_results.params
    strip_val = len('_auto_loc')
    init_values = {key[:-strip_val]:value for (key, value) in map_est.items()}
    
    # run HMC
    train_key = random.PRNGKey(cfg.train.seed)
    mcmc_config = {'num_warmup' : cfg.train.n_warmup, 'num_samples' : cfg.train.n_samples, 'num_chains' : cfg.train.n_chains, 'thinning' : cfg.train.n_thinning, 'init_strategy' : init_to_value(values=init_values)}
    print("HMC starting.")
    mcmc = run_inference(train_key, mcmc_config, model, s_train, log_Ws_train)
    
    if save_stdout:
        original_stdout = sys.stdout
        with open('hmc_log.txt', 'w') as f:
            sys.stdout = f
            mcmc.print_summary()
            sys.stdout = original_stdout
    
    samples = mcmc.get_samples()
    inference_data = samples.copy()
    for param, initial_val in init_values.items():
        inference_data[f"{param}-initial_value"] = initial_val
    
    if save_results:
        pickle_save(inference_data, "inference_data.pickle")
    
    samples = dict(filter(lambda elem: 'initial_value' not in elem[0], inference_data.items()))
    initial_values = dict(filter(lambda elem: 'initial_value' in elem[0], inference_data.items()))
    assert set(samples.keys()).union(initial_values.keys()) == set(inference_data.keys())
    
    if plot_figs:
        my_samples = flatten_samples(samples, ignore=[])
        trace_plot_vars = ['kernel_length', 'ell']
        for key in my_samples.keys():
            if 'Omega' in key:
                trace_plot_vars.append(key)

        my_samples[trace_plot_vars].plot(subplots=True, figsize=(10,40), sharey=False)
        plt.show()
        
        for var in trace_plot_vars:
            sm.graphics.tsa.plot_acf(my_samples[var], lags=cfg.plots.acf_lags)
            plt.title(f"acf for {var}")
            plt.show()
    
    # compute Ws's from mcmc samples
    tol=1e-5
    samples_Ws_train = vmap(lambda Deltas: convert_to_projs(Deltas, anchor_point, reorthonormalize=False))(samples['Deltas'])

    for ws in samples_Ws_train:
        assert vmap(lambda w: valid_grass_point(w, tol=tol))(ws).all()
    
    if save_results:
        pickle_save(samples_Ws_train, "samples_Ws_train.pickle")
        
    # compute barycenters for mcmc results
    mcmc_barycenters = []
    results = []
    inits = []
    for i in tqdm(range(s_train.shape[0])):
        barycenter, result, mu_0 = sample_karcher_mean(samples_Ws_train[:,i,:,:])
        mcmc_barycenters.append(barycenter)
        results.append(result)
        inits.append(mu_0)
    
    mcmc_barycenters = np.array(mcmc_barycenters)
    if save_results:
        pickle_save(mcmc_barycenters, "mcmc_barycenters.pickle")
    
    if plot_figs:
        bary_losses = []
        for i in tqdm(range(s_train.shape[0])):
            loss = (vmap(lambda W: grass_dist(mcmc_barycenters[i], W))(samples_Ws_train[:,i,:,:]) ** 2).sum()
            bary_losses.append(loss)

        plt.plot(bary_losses)
        plt.title("Final loss for computed barycenters")
        plt.show()
    
    in_sample_errors = vmap(grass_dist)(Ws_train, mcmc_barycenters)
    
    if plot_figs:
        plt.plot(s_train,in_sample_errors)
        plt.show()
    
    sd_s_train = []
    for i in tqdm(range(s_train.shape[0])):
        fixed = mcmc_barycenters[i]
        dists = vmap(lambda W: grass_dist(W, fixed))(samples_Ws_train[:,i,:,:])
        dists_Sq = dists**2
        sd_s_train.append(np.sqrt(dists_Sq.mean()))
        
    sd_s_train = np.array(sd_s_train)
    
    pd_data = {'s': s_train, 'errors': in_sample_errors, 'sd': sd_s_train}
    in_sample_errors_df = pd.DataFrame(data=pd_data)
    
    if save_results:
        pickle_save(in_sample_errors_df, "in_sample_errors_df.pickle")
        
    print("Prediction starting")
    pred_key = random.PRNGKey(cfg.predict.seed)
    splits = cfg.predict.splits
    pred_keys = random.split(pred_key, splits)
    pred_results_chunked = {}
    for (i, s_test_chunk) in tqdm(enumerate(np.split(s_test, splits))):
        p_key = pred_keys[i]
        pred_results_chunked[i] = predict_tangents(p_key, s_test_chunk, s_train, log_Ws_train, cfg, samples)

    Deltas_means_list = []
    Deltas_preds_list = []
    for i in range(splits):
        means, preds = pred_results_chunked[i]
        Deltas_means_list.append(means)
        Deltas_preds_list.append(preds)
        
    Deltas_means = np.concatenate(Deltas_means_list, axis=1)
    Deltas_preds = np.concatenate(Deltas_preds_list, axis=1)
    
    assert np.isnan(Deltas_means).sum() == 0
    assert np.isnan(Deltas_preds).sum() == 0
    
    if save_results:
        pickle_save(Deltas_means, "Deltas_means.pickle")
        pickle_save(Deltas_preds, "Deltas_preds.pickle")
        
    d = Ws_train.shape[1]
    if plot_figs:
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
            plt.vlines(s_train, 0.99*lower.min(), 1.01*upper.max(), colors='green', linestyles='dashed')
            plt.title(f"{i+1}th component of tangents")
            plt.show()
            
    Ws_test_means = vmap(lambda Deltas: convert_to_projs(Deltas, anchor_point, reorthonormalize=False))(Deltas_means)

    for ws in Ws_test_means:
        assert vmap(lambda w: valid_grass_point(w, tol=tol))(ws).all()
        
    Ws_test_preds = vmap(lambda Deltas: convert_to_projs(Deltas, anchor_point, reorthonormalize=False))(Deltas_preds)

    for ws in Ws_test_preds:
        assert vmap(lambda w: valid_grass_point(w, tol=tol))(ws).all()
        
    if save_results:
        pickle_save(Ws_test_means, "Ws_test_means.pickle")
        pickle_save(Ws_test_preds, "Ws_test_preds.pickle")
        
    if plot_figs:
        plt.rcParams["figure.figsize"] = (12,6)
        percentile_levels = [2.5, 97.5]
        conf_level = percentile_levels[-1] - percentile_levels[0]
        for i in range(d):
            obs = Ws_train[:,i,0]
            means = Ws_test_means[:,:,i,0]
            means_avg = np.mean(means, axis=0)
            preds = Ws_test_preds[:,:,i,0]
            percentiles = np.percentile(preds, np.array(percentile_levels), axis=0)
            lower = percentiles[0,:]
            upper = percentiles[1,:]
            plt.plot(s_test, Ws_test[:,i,0], label='full data',c='black', alpha=0.75, linestyle='dashed')
            plt.scatter(s_train, Ws_train[:,i,0], label='training data', c='g')
            plt.plot(s_test, means_avg, label='averaged mean prediction', c='r', alpha=0.75)
            plt.fill_between(s_test, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
            plt.xlabel(r"$s$")
            plt.legend()
            plt.vlines(s_train, 0.99*lower.min(), 1.01*upper.max(), colors='green', linestyles='dashed')
            plt.title(f"{i+1}th component of projections")
            plt.show()
            
    # compute barycenters of test means and preds
    test_means_mcmc_barycenters = []
    test_means_results = []
    test_means_inits = []
    for i in tqdm(range(s_test.shape[0])):
        barycenter, result, mu_0 = sample_karcher_mean(Ws_test_means[:,i,:,:])
        test_means_mcmc_barycenters.append(barycenter)
        test_means_results.append(result)
        test_means_inits.append(mu_0)
        
    test_preds_mcmc_barycenters = []
    test_preds_results = []
    test_preds_inits = []
    for i in tqdm(range(s_test.shape[0])):
        barycenter, result, mu_0 = sample_karcher_mean(Ws_test_preds[:,i,:,:])
        test_preds_mcmc_barycenters.append(barycenter)
        test_preds_results.append(result)
        test_preds_inits.append(mu_0)
        
    
    test_means_mcmc_barycenters = np.array(test_means_mcmc_barycenters)
    test_preds_mcmc_barycenters = np.array(test_preds_mcmc_barycenters)
    
    if save_results:
        pickle_save(test_means_mcmc_barycenters, "test_means_mcmc_barycenters.pickle")
        pickle_save(test_preds_mcmc_barycenters, "test_preds_mcmc_barycenters.pickle")
        
    sd_s_test = []
    for i in tqdm(range(s_test.shape[0])):
        fixed = test_preds_mcmc_barycenters[i]
        dists = vmap(lambda W: grass_dist(W, fixed))(Ws_test_preds[:,i,:,:])
        dists_Sq = dists**2
        sd_s_test.append(np.sqrt(dists_Sq.mean()))
        
    sd_s_test = np.array(sd_s_test)
    out_sample_mean_errors = vmap(grass_dist)(Ws_test, test_means_mcmc_barycenters)
    out_sample_pred_errors = vmap(grass_dist)(Ws_test, test_preds_mcmc_barycenters)
    
    if plot_figs:
        plt.plot(s_test,out_sample_mean_errors, label='out of sample errors using means')
        plt.plot(s_test,out_sample_pred_errors, label='out of sample errors using preds')
        plt.vlines(s_train,ymin=0,ymax=0.00575, colors='green',linestyles='dotted')
        plt.legend()
        plt.show()
        
    test_pd_data = {'s': s_test, 'errors_mean': out_sample_mean_errors, 'errors_pred': out_sample_pred_errors, 'sd': sd_s_test}
    out_sample_errors_df = pd.DataFrame(data=test_pd_data)
    
    if save_results:
        pickle_save(out_sample_errors_df, "out_sample_errors_df.pickle")

# %%
jobs = launch(
    Config,
    train_analyse,
    overrides=[
        "plots.plot=False",
        "model.savvas_param=True,False",
        "model.b=0.001,0.01,0.1,0.5",
        "model.cov_jitter=0.0001,0.00001",
    ],
    version_base="1.1",
    multirun=True,
)
