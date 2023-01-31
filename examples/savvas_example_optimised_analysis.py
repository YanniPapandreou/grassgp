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
import os
import time
from hydra_zen import instantiate, make_config, builds, launch, load_from_yaml, to_yaml
from pathlib import Path
import sys

import chex
from typing import Tuple

# %%
import jax.numpy as np
from jax import random, vmap, grad, jit
from scipy.stats import special_ortho_group
from numpy.random import seed as set_numpy_seed

# %%
from grassgp.utils import get_save_path
from grassgp.utils import safe_save_jax_array_dict as safe_save
from grassgp.utils import load_and_convert_to_samples_dict as load_data
from grassgp.grassmann import valid_grass_point, grass_log, compute_barycenter, grass_exp, convert_to_projs, grass_dist, sample_karcher_mean
from grassgp.kernels import rbf
from grassgp.models_optimised import GrassGP
from grassgp.means import zero_mean
from grassgp.plot_utils import flatten_samples, plot_grass_dists

# %%
import numpyro
# from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_value
import numpyro.distributions as dist

# %%
from tqdm import tqdm
import pandas as pd
import statsmodels.api as sm

# %%
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (10,6)


# %%
def print_file(x: Path):
    with x.open("r") as f:
        print(f.read())


# %%
base_path = Path(os.getcwd())
output_folder = "outputs"
date = "2023-01-21"
time = "16-14-08"
# tol = 1e-05
# id_num = "1"

# %%
# job path
job_path = base_path / output_folder / date / time
assert job_path.exists()

# %%
dataset_path = job_path / "training_test_data.npz"
assert dataset_path.exists()

# %%
training_test_data = load_data(str(dataset_path))

# %%
training_test_data.keys()

# %%
s_train = np.array(training_test_data['s_train'])
s_test = np.array(training_test_data['s_test'])

Ws_train = np.array(training_test_data['Ws_train'])
Ws_test = np.array(training_test_data['Ws_test'])

log_Ws_train = np.array(training_test_data['log_Ws_train'])
log_Ws_test = np.array(training_test_data['log_Ws_test'])

anchor_point = np.array(training_test_data['anchor_point'])

# %%
d, n = anchor_point.shape

# %%
plt.scatter(s_train[:,0],s_train[:, 1], label='training locs')
plt.scatter(s_test[:,0],s_test[:, 1], label='test locs')
plt.legend()
plt.show()

# %%
# load config in
config = load_from_yaml(job_path / ".hydra" / "config.yaml")
overrides = load_from_yaml(job_path / ".hydra" / "overrides.yaml")

# %%
print(to_yaml(config))

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
model_config = {
    'anchor_point': anchor_point.tolist(),
    'd_in': 2,
    'Omega_diag_chol' : None,
    # 'Omega_diag_chol' : Omega.tolist(),
    'k_include_noise': True,
    'var' : 1.0,
    'length' : None, 
    'noise' : None,
    'require_noise' : False,
    'jitter' : 1e-06,
    'cov_jitter' : 1e-4,
    'L_jitter' : 1e-8,
    'reorthonormalize' : False,
    # 'b' : 0.5, # mine
    'b' : 0.001, # savvas
    # 'ell': 0.0075
    # 'ell': 0.01, # mine
    'use_kron_chol': False
}
def model(s, log_Ws, grass_config = model_config):
    U = np.array(grass_config['anchor_point'])
    d, n = U.shape
    N = s.shape[0]
    d_n = d * n
    # N_params = N * d_n
    if log_Ws is not None:
        assert log_Ws.shape == (N, d, n), f"log_Ws.shape=({log_Ws.shape}) instead of {(N, d, n)} as expected"

    # get/sample Omega
    if grass_config['Omega_diag_chol'] is None:
        # full example
        # sigmas = numpyro.sample('sigmas', dist.LogNormal(0.0, 1.0).expand([d_n]))
        # L_factor = numpyro.sample('L_factor', dist.LKJ(d_n, 1.0))
        # L = numpyro.deterministic('L', L_factor + grass_config['L_jitter'] * np.eye(d_n))
        # Omega = numpyro.deterministic('Omega', np.outer(sigmas, sigmas) * L)
        # simpler diagonal structure
        # sigmas = numpyro.sample('sigmas', dist.LogNormal(0.0, 1.0).expand([d_n]))
        # Omega_diag = numpyro.deterministic('Omega_diag', sigmas**2)
        Omega_diag_chol = numpyro.sample('Omega_diag_chol', dist.LogNormal(0.0, 1.0).expand([d_n]))
    else:
        Omega_diag_chol = np.array(grass_config['Omega_diag_chol'])
        
    # get/sample kernel params
    if grass_config['var'] is None:
        # sample var
        var = numpyro.sample("kernel_var", dist.LogNormal(0.0, grass_config['b']))
    else:
        var = grass_config['var']

    if grass_config['length'] is None:
        # sample length
        # # ! my parametrisation
        # length = numpyro.sample("kernel_length", dist.LogNormal(0.0, grass_config['b']))
        # # ! savvas parametrisation
        length = numpyro.sample("kernel_length", dist.LogNormal(-0.7, grass_config['b']))
    else:
        length = grass_config['length']

    if grass_config['require_noise']:
        if grass_config['noise'] is None:
            # sample noise
            noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, grass_config['b']))
        else:
            noise = grass_config['noise']
    else:
        noise = 0.0
    

    # kernel_params = {'var': var, 'length': length, 'noise': noise} # mine
    kernel_params = {'var': var, 'length': np.sqrt(1/length), 'noise': noise} # savvas
    # create kernel function
    k = lambda t, s: rbf(t, s, kernel_params, jitter=grass_config['jitter'], include_noise=grass_config['k_include_noise'])
    # create mean function
    mu = lambda s: zero_mean(s, d, n)

    # initialize GrassGp
    grass_gp = GrassGP(d_in=grass_config['d_in'], d_out=(d,n), mu=mu, k=k, Omega_diag_chol=Omega_diag_chol, U=U, cov_jitter=grass_config['cov_jitter'])

    # sample Deltas
    Deltas = grass_gp.tangent_model(s)

    # # # # # ! check what power this should be
    # likelihood
    # ell = grass_config['ell'] # mine
    ell = numpyro.sample("ell", dist.LogNormal(-6, 0.0015))
    with numpyro.plate("N", N):
        numpyro.sample("log_W", dist.continuous.MatrixNormal(loc=Deltas, scale_tril_row=ell * np.eye(d), scale_tril_column=np.eye(n)), obs=log_Ws)

TangentSpaceModelConf = builds(model, grass_config=model_config, zen_partial=True)

# %%
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

Config = make_config(
    model = TangentSpaceModelConf,
    svi = SVIConfig,
    train = TrainConfig
)

# %%
numpyro.render_model(instantiate(Config.model), model_args=(s_train,log_Ws_train))

# %% tags=[]
print_file(job_path / "hmc_log.txt")

# %%
my_samples = flatten_samples(samples, ignore=[])

# %%
samples.keys()

# %%
trace_plot_vars = ['kernel_length', 'ell']
for key in my_samples.keys():
    if 'Omega' in key:
        trace_plot_vars.append(key)

my_samples[trace_plot_vars].plot(subplots=True, figsize=(10,40), sharey=False)
plt.show()

# %% tags=[] jupyter={"outputs_hidden": true}
for var in trace_plot_vars:
    acf_plot = pd.plotting.autocorrelation_plot(my_samples['ell'],)
    acf_plot.plot()
    # plt.ylim((-.25,.25))
    plt.title(f"acf for {var}")
    plt.show()

# %% tags=[] jupyter={"outputs_hidden": true}
for var in trace_plot_vars:
    plt.acorr((my_samples[var] - my_samples[var].mean())/my_samples[var].std())
    plt.grid()
    plt.title(f"acf for {var}")
    plt.show()

# %% tags=[] jupyter={"outputs_hidden": true}
for var in trace_plot_vars:
    sm.graphics.tsa.plot_acf(my_samples[var], lags=100)
    plt.title(f"acf for {var}")
    plt.show()

# %%
tol=1e-5

samples_Ws_train = vmap(lambda Deltas: convert_to_projs(Deltas, anchor_point, reorthonormalize=False))(samples['Deltas'])

for ws in samples_Ws_train:
    assert vmap(lambda w: valid_grass_point(w, tol=tol))(ws).all()

# %%
mcmc_barycenters = []
results = []
inits = []
for i in tqdm(range(s_train.shape[0])):
    barycenter, result, mu_0 = sample_karcher_mean(samples_Ws_train[:,i,:,:])
    mcmc_barycenters.append(barycenter)
    results.append(result)
    inits.append(mu_0)

# %%
mcmc_barycenters = np.array(mcmc_barycenters)

# %%
bary_losses = []
for i in tqdm(range(s_train.shape[0])):
    loss = (vmap(lambda W: grass_dist(mcmc_barycenters[i], W))(samples_Ws_train[:,i,:,:]) ** 2).sum()
    bary_losses.append(loss)

plt.plot(bary_losses)
plt.title("Final loss for computed barycenters")
plt.show()

# %% jupyter={"source_hidden": true} tags=[]
# def compute_losses(mu_s, points):
#     def cost(X):
#         dists = vmap(lambda W: grass_dist(X, W))(points)
#         dists_Sq = dists ** 2
#         return dists_Sq.sum()
    
#     return vmap(cost)(mu_s)

# losses = []
# for i in tqdm(range(s_train.shape[0])):
#     loss = compute_losses(results[i], samples_Ws_train[:,i,:,:])
#     losses.append(loss)

# losses = np.array(losses)

# losses.shape

# plt.plot(losses.T)
# plt.show()

# %%
barycenter_save_path = str(job_path) + "/mcmc_barycenters.npz"
if os.path.exists(barycenter_save_path):
    print("File exists so not saving")
else:
    np.savez(barycenter_save_path, mcmc_barycenters)

# %%
in_sample_errors = vmap(grass_dist)(Ws_train, mcmc_barycenters)

# %%
plt.plot(in_sample_errors)
plt.show()

# %%
pd_data = {'x': s_train[:,0], 'y': s_train[:,1], 'errors': in_sample_errors}
in_sample_errors_df = pd.DataFrame(data=pd_data)
in_sample_errors_df.head()

# %%
in_sample_errors_df.describe()

# %%
sns.scatterplot(data=in_sample_errors_df, x="x", y="y", size="errors", legend=True, alpha=0.75)
plt.title("In sample errors")
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
    
    cfg_Omega_diag_chol = dict_cfg.model.grass_config.Omega_diag_chol
    cfg_var = dict_cfg.model.grass_config.var
    cfg_length = dict_cfg.model.grass_config.length
    cfg_noise = dict_cfg.model.grass_config.noise
    cfg_require_noise = dict_cfg.model.grass_config.require_noise
    
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
pred_key = random.PRNGKey(6578)
Deltas_means, Deltas_preds = predict_tangents(pred_key, s_test, s_train, log_Ws_train, config, samples)
assert np.isnan(Deltas_means).sum() == 0
assert np.isnan(Deltas_preds).sum() == 0

# %%
Deltas_means_save_path = str(job_path) + "/Deltas_means.npz"
if os.path.exists(Deltas_means_save_path):
    print("File exists so not saving")
else:
    np.savez(Deltas_means_save_path, Deltas_means)

# %%
Deltas_preds_save_path = str(job_path) + "/Deltas_preds.npz"
if os.path.exists(Deltas_preds_save_path):
    print("File exists so not saving")
else:
    np.savez(Deltas_preds_save_path, Deltas_preds)

# %%
Ws_test_means = vmap(lambda Deltas: convert_to_projs(Deltas, anchor_point, reorthonormalize=False))(Deltas_means)

for ws in Ws_test_means:
    assert vmap(lambda w: valid_grass_point(w, tol=tol))(ws).all()

# %%
Ws_test_preds = vmap(lambda Deltas: convert_to_projs(Deltas, anchor_point, reorthonormalize=False))(Deltas_preds)

for ws in Ws_test_preds:
    assert vmap(lambda w: valid_grass_point(w, tol=tol))(ws).all()

# %%
Ws_test_means.shape

# %%
Ws_test_preds.shape

# %%
test_means_mcmc_barycenters = []
test_means_results = []
test_means_inits = []
for i in tqdm(range(s_test.shape[0])):
    barycenter, result, mu_0 = sample_karcher_mean(Ws_test_means[:,i,:,:])
    test_means_mcmc_barycenters.append(barycenter)
    test_means_results.append(result)
    test_means_inits.append(mu_0)

# %%
test_preds_mcmc_barycenters = []
test_preds_results = []
test_preds_inits = []
for i in tqdm(range(s_test.shape[0])):
    barycenter, result, mu_0 = sample_karcher_mean(Ws_test_preds[:,i,:,:])
    test_preds_mcmc_barycenters.append(barycenter)
    test_preds_results.append(result)
    test_preds_inits.append(mu_0)

# %%
test_means_mcmc_barycenters = np.array(test_means_mcmc_barycenters)

# %%
test_preds_mcmc_barycenters = np.array(test_preds_mcmc_barycenters)

# %%
test_means_mcmc_barycenter_save_path = str(job_path) + "/test_means_mcmc_barycenters.npz"
if os.path.exists(test_means_mcmc_barycenter_save_path):
    print("File exists so not saving")
else:
    np.savez(test_means_mcmc_barycenter_save_path, test_means_mcmc_barycenters)

# %%
test_preds_mcmc_barycenter_save_path = str(job_path) + "/test_preds_mcmc_barycenters.npz"
if os.path.exists(test_preds_mcmc_barycenter_save_path):
    print("File exists so not saving")
else:
    np.savez(test_preds_mcmc_barycenter_save_path, test_preds_mcmc_barycenters)

# %%
out_sample_mean_errors = vmap(grass_dist)(Ws_test, test_means_mcmc_barycenters)
out_sample_pred_errors = vmap(grass_dist)(Ws_test, test_preds_mcmc_barycenters)

# %%
plt.plot(out_sample_mean_errors, label='error using means')
plt.plot(out_sample_pred_errors, label='error using preds')
plt.legend()
plt.show()

# %%
test_pd_data = {'x': s_test[:,0], 'y': s_test[:,1], 'errors_mean': out_sample_mean_errors, 'errors_pred': out_sample_pred_errors}
out_sample_errors_df = pd.DataFrame(data=test_pd_data)
out_sample_errors_df.head()

# %%
out_sample_errors_df.describe()

# %%
sns.scatterplot(data=out_sample_errors_df, x="x", y="y", size="errors_mean", legend=True, alpha=0.75)
plt.title("out sample errors using means")
plt.show()

# %%
sns.scatterplot(data=out_sample_errors_df, x="x", y="y", size="errors_pred", legend=True, alpha=0.75)
plt.title("out sample errors using preds")
plt.show()

# %%
sns.scatterplot(data=in_sample_errors_df, x="x", y="y", size="errors", legend=False, label='in sample', alpha=0.75, c='blue')
sns.scatterplot(data=out_sample_errors_df, x="x", y="y", size="errors_mean", legend=False, label='out sample using means', alpha=0.75, c='red')
sns.scatterplot(data=out_sample_errors_df, x="x", y="y", size="errors_pred", legend=False, label='out sample using preds', alpha=0.75, c='purple')
plt.title("in and out of sample errors")
plt.legend()
plt.show()

# %%
test_preds_mcmc_barycenters.shape

# %%
import procrustes
import numpy

# %%
test_preds_mcmc_barycenters_aligned, dist_gpa_test_preds = procrustes.generalized(numpy.array(test_preds_mcmc_barycenters))

# %%
test_preds_mcmc_barycenters_aligned = np.array(test_preds_mcmc_barycenters_aligned)

# %%
Ws_train_aligned, dist_gpa_train = procrustes.generalized(numpy.array(Ws_train))
Ws_train_aligned = np.array(Ws_train_aligned)

Ws_test_aligned, dist_gpa_train = procrustes.generalized(numpy.array(Ws_test))
Ws_test_aligned = np.array(Ws_test_aligned)

# %% jupyter={"outputs_hidden": true} tags=[]
plt.plot(vmap(grass_dist)(Ws_train_aligned, Ws_train))
plt.show()

plt.plot(vmap(grass_dist)(Ws_test_aligned, Ws_test))
plt.show()

plt.plot(vmap(grass_dist)(test_preds_mcmc_barycenters_aligned, test_preds_mcmc_barycenters))
plt.show()

# %%
from sklearn.decomposition import PCA

# %%
pca_train_data = PCA(n_components=2)
pca_train_data.fit(numpy.array(Ws_train_aligned[:,:,0]))
print(f"{pca_train_data.explained_variance_ratio_ = }")
print(f"{pca_train_data.singular_values_ = }")
Ws_train_aligned_pca_comps = np.array(pca_train_data.transform(numpy.array(Ws_train_aligned[:,:,0])))

# %%
pca_test_data = PCA(n_components=2)
pca_test_data.fit(numpy.array(Ws_test_aligned[:,:,0]))
print(f"{pca_test_data.explained_variance_ratio_ = }")
print(f"{pca_test_data.singular_values_ = }")
Ws_test_aligned_pca_comps = np.array(pca_test_data.transform(numpy.array(Ws_test_aligned[:,:,0])))

# %%
pca_test_preds = PCA(n_components=2)
pca_test_preds.fit(numpy.array(test_preds_mcmc_barycenters_aligned[:,:,0]))
print(f"{pca_test_preds.explained_variance_ratio_ = }")
print(f"{pca_test_preds.singular_values_ = }")
Ws_test_preds_aligned_pca_comps = np.array(pca_test_preds.transform(numpy.array(test_preds_mcmc_barycenters_aligned[:,:,0])))

# %%
test_preds_mcmc_barycenters.shape

# %%
Ws_test_preds.shape

# %%
vmap(lambda W: grass_dist(W,test_preds_mcmc_barycenters[0]))(Ws_test_preds[:,0,:,:])

# %%
sd_s = []
for i in tqdm(range(s_test.shape[0])):
    fixed = test_preds_mcmc_barycenters[i]
    dists = vmap(lambda W: grass_dist(W, fixed))(Ws_test_preds[:,i,:,:])
    dists_Sq = dists**2
    sd_s.append(np.sqrt(dists_Sq.mean()))


# %%
sd_s = np.array(sd_s)

# %%
Ws_train_aligned_pca_comps.shape

# %%
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %%
norm = matplotlib.colors.Normalize()
norm.autoscale(sd_s)
cm = matplotlib.cm.plasma

sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])

fig, ax = plt.subplots()
ax.quiver(s_train[:,0], s_train[:,1], Ws_train_aligned_pca_comps[:,0], Ws_train_aligned_pca_comps[:,1], color='blue', label='training data')
ax.quiver(s_test[:,0], s_test[:,1], Ws_test_preds_aligned_pca_comps[:,0], Ws_test_preds_aligned_pca_comps[:,1], color=cm(norm(sd_s)))
ax.grid()
ax.set_xlabel('pca_1')
ax.set_ylabel('pca_2')
fig.legend()
# create an Axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
plt.colorbar(sm, cax=cax)
plt.show()

# %%
