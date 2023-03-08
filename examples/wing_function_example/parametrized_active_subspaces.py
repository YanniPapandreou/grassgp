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
from hydra_zen import instantiate, make_config, builds, to_yaml, load_from_yaml
import sys

import jax.numpy as np
from jax import vmap, random, grad
import jax.numpy.linalg as lin
from functools import partial
from tqdm import tqdm

from grassgp.grassmann import valid_grass_point, sample_karcher_mean, grass_dist, grass_exp, valid_grass_tangent, grass_log, convert_to_projs, compute_barycenter
from grassgp.means import zero_mean
from grassgp.kernels import rbf
from grassgp.models_optimised import GrassGP
from grassgp.plot_utils import flatten_samples, plot_grass_dists
from grassgp.utils import to_dictconf, get_save_path
from grassgp.utils import safe_save_jax_array_dict as safe_save
from grassgp.utils import load_and_convert_to_samples_dict as load_data

import chex
from typing import Tuple

import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_value
import numpyro.distributions as dist

from grassgp.inference import run_inference

import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,6)

# %% [markdown]
# # Generate dataset

# %%
M = 1000 #This is the number of data points to use
#Sample the input space according to the distributions in the table above
key = random.PRNGKey(6753)
keys = random.split(key, 10)
Sw = random.uniform(keys[0], shape=(M,1), minval=150, maxval=200)
Wfw = random.uniform(keys[1], shape=(M,1), minval=220, maxval=300)
A = random.uniform(keys[2], shape=(M,1), minval=6, maxval=10)
L = random.uniform(keys[3], shape=(M,1), minval=-10, maxval=10)
q = random.uniform(keys[4], shape=(M,1), minval=16, maxval=45)
l = random.uniform(keys[5], shape=(M,1), minval=.5, maxval=1)
tc = random.uniform(keys[6], shape=(M,1), minval=.08, maxval=.18)
Nz = random.uniform(keys[7], shape=(M,1), minval=2.5, maxval=6)
Wdg = random.uniform(keys[8], shape=(M,1), minval=1700, maxval=2500)
Wp = random.uniform(keys[9], shape=(M,1), minval=.025, maxval=.08)

# %%
#Upper and lower limits for inputs
lb = np.array([150, 220, 6, -10, 16, .5, .08, 2.5, 1700, .025])
ub = np.array([200, 300, 10, 10, 45, 1, .18, 6, 2500, .08])

# %%
tau = lambda x: 0.5*((ub-lb) * x + ub + lb) # goes from [-1,1] -> [l,u]
tau_inv = lambda x: 2.0*(x - lb)/(ub - lb) - 1.0 # goes from [l, u] -> [-1,1]

# %%
wing_func = lambda Sw, Wfw, A, L, q, l, tc, Nz, Wdg, Wp: .036*Sw**.758*Wfw**.0035*A**.6*np.cos(L * np.pi/180)**-.9*q**.006*l**.04*100**-.3*tc**-.3*Nz**.49*Wdg**.49 + Sw*Wp


# %%
def wing_func_norm(x_norm):
    x = tau(x_norm)
    assert x.shape == (10,)
    Sw, Wfw, A, L, q, l, tc, Nz, Wdg, Wp = x
    return wing_func(Sw, Wfw, A, L, q, l, tc, Nz, Wdg, Wp)


# %%
x = np.hstack((Sw, Wfw, A, L, q, l, tc, Nz, Wdg, Wp))
XX = vmap(tau_inv)(x)
print(f"{XX.shape = }")

# %%
vmap(wing_func_norm)(XX).shape


# %%
def f_parametrized(Sw_norm, x_norm_remaining):
    x_norm = np.hstack((Sw_norm, x_norm_remaining))
    assert x_norm.shape == (10,)
    return wing_func_norm(x_norm)


# %%
Sw_norm_tmp = XX[0][0]
x_norm_remaining_tmp = XX[0][1:]
assert f_parametrized(Sw_norm_tmp, x_norm_remaining_tmp) == vmap(wing_func_norm)(XX)[0]

# %%
f = partial(f_parametrized,Sw_norm_tmp)
assert f(x_norm_remaining_tmp) == f_parametrized(Sw_norm_tmp, x_norm_remaining_tmp)

# %%
grad_f = grad(f)
assert (grad_f(x_norm_remaining_tmp) == grad(wing_func_norm)(XX[0])[1:]).all()


# %%
def C_hat(Sw_norm,x_norm_remaining):
    f = partial(f_parametrized, Sw_norm)
    grad_f = grad(f)
    assert x_norm_remaining.shape[1] == 9
    df = vmap(grad_f)(x_norm_remaining)
    C_x = vmap(lambda gradient: np.outer(gradient, gradient))(df)
    assert C_x.shape[1:] == (9,9)
    return C_x.mean(axis=0)


# %%
assert ((2*(Sw-lb[0])/(ub[0]-lb[0]) - 1.0).flatten() == XX[:,0]).all()

# %%
C_hat_s = vmap(C_hat, in_axes=(0, None))(XX[:,0],XX[:,1:])

# %%
print(f"{C_hat_s.shape = }")

# %%
# for i, sw in tqdm(enumerate(XX[:,0].flatten())):
#     assert (C_hat(sw, XX[:,1:]) == C_hat_s[i]).all()

# %%
C = C_hat_s[0]
C.shape


# %%
def get_AS(C):
    eigs, eig_vecs = lin.eigh(C)
    inds = eigs.argsort()[::-1]
    max_eig = eigs[inds[0]]
    prop = max_eig / eigs.sum()
    w = eig_vecs[:,inds[0]]
    return w, max_eig, prop


# %%
Ws, max_eigs, props = vmap(get_AS)(C_hat_s)
Ws = Ws[:,:,None]
assert Ws.shape == (M,9,1)
assert max_eigs.shape == (M,)
assert props.shape == (M,)
assert (vmap(valid_grass_point)(Ws)).all()

# %%
props.mean()

# %%
props.var()

# %%
plt.plot(props)
plt.ylim(0.9,1.0)
plt.show()

# %%
plt.plot(vmap(lambda W: grass_dist(W, Ws[0]))(Ws))
plt.ylim(-0.001,0.01)
plt.show()


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
s_inds = XX[:,0].argsort()
s_gap = 100
s_train = XX[s_inds,0][::s_gap].copy()
Ws_train = Ws[s_inds][::s_gap].copy()
n_train = s_train.shape[0]

# %%
for i, i_s in enumerate(s_inds[::s_gap]):
    assert s_train[i] == XX[i_s,0]
    assert (Ws_train[i] == Ws[i_s]).all()

# %%
plt.scatter(range(n_train),s_train)
plt.show()

# %%
# compute barycenter of training data
anchor_point, _, _ = sample_karcher_mean(Ws_train)
assert valid_grass_point(anchor_point)

# %%
# compute log of training data
log_Ws_train = vmap(lambda W: grass_log(anchor_point, W))(Ws_train)

# %%
model_config = {
    'anchor_point': anchor_point.tolist(),
    'd_in': 1,
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
    'b' : 0.5, # mine
    # 'b' : 0.001, # savvas
    # 'ell': 0.0075
    # 'ell': 0.01, # mine
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
def train(cfg):
    # instantiate grass model
    model = instantiate(cfg.model)

    # # save training-test data
    # head = os.getcwd()
    # training_test_data = {'s_train': s_train, 's_test': s_test, 'Ws_train': Ws_train, 'Ws_test': Ws_test, 'log_Ws_train': log_Ws_train, 'log_Ws_test': log_Ws_test, 'anchor_point': anchor_point}
    # training_test_main_name = "training_test_data"
    # training_test_path = get_save_path(head, training_test_main_name)
    # try:
    #     safe_save(training_test_path, training_test_data)
    # except FileExistsError:
    #     print("File exists so not saving.")
    
    # run SVI to get MAP esimtate to initialise MCMC
    svi_key = random.PRNGKey(cfg.svi.seed)
    maxiter = cfg.svi.maxiter
    step_size = cfg.svi.step_size
    print("Running SVI for MAP estimate to initialise MCMC")
    svi_results = run_svi_for_map(svi_key, model, maxiter, step_size, s_train, log_Ws_train)
    
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
    # original_stdout = sys.stdout
    # with open('hmc_log.txt', 'w') as f:
    #     sys.stdout = f
    #     mcmc.print_summary()
    #     sys.stdout = original_stdout
    
    samples = mcmc.get_samples()
    inference_data = samples.copy()
    for param, initial_val in init_values.items():
        inference_data[f"{param}-initial_value"] = initial_val
    
    # inference_main_name = "inference_data"
    # inference_path = get_save_path(head, inference_main_name)
    # try:
    #     safe_save(inference_path, inference_data)
    # except FileExistsError:
    #     print("File exists so not saving.")
    return inference_data


# %%
inference_data = train(Config)

# %%
head = os.getcwd()
inference_main_name = "inference_data"
inference_path = get_save_path(head + '/saved_results', inference_main_name)
try:
    safe_save(inference_path, inference_data)
except FileExistsError:
    print("File exists so not saving.")

# %%
# head = os.getcwd()
# inference_data = {}
# for key, value in load_data(head + '/saved_results/inference_data.npz').items():
#     inference_data[key] = np.array(value)

# %%
samples = dict(filter(lambda elem: 'initial_value' not in elem[0], inference_data.items()))
initial_values = dict(filter(lambda elem: 'initial_value' in elem[0], inference_data.items()))
assert set(samples.keys()).union(initial_values.keys()) == set(inference_data.keys())

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

# %% tags=[]
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
barycenter_save_path = head + "/saved_results/mcmc_barycenters.npz"
if os.path.exists(barycenter_save_path):
    print("File exists so not saving")
else:
    np.savez(barycenter_save_path, mcmc_barycenters)

# %%
# mcmc_barycenters = np.array(load_data(head + "/saved_results/mcmc_barycenters.npz")['arr_0'])

# %%
bary_losses = []
for i in tqdm(range(s_train.shape[0])):
    loss = (vmap(lambda W: grass_dist(mcmc_barycenters[i], W))(samples_Ws_train[:,i,:,:]) ** 2).sum()
    bary_losses.append(loss)

plt.plot(bary_losses)
plt.title("Final loss for computed barycenters")
plt.show()

# %%
in_sample_errors = vmap(grass_dist)(Ws_train, mcmc_barycenters)

# %%
plt.plot(s_train,in_sample_errors)
plt.show()

# %%
sd_s_train = []
for i in tqdm(range(s_train.shape[0])):
    fixed = mcmc_barycenters[i]
    dists = vmap(lambda W: grass_dist(W, fixed))(samples_Ws_train[:,i,:,:])
    dists_Sq = dists**2
    sd_s_train.append(np.sqrt(dists_Sq.mean()))

# %%
sd_s_train = np.array(sd_s_train)

# %%
pd_data = {'s': s_train, 'errors': in_sample_errors, 'sd': sd_s_train}
in_sample_errors_df = pd.DataFrame(data=pd_data)
in_sample_errors_df.head()

# %%
in_sample_errors_df.describe()

# %%
s_test = XX[s_inds,0].copy()
Ws_test = Ws[s_inds].copy()
log_Ws_test = vmap(lambda W: grass_log(anchor_point, W))(Ws_test)


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
config = to_dictconf(Config)

# %%
# %%time
pred_key = random.PRNGKey(6578)
splits = 25
pred_keys = random.split(pred_key, splits)
pred_results_chunked = {}
for (i, s_test_chunk) in tqdm(enumerate(np.split(s_test, splits))):
    p_key = pred_keys[i]
    pred_results_chunked[i] = predict_tangents(p_key, s_test_chunk, s_train, log_Ws_train, config, samples)

# %%
Deltas_means_list = []
Deltas_preds_list = []
for i in range(splits):
    means, preds = pred_results_chunked[i]
    Deltas_means_list.append(means)
    Deltas_preds_list.append(preds)

# %%
Deltas_means = np.concatenate(Deltas_means_list, axis=1)
Deltas_preds = np.concatenate(Deltas_preds_list, axis=1)

# %%
assert np.isnan(Deltas_means).sum() == 0
assert np.isnan(Deltas_preds).sum() == 0

# %%
Deltas_means_save_path = head + "/saved_results/Deltas_means.npz"
if os.path.exists(Deltas_means_save_path):
    print("File exists so not saving")
else:
    np.savez(Deltas_means_save_path, Deltas_means)

# %%
# Deltas_means = np.array(load_data(head + "/saved_results/Deltas_means.npz")['arr_0'])

# %%
Deltas_preds_save_path = head + "/saved_results/Deltas_preds.npz"
if os.path.exists(Deltas_preds_save_path):
    print("File exists so not saving")
else:
    np.savez(Deltas_preds_save_path, Deltas_preds)

# %%
# Deltas_preds = np.array(load_data(head + "/saved_results/Deltas_preds.npz")['arr_0'])

# %%
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
test_means_mcmc_barycenter_save_path = head + "/saved_results/test_means_mcmc_barycenters.npz"
if os.path.exists(test_means_mcmc_barycenter_save_path):
    print("File exists so not saving")
else:
    np.savez(test_means_mcmc_barycenter_save_path, test_means_mcmc_barycenters)

test_preds_mcmc_barycenter_save_path = head + "/saved_results/test_preds_mcmc_barycenters.npz"
if os.path.exists(test_preds_mcmc_barycenter_save_path):
    print("File exists so not saving")
else:
    np.savez(test_preds_mcmc_barycenter_save_path, test_preds_mcmc_barycenters)

# %%
# test_means_mcmc_barycenters = np.array(load_data(head + "/saved_results/test_means_mcmc_barycenters.npz")['arr_0'])
# test_preds_mcmc_barycenters = np.array(load_data(head + "/saved_results/test_preds_mcmc_barycenters.npz")['arr_0'])

# %%
sd_s_test = []
for i in tqdm(range(s_test.shape[0])):
    fixed = test_preds_mcmc_barycenters[i]
    dists = vmap(lambda W: grass_dist(W, fixed))(Ws_test_preds[:,i,:,:])
    dists_Sq = dists**2
    sd_s_test.append(np.sqrt(dists_Sq.mean()))


# %%
sd_s_test = np.array(sd_s_test)

# %%
out_sample_mean_errors = vmap(grass_dist)(Ws_test, test_means_mcmc_barycenters)
out_sample_pred_errors = vmap(grass_dist)(Ws_test, test_preds_mcmc_barycenters)

# %%
plt.plot(s_test,out_sample_mean_errors, label='out of sample errors using means')
plt.plot(s_test,out_sample_pred_errors, label='out of sample errors using preds')
plt.vlines(s_train,ymin=0,ymax=0.00575, colors='green',linestyles='dotted')
plt.legend()
plt.show()

# %%
test_pd_data = {'s': s_test, 'errors_mean': out_sample_mean_errors, 'errors_pred': out_sample_pred_errors, 'sd': sd_s_test}
out_sample_errors_df = pd.DataFrame(data=test_pd_data)
out_sample_errors_df.head()

# %%
out_sample_errors_df.describe()

# %%
