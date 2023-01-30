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

# %%
import jax.numpy as np
from jax import random, vmap, grad
from scipy.stats import special_ortho_group
from numpy.random import seed as set_numpy_seed

# %%
from grassgp.utils import get_save_path
from grassgp.utils import safe_save_jax_array_dict as safe_save
from grassgp.utils import load_and_convert_to_samples_dict as load_data
from grassgp.grassmann import valid_grass_point, grass_log, compute_barycenter, grass_exp, convert_to_projs, grass_dist
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

# %% jupyter={"outputs_hidden": true} tags=[]
for var in trace_plot_vars:
    acf_plot = pd.plotting.autocorrelation_plot(my_samples['ell'],)
    acf_plot.plot()
    # plt.ylim((-.25,.25))
    plt.title(f"acf for {var}")
    plt.show()

# %% jupyter={"outputs_hidden": true} tags=[]
for var in trace_plot_vars:
    plt.acorr((my_samples[var] - my_samples[var].mean())/my_samples[var].std())
    plt.grid()
    plt.title(f"acf for {var}")
    plt.show()

# %% jupyter={"outputs_hidden": true} tags=[]
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
for i in tqdm(range(s_train.shape[0])):
    barycenter = compute_barycenter(samples_Ws_train[:,i,:,:])
    mcmc_barycenters.append(barycenter)

# %%
samples_Ws_train.shape

# %%
Ws_train.shape

# %%
compute_dists_at_single_loc = lambda i: vmap(lambda proj: grass_dist(Ws_train[i], proj[i]))(samples_Ws_train)
dists = vmap(compute_dists_at_single_loc)(np.arange(s_train.shape[0]))

# %%
dists.shape

# %%
plt.plot(dists.mean(axis=1))

# %%
pd_data = {'x': s_train[:,0], 'y': s_train[:,1], 'errors': dists.mean(axis=1)}
in_sample_errors_df = pd.DataFrame(data=pd_data)
in_sample_errors_df.head()

# %%
sns.scatterplot(data=in_sample_errors_df, x="x", y="y", size="errors", legend=True, alpha=0.75)
plt.title("In sample errors")
plt.show()

# %%
