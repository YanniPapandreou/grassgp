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
#     display_name: projection_pursuit
#     language: python
#     name: projection_pursuit
# ---

# %% [markdown]
# # New version with map estimation at start

# %%
# # %load_ext autoreload
# # %autoreload 2
# %%
import time
import jax.numpy as np
import jax.numpy.linalg as lin
from jax import random, vmap

import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_value
from numpyro.handlers import scope

from projection_pursuit.utils import unvec, vec, kron_chol, get_save_path
from projection_pursuit.kernels import rbf_covariance
from projection_pursuit.models import grassmann_process, univariate_gp_model
from projection_pursuit.generate_data import generate_input_grid, gen_proj_from_grass_process, gen_from_gp
from projection_pursuit.grassmann import grass_dist, convert_to_projs
from projection_pursuit.inference import run_inference
from projection_pursuit.utils import safe_save_jax_array_dict as safe_save

# import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (12,8)

# %%
input_key = random.PRNGKey(4537658)
projs_key = random.PRNGKey(658769)
outer_key = random.PRNGKey(4357)
# params dict controls data generation
# D is dimension of full spatial part
# N is number of spatial points to create
# n_s is number of time points
# var, length and noise are kernel parameters for OUTER gp
params = {'D': 2, 'N_fine_sqrt': 50, 'n_s_fine': 10, 'active_dimension':1, 'var': 1.0, 'length': 0.5, 'noise': 0.1}
D = params['D']
N_fine_sqrt = params['N_fine_sqrt']

n_s_fine = params['n_s_fine']

active_dimension = params['active_dimension']

var = params['var']
length = params['length']
noise = params['noise']
outer_jitter = 1e-5

# use generate_inputs to draw spatio-temporal points (spatial part is random, temporal is not)
X_fine, s_fine = generate_input_grid(input_key, D, N_fine_sqrt, n_s_fine) 
anchor_point = np.eye(D, active_dimension)

# parameters for grass_process
# proj_locs = np.zeros(shape=(D*active_dimension*n_s_fine,))
# model_params = {'var': 1.0, 'Omega': np.eye(D * active_dimension), 'proj_locs': proj_locs}
model_params = {}
grass_inputs = {'s': s_fine, 'anchor_point': anchor_point,'model_params': model_params, 'jitter': 1e-06, 'proj_jitter': 1e-4, 'L_jitter': 1e-8, 'require_noise': False, 'reorthonormalize':True}

# draw random projections from grass_mann process prior
Ps_fine, Ps_params = gen_proj_from_grass_process(projs_key, **grass_inputs)

# project and collect data
X_fine_projs = np.einsum('ij,ljk->lik', X_fine, Ps_fine)

X_fine_projs_all = np.vstack([X_fine_projs[i,:,:] for i in range(n_s_fine)])

# sample from GP using projected data
# mean function for GP
def mu_f(x):
    # y = x + 0.2 * (x ** 3) + 0.5 * ((0.5 + x) ** 2) * np.sin(4.0 * x)
    # return y
    return 0.0

kernel_inputs = {'var': var, 'length': length, 'noise': noise, 'jitter': outer_jitter}
Y_fine = gen_from_gp(outer_key, X_fine_projs_all, mu_f, rbf_covariance, **kernel_inputs)

Ys_fine = unvec(Y_fine, N_fine_sqrt**2, n_s_fine)

Ys_fine_means = Ys_fine.mean(axis=0)

Ys_fine_centred = vmap(lambda y, m: y - m, in_axes=(1,0), out_axes=1)(Ys_fine, Ys_fine_means)

print('Data has been generated.')

# %%
# import pickle

# with open('../data/datasets/mcmc_initialised_with_map_data_2022-10-11--11:40-grass-params.pickle', 'wb') as f:
#     pickle.dump(Ps_params, f)

# %%
print(f"Grass_kernel_length_sampled: {Ps_params['kernel_length']}")


# %%
def sub_grid_inds(h_gap, v_gap, N_sqrt):
    inds = []
    for i in range(0,N_sqrt,h_gap):
        v_inds = [50 * i + j for j in range(0, N_sqrt, v_gap)]
        inds.extend(v_inds)
    return inds

# %%
subsample_key = random.PRNGKey(64879)
# N = 50
# inds = random.choice(subsample_key, N_fine_sqrt ** 2, shape=(N,), replace=False)
# X = X_fine[inds,:].copy()
N_gap = 8
inds = sub_grid_inds(N_gap,N_gap,N_fine_sqrt)
X = X_fine[inds,:].copy()
print(X.shape)
s_gap = 2
s = s_fine[::s_gap].copy()
n_s = s.shape[0]
Ps = Ps_fine[::s_gap,:,:].copy()
Ys = Ys_fine[inds, ::s_gap].copy()
Ys_centred = Ys_fine_centred[inds, ::s_gap].copy()
X_projs = np.einsum('ij,ljk->lik', X, Ps)
X = np.array(X)
s = np.array(s)
Ps = np.array(Ps)
Ys = np.array(Ys)
Ys_centred = np.array(Ys_centred)
X_projs = np.array(X_projs)
# %%
# save dataset - uncomment this if not needed
data = {'X_fine': X_fine, 's_fine': s_fine,'Ys_fine': Ys_fine, 'Ys_fine_means': Ys_fine_means, 'Ys_fine_centred': Ys_fine_centred, 'Ps_fine': Ps_fine, 'anchor_point': anchor_point, 'X': X, 's': s,'Ys': Ys, 'Ys_centred': Ys_centred, 'Ps': Ps}

head = '../data/datasets'
main_name = 'mcmc_initialised_with_map_data'
path = get_save_path(head, main_name)
try:
    safe_save(path, data)
except FileExistsError:
    print("File exists so not saving.")

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
# # ! fix this!!!!!!!!!!!
# function to obtain MAP estimate of Ps from SVI results
def get_Ps_map(map_est, anchor_modelled = True, anchor = None):
    if anchor_modelled:
        anchor_params = map_est['anchor_params_auto_loc']
        anchor_point = np.linalg.qr(anchor_params)[0]
    elif anchor is not None:
        anchor_point = anchor
    else:
        raise ValueError(f"anchor must be specified as anchor_modelled={anchor_modelled}")
    
    # form Omega
    sigmas = map_est['grass-sigmas_auto_loc']
    L_factor = map_est['grass-L_factor_auto_loc']
    L = L_factor + grass_inputs['L_jitter'] * np.eye(D * active_dimension)
    Omega = np.outer(sigmas, sigmas) * L
    
    # for proj_locs and get proj_params
    proj_mean = map_est['grass-proj_mean_auto_loc']
    proj_locs = np.tile(proj_mean, n_s)
    proj_params = map_est['grass-standard_proj_params_auto_loc']
    
    # get grassmann kernel params
    grass_var = map_est['grass-kernel_var_auto_loc']
    grass_length = map_est['grass-kernel_length_auto_loc']
    
    # form kernel mat
    K = rbf_covariance(s.reshape(n_s,-1), s.reshape(n_s,-1), grass_var, grass_length, 0.0, jitter=grass_inputs['jitter'])
    
    # form M_chol
    M = np.kron(K, Omega) + proj_jitter * np.eye(N_projs)
    M_chol = lin.cholesky(M)
    # M_chol = kron_chol(K, Omega)
    
    # form projection_parameters
    # projection_parameters = proj_locs + (M_chol + grass_inputs['proj_jitter']) @ proj_params
    projection_parameters = proj_locs + M_chol @ proj_params
    
    # split and unvec
    projection_parameters_split = np.array(projection_parameters.split(n_s))
    unvec_Vs = vmap(lambda params: unvec(params, D, active_dimension))(projection_parameters_split)
    
    # form projector
    I_UUT = (np.eye(D) - anchor_point @ anchor_point.T)
    
    # project and convert to projs
    Deltas = np.einsum('ij,ljk->lik', I_UUT, unvec_Vs)
    Ps_map = convert_to_projs(Deltas, anchor_point, reorthonormalize=grass_inputs['reorthonormalize'])
    if anchor_modelled:
        return Ps_map, anchor_point
    else:
        return Ps_map
# %%
# # Complete model
def model(X, s, Ys, anchor_point, model_params, grass_jitter, reg_jitter, proj_jitter, grassmann_noise, reorthonormalize, b, params, know_reg_kernel_params: bool = True, n_subsample_gap = 1):
    # get num of aux params
    n_s = s.shape[0]
    
    # get projections using Grassmann Process
    Ps = scope(grassmann_process, prefix="grass", divider="-")(s, anchor_point,model_params=model_params, jitter=grass_jitter, proj_jitter=proj_jitter, require_noise=grassmann_noise, reorthonormalize=reorthonormalize, b=b)
    
    # project the data
    projected_data = np.einsum('ij,ljk->lik', X, Ps)
    
    # collect the projected data over aux axis and subsample
    projected_data_all = np.vstack([projected_data[i,:,:] for i in range(n_s)])[::n_subsample_gap,:]
    
    # subsample vec(Ys)
    vec_Ys_subsampled = vec(Ys)[::n_subsample_gap]
    
    
    # fit outer univariate gp
    scope(univariate_gp_model, prefix="reg", divider="-")(projected_data_all, vec_Ys_subsampled, params, jitter=reg_jitter, know_kernel_params=know_reg_kernel_params)

# %%
# parameters for grass_process
train_proj_locs = np.zeros(shape=(D*active_dimension*n_s,))

train_model_params = {'var': 1.0, 'Omega': np.eye(D * active_dimension), 'proj_locs': train_proj_locs}

model_options = {'model_params': train_model_params, 'reg_jitter': 1e-06, 'grass_jitter': 1e-06, 'grassmann_noise': False, 'reorthonormalize': True, 'b': 5.0, 'params': params, 'proj_jitter': 1e-4, 'know_reg_kernel_params': False, 'n_subsample_gap': 4}
# numpyro.render_model(model, model_args=(X, s, Ys_centred, anchor_point, model_options['model_params'], model_options['grass_jitter'], model_options['reg_jitter'], model_options['proj_jitter'], model_options['grassmann_noise'], model_options['reorthonormalize'], model_options['b'], model_options['params'], model_options['know_reg_kernel_params'], model_options['n_subsample_gap']))
# %%
key = random.PRNGKey(123514354575)
# maxiter=10000
maxiter=30000
step_size=0.001
anchor_point_guess = anchor_point
print(f'Grass-dist btw true anchor point and anchor point used for model = {grass_dist(anchor_point, anchor_point_guess)}')
# svi_results = run_svi_for_map(key, model, maxiter, step_size, X, s, Ys_centred, anchor_point_guess, model_options['grass_jitter'], model_options['reg_jitter'], model_options['proj_jitter'], model_options['grassmann_noise'], model_options['reorthonormalize'], model_options['params'], model_options['know_reg_kernel_params'])
# use 1 for n_subsample_gap for SVI
svi_results = run_svi_for_map(key, model, maxiter, step_size, X, s, Ys, anchor_point_guess, model_options['model_params'], model_options['grass_jitter'], model_options['reg_jitter'], model_options['proj_jitter'], model_options['grassmann_noise'], model_options['reorthonormalize'],model_options['b'], model_options['params'], model_options['know_reg_kernel_params'], 1)

# %%
map_est = svi_results.params
# %%
strip_val = len('_auto_loc')
# %%
init_values = {key[:-9]:value for (key, value) in map_est.items()}
# %%
seed = 9870687
train_key = random.PRNGKey(seed)
mcmc_config = {'num_warmup' : 1000, 'num_samples' : 1000, 'num_chains' : 1, 'thinning' : 2, 'init_strategy' : init_to_value(values=init_values)}
print("Inference starting.")
# mcmc = run_inference(train_key, mcmc_config, model, X, s, Ys_centred, anchor_point, model_options['grass_jitter'], model_options['reg_jitter'], model_options['proj_jitter'], model_options['grassmann_noise'], model_options['reorthonormalize'], model_options['params'], model_options['know_reg_kernel_params'])
# use n_subsample_gap > 1 here
mcmc = run_inference(train_key, mcmc_config, model, X, s, Ys, anchor_point, model_options['model_params'], model_options['grass_jitter'], model_options['reg_jitter'], model_options['proj_jitter'], model_options['grassmann_noise'], model_options['reorthonormalize'], model_options['b'], model_options['params'], model_options['know_reg_kernel_params'], model_options['n_subsample_gap'])

# %%
samples = mcmc.get_samples()
head_samples = '../data/mcmc_samples'
main_name_samples = 'mcmc_initialised_with_map_samples'
path_samples = get_save_path(head_samples, main_name_samples)
try:
    safe_save(path_samples, samples)
except FileExistsError:
    print("File exists so not saving.")

# %%