# +
import time
from hydra_zen import instantiate, make_config, builds, to_yaml
from pathlib import Path
import os
import sys

import jax.numpy as np
from jax import random, vmap
import jax.numpy.linalg as lin

from grassgp.grassmann import grass_log, grass_exp, valid_grass_point, valid_grass_tangent, compute_barycenter, grass_dist, convert_to_projs
from grassgp.utils import vec, unvec, kron_solve, get_save_path, to_dictconf
from grassgp.kernels import rbf
from grassgp.utils import safe_save_jax_array_dict as safe_save

import numpyro
from numpyro.infer import init_to_median
import numpyro.distributions as dist

from grassgp.inference import run_inference

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,6)


# -

def print_cfg(cfg):
    print(to_yaml(cfg))


# +
# generate dataset
N = 20
s = np.linspace(0, 1, N)
k = 2 * np.pi
x = np.cos(k * s).reshape(-1, 1)
y = np.sin(k * s).reshape(-1, 1)
Ws = np.hstack((x,y))[:,:,None]
assert vmap(valid_grass_point)(Ws).all()
d, n = Ws.shape[1:]

# plot dataset
for i in range(d):
    plt.plot(s, Ws[:,i,0])
    plt.title(f'{i+1}th component of projection')
    plt.grid()
    plt.xlabel(r'$s$')
    plt.show()
# -

# compute barycenter of train data
s_gap = 2
anchor_point = compute_barycenter(Ws[::s_gap])
assert valid_grass_point(anchor_point)
print(f"anchor_point = {anchor_point.tolist()}")

# compute logarithm of data
log_Ws = vmap(lambda W: grass_log(anchor_point, W))(Ws)
log_Ws.shape

# +
model_config = {
    'anchor_point': anchor_point.tolist(),
    'Omega' : None,
    'proj_locs' : np.zeros(20*2).tolist(),
    'var' : None,
    'length' : None, 
    'noise' : 0.01,
    'require_noise' : True,
    'jitter' : 1e-06,
    'proj_jitter' : 1e-4,
    'L_jitter' : 1e-8,
    'reorthonormalize' : False,
    'b' : 1.0,
    'ell': 0.002,
    's_gap': s_gap
}

def model(s, log_Ws, grass_config = model_config):
    anchor_point = np.array(grass_config['anchor_point'])
    d, n = anchor_point.shape
    N = s.shape[0]
    s_gap = grass_config['s_gap']
    N_train = N // s_gap
    proj_dim = d * n
    N_projs = N * proj_dim
    assert log_Ws.shape == (N, d, n)
    
    if grass_config['Omega'] is None:
        # sample Omega
        sigmas = numpyro.sample("sigmas", dist.LogNormal(0.0, 1.0).expand([proj_dim]))
        L_factor = numpyro.sample("L_factor", dist.LKJ(proj_dim, 1.0))
        L = numpyro.deterministic("L", L_factor + grass_config['L_jitter'] * np.eye(proj_dim))
        Omega = numpyro.deterministic("Omega", np.outer(sigmas, sigmas) * L)
    else:
        Omega = np.array(grass_config['Omega'])
        
    if grass_config['proj_locs'] is None:
        # sample proj_locs
        # TODO: look into how these are sampled and if we need to be clear that these are copied across
        # # # ! old way where mean is same for each time
        # proj_mean = numpyro.sample("proj_mean", dist.MultivariateNormal(scale_tril=np.eye(proj_dim)))
        # proj_locs = np.tile(proj_mean, n_s)
        # new way using different means
        proj_mean = numpyro.sample("proj_mean", dist.MultivariateNormal(scale_tril=np.eye(proj_dim)).expand([N]))
        proj_locs = numpyro.deterministic("proj_locs", vec(proj_mean.T))
    else:
        proj_locs = np.array(grass_config['proj_locs'])

    Z = numpyro.sample("Z", dist.MultivariateNormal(covariance_matrix=np.eye(N_projs)))
    
    if N > 1:
        # parameters for the kernel of the Grassmann Process
        if grass_config['var'] is None:
            # sample var
            var = numpyro.sample("kernel_var", dist.LogNormal(0.0, grass_config['b']))
        else:
            var = grass_config['var']
        
        if grass_config['length'] is None:
            # sample length
            length = numpyro.sample("kernel_length", dist.LogNormal(0.0, grass_config['b']))
        else:
            length = grass_config['length']
        
        if grass_config['require_noise']:
            if grass_config['noise'] is None:
                # sample noise
                noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, grass_config['b']))
            else:
                noise = grass_config['noise']
            
            params = {'var': var, 'length': length, 'noise': noise}
            K = rbf(s, s, params, jitter=grass_config['jitter'])
        else:
            params = {'var': var, 'length': length, 'noise': 0.0}
            K = rbf(s, s, params, jitter=grass_config['jitter'])
        
        M = np.kron(K, Omega) + grass_config['proj_jitter'] * np.eye(N_projs)
        M_chol = lin.cholesky(M)
    else:
        M_chol = lin.cholesky(Omega)
        
    vec_Vs = numpyro.deterministic("vec_Vs", proj_locs + M_chol @ Z)
    vec_Vs_split = np.array(vec_Vs.split(N))
    Vs = vmap(lambda params: unvec(params, d, n))(vec_Vs_split)
    I_UUT = (np.eye(d) - anchor_point @ anchor_point.T)
    Deltas = np.einsum('ij,ljk->lik', I_UUT, Vs)
    assert Deltas.shape == (N, d, n)
    
    # # ! check what power this should be
    ell = grass_config['ell']
    with numpyro.plate("N_train", N_train):
        numpyro.sample("log_W", dist.continuous.MatrixNormal(loc=Deltas[::s_gap], scale_tril_row=ell * np.eye(d),scale_tril_column=np.eye(n)), obs=log_Ws[::s_gap])
        

TangentSpaceModelConf = builds(model, grass_config=model_config, zen_partial=True)
# -

# subsample data
s_gap = 2
s_train = s[::s_gap].copy()
Ws_train = Ws[::s_gap,:,:].copy()
log_Ws_train = vmap(lambda W: grass_log(anchor_point, W))(Ws_train)
for i in range(d):
    plt.plot(s, Ws[:,i,0])
    plt.scatter(s_train, Ws_train[:,i,0], c='r')
    plt.title(f'{i+1}th component of projection')
    plt.grid()
    plt.xlabel(r'$s$')
    plt.show()

# +
TrainConfig = make_config(
    seed = 9870687,
    n_warmup = 1000,
    n_samples = 1000,
    n_chains = 1,
    n_thinning = 2,
    init_strategy = init_to_median(num_samples=10)
)

Config = make_config(
    model = TangentSpaceModelConf,
    train = TrainConfig
)
# -

numpyro.render_model(model, model_args=(s,log_Ws))


def train(cfg):
    # save training and testing data
    # training_test_data = {'s_train': s_train, 'Ws_train': Ws_train, 's': s, 'Ws': Ws}
    # head = os.getcwd()
    # main_name_training_test = "training_test_data"
    # path_training_test = get_save_path(head, main_name_training_test)
    # try:
    #     safe_save(path_training_test, training_test_data)
    # except FileExistsError:
    #     print("File exists so not saving.")
    
    # instantiate grass model
    grass_model = instantiate(cfg.model)
    
    # run HMC
    train_key = random.PRNGKey(cfg.train.seed)
    mcmc_config = {'num_warmup' : cfg.train.n_warmup, 'num_samples' : cfg.train.n_samples, 'num_chains' : cfg.train.n_chains, 'thinning' : cfg.train.n_thinning, 'init_strategy' : instantiate(cfg.train.init_strategy)}
    print("HMC starting.")
    mcmc = run_inference(train_key, mcmc_config, grass_model, s, log_Ws)
    # original_stdout = sys.stdout
    # with open('hmc_log.txt', 'w') as f:
    #     sys.stdout = f
    #     mcmc.print_summary()
    #     sys.stdout = original_stdout
    
    samples = mcmc.get_samples()
    
    # head = os.getcwd()
    # main_name = "samples_data"
    # path = get_save_path(head, main_name)
    # try:
    #     safe_save(path, samples)
    # except FileExistsError:
    #     print("File exists so not saving.")
    
    return samples


cfg = Config
samples = train(cfg)

# +
from grassgp.plot_utils import flatten_samples, traceplots

my_samples = flatten_samples(samples)

traceplots(my_samples)
# -

samples.keys()

# +
vec_Vs_samples_train = samples['vec_Vs']
def get_projs_from_vec_Vs(vec_Vs, reortho=False):
    vec_Vs_split = np.array(vec_Vs.split(len(s_train)))
    Vs = vmap(lambda params: unvec(params, d, n))(vec_Vs_split)
    I_UUT = (np.eye(d) - anchor_point @ anchor_point.T)
    Deltas = np.einsum('ij,ljk->lik', I_UUT, Vs)
    Ws = convert_to_projs(Deltas, anchor_point, reorthonormalize=reortho)
    return Ws

preds_Ws_train = vmap(get_projs_from_vec_Vs)(vec_Vs_samples)
for i in range(500):
    assert vmap(valid_grass_point)(preds_Ws_train[i]).all()
# -

preds_Ws_train.shape

from grassgp.plot_utils import plot_grass_dists

vec_Vs_samples_mean = vec_Vs_samples.mean(axis=0)
preds_Ws_train_means = get_projs_from_vec_Vs(vec_Vs_samples_mean)
for i in range(d):
    plt.scatter(s_train, preds_Ws_train_means[:,i,0],label='mean from MCMC',c='r')
    plt.plot(s_train, Ws_train[:,i,0], label='true')
    plt.legend()
    plt.show()

plot_grass_dists(preds_Ws_train, Ws_train, s_train)


def grass_predict(key, s_test, s, anchor_point, Omega, var, length, noise, proj_params, reortho, jitter=5e-4):
    d, n = anchor_point.shape
    N_train = s.shape[0]
    N_test = s_test.shape[0]
    
    # compute kernels between train and test locs
    grass_kernel_params = {'var': var, 'length': length, 'noise': noise}
    K_pp = rbf(s_test, s_test, grass_kernel_params, include_noise=False)
    K_pt = rbf(s_test, s, grass_kernel_params, include_noise=False)
    K_tt = rbf(s, s, grass_kernel_params)
    
    # form M_covs between train and test locs
    M_cov_pp = np.kron(K_pp, Omega)
    M_cov_pt = np.kron(K_pt, Omega)
    
    # add jitter to M_cov_pp
    M_cov_pp += jitter * np.eye(M_cov_pp.shape[0])
    
    # Get posterior cov for tangent space
    # # ! check this
    K = M_cov_pp - np.matmul(M_cov_pt, vmap(lambda v: kron_solve(K_tt, Omega, v), in_axes=1, out_axes=1)(M_cov_pt.T))
    
    # Get posterior mean for tangent space
    # # ! include prior mean
    mean = np.matmul(M_cov_pt, kron_solve(K_tt, Omega, proj_params))
    
    # sample projection params for test locs
    sample = dist.MultivariateNormal(loc=mean, covariance_matrix=K).sample(key)
    
    # split each up into params for each time
    mean_split = np.array(mean.split(N_test))
    sample_split = np.array(sample.split(N_test))
    
    # unvec each
    unvec_mean = vmap(lambda params: unvec(params, d, n))(mean_split)
    unvec_sample = vmap(lambda params: unvec(params, d, n))(sample_split)

    
    # form projector
    I_UUT = (np.eye(d) - anchor_point @ anchor_point.T)
    
    # apply this to each
    Deltas_mean = np.einsum('ij,ljk->lik', I_UUT, unvec_mean)
    Deltas_sample = np.einsum('ij,ljk->lik', I_UUT, unvec_sample)
    
    # convert posterior means to projections
    Ps_mean = convert_to_projs(Deltas_mean, anchor_point, reorthonormalize=reortho)
    Ps_sample = convert_to_projs(Deltas_sample, anchor_point, reorthonormalize=reortho)
    
    return Ps_mean, Ps_sample

samples.keys()


def run_grass_predict(pred_key, s_test, s, dict_cfg, samples:dict, jitter=5e-4):
    anchor_point = np.array(dict_cfg.model.grass_config.anchor_point)
    n_samples = dict_cfg.train.n_samples // dict_cfg.train.n_thinning
    proj_params_samples = samples['vec_Vs']
    assert n_samples == proj_params_samples.shape[0]
    
    # initialize vmap_args
    vmap_args = (random.split(pred_key, n_samples), proj_params_samples)

    cfg_Omega = dict_cfg.model.grass_config.Omega
    cfg_var = dict_cfg.model.grass_config.var
    cfg_length = dict_cfg.model.grass_config.length
    cfg_noise = dict_cfg.model.grass_config.noise
    require_noise = dict_cfg.model.grass_config.require_noise
    reortho = dict_cfg.model.grass_config.reorthonormalize
    
    if cfg_Omega is None:
        vmap_args += (samples['Omega'],)
    else:
        cfg_Omega = np.array(cfg_Omega)
        vmap_args += (np.repeat(cfg_Omega[None,:,:], n_samples, axis=0),)
        
    if cfg_var is None:
        vmap_args += (samples['kernel_var'],)
    else:
        vmap_args += (cfg_var * np.ones(n_samples),)
        
    if cfg_length is None:
        vmap_args += (samples['kernel_length'],)
    else:
        vmap_args += (cfg_length * np.ones(n_samples),)
    
    if require_noise:
        if cfg_noise is None:
            vmap_args += (samples['kernel_noise'],)
        else:
            vmap_args += (cfg_noise * np.ones(n_samples),)
    else:
        vmap_args += (np.zeros(n_samples),)
        
    Ps_means, Ps_preds = vmap(lambda key, proj_params, Omega, var, length, noise: grass_predict(key, s_test, s, anchor_point, Omega, var, length, noise, proj_params, reortho, jitter=jitter))(*vmap_args)
    return Ps_means, Ps_preds


pred_key = random.PRNGKey(457657)
dict_cfg = to_dictconf(cfg)
Ws_means, Ws_preds = run_grass_predict(pred_key, s, s_train, dict_cfg, samples)

from grassgp.plot_utils import plot_grass_preds, plot_AS_dir_preds

plot_grass_preds(s_train, s, Ws_means, Ws_preds, Ws, [2.5, 97.5])

plot_AS_dir_preds(Ws_preds, Ws, s, s_train)

plot_grass_dists(Ws_preds, Ws, s)




