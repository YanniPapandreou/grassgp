import time
from hydra_zen import instantiate, make_config, builds 
import os
import sys

import jax.numpy as np
from jax import random 

from grassgp.utils import get_save_path
from grassgp.utils import safe_save_jax_array_dict as safe_save
from grassgp.utils import load_and_convert_to_samples_dict as load_data
from grassgp.kernels import rbf
from grassgp.models import GrassGP
from grassgp.means import zero_mean

import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_value
import numpyro.distributions as dist

from grassgp.inference import run_inference

def run_svi_for_map(rng_key, model, maxiter, step_size, *args):
    start = time.time()
    guide = autoguide.AutoDelta(model)
    optimzer = numpyro.optim.Adam(step_size)
    svi = SVI(model, guide, optimzer, Trace_ELBO())
    svi_results = svi.run(rng_key, maxiter, *args)
    print('\nSVI elapsed time:', time.time() - start)
    return svi_results

# +
########################################
# ## Choose default dataset here ########
path = './datasets/training_test_data_gpsr_example.npz'
""
# load dataset
dataset = load_data(path)

# +
s_train = np.array(dataset['s_train'])
s_test = np.array(dataset['s_test'])

Ws_train = np.array(dataset['Ws_train'])
Ws_test = np.array(dataset['Ws_test'])

log_Ws_train = np.array(dataset['log_Ws_train'])
log_Ws_test = np.array(dataset['log_Ws_test'])

anchor_point = np.array(dataset['anchor_point'])

# +
model_config = {
    'anchor_point': anchor_point.tolist(),
    'd_in': 1,
    'Omega' : None,
    'k_include_noise': True,
    'var' : None,
    'length' : None, 
    'noise' : None,
    'require_noise' : False,
    'jitter' : 1e-06,
    'cov_jitter' : 1e-4,
    'L_jitter' : 1e-8,
    'reorthonormalize' : False,
    'b' : 1.0,
    'ell': 0.0075
}

def model(s, log_Ws, grass_config = model_config):
    U = np.array(grass_config['anchor_point'])
    d, n = U.shape
    N = s.shape[0]
    d_n = d * n
    # N_params = N * d_n
    if log_Ws is not None:
        assert log_Ws.shape == (N, d, n)
    
    # get/sample Omega
    if grass_config['Omega'] is None:
        sigmas = numpyro.sample('sigmas', dist.LogNormal(0.0, 1.0).expand([d_n]))
        L_factor = numpyro.sample('L_factor', dist.LKJ(d_n, 1.0))
        L = numpyro.deterministic('L', L_factor + grass_config['L_jitter'] * np.eye(d_n))
        Omega = numpyro.deterministic('Omega', np.outer(sigmas, sigmas) * L)
    else:
        Omega = np.array(grass_config['Omega'])
        
    # get/sample kernel params
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
    else:
        noise = 0.0
    
    kernel_params = {'var': var, 'length': length, 'noise': noise}
    # create kernel function
    k = lambda t, s: rbf(t, s, kernel_params, jitter=grass_config['jitter'], include_noise=grass_config['k_include_noise'])
    # create mean function
    mu = lambda s: zero_mean(s, d, n)

    # initialize GrassGP
    grass_gp = GrassGP(d_in=grass_config['d_in'], d_out=(d,n), mu=mu, k=k, Omega=Omega, U=U, cov_jitter=grass_config['cov_jitter'])
    
    # sample Deltas
    Deltas = grass_gp.tangent_model(s)
    
    # # # ! check what power this should be
    # likelihood
    ell = grass_config['ell']
    with numpyro.plate("N", N):
        numpyro.sample("log_W", dist.continuous.MatrixNormal(loc=Deltas, scale_tril_row=ell * np.eye(d),scale_tril_column=np.eye(n)), obs=log_Ws)

TangentSpaceModelConf = builds(model, grass_config=model_config, zen_partial=True)

# +
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
    n_thinning = 2
)

Config = make_config(
    model = TangentSpaceModelConf,
    svi = SVIConfig,
    train = TrainConfig
)


# -

def train(cfg):
    # instantiate grass model
    model = instantiate(cfg.model)
    
    # run SVI to get MAP esimtate to initialise MCMC
    svi_key = random.PRNGKey(cfg.svi.seed)
    maxiter = cfg.svi.maxiter
    step_size = cfg.svi.step_size
    print("Running SVI for MAP estimate to initialise MCMC")
    svi_results = run_svi_for_map(svi_key, model, maxiter, step_size, s_train, log_Ws_train)
    
    # get initialisation from SVI results
    map_est = svi_results.params
    strip_val = len('_auto_loc')
    init_values = {key[:-strip_val]:value for (key, value) in map_est.items()}
    
    # run HMC
    train_key = random.PRNGKey(cfg.train.seed)
    mcmc_config = {'num_warmup' : cfg.train.n_warmup, 'num_samples' : cfg.train.n_samples, 'num_chains' : cfg.train.n_chains, 'thinning' : cfg.train.n_thinning, 'init_strategy' : init_to_value(values=init_values)}
    print("HMC starting.")
    mcmc = run_inference(train_key, mcmc_config, model, s_train, log_Ws_train)    
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
    main_name = "inference_data"
    path = get_save_path(head, main_name)
    try:
        safe_save(path, samples)
    except FileExistsError:
        print("File exists so not saving.")



""

