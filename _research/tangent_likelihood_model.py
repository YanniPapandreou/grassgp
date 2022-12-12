# %%
# +
import jax.numpy as np
from jax import random, vmap
import jax.numpy.linalg as lin

from grassgp.grassmann import grass_exp, grass_log
from grassgp.utils import vec, unvec, get_config_and_data, get_save_path
from grassgp.kernels import rbf

import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_value, init_to_median
import numpyro.distributions as dist
from numpyro.distributions.continuous import MatrixNormal

from grassgp.inference import run_inference

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,6)


# -

def model(s,log_Ws,grass_config: dict = {
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
            'b' : 1.0,
            'ell': 1.0}
        ):
    anchor_point = np.array(grass_config['anchor_point']) 
    d, n = anchor_point.shape
    N = s.shape[0]
    proj_dim = d * n
    N_projs = N * proj_dim
    assert log_Ws.shape == (N,d,n)

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
        # # ! old way where mean is same for each time
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
    assert Deltas.shape == (N,d,n)

    ellSqN = grass_config['ell']**(2/n)
    with numpyro.plate("N", N):
        numpyro.sample("log_W", MatrixNormal(loc=Deltas, scale_tril_row=ellSqN * np.eye(d),scale_tril_column=np.eye(n)), obs=log_Ws)


# +
dataset_path = '/home/yanni/projects/grassgp/scripts/outputs/2022-11-02/16-27-53/'
def load_dataset(dataset_path):
    config_and_dataset = get_config_and_data(dataset_path)
    print("Dataset loaded; overrides used in generation:")
    for override in config_and_dataset['overrides']:
        print(f"-{override}")
    return config_and_dataset

config_and_data = load_dataset(dataset_path)
data = config_and_data['data']
# -

data = {k:np.array(array) for k, array in data.items()}
X_fine = data['X']
s_fine = data['s']
Ps_fine = data['Ps']
Ys_fine = data['Ys']


# +
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


# -

subsample_key = random.PRNGKey(325)
n_samples = 49
s_gap = 2
X, s, Ys, Ps = subsample(subsample_key, X_fine, s_fine, Ys_fine, Ps_fine, n_samples, s_gap)

X.shape

Ps_fine.shape

# get logarithm
anchor_point = np.eye(2,1)
log_Ws = vmap(lambda W: grass_log(anchor_point, W))(Ps)

numpyro.render_model(model, model_args=(s,log_Ws))

train_key = random.PRNGKey(23523)
mcmc_config = {'num_warmup' : 1000, 'num_samples' : 1000, 'num_chains' : 1, 'thinning' : 2, 'init_strategy' : init_to_median(num_samples=10)}
mcmc = run_inference(train_key, mcmc_config, model, s, log_Ws)

samples = mcmc.get_samples()

from grassgp.plot_utils import flatten_samples, plot_projected_data, traceplots, plot_grids, plot_preds_train_locs, plot_grass_preds, plot_grass_dists, plot_preds, plot_AS_dir_preds, plot_fixed_x_preds_vs_time
my_samples = flatten_samples(samples)

traceplots(my_samples, a=1.0)



