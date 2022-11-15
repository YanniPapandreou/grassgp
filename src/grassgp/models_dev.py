import jax.numpy as np
from jax import vmap
import jax.numpy.linalg as lin

import numpyro
import numpyro.distributions as dist

from grassgp.utils import unvec, vec
from grassgp.kernels import rbf
from grassgp.grassmann import convert_to_projs

def grassmann_process(s, grass_config: dict = {'anchor_point': [[1.0], [0.0]], 'Omega' : None, 'proj_locs' : None, 'var' : None, 'length' : None, 'noise' : None, 'require_noise' : False, 'jitter' : 1e-06, 'proj_jitter' : 1e-4, 'L_jitter' : 1e-8, 'reorthonormalize' : True, 'b' : 1.0}):
    anchor_point = np.array(grass_config['anchor_point'])
    D, n = anchor_point.shape
    n_s = s.shape[0]
    proj_dim = D * n

    N_projs = n_s * proj_dim
    
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
        # ! old way where mean is same for each time
        # proj_mean = numpyro.sample("proj_mean", dist.MultivariateNormal(scale_tril=np.eye(proj_dim)))
        # proj_locs = np.tile(proj_mean, n_s)
        # new way using different means
        proj_mean = numpyro.sample("proj_mean", dist.MultivariateNormal(scale_tril=np.eye(proj_dim)).expand([n_s]))
        proj_locs = numpyro.deterministic("proj_locs", vec(proj_mean.T))
    else:
        proj_locs = np.array(grass_config['proj_locs'])
        
    proj_params = numpyro.sample("standard_proj_params",
        dist.MultivariateNormal(covariance_matrix=np.eye(N_projs))
    )
    
    if n_s > 1:
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
    
    
    projection_parameters = numpyro.deterministic("proj_params", proj_locs + M_chol @ proj_params)
    
    # split projection_parameters up into params for each time
    projection_parameters_split = np.array(projection_parameters.split(n_s))

    # unvec each chunk
    unvec_Vs = vmap(lambda params: unvec(params, D, n))(projection_parameters_split)

    # form projector
    I_UUT = (np.eye(D) - anchor_point @ anchor_point.T)

    # apply this to each unvec_V
    Deltas = np.einsum('ij,ljk->lik', I_UUT, unvec_Vs)

    # convert to projections
    Ps = numpyro.deterministic("Ps", convert_to_projs(Deltas, anchor_point, reorthonormalize=grass_config['reorthonormalize']))
    
    return Ps


def univariate_gp_model(x, y, gp_config: dict = {'params': {'var': None, 'length': None, 'noise': None}, 'jitter': 1e-06, 'b': 10.0}):
    params = gp_config['params']
    
    # # loop over params and sample any missing
    # for param, value in params.items():
    #     if value is None:
    #         params[param] = numpyro.sample(f"kernel_{param}", dist.LogNormal(0.0, gp_config['b']))
    # numpyro.sample(f"kernel_{param}", dist.LogNormal(0.0, gp_config['b']))
    
    if params['var'] is None:
        var = numpyro.sample("kernel_var", dist.LogNormal(0.0, gp_config['b']))
    else:
        var = params['var']
        
    if params['length'] is None:
        length = numpyro.sample("kernel_length", dist.LogNormal(0.0, gp_config['b']))
    else:
        length = params['length']
        
    if params['noise'] is None:
        noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, gp_config['b']))
    else:
        noise = params['noise']

    kernel_params = {'var': var, 'length': length, 'noise': noise}
    K = rbf(x, x, kernel_params, jitter = gp_config['jitter'])
    
    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "obs_y",
        dist.MultivariateNormal(loc=np.zeros(K.shape[0]), covariance_matrix=K),
        obs=y,
    )   
