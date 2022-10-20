import jax.numpy as np
from jax import vmap
import jax.numpy.linalg as lin

import numpyro
import numpyro.distributions as dist

from grassgp.utils import unvec
from grassgp.kernels import rbf, spatio_temporal_matern_52 
from grassgp.grassmann import convert_to_projs

# def grassmann_process(s, anchor_point, model_params = {}, jitter=1e-06, proj_jitter=1e-4, L_jitter=1e-8, require_noise = False, reorthonormalize = True, b = 1.0):
#     D, n = anchor_point.shape
#     n_s = s.shape[0]
#     proj_dim = D * n

#     N_projs = n_s * proj_dim
    
#     # ! LogNormal params?
#     if 'Omega' not in model_params.keys():
#         sigmas = numpyro.sample("sigmas", dist.LogNormal(0.0, 1.0).expand([proj_dim]))
#         L_factor = numpyro.sample("L_factor", dist.LKJ(proj_dim, 1.0)) 
#         L = numpyro.deterministic("L", L_factor + L_jitter * np.eye(proj_dim))
#         Omega = numpyro.deterministic("Omega", np.outer(sigmas, sigmas) * L)
#     else:
#         Omega = model_params['Omega']
    
#     if 'proj_locs' not in model_params.keys():
#         proj_mean = numpyro.sample("proj_mean", dist.MultivariateNormal(scale_tril=np.eye(proj_dim)))
#         proj_locs = np.tile(proj_mean, n_s)
#     else:
#         proj_locs = model_params['proj_locs']

#     proj_params = numpyro.sample("standard_proj_params",
#         dist.MultivariateNormal(covariance_matrix=np.eye(N_projs))
#     )

#     if n_s > 1:
#         # parameters for the kernel of the Grassmann Process
#         if 'var' not in model_params.keys():
#             var = numpyro.sample("kernel_var", dist.LogNormal(0.0, b))
#         else:
#             var = model_params['var']

#         if 'length' not in model_params.keys():
#             length = numpyro.sample("kernel_length", dist.LogNormal(0.0, b))
#         else:
#             length = model_params['length']
        
#         if require_noise:
#             if 'noise' not in model_params.keys():
#                 noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, b))
#             else:
#                 noise = model_params['noise']

#             params = {'var': var, 'length': length, 'noise': noise}
#             K = rbf(s, s, params, jitter=jitter)
#         else:
#             params = {'var': var, 'length': length, 'noise': 0.0}
#             K = rbf(s, s, params, jitter=jitter)

#         M = np.kron(K, Omega) + proj_jitter * np.eye(N_projs)
#         M_chol = lin.cholesky(M) 
#         # M_chol = kron_chol(K, Omega) # ! this cannot be used if we add a jitter
#     else:
#         M_chol = lin.cholesky(Omega) # ! this should not be just M_chol = Omega ?
    
#     projection_parameters = numpyro.deterministic("proj_params", proj_locs + M_chol @ proj_params) 
#     # projection_parameters = numpyro.deterministic("proj_params", mu * np.ones(N_projs) + (M_chol+proj_jitter) @ proj_params)
#     # projection_parameters = numpyro.deterministic("proj_params", proj_locs + (M_chol+proj_jitter) @ proj_params) # ! this is incorrect we are doing (L + mat with values all jitter) -> two mistakes!
#     # projection_parameters = numpyro.deterministic("proj_params", (M_chol+proj_jitter) @ proj_params)

#     # split projection_parameters up into params for each time
#     projection_parameters_split = np.array(projection_parameters.split(n_s))

#     # unvec each chunk
#     unvec_Vs = vmap(lambda params: unvec(params, D, n))(projection_parameters_split)

#     # form projector
#     I_UUT = (np.eye(D) - anchor_point @ anchor_point.T)

#     # apply this to each unvec_V
#     Deltas = np.einsum('ij,ljk->lik', I_UUT, unvec_Vs)

#     # convert to projections
#     Ps = numpyro.deterministic("Ps", convert_to_projs(Deltas, anchor_point, reorthonormalize=reorthonormalize))

#     return Ps


# # def univariate_gp_model(x, y, params:dict, jitter=1e-06, know_kernel_params: bool = True):
# #     if know_kernel_params:
# #         var = params['var']
# #         noise = params['noise']
# #         length = params['noise']
# #     else:
# #         # set uninformative log-normal priors on our three kernel hyperparameters
# #         # ! log-normal params?
# #         var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
# #         noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
# #         length = numpyro.sample("kernel_length", dist.LogNormal(0.0, 10.0))
# #
# #     # compute kernel
# #     K = rbf_covariance(x.reshape(-1,1), x.reshape(-1,1), var, length, noise, jitter=jitter)
# #
# #     # sample Y according to the standard gaussian process formula
# #     numpyro.sample(
# #         "obs_y",
# #         dist.MultivariateNormal(loc=np.zeros(K.shape[0]), covariance_matrix=K),
# #         obs=y,
# #     )

# def univariate_gp_model(x, y, params:dict, jitter=1e-06, know_kernel_params: bool = True):
#     if know_kernel_params:
#         K = rbf(x, x, params, jitter = jitter)
#     else:
#         # set uninformative log-normal priors on our three kernel hyperparameters
#         # ! log-normal params?
#         var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
#         noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
#         length = numpyro.sample("kernel_length", dist.LogNormal(0.0, 10.0))
#         kernel_params = {'var': var, 'length': length, 'noise': noise}
#         K = rbf(x, x, kernel_params, jitter = jitter)

#     # sample Y according to the standard gaussian process formula
#     numpyro.sample(
#         "obs_y",
#         dist.MultivariateNormal(loc=np.zeros(K.shape[0]), covariance_matrix=K),
#         obs=y,
#     )

# def grassmann_process_new(s, anchor_point, jitter=1e-06, proj_jitter=1e-4, L_jitter=1e-8, require_noise = False, reorthonormalize = True):
#     D, n = anchor_point.shape
#     n_s = s.shape[0]
#     proj_dim = D * n

#     N_projs = n_s * proj_dim
    
#     # ! LogNormal params?
#     sigmas = numpyro.sample("sigmas", dist.LogNormal(0.0, 1.0).expand([proj_dim]))
#     L_factor = numpyro.sample("L_factor", dist.LKJ(proj_dim, 1.0)) 
#     L = numpyro.deterministic("L", L_factor + L_jitter * np.eye(proj_dim))
#     Omega = numpyro.deterministic("Omega", np.outer(sigmas, sigmas) * L)

#     proj_mean = numpyro.sample("proj_mean", dist.MultivariateNormal(scale_tril=np.eye(proj_dim)))
#     proj_locs = np.tile(proj_mean, n_s)

#     proj_params = numpyro.sample("standard_proj_params",
#         dist.MultivariateNormal(covariance_matrix=np.eye(N_projs))
#     )

#     if n_s > 1:
#         # parameters for the kernel of the Grassmann Process
#         var = numpyro.sample("kernel_var", dist.LogNormal(0.0, 1.0))
#         length = numpyro.sample("kernel_length", dist.LogNormal(0.0, 1.0))
        
#         if require_noise:
#             noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 1.0))
#             params = {'var': var, 'length': length, 'noise': noise}
#             K = rbf(s, s, params, jitter=jitter)
#         else:
#             params = {'var': var, 'length': length, 'noise': 0.0}
#             K = rbf(s, s, params, jitter=jitter)

#         M = np.kron(K, Omega) + proj_jitter * np.eye(N_projs)
#         M_chol = lin.cholesky(M) 
#         # M_chol = kron_chol(K, Omega) # ! this cannot be used if we add a jitter
#     else:
#         M_chol = lin.cholesky(Omega) # ! this should not be just M_chol = Omega ?
    
#     projection_parameters = numpyro.deterministic("proj_params", proj_locs + M_chol @ proj_params) 

#     # split projection_parameters up into params for each time
#     projection_parameters_split = np.array(projection_parameters.split(n_s))

#     # unvec each chunk
#     unvec_Vs = vmap(lambda params: unvec(params, D, n))(projection_parameters_split)

#     # form projector
#     I_UUT = (np.eye(D) - anchor_point @ anchor_point.T)

#     # apply this to each unvec_V
#     Deltas = np.einsum('ij,ljk->lik', I_UUT, unvec_Vs)

#     # convert to projections
#     Ps = numpyro.deterministic("Ps", convert_to_projs(Deltas, anchor_point, reorthonormalize=reorthonormalize))

#     return Ps


# def spatio_temporal_gp_model_matern_52(X_T, Y, jitter=1e-06):
#     # set uninformative log-normal priors on our kernel hyperparameters
#     var_x = numpyro.sample("kernel_var_x", dist.LogNormal(0.0, 10.0))
#     length_x = numpyro.sample("kernel_length_x", dist.LogNormal(0.0, 10.0))
    
#     var_t = numpyro.sample("kernel_var_t", dist.LogNormal(0.0, 10.0))
#     length_t = numpyro.sample("kernel_length_t", dist.LogNormal(0.0, 10.0))
    
#     noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))

#     params = {'var_x': var_x, 'var_t': var_t, 'length_x': length_x, 'length_t': length_t, 'noise': noise}
    
#     k = spatio_temporal_matern_52(X_T, X_T, params)
    
#     # sample Y according to the standard gaussian process formula
#     numpyro.sample(
#         "Y",
#         dist.MultivariateNormal(loc=np.zeros(X_T.shape[0]), covariance_matrix=k),
#         obs=Y,
#     )

# def grassmann_process(s, anchor_point, Omega = None, proj_locs = None, var = None, length = None, noise = None, require_noise: bool = False, jitter: float = 1e-06, proj_jitter: float = 1e-4, L_jitter: float = 1e-8, reorthonormalize: bool = True, b: float = 1.0):
#     D, n = anchor_point.shape
#     n_s = s.shape[0]
#     proj_dim = D * n

#     N_projs = n_s * proj_dim
    
#     if Omega is None:
#         # sample Omega
#         sigmas = numpyro.sample("sigmas", dist.LogNormal(0.0, 1.0).expand([proj_dim]))
#         L_factor = numpyro.sample("L_factor", dist.LKJ(proj_dim, 1.0)) 
#         L = numpyro.deterministic("L", L_factor + L_jitter * np.eye(proj_dim))
#         Omega = numpyro.deterministic("Omega", np.outer(sigmas, sigmas) * L)
#     else:
#         Omega = np.array(Omega)
        
    
#     if proj_locs is None:
#         # sample proj_locs
#         proj_mean = numpyro.sample("proj_mean", dist.MultivariateNormal(scale_tril=np.eye(proj_dim)))
#         proj_locs = np.tile(proj_mean, n_s)
#     else:
#         proj_locs = np.array(proj_locs)
        
#     proj_params = numpyro.sample("standard_proj_params",
#         dist.MultivariateNormal(covariance_matrix=np.eye(N_projs))
#     )
    
#     if n_s > 1:
#         # parameters for the kernel of the Grassmann Process
#         if var is None:
#             # sample var
#             var = numpyro.sample("kernel_var", dist.LogNormal(0.0, b))
        
#         if length is None:
#             # sample length
#             length = numpyro.sample("kernel_length", dist.LogNormal(0.0, b))
        
#         if require_noise:
#             if noise is None:
#                 # sample noise
#                 noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, b))
            
#             params = {'var': var, 'length': length, 'noise': noise}
#             K = rbf(s, s, params, jitter=jitter)
#         else:
#             params = {'var': var, 'length': length, 'noise': 0.0}
#             K = rbf(s, s, params, jitter=jitter)
        
#         M = np.kron(K, Omega) + proj_jitter * np.eye(N_projs)
#         M_chol = lin.cholesky(M)
#     else:
#         M_chol = lin.cholesky(Omega)
    
    
#     projection_parameters = numpyro.deterministic("proj_params", proj_locs + M_chol @ proj_params)
    
#     # split projection_parameters up into params for each time
#     projection_parameters_split = np.array(projection_parameters.split(n_s))

#     # unvec each chunk
#     unvec_Vs = vmap(lambda params: unvec(params, D, n))(projection_parameters_split)

#     # form projector
#     I_UUT = (np.eye(D) - anchor_point @ anchor_point.T)

#     # apply this to each unvec_V
#     Deltas = np.einsum('ij,ljk->lik', I_UUT, unvec_Vs)

#     # convert to projections
#     Ps = numpyro.deterministic("Ps", convert_to_projs(Deltas, anchor_point, reorthonormalize=reorthonormalize))
    
#     return Ps

def grassmann_process(s, model_config: dict = {'anchor_point': [[1.0], [0.0]], 'Omega' : None, 'proj_locs' : None, 'var' : None, 'length' : None, 'noise' : None, 'require_noise' : False, 'jitter' : 1e-06, 'proj_jitter' : 1e-4, 'L_jitter' : 1e-8, 'reorthonormalize' : True, 'b' : 1.0}):
    anchor_point = np.array(model_config['anchor_point'])
    D, n = anchor_point.shape
    n_s = s.shape[0]
    proj_dim = D * n

    N_projs = n_s * proj_dim
    
    if model_config['Omega'] is None:
        # sample Omega
        sigmas = numpyro.sample("sigmas", dist.LogNormal(0.0, 1.0).expand([proj_dim]))
        L_factor = numpyro.sample("L_factor", dist.LKJ(proj_dim, 1.0)) 
        L = numpyro.deterministic("L", L_factor + model_config['L_jitter'] * np.eye(proj_dim))
        Omega = numpyro.deterministic("Omega", np.outer(sigmas, sigmas) * L)
    else:
        Omega = np.array(model_config['Omega'])
        
    
    if model_config['proj_locs'] is None:
        # sample proj_locs
        proj_mean = numpyro.sample("proj_mean", dist.MultivariateNormal(scale_tril=np.eye(proj_dim)))
        proj_locs = np.tile(proj_mean, n_s)
    else:
        proj_locs = np.array(model_config['proj_locs'])
        
    proj_params = numpyro.sample("standard_proj_params",
        dist.MultivariateNormal(covariance_matrix=np.eye(N_projs))
    )
    
    if n_s > 1:
        # parameters for the kernel of the Grassmann Process
        if model_config['var'] is None:
            # sample var
            var = numpyro.sample("kernel_var", dist.LogNormal(0.0, model_config['b']))
        else:
            var = model_config['var']
        
        if model_config['length'] is None:
            # sample length
            length = numpyro.sample("kernel_length", dist.LogNormal(0.0, model_config['b']))
        else:
            length = model_config['length']
        
        if model_config['require_noise']:
            if model_config['noise'] is None:
                # sample noise
                noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, model_config['b']))
            else:
                noise = model_config['noise']
            
            params = {'var': var, 'length': length, 'noise': noise}
            K = rbf(s, s, params, jitter=model_config['jitter'])
        else:
            params = {'var': var, 'length': length, 'noise': 0.0}
            K = rbf(s, s, params, jitter=model_config['jitter'])
        
        M = np.kron(K, Omega) + model_config['proj_jitter'] * np.eye(N_projs)
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
    Ps = numpyro.deterministic("Ps", convert_to_projs(Deltas, anchor_point, reorthonormalize=model_config['reorthonormalize']))
    
    return Ps