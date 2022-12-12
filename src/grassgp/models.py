import jax.numpy as np
from jax import vmap, random
import jax.numpy.linalg as lin

from dataclasses import field
from typing import Callable, Tuple
import chex
from chex import assert_shape, assert_rank

import numpyro
import numpyro.distributions as dist
import numpyro.handlers as handlers

from grassgp.utils import unvec, vec, kron_solve
from grassgp.kernels import rbf
from grassgp.grassmann import convert_to_projs, valid_grass_tangent, valid_grass_point

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


@chex.dataclass
class MatGP:
    d_in: int
    d_out: Tuple[int, int]
    mu: Callable = field(repr=False)
    k: Callable = field(repr=False)
    Omega: chex.ArrayDevice = field(repr=False)
    cov_jitter: float = field(default=1e-4, repr=False)

    
    def __post_init__(self):
        d, n = self.d_out
        d_n = d * n
        assert_shape(self.Omega, (d_n, d_n),
                    custom_message=f"Omega has shape {self.Omega.shape}; expected shape {(d_n, d_n)}")


    def model(self, s: chex.ArrayDevice) -> chex.ArrayDevice:
        d, n = self.d_out
        d_n = d * n
        assert_rank(s, self.d_in)
        N = s.shape[0]

        # compute mean matrix M = [mu(s[1]), mu(s[2]), ..., mu(s[N])]
        M = np.hstack(vmap(self.mu)(s))
        assert_shape(M, (d, n*N))

        # compute kernel matrix
        K = self.k(s, s)
        assert_shape(K, (N, N))

        # compute covariance matrix and cholesky factor
        Cov = np.kron(K, self.Omega) + self.cov_jitter * np.eye(N*d_n)
        Chol = lin.cholesky(Cov)

        # sample vec_Vs
        Z = numpyro.sample("Z", dist.MultivariateNormal(covariance_matrix=np.eye(N*d_n)))
        vec_Vs = numpyro.deterministic("vec_Vs", vec(M) + Chol @ Z)

        # form Vs
        Vs = numpyro.deterministic("Vs", vmap(lambda params: unvec(params, d, n))(np.array(vec_Vs.split(N))))
        return Vs


    def sample(self, seed: int, s: chex.ArrayDevice) -> chex.ArrayDevice:
        model = self.model
        seeded_model = handlers.seed(model, rng_seed=seed)
        return seeded_model(s)


    # def predict(self, seed: int, s_test: chex.ArrayDevice, s_train: chex.ArrayDevice, Vs_train: chex.ArrayDevice, jitter: float = 1e-8):
    #     d, n = self.d_out
    #     d_in = self.d_in
    #     N_train = s_train.shape[0]
    #     N_test = s_test.shape[0]
    #     if d_in > 1:
    #         assert s_train.shape[1] == d_in
    #         assert s_test.shape[1] == d_in

    #     # compute means
    #     M_train = np.hstack(vmap(self.mu)(s_train))
    #     M_test = np.hstack(vmap(self.mu)(s_test))
    #     assert_shape(M_train, (d, n*N_train))
    #     assert_shape(M_test, (d, n*N_test))

    #     # compute kernels between train and test locs
    #     K_train_train = self.k(s_train, s_train)
    #     assert_shape(K_train_train, (N_train, N_train))
    #     K_train_test = self.k(s_train, s_test)
    #     assert_shape(K_train_test, (N_train, N_test))
    #     K_test_test = self.k(s_test, s_test)
    #     assert_shape(K_test_test, (N_test, N_test))

    #     # compute posterior mean and cov
    #     K_train_train_inv = lin.inv(K_train_train + jitter * np.eye(N_train))
    #     vec_post_mean = vec(M_test) + np.kron(K_train_test.T @ K_train_train_inv, np.eye(d*n)) @ (vec(Vs_train) - vec(M_train))
    #     assert_shape(vec_post_mean, (d*n*N_test,),
    #                  custom_message=f"vec_post_mean should have shape {(d*n*N_test,)}; obtained {vec_post_mean.shape}")
    #     post_cov = np.kron(K_test_test, self.Omega) - np.kron(K_train_test.T @ (K_train_train_inv @ K_train_test), self.Omega)
    #     assert_shape(post_cov, (d*n*N_test, d*n*N_test),
    #                  custom_message=f"post_cov should have shape {(d*n*N_test,d*n*N_test)}; obtained {post_cov.shape}")

    #     # sample predictions
    #     vec_preds = dist.MultivariateNormal(loc=vec_post_mean, covariance_matrix=post_cov).sample(random.PRNGKey(seed))

    #     # unvec mean and preds and return
    #     post_mean = vmap(lambda params: unvec(params, d, n))(np.array(vec_post_mean.split(N_test)))
    #     preds = vmap(lambda params: unvec(params, d, n))(np.array(vec_preds.split(N_test)))
    #     return post_mean, preds

    def predict(self, key: chex.ArrayDevice, s_test: chex.ArrayDevice, s_train: chex.ArrayDevice, Vs_train: chex.ArrayDevice, jitter: float = 1e-8) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        d, n = self.d_out
        d_in = self.d_in
        N_train = s_train.shape[0]
        N_test = s_test.shape[0]
        if d_in > 1:
            assert s_train.shape[1] == d_in
            assert s_test.shape[1] == d_in

        # compute means
        M_train = np.hstack(vmap(self.mu)(s_train))
        M_test = np.hstack(vmap(self.mu)(s_test))
        assert_shape(M_train, (d, n*N_train))
        assert_shape(M_test, (d, n*N_test))

        # compute kernels between train and test locs
        K_train_train = self.k(s_train, s_train)
        assert_shape(K_train_train, (N_train, N_train))
        K_train_test = self.k(s_train, s_test)
        assert_shape(K_train_test, (N_train, N_test))
        K_test_train = K_train_test.T
        K_test_test = self.k(s_test, s_test)
        assert_shape(K_test_test, (N_test, N_test))

        # compute posterior mean and cov
        K_test_train_Omega = np.kron(K_test_train, self.Omega)
        K_train_test_Omega = np.kron(K_train_test, self.Omega)
        K_test_test_Omega = np.kron(K_test_test, self.Omega)
        mean_sols = kron_solve(K_train_train, self.Omega, vec(np.hstack(Vs_train)) - vec(M_train))
        vec_post_mean = vec(M_test) + K_test_train_Omega @ mean_sols
        assert_shape(vec_post_mean, (d*n*N_test,),
                     custom_message=f"vec_post_mean should have shape {(d*n*N_test,)}; obtained {vec_post_mean.shape}")

        cov_sols = vmap(lambda v: kron_solve(K_train_train, self.Omega, v), in_axes=1, out_axes=1)(K_train_test_Omega)
        post_cov = K_test_test_Omega - K_test_train_Omega @ cov_sols
        assert_shape(post_cov, (d*n*N_test, d*n*N_test),
                     custom_message=f"post_cov should have shape {(d*n*N_test,d*n*N_test)}; obtained {post_cov.shape}")

        # sample predictions
        post_cov += jitter * np.eye(d*n*N_test)
        vec_pred = dist.MultivariateNormal(loc=vec_post_mean, covariance_matrix=post_cov).sample(key)
        assert_shape(vec_pred, (d*n*N_test,),
                     custom_message=f"vec_pred should have shape {(d*n*N_test,)}; obtained {vec_pred.shape}")

        # unvec mean and preds and return
        post_mean = vmap(lambda params: unvec(params, d, n))(np.array(vec_post_mean.split(N_test)))
        pred = vmap(lambda params: unvec(params, d, n))(np.array(vec_pred.split(N_test)))
        return post_mean, pred

    
    
@chex.dataclass
class GrassGP:
    d_in: int
    d_out: Tuple[int, int]
    mu: Callable = field(repr=False)
    k: Callable = field(repr=False)
    Omega: chex.ArrayDevice = field(repr=False)
    U: chex.ArrayDevice
    cov_jitter: float = field(default=1e-4, repr=False)


    def __post_init__(self):
        d, n = self.d_out
        d_n = d * n
        assert_shape(self.Omega, (d_n, d_n),
                    custom_message=f"Omega has shape {self.Omega.shape}; expected shape {(d_n, d_n)}")
        assert_shape(self.U, (d, n),
                    custom_message=f"U has shape {self.U.shape}; expected shape {(d, n)}")
        tol = 1e-06
        # assert valid_grass_point(self.U), f"U is not a valid point on Grassmann manifold G({d},{n}) at tolerance level {tol = }"


    @property
    def V(self) -> MatGP:
        mat_gp = MatGP(d_in=self.d_in, d_out=self.d_out, mu=self.mu, k=self.k, Omega=self.Omega, cov_jitter=self.cov_jitter)
        return mat_gp


    def tangent_model(self, s: chex.ArrayDevice) -> chex.ArrayDevice:
        d, n = self.d_out
        N = s.shape[0]
        Vs = self.V.model(s)
        I_UUT = np.eye(d) - self.U @ self.U.T
        Deltas = numpyro.deterministic("Deltas", np.einsum('ij,ljk->lik', I_UUT, Vs))
        assert_shape(Deltas, (N, d, n),
                    custom_message=f"Deltas has shape {Deltas.shape}; expected shape {(N, d, n)}")
        return Deltas


    def sample_tangents(self, seed: int, s: chex.ArrayDevice) -> chex.ArrayDevice:
        tangent_model = self.tangent_model
        seeded_model = handlers.seed(tangent_model, rng_seed=seed)
        Deltas = seeded_model(s)
        assert vmap(lambda Delta: valid_grass_tangent(self.U, Delta))(Deltas).all()
        return Deltas


    def sample_grass(self, seed: int, s: chex.ArrayDevice, reortho: bool = False) -> chex.ArrayDevice:
        Deltas = self.sample_tangents(seed, s)
        Ws = convert_to_projs(Deltas, self.U, reorthonormalize=reortho)
        return Ws


    def predict_tangents(self, key: chex.ArrayDevice, s_test: chex.ArrayDevice, s_train: chex.ArrayDevice, Vs_train: chex.ArrayDevice, jitter: float = 1e-8) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        d, _ = self.d_out
        Vs_mean, Vs_pred = self.V.predict(key, s_test, s_train, Vs_train, jitter=jitter)
        I_UUT = np.eye(d) - self.U @ self.U.T
        Deltas_mean = np.einsum('ij,ljk->lik', I_UUT, Vs_mean)
        Deltas_pred = np.einsum('ij,ljk->lik', I_UUT, Vs_pred)
        return Deltas_mean, Deltas_pred


    def predict_grass(self, key: chex.ArrayDevice, s_test: chex.ArrayDevice, s_train: chex.ArrayDevice, Vs_train: chex.ArrayDevice, jitter: float = 1e-8, reortho: bool = False) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        Deltas_mean, Deltas_pred = self.predict_tangents(key, s_test, s_train, Vs_train, jitter=jitter)
        Ws_mean = convert_to_projs(Deltas_mean, self.U, reorthonormalize=reortho)
        Ws_pred = convert_to_projs(Deltas_pred, self.U, reorthonormalize=reortho)
        return Ws_mean, Ws_pred
        

@chex.dataclass
class GP:
    d_in: int
    mu: Callable = field(repr=False)
    k: Callable = field(repr=False)
    jitter: float = field(default=1e-8, repr=False)


    def model(self, s: chex.ArrayDevice) -> chex.ArrayDevice:
        d_in = self.d_in
        N = s.shape[0]
        if d_in > 1:
            assert_shape(s, (N, d_in))

        # compute mean
        m = vmap(self.mu)(s)
        assert_shape(m, (N,))

        # compute cov
        K = self.k(s, s)
        assert_shape(K, (N, N))

        Cov = K + self.jitter * np.eye(N)
        Chol = lin.cholesky(Cov)

        # sample
        Z = numpyro.sample("Z", dist.MultivariateNormal(covariance_matrix=np.eye(N)))
        Y = numpyro.deterministic("Y", m + Chol @ Z)
        return Y


    def sample(self, seed: int, s: chex.ArrayDevice) -> chex.ArrayDevice:
        model = self.model
        seeded_model = handlers.seed(model, rng_seed=seed)
        return seeded_model(s)


    def predict(self, seed: int, s_test: chex.ArrayDevice, s_train: chex.ArrayDevice, obs: chex.ArrayDevice, jitter: float = 1e-8) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        d_in = self.d_in
        N_train = s_train.shape[0]
        N_test = s_test.shape[0]
        if d_in > 1:
            assert_shape(s_train, (N_train, d_in))
            assert_shape(s_test, (N_test, d_in))

        # compute means
        m_train = vmap(self.mu)(s_train)
        m_test = vmap(self.mu)(s_test)
        assert_shape(m_train, (N_train,))
        assert_shape(m_test, (N_test,))

        # compute kernels
        K_train_train = self.k(s_train, s_train)
        K_train_test = self.k(s_train, s_test)
        K_test_train = K_train_test.T
        K_test_test = self.k(s_test, s_test)

        # compute post mean and cov
        post_mean = m_test + K_test_train @ lin.solve(K_train_train, obs - m_train)
        post_cov = K_test_test - K_test_train @ vmap(lambda x: lin.solve(K_train_train, x), in_axes=1, out_axes=1)(K_train_test)
        assert_shape(post_mean, (N_test,))
        assert_shape(post_cov, (N_test, N_test))

        # sample prediction
        post_cov += jitter * np.eye(N_test)
        pred = dist.MultivariateNormal(loc=post_mean, covariance_matrix=post_cov).sample(random.PRNGKey(seed))

        return post_mean, pred

