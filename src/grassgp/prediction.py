import jax.numpy as np
import jax.numpy.linalg as lin
import jax.random as random
from jax import vmap
from grassgp.grassmann import convert_to_projs
from tqdm import tqdm

from grassgp.utils import vec, unvec, kron_solve
from grassgp.kernels import rbf
import numpyro.distributions as dist

# def predict_at_train_locs(rng_key, X, s, Ys, X_test, var, length, noise, Ps, jitter=7.5e-4):
#     """function to predict at train locs"""
#     n_s = s.shape[0]

#     # project train, test data at each train loc
#     X_projs = np.einsum('ij,ljk->lik', X, Ps)
#     X_test_projs = np.einsum('ij,ljk->lik', X_test, Ps)
    
#     # group these over all locs into long vectors
#     Train = np.vstack([X_projs[i,:,:] for i in range(n_s)])
#     Test = np.vstack([X_test_projs[i,:,:] for i in range(n_s)])
    
#     # compute kernels between train and test data
#     params = {'var': var, 'length': length, 'noise': noise}
#     # K_pp = rbf_covariance(Test, Test, var, length, noise, include_noise=False)
#     # K_pt = rbf_covariance(Test, Train, var, length, noise, include_noise=False)
#     # K_tt = rbf_covariance(Train, Train, var, length, noise)
#     K_pp = rbf(Test, Test, params, include_noise=False)
#     K_pt = rbf(Test, Train, params, include_noise=False)
#     K_tt = rbf(Train, Train, params)

#     # perform conditioning
#     # covariance
#     K = K_pp - np.matmul(K_pt, lin.solve(K_tt, np.transpose(K_pt)))
    
#     # mean
#     means = np.matmul(K_pt, lin.solve(K_tt, vec(Ys)))
    
#     # predictions
#     preds = dist.MultivariateNormal(loc=means,covariance_matrix=K + jitter * np.eye(K.shape[0])).sample(rng_key)

#     # return unveced means and preds
#     return unvec(means, X_test.shape[0], n_s), unvec(preds, X_test.shape[0], n_s)

def predict_at_train_locs(X_test, X, s, Ys, rng_key, Ps, var, length, noise, jitter=7.5e-4):
    """function to predict at train locs"""
    n_s = s.shape[0]

    # project train, test data at each train loc
    X_projs = np.einsum('ij,ljk->lik', X, Ps)
    X_test_projs = np.einsum('ij,ljk->lik', X_test, Ps)
    
    # group these over all locs into long vectors
    Train = np.vstack([X_projs[i,:,:] for i in range(n_s)])
    Test = np.vstack([X_test_projs[i,:,:] for i in range(n_s)])
    
    # compute kernels between train and test data
    params = {'var': var, 'length': length, 'noise': noise}
    # K_pp = rbf_covariance(Test, Test, var, length, noise, include_noise=False)
    # K_pt = rbf_covariance(Test, Train, var, length, noise, include_noise=False)
    # K_tt = rbf_covariance(Train, Train, var, length, noise)
    K_pp = rbf(Test, Test, params, include_noise=False)
    K_pt = rbf(Test, Train, params, include_noise=False)
    K_tt = rbf(Train, Train, params)

    # perform conditioning
    # covariance
    K = K_pp - np.matmul(K_pt, lin.solve(K_tt, np.transpose(K_pt)))
    
    # mean
    means = np.matmul(K_pt, lin.solve(K_tt, vec(Ys)))
    
    # predictions
    preds = dist.MultivariateNormal(loc=means,covariance_matrix=K + jitter * np.eye(K.shape[0])).sample(rng_key)

    # return unveced means and preds
    return unvec(means, X_test.shape[0], n_s), unvec(preds, X_test.shape[0], n_s)

## function to run prediction at train locs
# def run_prediction_train_locs(pred_key, X, s, Ys, samples:dict, jitter=7.5e-4, know_reg_params: bool = True, params: dict = None):
#     """function to run prediction at train locs"""
#     if know_reg_params and params:
#         vmap_args = (random.split(pred_key, samples['grass-Ps'].shape[0]), samples['grass-Ps'])
#         means, predictions = vmap(lambda rng_key, Ps: predict_at_train_locs(rng_key, X, s, Ys, X, params['var'], params['length'], params['noise'], Ps, jitter=jitter))(*vmap_args)
#     elif not know_reg_params:
#         vmap_args = (random.split(pred_key, samples['grass-Ps'].shape[0]), samples['reg-var'], samples['reg-length'], samples['reg-noise'], samples['grass-Ps'])
#         means, predictions = vmap(lambda rng_key, var, length, noise, Ps: predict_at_train_locs(rng_key, X, s, Ys, X, var, length, noise, Ps, jitter=jitter))(*vmap_args)
#     return means, predictions

def run_prediction_at_train_times(pred_key, X_test, X, s, Ys, cfg, samples:dict, jitter=7.5e-4):    
    # get number of samples
    n_samples = cfg.train.n_samples // cfg.train.n_thinning
    Ps_samples = samples['grass-Ps']
    assert n_samples == Ps_samples.shape[0]
    
    # initialize vmap_args
    vmap_args = (random.split(pred_key, n_samples), Ps_samples)

    cfg_var = cfg.outer_model.gp_config.params.var
    cfg_length = cfg.outer_model.gp_config.params.length
    cfg_noise = cfg.outer_model.gp_config.params.noise
    
    if cfg_var is None:
        vmap_args += (samples['reg-kernel_var'],)
    else:
        vmap_args += (cfg_var * np.ones(n_samples),)
        
    if cfg_length is None:
        vmap_args += (samples['reg-kernel_length'],)
    else:
        vmap_args += (cfg_length * np.ones(n_samples),)
        
    if cfg_noise is None:
        vmap_args += (samples['reg-kernel_noise'],)
    else:
        vmap_args += (cfg_noise * np.ones(n_samples),)
    
    means, predictions = vmap(lambda key, Ps, var, length, noise: predict_at_train_locs(X_test, X, s, Ys, key, Ps, var, length, noise, jitter=jitter))(*vmap_args)
    return means, predictions

## function to perform prediction for the grassmann process
# def grass_predict(key, s, s_test, T_var, T_length, T_noise, sigmas, L, projection_params, anchor_point, jitter=5e-4, reortho=True):
#     D, n = anchor_point.shape
#     n_train = s.shape[0]
#     n_test = s_test.shape[0]
#
#     # form Omega
#     Omega = np.outer(sigmas, sigmas) * L
#
#     # compute (temporal) kernels between train and test locs
#     grass_kernel_params = {'var': T_var, 'length': T_length, 'noise': T_noise}
#     # T_K_pp = rbf_covariance(s_test.reshape(n_test,-1), s_test.reshape(n_test,-1), T_var, T_length, T_noise, include_noise=False)
#     # T_K_pt = rbf_covariance(s_test.reshape(n_test,-1), s.reshape(n_train,-1), T_var, T_length, T_noise, include_noise=False)
#     # T_K_tt = rbf_covariance(s.reshape(n_train,-1), s.reshape(n_train,-1), T_var, T_length, T_noise)
#     T_K_pp = rbf(s_test, s_test, grass_kernel_params, include_noise=False)
#     T_K_pt = rbf(s_test, s, grass_kernel_params, include_noise=False)
#     T_K_tt = rbf(s, s, grass_kernel_params)
#
#     # form M_covs between train and test locs
#     M_cov_pp = np.kron(T_K_pp, Omega)
#     M_cov_pt = np.kron(T_K_pt, Omega)
#
#     # add jitter to M_cov_tt, M_cov_pp
#     M_cov_pp += jitter * np.eye(M_cov_pp.shape[0])
#
#     # Get posterior cov for grass part    
#     T_K = M_cov_pp - np.matmul(M_cov_pt, vmap(lambda v: kron_solve(T_K_tt, Omega, v), in_axes=1, out_axes=1)(M_cov_pt.T))
#
#     # Get posterior mean for grass part
#     T_mean = np.matmul(M_cov_pt, kron_solve(T_K_tt, Omega, projection_params))
#     
#     # sample projection params for test locs
#     T_sample = dist.MultivariateNormal(loc=T_mean, covariance_matrix=T_K).sample(key)
#     
#     # Convert mean and samples to projections
#     
#     # split each up into params for each time
#     T_mean_split = np.array(T_mean.split(n_test))
#     T_sample_split = np.array(T_sample.split(n_test))
#     
#     # unvec each
#     unvec_T_mean = vmap(lambda params: unvec(params, D, n))(T_mean_split)
#     unvec_T_sample = vmap(lambda params: unvec(params, D, n))(T_sample_split)
#     
#     # form projector
#     I_UUT = (np.eye(D) - anchor_point @ anchor_point.T)
#     
#     # apply this to each
#     Deltas_mean = np.einsum('ij,ljk->lik', I_UUT, unvec_T_mean)
#     Deltas_sample = np.einsum('ij,ljk->lik', I_UUT, unvec_T_sample)
#     
#     # convert posterior means to projections
#     Ps_mean = convert_to_projs(Deltas_mean, anchor_point, reorthonormalize=reortho)
#     Ps_sample = convert_to_projs(Deltas_sample, anchor_point, reorthonormalize=reortho)
#                         
#     # return Ps_mean and Ps_sample
#     return Ps_mean, Ps_sample
#
#
# ## function to perform prediction for the grassmann process
# def grass_predict_no_noise(key, s, s_test, T_var, T_length, sigmas, L, projection_params, anchor_point, jitter=5e-4, reortho=True):
#     D, n = anchor_point.shape
#     n_train = s.shape[0]
#     n_test = s_test.shape[0]
#
#     # form Omega
#     Omega = np.outer(sigmas, sigmas) * L
#
#     # compute (temporal) kernels between train and test locs
#     grass_kernel_params = {'var': T_var, 'length': T_length, 'noise': 0.0}
#     # T_K_pp = rbf_covariance(s_test.reshape(n_test,-1), s_test.reshape(n_test,-1), T_var, T_length, 0.0, include_noise=False)
#     # T_K_pt = rbf_covariance(s_test.reshape(n_test,-1), s.reshape(n_train,-1), T_var, T_length, 0.0, include_noise=False)
#     # T_K_tt = rbf_covariance(s.reshape(n_train,-1), s.reshape(n_train,-1), T_var, T_length, 0.0)
#     T_K_pp = rbf(s_test, s_test, grass_kernel_params, include_noise=False)
#     T_K_pt = rbf(s_test, s, grass_kernel_params, include_noise=False)
#     T_K_tt = rbf(s, s, grass_kernel_params)
#
#     # form M_covs between train and test locs
#     M_cov_pp = np.kron(T_K_pp, Omega)
#     M_cov_pt = np.kron(T_K_pt, Omega)
#
#     # add jitter to M_cov_tt, M_cov_pp
#     M_cov_pp += jitter * np.eye(M_cov_pp.shape[0])
#
#     # Get posterior cov for grass part    
#     T_K = M_cov_pp - np.matmul(M_cov_pt, vmap(lambda v: kron_solve(T_K_tt, Omega, v), in_axes=1, out_axes=1)(M_cov_pt.T))
#
#     # Get posterior mean for grass part
#     T_mean = np.matmul(M_cov_pt, kron_solve(T_K_tt, Omega, projection_params))
#     
#     # sample projection params for test locs
#     T_sample = dist.MultivariateNormal(loc=T_mean, covariance_matrix=T_K).sample(key)
#     
#     # Convert mean and samples to projections
#     
#     # split each up into params for each time
#     T_mean_split = np.array(T_mean.split(n_test))
#     T_sample_split = np.array(T_sample.split(n_test))
#     
#     # unvec each
#     unvec_T_mean = vmap(lambda params: unvec(params, D, n))(T_mean_split)
#     unvec_T_sample = vmap(lambda params: unvec(params, D, n))(T_sample_split)
#     
#     # form projector
#     I_UUT = (np.eye(D) - anchor_point @ anchor_point.T)
#     
#     # apply this to each
#     Deltas_mean = np.einsum('ij,ljk->lik', I_UUT, unvec_T_mean)
#     Deltas_sample = np.einsum('ij,ljk->lik', I_UUT, unvec_T_sample)
#     
#     # convert posterior means to projections
#     Ps_mean = convert_to_projs(Deltas_mean, anchor_point, reorthonormalize=reortho)
#     Ps_sample = convert_to_projs(Deltas_sample, anchor_point, reorthonormalize=reortho)
#                         
#     # return Ps_mean and Ps_sample
#     return Ps_mean, Ps_sample
#
#
# ## runs grass_predict for each sample in samples
# def run_grass_predict(pred_key, samples:dict, s, s_test, anchor_point, jitter=5e-4, grass_noise=False):
#     if grass_noise:
#         vmap_args = (random.split(pred_key, samples['grass-kernel_var'].shape[0]), samples['grass-kernel_var'], samples['grass-kernel_length'], samples['grass-kernel_noise'], samples['grass-sigmas'],samples['grass-L'], samples['grass-proj_params'])
#         Ps_mean_samples, Ps_test_preds = vmap(lambda rng_key, T_var, T_length, T_noise, sigmas, L, projection_params: grass_predict(rng_key,  s, s_test, T_var, T_length, T_noise, sigmas, L, projection_params, anchor_point, jitter=jitter))(*vmap_args)
#     else:
#         vmap_args = (random.split(pred_key, samples['grass-kernel_var'].shape[0]), samples['grass-kernel_var'], samples['grass-kernel_length'], samples['grass-sigmas'],samples['grass-L'], samples['grass-proj_params'])
#         Ps_mean_samples, Ps_test_preds = vmap(lambda rng_key, T_var, T_length, sigmas, L, projection_params: grass_predict_no_noise(rng_key, s, s_test, T_var, T_length, sigmas, L, projection_params, anchor_point, jitter=jitter))(*vmap_args)
#     return Ps_mean_samples, Ps_test_preds
#
# # function to predict at test spaces and test locs
# def predict_no_grass_noise(rng_key, X, s, Y, X_test, s_test, var, length, noise, T_var, T_length, sigmas, L, projection_params, projection_params_Ps, anchor_point, active_dimension, grass_jitter=2.5e-3, reg_jitter=7.5e-4, use_means=False):
#     # get keys for loc and space
#     loc_key, space_key = random.split(rng_key, 2)
#
#     # grassmann predictions
#     Ps_means, Ps_preds = grass_predict_no_noise(loc_key, s, s_test, T_var, T_length, sigmas, L, projection_params, anchor_point, jitter = grass_jitter)
#
#     # project data
#     X_train_projs_train = np.einsum('ij,ljk->lik', X, projection_params_Ps) # spatial train, loc train
#     if use_means:
#         X_test_projs_test = np.einsum('ij,ljk->lik', X_test, Ps_means)
#     else:
#         X_test_projs_test = np.einsum('ij,ljk->lik', X_test, Ps_preds)
#
#     # stack train and test projects across loc
#     Train = np.vstack([X_train_projs_train[i,:,:] for i in range(s.shape[0])])
#     Test = np.vstack([X_test_projs_test[i,:,:] for i in range(s_test.shape[0])])
#
#     # compute kernels between train and test data
#     kernel_params = {'var': var, 'length': length, 'noise': noise}
#     K_pp = rbf(Test, Test, kernel_params, include_noise=False)
#     K_pt = rbf(Test, Train, kernel_params, include_noise=False)
#     K_tt = rbf(Train, Train, kernel_params)
#
#     # perform conditioning
#     K = K_pp - np.matmul(K_pt, lin.solve(K_tt, np.transpose(K_pt)))
#
#     # # variances
#     # sigma_noises = np.sqrt(np.clip(np.diag(K), a_min=0.0)) * random.normal(
#     #                         space_key, K.shape[1:]
#     #                       )
#
#     # means
#     means = np.matmul(K_pt, lin.solve(K_tt, Y))
#
#     # predictions
#     # preds = means + sigma_noises
#
#     # predictions
#     preds = dist.MultivariateNormal(loc=means,covariance_matrix=K + reg_jitter * np.eye(K.shape[0])).sample(space_key)
#
#     # # we only want test_test means and preds
#     # means_test_test = means[n_train_test + n_test_train:]
#     # assert len(means_test_test) == n_test_test
#     # preds_test_test = preds[n_train_test + n_test_train:]
#     # assert len(preds_test_test) == n_test_test
#
#     # unvec and return
#     if use_means:
#         return unvec(means, X_test.shape[0], s_test.shape[0]), unvec(preds, X_test.shape[0], s_test.shape[0]), Ps_means
#     else:
#         return unvec(means, X_test.shape[0], s_test.shape[0]), unvec(preds, X_test.shape[0], s_test.shape[0]), Ps_preds
#
#
# def run_prediction(pred_key, samples:dict, X, s, Ys, X_test, s_test, anchor_point, active_dimension, grass_jitter=2.5e-3, reg_jitter=7.5e-4, use_means=False, grass_noise=False):
#     if grass_noise:
#         vmap_args = (random.split(pred_key, samples['reg-kernel_var'].shape[0]), samples['reg-kernel_var'], samples['reg-kernel_length'], samples['reg-kernel_noise'], samples['grass-kernel_var'],samples['grass-kernel_length'], samples['grass-kernel_noise'], samples['grass-sigmas'],samples['grass-L'], samples['grass-proj_params'], samples['grass-Ps'])
#         vmap_inds = {'rng_key': 0, 'var': 1, 'length': 2, 'noise': 3, 'T_var': 4, 'T_length': 5, 'T_noise': 6, 'sigmas': 7, 'L': 8, 'projection_params': 9, 'projection_params_Ps': 10}
#         means = np.zeros((samples['reg-kernel_noise'].shape[0], X_test.shape[0], s_test.shape[0]))
#         predictions = np.zeros((samples['reg-kernel_noise'].shape[0], X_test.shape[0], s_test.shape[0]))
#         projs = np.zeros((samples['reg-kernel_noise'].shape[0], s_test.shape[0], X_test.shape[1], active_dimension))
#         for i in tqdm(range(samples['reg-kernel_length'].shape[0])):
#             rng_key = vmap_args[vmap_inds['rng_key']][i]
#             var = vmap_args[vmap_inds['var']][i]
#             length = vmap_args[vmap_inds['length']][i]
#             noise = vmap_args[vmap_inds['noise']][i]
#             T_var = vmap_args[vmap_inds['T_var']][i]
#             T_length = vmap_args[vmap_inds['T_length']][i]
#             T_noise = vmap_args[vmap_inds['T_noise']][i]
#             sigmas = vmap_args[vmap_inds['sigmas']][i]
#             L = vmap_args[vmap_inds['L']][i]
#             projection_params = vmap_args[vmap_inds['projection_params']][i]
#             projection_params_Ps = vmap_args[vmap_inds['projection_params_Ps']][i]
#             m, p, proj = predict(rng_key, X, s, vec(Ys), X_test, s_test, var, length, noise, T_var, T_length, T_noise, sigmas, L, projection_params, projection_params_Ps, anchor_point, active_dimension, grass_jitter=grass_jitter, reg_jitter=reg_jitter, use_means=use_means)
#             means = means.at[i,:,:].set(m)
#             predictions = predictions.at[i,:,:].set(p)
#             projs = projs.at[i,:,:].set(proj)
#     else:
#         vmap_args = (random.split(pred_key, samples['reg-kernel_var'].shape[0]), samples['reg-kernel_var'], samples['reg-kernel_length'], samples['reg-kernel_noise'], samples['grass-kernel_var'],samples['grass-kernel_length'], samples['grass-sigmas'],samples['grass-L'], samples['grass-proj_params'], samples['grass-Ps'])
#         vmap_inds = {'rng_key': 0, 'var': 1, 'length': 2, 'noise': 3, 'T_var': 4, 'T_length': 5, 'sigmas': 6, 'L': 7, 'projection_params': 8, 'projection_params_Ps': 9}
#         means = np.zeros((samples['reg-kernel_noise'].shape[0], X_test.shape[0], s_test.shape[0]))
#         predictions = np.zeros((samples['reg-kernel_noise'].shape[0], X_test.shape[0], s_test.shape[0]))
#         projs = np.zeros((samples['reg-kernel_noise'].shape[0], s_test.shape[0], X_test.shape[1], active_dimension))
#         for i in tqdm(range(samples['reg-kernel_length'].shape[0])):
#             rng_key = vmap_args[vmap_inds['rng_key']][i]
#             var = vmap_args[vmap_inds['var']][i]
#             length = vmap_args[vmap_inds['length']][i]
#             noise = vmap_args[vmap_inds['noise']][i]
#             T_var = vmap_args[vmap_inds['T_var']][i]
#             T_length = vmap_args[vmap_inds['T_length']][i]
#             sigmas = vmap_args[vmap_inds['sigmas']][i]
#             L = vmap_args[vmap_inds['L']][i]
#             projection_params = vmap_args[vmap_inds['projection_params']][i]
#             projection_params_Ps = vmap_args[vmap_inds['projection_params_Ps']][i]
#             m, p, proj = predict_no_grass_noise(rng_key, X, s, vec(Ys), X_test, s_test, var, length, noise, T_var, T_length, sigmas, L, projection_params, projection_params_Ps, anchor_point, active_dimension, grass_jitter=grass_jitter, reg_jitter=reg_jitter, use_means=use_means)
#             means = means.at[i,:,:].set(m)
#             predictions = predictions.at[i,:,:].set(p)
#             projs = projs.at[i,:,:].set(proj)
#     return means, predictions, projs


def grass_predict(key, s_test, s, anchor_point, Omega, var, length, noise, proj_params, reortho, jitter=5e-4):
    D, n = anchor_point.shape
    n_train = s.shape[0]
    n_test = s_test.shape[0]
    
    # compute (temporal) kernels between train and test locs
    grass_kernel_params = {'var': var, 'length': length, 'noise': noise}
    T_K_pp = rbf(s_test, s_test, grass_kernel_params, include_noise=False)
    T_K_pt = rbf(s_test, s, grass_kernel_params, include_noise=False)
    T_K_tt = rbf(s, s, grass_kernel_params)
    
    # form M_covs between train and test locs
    M_cov_pp = np.kron(T_K_pp, Omega)
    M_cov_pt = np.kron(T_K_pt, Omega)
    
    # add jitter to M_cov_tt, M_cov_pp
    M_cov_pp += jitter * np.eye(M_cov_pp.shape[0])
    
    # Get posterior cov for grass part    
    T_K = M_cov_pp - np.matmul(M_cov_pt, vmap(lambda v: kron_solve(T_K_tt, Omega, v), in_axes=1, out_axes=1)(M_cov_pt.T))
    
    # Get posterior mean for grass part
    T_mean = np.matmul(M_cov_pt, kron_solve(T_K_tt, Omega, proj_params))
    
    # sample projection params for test locs
    T_sample = dist.MultivariateNormal(loc=T_mean, covariance_matrix=T_K).sample(key)
    
    
    # split each up into params for each time
    T_mean_split = np.array(T_mean.split(n_test))
    T_sample_split = np.array(T_sample.split(n_test))
    
    # unvec each
    unvec_T_mean = vmap(lambda params: unvec(params, D, n))(T_mean_split)
    unvec_T_sample = vmap(lambda params: unvec(params, D, n))(T_sample_split)
    
    # form projector
    I_UUT = (np.eye(D) - anchor_point @ anchor_point.T)
    
    # apply this to each
    Deltas_mean = np.einsum('ij,ljk->lik', I_UUT, unvec_T_mean)
    Deltas_sample = np.einsum('ij,ljk->lik', I_UUT, unvec_T_sample)
    
    # convert posterior means to projections
    Ps_mean = convert_to_projs(Deltas_mean, anchor_point, reorthonormalize=reortho)
    Ps_sample = convert_to_projs(Deltas_sample, anchor_point, reorthonormalize=reortho)
    
    # return Ps_mean and Ps_sample
    return Ps_mean, Ps_sample

    
def run_grass_predict(pred_key, s_test, s, cfg, samples:dict, jitter=5e-4):
    anchor_point = np.array(cfg.inner_model.grass_config.anchor_point)
    n_samples = cfg.train.n_samples // cfg.train.n_thinning
    proj_params_samples = samples['grass-proj_params']
    assert n_samples == proj_params_samples.shape[0]
    
    # initialize vmap_args
    vmap_args = (random.split(pred_key, n_samples), proj_params_samples)

    cfg_Omega = cfg.inner_model.grass_config.Omega
    cfg_var = cfg.inner_model.grass_config.var
    cfg_length = cfg.inner_model.grass_config.length
    cfg_noise = cfg.inner_model.grass_config.noise
    require_noise = cfg.inner_model.grass_config.require_noise
    reortho = cfg.inner_model.grass_config.reorthonormalize
    
    if cfg_Omega is None:
        vmap_args += (samples['grass-Omega'],)
    else:
        cfg_Omega = np.array(cfg_Omega)
        vmap_args += (np.repeat(cfg_Omega[None,:,:], n_samples, axis=0),)
        
    if cfg_var is None:
        vmap_args += (samples['grass-kernel_var'],)
    else:
        vmap_args += (cfg_var * np.ones(n_samples),)
        
    if cfg_length is None:
        vmap_args += (samples['grass-kernel_length'],)
    else:
        vmap_args += (cfg_length * np.ones(n_samples),)
    
    if require_noise:
        if cfg_noise is None:
            vmap_args += (samples['grass-kernel_noise'],)
        else:
            vmap_args += (cfg_noise * np.ones(n_samples),)
    else:
        vmap_args += (np.zeros(n_samples),)
        
    Ps_means, Ps_preds = vmap(lambda key, proj_params, Omega, var, length, noise: grass_predict(key, s_test, s, anchor_point, Omega, var, length, noise, proj_params, reortho, jitter=jitter))(*vmap_args)
    return Ps_means, Ps_preds


def predict(key, X_test, s_test, X, s, Y, anchor_point, Omega, inner_var, inner_length, inner_noise, proj_params, proj_params_Ps, outer_var, outer_length, outer_noise, inner_jitter=2.5e-3, outer_jitter=7.5e-4, reortho=True, use_means=False):    
    # get keys for loc and space
    loc_key, space_key = random.split(key, 2)

    # grassmann predictions
    Ps_mean, Ps_sample = grass_predict(loc_key, s_test, s, anchor_point, Omega, inner_var, inner_length, inner_noise, proj_params, reortho, jitter=inner_jitter)

    # project data
    X_train_projs_train = np.einsum('ij,ljk->lik', X, proj_params_Ps) # spatial train, loc train
    if use_means:
        X_test_projs_test = np.einsum('ij,ljk->lik', X_test, Ps_mean)
    else:
        X_test_projs_test = np.einsum('ij,ljk->lik', X_test, Ps_sample)

    # stack train and test projects across loc
    Train = np.vstack([X_train_projs_train[i,:,:] for i in range(s.shape[0])])
    Test = np.vstack([X_test_projs_test[i,:,:] for i in range(s_test.shape[0])])

    # compute kernels between train and test data
    kernel_params = {'var': outer_var, 'length': outer_length, 'noise': outer_noise}
    K_pp = rbf(Test, Test, kernel_params, include_noise=False)
    K_pt = rbf(Test, Train, kernel_params, include_noise=False)
    K_tt = rbf(Train, Train, kernel_params)

    # perform conditioning
    K = K_pp - np.matmul(K_pt, lin.solve(K_tt, np.transpose(K_pt)))

    # # variances
    # sigma_noises = np.sqrt(np.clip(np.diag(K), a_min=0.0)) * random.normal(
    #                         space_key, K.shape[1:]
    #                       )

    # means
    means = np.matmul(K_pt, lin.solve(K_tt, Y))

    # predictions
    # preds = means + sigma_noises

    # predictions
    preds = dist.MultivariateNormal(loc=means,covariance_matrix=K + outer_jitter * np.eye(K.shape[0])).sample(space_key)

    # # we only want test_test means and preds
    # means_test_test = means[n_train_test + n_test_train:]
    # assert len(means_test_test) == n_test_test
    # preds_test_test = preds[n_train_test + n_test_train:]
    # assert len(preds_test_test) == n_test_test

    # unvec and return
    if use_means:
        return unvec(means, X_test.shape[0], s_test.shape[0]), unvec(preds, X_test.shape[0], s_test.shape[0]), Ps_mean
    else:
        return unvec(means, X_test.shape[0], s_test.shape[0]), unvec(preds, X_test.shape[0], s_test.shape[0]), Ps_sample


def run_predict(key, X_test, s_test, X, s, Ys, cfg, samples:dict, inner_jitter=2.5e-3, outer_jitter=7.5e-4, use_means=False):
    anchor_point = np.array(cfg.inner_model.grass_config.anchor_point)
    active_dimension = anchor_point.shape[1]
    n_samples = cfg.train.n_samples // cfg.train.n_thinning
    proj_params_samples = samples['grass-proj_params']
    proj_params_Ps_samples = samples['grass-Ps']
    assert n_samples == proj_params_samples.shape[0]
    
    # initialize loop args
    loop_args = (random.split(key, n_samples), proj_params_samples, proj_params_Ps_samples) # key: 0, proj_params: 1, proj_params_Ps: 2
    
    cfg_Omega = cfg.inner_model.grass_config.Omega
    cfg_inner_var = cfg.inner_model.grass_config.var
    cfg_inner_length = cfg.inner_model.grass_config.length
    cfg_inner_noise = cfg.inner_model.grass_config.noise
    reortho = cfg.inner_model.grass_config.reorthonormalize
    require_noise = cfg.inner_model.grass_config.require_noise
    
    cfg_outer_var = cfg.outer_model.gp_config.params.var
    cfg_outer_length = cfg.outer_model.gp_config.params.length
    cfg_outer_noise = cfg.outer_model.gp_config.params.noise
    
    if cfg_Omega is None:
        loop_args += (samples['grass-Omega'],) # Omega: 3
    else:
        cfg_Omega = np.array(cfg_Omega)
        loop_args += (np.repeat(cfg_Omega[None,:,:], n_samples, axis=0),)
    
    if cfg_inner_var is None:
        loop_args += (samples['grass-kernel_var'],) # inner_var: 4
    else:
        loop_args += (cfg_inner_var * np.ones(n_samples),)
        
    if cfg_inner_length is None:
        loop_args += (samples['grass-kernel_length'],) # inner_length: 5
    else:
        loop_args += (cfg_inner_length * np.ones(n_samples),)
    
    if require_noise:
        if cfg_inner_noise is None:
            loop_args += (samples['grass-kernel_noise'],) # inner_noise: 6
        else:
            loop_args += (cfg_inner_noise * np.ones(n_samples),)
    else:
        loop_args += (np.zeros(n_samples),)
    
    if cfg_outer_var is None:
        loop_args += (samples['reg-kernel_var'],) # outer_var: 7
    else:
        loop_args += (cfg_outer_var * np.ones(n_samples),)
        
    if cfg_outer_length is None:
        loop_args += (samples['reg-kernel_length'],) # outer_length: 8
    else:
        loop_args += (cfg_outer_length * np.ones(n_samples),)
    
    if cfg_outer_noise is None:
        loop_args += (samples['reg-kernel_noise'],) # outer_noise: 9
    else:
        loop_args += (cfg_outer_noise * np.ones(n_samples),)
    
    # indices of loop args
    loop_inds = {'key': 0, 'proj_params': 1, 'proj_params_Ps': 2, 'Omega': 3, 'inner_var': 4, 'inner_length': 5, 'inner_noise': 6, 'outer_var': 7, 'outer_length': 8, 'outer_noise': 9}
    
    # initialise arrays to hold results
    means = np.zeros((n_samples, X_test.shape[0], s_test.shape[0]))
    predictions = np.zeros((n_samples, X_test.shape[0], s_test.shape[0]))
    projs = np.zeros((n_samples, s_test.shape[0], anchor_point.shape[0], active_dimension))
    
    # loop over samples and predict
    for i in tqdm(range(n_samples)):
        key = loop_args[loop_inds['key']][i]
        proj_params = loop_args[loop_inds['proj_params']][i]
        proj_params_Ps = loop_args[loop_inds['proj_params_Ps']][i]
        Omega = loop_args[loop_inds['Omega']][i]
        inner_var = loop_args[loop_inds['inner_var']][i]
        inner_length = loop_args[loop_inds['inner_length']][i]
        inner_noise = loop_args[loop_inds['inner_noise']][i]
        outer_var = loop_args[loop_inds['outer_var']][i]
        outer_length = loop_args[loop_inds['outer_length']][i]
        outer_noise = loop_args[loop_inds['outer_noise']][i]
        m, p, proj = predict(key, X_test, s_test, X, s, vec(Ys), anchor_point, Omega, inner_var, inner_length, inner_noise, proj_params, proj_params_Ps, outer_var, outer_length, outer_noise, inner_jitter=inner_jitter, outer_jitter=outer_jitter, reortho=reortho, use_means=use_means)
        means = means.at[i,:,:].set(m)
        predictions = predictions.at[i,:,:].set(p)
        projs = projs.at[i,:,:,:].set(proj)
    
    return means, predictions, projs
