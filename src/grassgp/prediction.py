import jax.numpy as np
import jax.numpy.linalg as lin
import jax.random as random
from jax import vmap
from grassgp.grassmann import convert_to_projs
from tqdm import tqdm

from grassgp.utils import vec, unvec, kron_solve
from grassgp.kernels import rbf
import numpyro.distributions as dist

def predict_at_train_locs(rng_key, X, s, Ys, X_test, var, length, noise, Ps, jitter=7.5e-4):
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
def run_prediction_train_locs(pred_key, X, s, Ys, samples:dict, jitter=7.5e-4, know_reg_params: bool = True, params: dict = None):
    """function to run prediction at train locs"""
    if know_reg_params and params:
        vmap_args = (random.split(pred_key, samples['grass-Ps'].shape[0]), samples['grass-Ps'])
        means, predictions = vmap(lambda rng_key, Ps: predict_at_train_locs(rng_key, X, s, Ys, X, params['var'], params['length'], params['noise'], Ps, jitter=jitter))(*vmap_args)
    elif not know_reg_params:
        vmap_args = (random.split(pred_key, samples['grass-Ps'].shape[0]), samples['reg-var'], samples['reg-length'], samples['reg-noise'], samples['grass-Ps'])
        means, predictions = vmap(lambda rng_key, var, length, noise, Ps: predict_at_train_locs(rng_key, X, s, Ys, X, var, length, noise, Ps, jitter=jitter))(*vmap_args)
    return means, predictions

## function to perform prediction for the grassmann process
def grass_predict(key, s, s_test, T_var, T_length, T_noise, sigmas, L, projection_params, anchor_point, jitter=5e-4, reortho=True):
    D, n = anchor_point.shape
    n_train = s.shape[0]
    n_test = s_test.shape[0]

    # form Omega
    Omega = np.outer(sigmas, sigmas) * L

    # compute (temporal) kernels between train and test locs
    grass_kernel_params = {'var': T_var, 'length': T_length, 'noise': T_noise}
    # T_K_pp = rbf_covariance(s_test.reshape(n_test,-1), s_test.reshape(n_test,-1), T_var, T_length, T_noise, include_noise=False)
    # T_K_pt = rbf_covariance(s_test.reshape(n_test,-1), s.reshape(n_train,-1), T_var, T_length, T_noise, include_noise=False)
    # T_K_tt = rbf_covariance(s.reshape(n_train,-1), s.reshape(n_train,-1), T_var, T_length, T_noise)
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
    T_mean = np.matmul(M_cov_pt, kron_solve(T_K_tt, Omega, projection_params))
    
    # sample projection params for test locs
    T_sample = dist.MultivariateNormal(loc=T_mean, covariance_matrix=T_K).sample(key)
    
    # Convert mean and samples to projections
    
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


## function to perform prediction for the grassmann process
def grass_predict_no_noise(key, s, s_test, T_var, T_length, sigmas, L, projection_params, anchor_point, jitter=5e-4, reortho=True):
    D, n = anchor_point.shape
    n_train = s.shape[0]
    n_test = s_test.shape[0]

    # form Omega
    Omega = np.outer(sigmas, sigmas) * L

    # compute (temporal) kernels between train and test locs
    grass_kernel_params = {'var': T_var, 'length': T_length, 'noise': 0.0}
    # T_K_pp = rbf_covariance(s_test.reshape(n_test,-1), s_test.reshape(n_test,-1), T_var, T_length, 0.0, include_noise=False)
    # T_K_pt = rbf_covariance(s_test.reshape(n_test,-1), s.reshape(n_train,-1), T_var, T_length, 0.0, include_noise=False)
    # T_K_tt = rbf_covariance(s.reshape(n_train,-1), s.reshape(n_train,-1), T_var, T_length, 0.0)
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
    T_mean = np.matmul(M_cov_pt, kron_solve(T_K_tt, Omega, projection_params))
    
    # sample projection params for test locs
    T_sample = dist.MultivariateNormal(loc=T_mean, covariance_matrix=T_K).sample(key)
    
    # Convert mean and samples to projections
    
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


## runs grass_predict for each sample in samples
def run_grass_predict(pred_key, samples:dict, s, s_test, anchor_point, jitter=5e-4, grass_noise=False):
    if grass_noise:
        vmap_args = (random.split(pred_key, samples['grass-kernel_var'].shape[0]), samples['grass-kernel_var'], samples['grass-kernel_length'], samples['grass-kernel_noise'], samples['grass-sigmas'],samples['grass-L'], samples['grass-proj_params'])
        Ps_mean_samples, Ps_test_preds = vmap(lambda rng_key, T_var, T_length, T_noise, sigmas, L, projection_params: grass_predict(rng_key,  s, s_test, T_var, T_length, T_noise, sigmas, L, projection_params, anchor_point, jitter=jitter))(*vmap_args)
    else:
        vmap_args = (random.split(pred_key, samples['grass-kernel_var'].shape[0]), samples['grass-kernel_var'], samples['grass-kernel_length'], samples['grass-sigmas'],samples['grass-L'], samples['grass-proj_params'])
        Ps_mean_samples, Ps_test_preds = vmap(lambda rng_key, T_var, T_length, sigmas, L, projection_params: grass_predict_no_noise(rng_key, s, s_test, T_var, T_length, sigmas, L, projection_params, anchor_point, jitter=jitter))(*vmap_args)
    return Ps_mean_samples, Ps_test_preds

# function to predict at test spaces and test locs
def predict_no_grass_noise(rng_key, X, s, Y, X_test, s_test, var, length, noise, T_var, T_length, sigmas, L, projection_params, projection_params_Ps, anchor_point, active_dimension, grass_jitter=2.5e-3, reg_jitter=7.5e-4, use_means=False):
    # get keys for loc and space
    loc_key, space_key = random.split(rng_key, 2)

    # grassmann predictions
    Ps_means, Ps_preds = grass_predict_no_noise(loc_key, s, s_test, T_var, T_length, sigmas, L, projection_params, anchor_point, jitter = grass_jitter)

    # project data
    X_train_projs_train = np.einsum('ij,ljk->lik', X, projection_params_Ps) # spatial train, loc train
    if use_means:
        X_test_projs_test = np.einsum('ij,ljk->lik', X_test, Ps_means)
    else:
        X_test_projs_test = np.einsum('ij,ljk->lik', X_test, Ps_preds)

    # stack train and test projects across loc
    Train = np.vstack([X_train_projs_train[i,:,:] for i in range(s.shape[0])])
    Test = np.vstack([X_test_projs_test[i,:,:] for i in range(s_test.shape[0])])

    # compute kernels between train and test data
    kernel_params = {'var': var, 'length': length, 'noise': noise}
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
    preds = dist.MultivariateNormal(loc=means,covariance_matrix=K + reg_jitter * np.eye(K.shape[0])).sample(space_key)

    # # we only want test_test means and preds
    # means_test_test = means[n_train_test + n_test_train:]
    # assert len(means_test_test) == n_test_test
    # preds_test_test = preds[n_train_test + n_test_train:]
    # assert len(preds_test_test) == n_test_test

    # unvec and return
    if use_means:
        return unvec(means, X_test.shape[0], s_test.shape[0]), unvec(preds, X_test.shape[0], s_test.shape[0]), Ps_means
    else:
        return unvec(means, X_test.shape[0], s_test.shape[0]), unvec(preds, X_test.shape[0], s_test.shape[0]), Ps_preds


def run_prediction(pred_key, samples:dict, X, s, Ys, X_test, s_test, anchor_point, active_dimension, grass_jitter=2.5e-3, reg_jitter=7.5e-4, use_means=False, grass_noise=False):
    if grass_noise:
        vmap_args = (random.split(pred_key, samples['reg-kernel_var'].shape[0]), samples['reg-kernel_var'], samples['reg-kernel_length'], samples['reg-kernel_noise'], samples['grass-kernel_var'],samples['grass-kernel_length'], samples['grass-kernel_noise'], samples['grass-sigmas'],samples['grass-L'], samples['grass-proj_params'], samples['grass-Ps'])
        vmap_inds = {'rng_key': 0, 'var': 1, 'length': 2, 'noise': 3, 'T_var': 4, 'T_length': 5, 'T_noise': 6, 'sigmas': 7, 'L': 8, 'projection_params': 9, 'projection_params_Ps': 10}
        means = np.zeros((samples['reg-kernel_noise'].shape[0], X_test.shape[0], s_test.shape[0]))
        predictions = np.zeros((samples['reg-kernel_noise'].shape[0], X_test.shape[0], s_test.shape[0]))
        projs = np.zeros((samples['reg-kernel_noise'].shape[0], s_test.shape[0], X_test.shape[1], active_dimension))
        for i in tqdm(range(samples['reg-kernel_length'].shape[0])):
            rng_key = vmap_args[vmap_inds['rng_key']][i]
            var = vmap_args[vmap_inds['var']][i]
            length = vmap_args[vmap_inds['length']][i]
            noise = vmap_args[vmap_inds['noise']][i]
            T_var = vmap_args[vmap_inds['T_var']][i]
            T_length = vmap_args[vmap_inds['T_length']][i]
            T_noise = vmap_args[vmap_inds['T_noise']][i]
            sigmas = vmap_args[vmap_inds['sigmas']][i]
            L = vmap_args[vmap_inds['L']][i]
            projection_params = vmap_args[vmap_inds['projection_params']][i]
            projection_params_Ps = vmap_args[vmap_inds['projection_params_Ps']][i]
            m, p, proj = predict(rng_key, X, s, vec(Ys), X_test, s_test, var, length, noise, T_var, T_length, T_noise, sigmas, L, projection_params, projection_params_Ps, anchor_point, active_dimension, grass_jitter=grass_jitter, reg_jitter=reg_jitter, use_means=use_means)
            means = means.at[i,:,:].set(m)
            predictions = predictions.at[i,:,:].set(p)
            projs = projs.at[i,:,:].set(proj)
    else:
        vmap_args = (random.split(pred_key, samples['reg-kernel_var'].shape[0]), samples['reg-kernel_var'], samples['reg-kernel_length'], samples['reg-kernel_noise'], samples['grass-kernel_var'],samples['grass-kernel_length'], samples['grass-sigmas'],samples['grass-L'], samples['grass-proj_params'], samples['grass-Ps'])
        vmap_inds = {'rng_key': 0, 'var': 1, 'length': 2, 'noise': 3, 'T_var': 4, 'T_length': 5, 'sigmas': 6, 'L': 7, 'projection_params': 8, 'projection_params_Ps': 9}
        means = np.zeros((samples['reg-kernel_noise'].shape[0], X_test.shape[0], s_test.shape[0]))
        predictions = np.zeros((samples['reg-kernel_noise'].shape[0], X_test.shape[0], s_test.shape[0]))
        projs = np.zeros((samples['reg-kernel_noise'].shape[0], s_test.shape[0], X_test.shape[1], active_dimension))
        for i in tqdm(range(samples['reg-kernel_length'].shape[0])):
            rng_key = vmap_args[vmap_inds['rng_key']][i]
            var = vmap_args[vmap_inds['var']][i]
            length = vmap_args[vmap_inds['length']][i]
            noise = vmap_args[vmap_inds['noise']][i]
            T_var = vmap_args[vmap_inds['T_var']][i]
            T_length = vmap_args[vmap_inds['T_length']][i]
            sigmas = vmap_args[vmap_inds['sigmas']][i]
            L = vmap_args[vmap_inds['L']][i]
            projection_params = vmap_args[vmap_inds['projection_params']][i]
            projection_params_Ps = vmap_args[vmap_inds['projection_params_Ps']][i]
            m, p, proj = predict_no_grass_noise(rng_key, X, s, vec(Ys), X_test, s_test, var, length, noise, T_var, T_length, sigmas, L, projection_params, projection_params_Ps, anchor_point, active_dimension, grass_jitter=grass_jitter, reg_jitter=reg_jitter, use_means=use_means)
            means = means.at[i,:,:].set(m)
            predictions = predictions.at[i,:,:].set(p)
            projs = projs.at[i,:,:].set(proj)
    return means, predictions, projs
