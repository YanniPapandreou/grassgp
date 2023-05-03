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
import time
from hydra_zen import instantiate, make_config, builds, launch, to_yaml
import sys

# from jax.config import config
# config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import vmap, random
import jax.numpy.linalg as lin
from tqdm import tqdm

from grassgp.utils import subspace_angle, unvec, vec, kron_chol
from grassgp.grassmann import valid_grass_point, grass_dist, grass_log, convert_to_projs, grass_exp, grass_dist
from grassgp.means import zero_mean
from grassgp.kernels import rbf
from grassgp.plot_utils import flatten_samples

import chex
from chex import assert_shape, assert_rank
from dataclasses import dataclass, field
from typing import Tuple, Union, Callable, Tuple

import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_value
import numpyro.distributions as dist

from grassgp.inference import run_inference

import pickle
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams["figure.figsize"] = (10,6)


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
@chex.dataclass
class MatGP:
    d_in: int
    d_out: Tuple[int, int]
    mu: Callable = field(repr=False)
    k: Callable = field(repr=False)
    Omega: chex.ArrayDevice = field(repr=False)
    cov_jitter: float = field(default=1e-8, repr=False)

    def __post_init__(self):
        d, n = self.d_out
        d_n = d * n
        assert_shape(self.Omega, (d_n, d_n),
                    custom_message=f"Omega has shape {self.Omega.shape}; expected shape {(d_n, d_n)}")

    def model(self, s: chex.ArrayDevice, use_kron_chol: bool = True) -> chex.ArrayDevice:
        d, n = self.d_out
        d_n = d * n
        assert_rank(s, self.d_in)
        N = s.shape[0]

        # compute mean matrix M = [mu(s[1]), mu(s[2]), ..., mu(s[N])]
        M = np.hstack(vmap(self.mu)(s))
        assert_shape(M, (d, n*N))

        # compute kernel matrix
        K = self.k(s, s)
        # cond_num = numpyro.deterministic("cond_num", lin.cond(K))
        assert_shape(K, (N, N))

        # compute covariance matrix and cholesky factor
        if use_kron_chol:
            Chol = kron_chol(K + self.cov_jitter * np.eye(N), self.Omega)
        else:
            Cov = np.kron(K + self.cov_jitter * np.eye(N), self.Omega)
            Chol = lin.cholesky(Cov)

        # sample vec_Vs
        # Z = numpyro.sample("Z", dist.MultivariateNormal(covariance_matrix=np.eye(N*d_n)))
        Z = numpyro.sample("Z", dist.Normal().expand([N*d_n]))
        vec_Vs = numpyro.deterministic("vec_Vs", vec(M) + Chol @ Z)

        # form Vs
        Vs = numpyro.deterministic("Vs", vmap(lambda params: unvec(params, d, n))(np.array(vec_Vs.split(N))))
        return Vs

    def sample(self, seed: int, s: chex.ArrayDevice) -> chex.ArrayDevice:
        model = self.model
        seeded_model = handlers.seed(model, rng_seed=seed)
        return seeded_model(s)

    def predict(self, key: chex.ArrayDevice, s_test: chex.ArrayDevice, s_train: chex.ArrayDevice, Vs_train: chex.ArrayDevice, jitter: float = 1e-8) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        d, n = self.d_out
        d_in = self.d_in
        d_n = d * n
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
        # FIX: change for singular Omega
        # print(f"Rank of Omega = {lin.matrix_rank(self.Omega)}. Shape of Omega = {self.Omega.shape}")
        # if lin.matrix_rank(self.Omega) == d_n:
        #     mean_sols = kron_solve(K_train_train, self.Omega, vec(np.hstack(Vs_train)) - vec(M_train))
        #     cov_sols = vmap(lambda v: kron_solve(K_train_train, self.Omega, v), in_axes=1, out_axes=1)(K_train_test_Omega)
        # else:
        #     K_train_train_inv = lin.inv(K_train_train)
        #     Omega_pinv = lin.pinv(self.Omega)
        #     K_train_train_Omega_pinv = np.kron(K_train_train_inv, Omega_pinv)
        #     mean_sols = K_train_train_Omega_pinv @ (vec(np.hstack(Vs_train)) - vec(M_train))
        #     cov_sols = K_train_train_Omega_pinv @ K_train_test_Omega
        K_train_train_inv = lin.inv(K_train_train)
        Omega_pinv = lin.pinv(self.Omega)
        K_train_train_Omega_pinv = np.kron(K_train_train_inv, Omega_pinv)
        mean_sols = K_train_train_Omega_pinv @ (vec(np.hstack(Vs_train)) - vec(M_train))
        cov_sols = K_train_train_Omega_pinv @ K_train_test_Omega
        
        vec_post_mean = vec(M_test) + K_test_train_Omega @ mean_sols
        assert_shape(vec_post_mean, (d*n*N_test,),
                     custom_message=f"vec_post_mean should have shape {(d*n*N_test,)}; obtained {vec_post_mean.shape}")

        # cov_sols = vmap(lambda v: kron_solve(K_train_train, self.Omega, v), in_axes=1, out_axes=1)(K_train_test_Omega)
        post_cov = K_test_test_Omega - K_test_train_Omega @ cov_sols
        assert_shape(post_cov, (d*n*N_test, d*n*N_test),
                     custom_message=f"post_cov should have shape {(d*n*N_test,d*n*N_test)}; obtained {post_cov.shape}")

        # sample predictions
        post_cov += jitter * np.eye(d*n*N_test)
        
        # FIX: change for singular post_cov
        # print(f"Rank of posterior cov = {lin.matrix_rank(post_cov)}. Shape of posterior cov = {post_cov.shape}")
        vec_pred = dist.MultivariateNormal(loc=vec_post_mean, covariance_matrix=post_cov).sample(key)
        assert_shape(vec_pred, (d*n*N_test,),
                     custom_message=f"vec_pred should have shape {(d*n*N_test,)}; obtained {vec_pred.shape}")

        # unvec mean and preds and return
        post_mean = vmap(lambda params: unvec(params, d, n))(np.array(vec_post_mean.split(N_test)))
        pred = vmap(lambda params: unvec(params, d, n))(np.array(vec_pred.split(N_test)))
        return post_mean, pred


# %%
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
        I_UUT = np.eye(d) - self.U @ self.U.T
        V_mu = lambda s: I_UUT @ self.mu(s)
        V_Omega = I_UUT @ self.Omega @ I_UUT.T
        V = MatGP(d_in=self.d_in, d_out=self.d_out, mu=V_mu, k=self.k, Omega=V_Omega, cov_jitter=self.cov_jitter)
        Deltas_mean, Deltas_pred = V.predict(key, s_test, s_train, Vs_train, jitter=jitter)
        return Deltas_mean, Deltas_pred

    def predict_grass(self, key: chex.ArrayDevice, s_test: chex.ArrayDevice, s_train: chex.ArrayDevice, Vs_train: chex.ArrayDevice, jitter: float = 1e-8, reortho: bool = False) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        Deltas_mean, Deltas_pred = self.predict_tangents(key, s_test, s_train, Vs_train, jitter=jitter)
        Ws_mean = convert_to_projs(Deltas_mean, self.U, reorthonormalize=reortho)
        Ws_pred = convert_to_projs(Deltas_pred, self.U, reorthonormalize=reortho)
        return Ws_mean, Ws_pred


# %% [markdown]
# # Generate dataset

# %%
# generate dataset
N = 40
s_test = np.linspace(0, 1, N)
k = 2 * np.pi
x = np.cos(k * s_test).reshape(-1, 1)
y = np.sin(k * s_test).reshape(-1, 1)
Ws_test = np.hstack((x,y))[:,:,None]
assert vmap(valid_grass_point)(Ws_test).all()
d, n = Ws_test.shape[1:]

# plot dataset
for i in range(d):
    plt.plot(s_test, Ws_test[:,i,0])
    plt.title(f'{i+1}th component of projection')
    plt.grid()
    plt.xlabel(r'$s$')
    plt.show()

# %%
# subsample data
s_gap = 3
s_train = s_test[::s_gap].copy()
print(f"Number of training points: {s_train.shape[0]}")
Ws_train = Ws_test[::s_gap,:,:].copy()

for i in range(d):
    plt.plot(s_test, Ws_test[:,i,0])
    plt.scatter(s_train, Ws_train[:,i,0], c='r')
    plt.title(f'{i+1}th component of projection')
    plt.grid()
    plt.xlabel(r'$s$')
    plt.show()


# %%
def subspace_angle_to_grass_pt(theta):
    x = np.cos(theta).reshape(-1,1)
    y = np.sin(theta).reshape(-1,1)
    W = np.hstack((x,y))[:,:,None]
    W = W[0]
    return W

def loss(theta, Ws):
    W = subspace_angle_to_grass_pt(theta)
    return vmap(lambda x: grass_dist(W, x)**2)(Ws).sum()


# %%
thetas = np.linspace(0, np.pi, 1000)
losses = vmap(lambda theta: loss(theta, Ws_train))(thetas)

theta_argmin = thetas[losses.argmin()]
anchor_point = subspace_angle_to_grass_pt(theta_argmin)
assert valid_grass_point(anchor_point)

plt.plot(thetas,losses)
plt.scatter(theta_argmin, losses.min(), color="red", label=r'value of $S(\theta)$ for anchor point')
plt.grid()
plt.xlabel(r"$\theta$")
plt.legend()
plt.savefig('karcher-mean-brute-force-loss-plot.png',dpi=300,bbox_inches='tight',facecolor="w")
# plt.title(r"Plot of $\theta$ vs $S(\theta)$")
plt.show()

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
ax.set_ylim((-1.25,1.25))
ax.set_xlim((-1.25,1.25))
ax.scatter(Ws_train[:,0,0], Ws_train[:,1,0], color="blue", alpha=0.25, label='training points')
ax.scatter(anchor_point[0,0], anchor_point[1,0], color="red", marker="*", label="anchor point")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.grid()
ax.legend()
plt.tight_layout()
fig.savefig('anchor-point-circle-rep.png',dpi=300,bbox_inches='tight',facecolor="w")
# ax.set_title("Plot of observed points with anchor point on circle representa of Gr(2,1)")

plt.show()

# %%
# compute log of training data and full data
log_Ws_train = vmap(lambda W: grass_log(anchor_point, W))(Ws_train)
log_Ws_test = vmap(lambda W: grass_log(anchor_point, W))(Ws_test)

# %%
alphas = np.array([subspace_angle(w) for w in Ws_test])
alphas_train = np.array([subspace_angle(w) for w in Ws_train])
alphas_test = np.array([subspace_angle(w) for w in Ws_test])

# %%
ordinals_dict = {1: 'st', 2: 'nd', 3: 'rd'}
fig, axs = plt.subplots(2,2, figsize=(16,12))
for i in range(d):
    axs[0,i].plot(s_test, Ws_test[:,i,0], label="test data")
    axs[0,i].scatter(s_train, Ws_train[:,i,0], c='r', label="train data")
    axs[0,i].set_title(rf'{i+1}{ordinals_dict[i+1]} component of $P(s)$')
    axs[0,i].set_xlabel(r'$s$')
    axs[0,i].legend()
    axs[0,i].grid()

axs[1,0].plot(s_test, alphas_test, label="test data")
axs[1,0].scatter(s_train, alphas_train,c='r', label="train data")
axs[1,0].set_title(r'Plot of $s$ vs $\alpha(s)$')
axs[1,0].set_xlabel(r'$s$')
axs[1,0].legend()
axs[1,0].grid()

axs[1,1].set_ylim((-1.25,1.25))
axs[1,1].set_xlim((-1.25,1.25))
axs[1,1].scatter(Ws_test[:,0,0], Ws_test[:,1,0], alpha=0.5, label='test data')
axs[1,1].scatter(Ws_train[:,0,0], Ws_train[:,1,0], color='r', marker='x', alpha=0.9, label='train data')
axs[1,1].scatter(anchor_point[0,0], anchor_point[1,0], color="green", marker="*", label="anchor point")
axs[1,1].set_title(r'Datasets plotted on $S^{1}$ representation of $\operatorname{Gr}(2,1)$')
axs[1,1].set_xlabel(r"$x$")
axs[1,1].set_ylabel(r"$y$")
axs[1,1].legend()
axs[1,1].grid()

fig.savefig('dataset-plot.png',dpi=300,bbox_inches='tight',facecolor="w")
plt.show()


# %%
@dataclass
class Model:
    name: str
    anchor_point: list
    d_in: int
    Omega: Union[list, None]
    k_include_noise: bool
    var: Union[float, None]
    length: Union[float, None]
    noise: Union[float, None]
    require_noise: bool
    jitter: float
    cov_jitter: float
    L_jitter: float
    reorthonormalize: bool
    b: float
    ell: float
    savvas_param: bool
    s_train: list
    s_test: list
    Ws_train: list
    Ws_test: list
    
    def model(self, s, log_Ws):
        U = np.array(self.anchor_point)
        d, n = U.shape
        N = s.shape[0]
        d_n = d * n
        # N_params = N * d_n
        if log_Ws is not None:
            assert log_Ws.shape == (N, d, n)

        # get/sample Omega
        if self.Omega is None:
            sigmas = numpyro.sample('sigmas', dist.LogNormal(0.0, 1.0).expand([d_n]))
            L_factor = numpyro.sample('L_factor', dist.LKJ(d_n, 1.0))
            L = numpyro.deterministic('L', L_factor + self.L_jitter * np.eye(d_n))
            Omega = numpyro.deterministic('Omega', np.outer(sigmas, sigmas) * L)
        else:
            Omega = np.array(self.Omega)

        # get/sample kernel params
        if self.var is None:
            # sample var
            var = numpyro.sample("kernel_var", dist.LogNormal(0.0, self.b))
        else:
            var = self.var

        if self.length is None:
            # sample length
            if self.savvas_param:
                length = numpyro.sample("kernel_length", dist.LogNormal(-0.7, self.b))
            else:
                length = numpyro.sample("kernel_length", dist.LogNormal(0.0, self.b))
        else:
            length = self.length

        if self.require_noise:
            if self.noise is None:
                # sample noise
                noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, self.b))
            else:
                noise = self.noise
        else:
            noise = 0.0

        if self.savvas_param:
            kernel_params = {'var': var, 'length': np.sqrt(1 / length), 'noise': noise}
        else:
            kernel_params = {'var': var, 'length': length, 'noise': noise}
        
        # create kernel function
        k = lambda t, s: rbf(t, s, kernel_params, jitter=self.jitter, include_noise=self.k_include_noise)
        # create mean function
        mu = lambda s: zero_mean(s, d, n)

        # initialize GrassGP
        grass_gp = GrassGP(d_in=self.d_in, d_out=(d,n), mu=mu, k=k, Omega=Omega, U=U, cov_jitter=self.cov_jitter)

        # sample Deltas
        Deltas = grass_gp.tangent_model(s)

        # # # # # ! check what power this should be
        # likelihood
        ell = self.ell
        with numpyro.plate("N", N):
            numpyro.sample("log_W", dist.continuous.MatrixNormal(loc=Deltas, scale_tril_row=ell * np.eye(d),scale_tril_column=np.eye(n)), obs=log_Ws)


# %%
# # subsample data
# s_gap = 3
# s_train = s_test[::s_gap].copy()
# print(f"Number of training points: {s_train.shape[0]}")
# Ws_train = Ws_test[::s_gap,:,:].copy()

# # compute log of training data and full data
# log_Ws_train = vmap(lambda W: grass_log(anchor_point, W))(Ws_train)
# log_Ws_test = vmap(lambda W: grass_log(anchor_point, W))(Ws_test)

# %%
TangentSpaceModelConf = builds(Model, populate_full_signature=True)

my_model_conf = TangentSpaceModelConf(
    name = "My Model",
    anchor_point = anchor_point.tolist(),
    d_in = 1,
    Omega = None,
    k_include_noise= True,
    var = 1.0,
    length = None,
    noise = None,
    require_noise = False,
    jitter = 1e-06,
    cov_jitter = 1e-4,
    L_jitter = 1e-8,
    reorthonormalize = False,
    b = 0.5,
    ell = 0.01,
    savvas_param = False,
    s_train = s_train.tolist(),
    s_test = s_test.tolist(),
    Ws_train = Ws_train.tolist(),
    Ws_test = Ws_test.tolist()
)

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

PredictConfig = make_config(
    seed = 6578,
    splits = 25
)

PlotsConfig = make_config(
    acf_lags = 100,
    plot = True,
)

Config = make_config(
    model = my_model_conf,
    svi = SVIConfig,
    train = TrainConfig,
    predict = PredictConfig,
    plots = PlotsConfig,
    save_results = True,
    save_stdout = True,
    load_saved = False,
)

# %%
numpyro.render_model(instantiate(Config.model).model, model_args=(s_train, log_Ws_train))


# %%
def pickle_save(obj, name: str):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


# %%
def pickle_load(name: str):
    with open(name, 'rb') as f:
        obj = pickle.load(f)
    return obj


# %%
def predict_tangents(
    key: chex.ArrayDevice,
    s_test: chex.ArrayDevice,
    s_train: chex.ArrayDevice,
    Vs_train: chex.ArrayDevice,
    cfg,
    samples: dict,
    jitter: float = 1e-8
) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
    
    d_in = cfg.model.d_in
    U = np.array(cfg.model.anchor_point)
    d, n = U.shape
    cov_jitter = cfg.model.cov_jitter
    k_include_noise = cfg.model.k_include_noise
    kern_jitter = cfg.model.jitter
    n_samples = cfg.train.n_samples // cfg.train.n_thinning
    assert n_samples == samples['Deltas'].shape[0]
    
    def predict(
        key: chex.ArrayDevice,
        Omega: chex.ArrayDevice,
        var: float,
        length: float,
        noise: float,
    ) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        # iniatilize GrassGP
        kernel_params = {'var': var, 'length': length, 'noise': noise}
        k = lambda t, s: rbf(t, s, kernel_params, jitter=kern_jitter, include_noise=k_include_noise)
        mu = lambda s: zero_mean(s, d, n)
        grass_gp = GrassGP(d_in=d_in, d_out=(d, n), mu=mu, k=k, Omega=Omega, U=U, cov_jitter=cov_jitter)

        # predict
        Deltas_mean, Deltas_pred = grass_gp.predict_tangents(key, s_test, s_train, Vs_train, jitter=jitter)
        return Deltas_mean, Deltas_pred

    # initialize vmap args
    vmap_args = (random.split(key, n_samples),)
    
    cfg_Omega = cfg.model.Omega
    cfg_var = cfg.model.var
    cfg_length = cfg.model.length
    cfg_noise = cfg.model.noise
    cfg_require_noise = cfg.model.require_noise
    
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
        
    if cfg_require_noise:
        if cfg_noise is None:
            vmap_args += (samples['kernel_noise'],)
        else:
            vmap_args += (cfg_noise * np.ones(n_samples),)
    else:
        vmap_args += (np.zeros(n_samples),)
    
    assert len(vmap_args) == 5
    Deltas_means, Deltas_preds = vmap(predict)(*vmap_args)
    return Deltas_means, Deltas_preds


# %%
def predict_grass(
    key: chex.ArrayDevice,
    s_test: chex.ArrayDevice,
    s_train: chex.ArrayDevice,
    Vs_train: chex.ArrayDevice,
    cfg,
    samples: dict,
    jitter: float = 1e-8,
    reortho: bool = False
) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
    
    d_in = cfg.model.d_in
    U = np.array(cfg.model.anchor_point)
    d, n = U.shape
    cov_jitter = cfg.model.cov_jitter
    k_include_noise = cfg.model.k_include_noise
    kern_jitter = cfg.model.jitter
    n_samples = cfg.train.n_samples // cfg.train.n_thinning
    assert n_samples == samples['Deltas'].shape[0]
    
    def predict(
        key: chex.ArrayDevice,
        Omega: chex.ArrayDevice,
        var: float,
        length: float,
        noise: float,
    ) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        # iniatilize GrassGP
        kernel_params = {'var': var, 'length': length, 'noise': noise}
        k = lambda t, s: rbf(t, s, kernel_params, jitter=kern_jitter, include_noise=k_include_noise)
        mu = lambda s: zero_mean(s, d, n)
        grass_gp = GrassGP(d_in=d_in, d_out=(d, n), mu=mu, k=k, Omega=Omega, U=U, cov_jitter=cov_jitter)

        # predict
        Ws_mean, Ws_pred = grass_gp.predict_grass(key, s_test, s_train, Vs_train, jitter=jitter, reortho=reortho)
        return Ws_mean, Ws_pred

    # initialize vmap args
    vmap_args = (random.split(key, n_samples),)
    
    cfg_Omega = cfg.model.Omega
    cfg_var = cfg.model.var
    cfg_length = cfg.model.length
    cfg_noise = cfg.model.noise
    cfg_require_noise = cfg.model.require_noise
    
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
        
    if cfg_require_noise:
        if cfg_noise is None:
            vmap_args += (samples['kernel_noise'],)
        else:
            vmap_args += (cfg_noise * np.ones(n_samples),)
    else:
        vmap_args += (np.zeros(n_samples),)
    
    assert len(vmap_args) == 5
    Ws_means, Ws_preds = vmap(predict)(*vmap_args)
    return Ws_means, Ws_preds


# %% tags=[]
def train_analyse(cfg):
    # instantiate grass model
    model = instantiate(cfg.model).model
    
    save_results = cfg.save_results
    plot_figs = cfg.plots.plot
    save_stdout = cfg.save_stdout
    
    anchor_point = np.array(cfg.model.anchor_point)
    s_train = np.array(cfg.model.s_train)
    Ws_train = np.array(cfg.model.Ws_train)
    s_test = np.array(cfg.model.s_test)
    Ws_test = np.array(cfg.model.Ws_test)
    
    log_Ws_train = vmap(lambda W: grass_log(anchor_point, W))(Ws_train)
    log_Ws_test = vmap(lambda W: grass_log(anchor_point, W))(Ws_test)
        
    if save_results:
        training_test_data = {'s_train': s_train, 's_test': s_test, 'Ws_train': Ws_train, 'Ws_test': Ws_test, 'log_Ws_train': log_Ws_train, 'log_Ws_test': log_Ws_test, 'anchor_point': anchor_point}
        pickle_save(training_test_data, 'training_test_data.pickle')
    
    # run SVI to get MAP esimtate to initialise MCMC
    svi_key = random.PRNGKey(cfg.svi.seed)
    maxiter = cfg.svi.maxiter
    step_size = cfg.svi.step_size
    print("Running SVI for MAP estimate to initialise MCMC")
    svi_results = run_svi_for_map(svi_key, model, maxiter, step_size, s_train, log_Ws_train)
    
    if save_results:
        pickle_save(svi_results, 'svi_results.pickle')
    
    if plot_figs:
        # plot svi losses
        plt.plot(svi_results.losses)
        plt.show()
        
    # get initialisation from SVI results
    map_est = svi_results.params
    strip_val = len('_auto_loc')
    init_values = {key[:-strip_val]:value for (key, value) in map_est.items()}
    
    # run HMC
    train_key = random.PRNGKey(cfg.train.seed)
    mcmc_config = {'num_warmup' : cfg.train.n_warmup, 'num_samples' : cfg.train.n_samples, 'num_chains' : cfg.train.n_chains, 'thinning' : cfg.train.n_thinning, 'init_strategy' : init_to_value(values=init_values)}
    print("HMC starting.")
    mcmc = run_inference(train_key, mcmc_config, model, s_train, log_Ws_train)
    
    if save_stdout:
        original_stdout = sys.stdout
        with open('hmc_log.txt', 'w') as f:
            sys.stdout = f
            mcmc.print_summary()
            sys.stdout = original_stdout
    
    samples = mcmc.get_samples()
    inference_data = samples.copy()
    for param, initial_val in init_values.items():
        inference_data[f"{param}-initial_value"] = initial_val
    
    if save_results:
        pickle_save(inference_data, "inference_data.pickle")
    
    samples = dict(filter(lambda elem: 'initial_value' not in elem[0], inference_data.items()))
    initial_values = dict(filter(lambda elem: 'initial_value' in elem[0], inference_data.items()))
    assert set(samples.keys()).union(initial_values.keys()) == set(inference_data.keys())
    
    if plot_figs:
        my_samples = flatten_samples(samples, ignore=[])
        trace_plot_vars = ['kernel_length']
        for key in my_samples.keys():
            if 'Omega' in key:
                trace_plot_vars.append(key)
            if 'sigmas' in key:
                trace_plot_vars.append(key)

        my_samples[trace_plot_vars].plot(subplots=True, figsize=(10,40), sharey=False)
        plt.show()
        
        for var in trace_plot_vars:
            sm.graphics.tsa.plot_acf(my_samples[var], lags=cfg.plots.acf_lags)
            plt.title(f"acf for {var}")
            plt.show()
    
    # compute Ws's from mcmc samples
    tol=1e-5
    samples_Ws_train = vmap(lambda Deltas: convert_to_projs(Deltas, anchor_point, reorthonormalize=False))(samples['Deltas'])
    for ws in samples_Ws_train:
        assert vmap(lambda w: valid_grass_point(w, tol=tol))(ws).all()
        
    if save_results:
        pickle_save(samples_Ws_train, "samples_Ws_train.pickle")
        
    # compute barycenters for mcmc results
    def subspace_angle_to_grass_pt(theta):
        x = np.cos(theta).reshape(-1,1)
        y = np.sin(theta).reshape(-1,1)
        W = np.hstack((x,y))[:,:,None]
        W = W[0]
        return W

    def loss_func(theta, Ws):
        W = subspace_angle_to_grass_pt(theta)
        return vmap(lambda x: grass_dist(W, x)**2)(Ws).sum()
    
    mcmc_barycenters = []
    for i in tqdm(range(s_train.shape[0])):
        points = samples_Ws_train[:,i,:,:]
        thetas = np.linspace(0, np.pi, 1000)
        mcmc_losses = vmap(lambda theta: loss_func(theta, points))(thetas)
        mcmc_theta_argmin = thetas[mcmc_losses.argmin()]
        barycenter = subspace_angle_to_grass_pt(mcmc_theta_argmin)
        mcmc_barycenters.append(barycenter)
        
    mcmc_barycenters = np.array(mcmc_barycenters)
    if save_results:
        pickle_save(mcmc_barycenters, "mcmc_barycenters.pickle")
        
    if plot_figs:
        bary_losses = []
        for i in tqdm(range(s_train.shape[0])):
            loss = vmap(lambda W: grass_dist(mcmc_barycenters[i], W)**2)(samples_Ws_train[:,i,:,:]).sum()
            bary_losses.append(loss)

        plt.plot(bary_losses)
        plt.title("Final loss for computed barycenters")
        plt.show()
    
    in_sample_errors = vmap(grass_dist)(Ws_train, mcmc_barycenters)
    if plot_figs:
        plt.plot(s_train,in_sample_errors)
        plt.show()
        
    sd_s_train = []
    for i in tqdm(range(s_train.shape[0])):
        fixed = mcmc_barycenters[i]
        dists = vmap(lambda W: grass_dist(W, fixed))(samples_Ws_train[:,i,:,:])
        dists_Sq = dists**2
        sd_s_train.append(np.sqrt(dists_Sq.mean()))
    sd_s_train = np.array(sd_s_train)
    
    pd_data = {'s': s_train, 'errors': in_sample_errors, 'sd': sd_s_train}
    in_sample_errors_df = pd.DataFrame(data=pd_data)
    
    if save_results:
        pickle_save(in_sample_errors_df, "in_sample_errors_df.pickle")
        
    samples_alphas_train = np.array([[subspace_angle(w)for w in Ws_sample] for Ws_sample in samples_Ws_train])
    if save_results:
        pickle_save(samples_alphas_train, "samples_alphas_train.pickle")
        
    if plot_figs:
        percentile_levels = [2.5, 97.5]
        conf_level = percentile_levels[-1] - percentile_levels[0]
        percentiles = np.percentile(samples_alphas_train, np.array(percentile_levels), axis=0)
        lower = percentiles[0,:]
        upper = percentiles[1,:]
        plt.plot(s_test, alphas, c='black', alpha=0.5, label='full data')
        plt.scatter(s_train, alphas_train, label='training data', c='g')
        plt.scatter(s_train, samples_alphas_train.mean(axis=0), label='mean samples', c='r')
        plt.fill_between(s_train, lower, upper,  color='lightblue', alpha=0.75,label=f'{conf_level}% credible interval')
        plt.xlabel(r"$s$")
        plt.ylabel("subspace angle")
        plt.legend()
        plt.show()
        
    print("Prediction starting")
    pred_key = random.PRNGKey(cfg.predict.seed)
    pred_key_tangent, pred_key_grass = random.split(pred_key, 2)
    
    Deltas_means, Deltas_preds = predict_tangents(pred_key_tangent, s_test, s_train, log_Ws_train, cfg, samples)
    assert np.isnan(Deltas_means).sum() == 0
    assert np.isnan(Deltas_preds).sum() == 0
    
    if save_results:
        pickle_save(Deltas_means, "Deltas_means.pickle")
        pickle_save(Deltas_preds, "Deltas_preds.pickle")

    if plot_figs:
        plt.rcParams["figure.figsize"] = (12,6)
        percentile_levels = [2.5, 97.5]
        conf_level = percentile_levels[-1] - percentile_levels[0]
        for i in range(d):
            obs = log_Ws_train[:,i,0]
            means = Deltas_means[:,:,i,0]
            means_avg = np.mean(means, axis=0)
            preds = Deltas_preds[:,:,i,0]
            percentiles = np.percentile(preds, np.array(percentile_levels), axis=0)
            lower = percentiles[0,:]
            upper = percentiles[1,:]
            plt.plot(s_test, log_Ws_test[:,i,0], label='full data',c='black', alpha=0.75, linestyle='dashed')
            plt.scatter(s_train, log_Ws_train[:,i,0], label='training data', c='g')
            plt.plot(s_test, means_avg, label='averaged mean prediction', c='r', alpha=0.75)
            plt.fill_between(s_test, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
            plt.xlabel(r"$s$")
            plt.legend()
            plt.vlines(s_train, 0.99*lower.min(), 1.01*upper.max(), colors='green', linestyles='dashed')
            plt.title(f"{i+1}th component of tangents")
            plt.show()
    
    Ws_means, Ws_preds = predict_grass(pred_key_grass, s_test, s_train, log_Ws_train, cfg, samples)
    assert np.isnan(Ws_means).sum() == 0
    assert np.isnan(Ws_preds).sum() == 0

    if save_results:
        pickle_save(Ws_means, "Ws_means.pickle")
        pickle_save(Ws_preds, "Ws_preds.pickle")
        
    if plot_figs:
        plt.rcParams["figure.figsize"] = (12,6)
        percentile_levels = [2.5, 97.5]
        conf_level = percentile_levels[-1] - percentile_levels[0]
        for i in range(d):
            obs = Ws_train[:,i,0]
            means = Ws_means[:,:,i,0]
            means_avg = np.mean(means, axis=0)
            preds = Ws_preds[:,:,i,0]
            percentiles = np.percentile(preds, np.array(percentile_levels), axis=0)
            lower = percentiles[0,:]
            upper = percentiles[1,:]
            plt.plot(s_test, Ws_test[:,i,0], label='full data',c='black', alpha=0.75, linestyle='dashed')
            plt.scatter(s_train, Ws_train[:,i,0], label='training data', c='g')
            plt.plot(s_test, means_avg, label='averaged mean prediction', c='r', alpha=0.75)
            plt.fill_between(s_test, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
            plt.xlabel(r"$s$")
            plt.legend()
            plt.vlines(s_train, 0.99*lower.min(), 1.01*upper.max(), colors='green', linestyles='dashed')
            plt.title(f"{i+1}th component of projections")
            plt.show()
        
    alphas_means = np.array([[subspace_angle(w) for w in mean] for mean in Ws_means])
    alphas_preds = np.array([[subspace_angle(w) for w in pred] for pred in Ws_preds])
    if save_results:
        pickle_save(alphas_means, "alpha_means.pickle")
        pickle_save(alphas_preds, "alpha_preds.pickle")
    
    if plot_figs:
        plt.rcParams["figure.figsize"] = (12,6)
        percentile_levels = [2.5, 97.5]
        conf_level = percentile_levels[-1] - percentile_levels[0]
        alphas_means_avg = np.mean(alphas_means, axis=0)
        percentiles = np.percentile(alphas_preds, np.array(percentile_levels), axis=0)
        lower = percentiles[0,:]
        upper = percentiles[1,:]
        plt.plot(s_test, alphas, label='full data',c='black', alpha=0.75, linestyle='dashed')
        plt.scatter(s_train, alphas_train, label='training data', c='g')
        plt.plot(s_test, alphas_means_avg, label='averaged mean prediction', c='r', alpha=0.75)
        plt.fill_between(s_test, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
        plt.xlabel(r"$s$")
        plt.ylabel("subspace angle")
        plt.legend()
        plt.vlines(s_train, 0, np.pi, colors='green', linestyles='dashed')
        plt.title(f"predictions for subspace angles")
        plt.show()
        
    test_means_mcmc_barycenters = []
    for i in tqdm(range(s_test.shape[0])):
        points = Ws_means[:,i,:,:]
        thetas = np.linspace(0, np.pi, 1000)
        test_means_mcmc_losses = vmap(lambda theta: loss_func(theta, points))(thetas)
        test_means_mcmc_theta_argmin = thetas[test_means_mcmc_losses.argmin()]
        barycenter = subspace_angle_to_grass_pt(test_means_mcmc_theta_argmin)
        test_means_mcmc_barycenters.append(barycenter)
        
    test_preds_mcmc_barycenters = []
    for i in tqdm(range(s_test.shape[0])):
        points = Ws_preds[:,i,:,:]
        thetas = np.linspace(0, np.pi, 1000)
        test_preds_mcmc_losses = vmap(lambda theta: loss_func(theta, points))(thetas)
        test_preds_mcmc_theta_argmin = thetas[test_preds_mcmc_losses.argmin()]
        barycenter = subspace_angle_to_grass_pt(test_preds_mcmc_theta_argmin)
        test_preds_mcmc_barycenters.append(barycenter)
        
    test_means_mcmc_barycenters = np.array(test_means_mcmc_barycenters)
    test_preds_mcmc_barycenters = np.array(test_preds_mcmc_barycenters)
    
    if save_results:
        pickle_save(test_means_mcmc_barycenters, "test_means_mcmc_barycenters.pickle")
        pickle_save(test_preds_mcmc_barycenters, "test_preds_mcmc_barycenters.pickle")
    
    out_sample_mean_errors = vmap(grass_dist)(Ws_test, test_means_mcmc_barycenters)
    out_sample_pred_errors = vmap(grass_dist)(Ws_test, test_preds_mcmc_barycenters)
    
    if plot_figs:
        plt.plot(s_test,out_sample_mean_errors, label='error using means')
        plt.plot(s_test,out_sample_pred_errors, label='error using preds')
        plt.vlines(s_train, 0, 1, colors="green", linestyles="dashed")
        plt.legend()
        plt.show()
    
    sd_s_test_means = []
    for i in tqdm(range(s_test.shape[0])):
        fixed = test_preds_mcmc_barycenters[i]
        dists = vmap(lambda W: grass_dist(W, fixed))(Ws_means[:,i,:,:])
        dists_Sq = dists**2
        sd_s_test_means.append(np.sqrt(dists_Sq.mean()))
        
    sd_s_test_preds = []
    for i in tqdm(range(s_test.shape[0])):
        fixed = test_preds_mcmc_barycenters[i]
        dists = vmap(lambda W: grass_dist(W, fixed))(Ws_preds[:,i,:,:])
        dists_Sq = dists**2
        sd_s_test_preds.append(np.sqrt(dists_Sq.mean()))
    
    sd_s_test_means = np.array(sd_s_test_means)
    sd_s_test_preds = np.array(sd_s_test_preds)
    
    test_pd_data = {'s': s_test, 'errors_mean': out_sample_mean_errors, 'errors_pred': out_sample_pred_errors, 'sd_mean': sd_s_test_means, 'sd_pred': sd_s_test_preds}
    out_sample_errors_df = pd.DataFrame(data=test_pd_data)
    
    if save_results:
        pickle_save(out_sample_errors_df, "out_sample_errors_df.pickle")


# %%
def load_analyse(cfg):
    # instantiate grass model
    model = instantiate(cfg.model).model
    
    plot_figs = cfg.plots.plot
    
    training_test_data = pickle_load('training_test_data.pickle')
    anchor_point = training_test_data['anchor_point']
    s_train = training_test_data['s_train']
    Ws_train = training_test_data['Ws_train']
    s_test = training_test_data['s_test']
    Ws_test = training_test_data['Ws_test']
    log_Ws_train = training_test_data['log_Ws_train']
    log_Ws_test = training_test_data['log_Ws_test']
    
    
    svi_key = random.PRNGKey(cfg.svi.seed)
    maxiter = cfg.svi.maxiter
    step_size = cfg.svi.step_size
    svi_results = pickle_load('svi_results.pickle')
        
    if plot_figs:
        # plot svi losses
        plt.plot(svi_results.losses)
        plt.show()
        
    with open('hmc_log.txt') as f:
        print(f.read())
    
    inference_data = pickle_load('inference_data.pickle')
    
    # get initialisation from SVI results
    map_est = svi_results.params
    strip_val = len('_auto_loc')
    init_values = {key[:-strip_val]:value for (key, value) in map_est.items()}
        
    samples = dict(filter(lambda elem: 'initial_value' not in elem[0], inference_data.items()))
    initial_values = dict(filter(lambda elem: 'initial_value' in elem[0], inference_data.items()))
    assert set(samples.keys()).union(initial_values.keys()) == set(inference_data.keys())
    
    if plot_figs:
        my_samples = flatten_samples(samples, ignore=[])
        trace_plot_vars = ['kernel_length']
        for key in my_samples.keys():
            if 'Omega' in key:
                trace_plot_vars.append(key)
            if 'sigmas' in key:
                trace_plot_vars.append(key)

        my_samples[trace_plot_vars].plot(subplots=True, figsize=(10,40), sharey=False)
        plt.show()
        
        for var in trace_plot_vars:
            sm.graphics.tsa.plot_acf(my_samples[var], lags=cfg.plots.acf_lags)
            plt.title(f"acf for {var}")
            plt.show()
    
    # compute Ws's from mcmc samples
    tol=1e-5
    samples_Ws_train = pickle_load("samples_Ws_train.pickle")
    for ws in samples_Ws_train:
        assert vmap(lambda w: valid_grass_point(w, tol=tol))(ws).all()
            
    # compute barycenters for mcmc results
    def subspace_angle_to_grass_pt(theta):
        x = np.cos(theta).reshape(-1,1)
        y = np.sin(theta).reshape(-1,1)
        W = np.hstack((x,y))[:,:,None]
        W = W[0]
        return W

    def loss_func(theta, Ws):
        W = subspace_angle_to_grass_pt(theta)
        return vmap(lambda x: grass_dist(W, x)**2)(Ws).sum()
    
        
    mcmc_barycenters = pickle_load("mcmc_barycenters.pickle")
        
    if plot_figs:
        bary_losses = []
        for i in tqdm(range(s_train.shape[0])):
            loss = vmap(lambda W: grass_dist(mcmc_barycenters[i], W)**2)(samples_Ws_train[:,i,:,:]).sum()
            bary_losses.append(loss)

        plt.plot(bary_losses)
        plt.title("Final loss for computed barycenters")
        plt.show()
    
    in_sample_errors_df = pickle_load("in_sample_errors_df.pickle")
    if plot_figs:
        plt.plot(s_train,in_sample_errors_df['errors'])
        plt.show()

    samples_alphas_train = pickle_load("samples_alphas_train.pickle")
        
    if plot_figs:
        percentile_levels = [2.5, 97.5]
        conf_level = percentile_levels[-1] - percentile_levels[0]
        percentiles = np.percentile(samples_alphas_train, np.array(percentile_levels), axis=0)
        lower = percentiles[0,:]
        upper = percentiles[1,:]
        plt.plot(s_test, alphas, c='black', alpha=0.5, label='full data')
        plt.scatter(s_train, alphas_train, label='training data', c='g')
        plt.scatter(s_train, samples_alphas_train.mean(axis=0), label='mean samples', c='r')
        plt.fill_between(s_train, lower, upper,  color='lightblue', alpha=0.75,label=f'{conf_level}% credible interval')
        plt.xlabel(r"$s$")
        plt.ylabel("subspace angle")
        plt.legend()
        plt.show()
        
    print("Prediction starting")
    pred_key = random.PRNGKey(cfg.predict.seed)
    pred_key_tangent, pred_key_grass = random.split(pred_key, 2)
    
    Deltas_means = pickle_load("Deltas_means.pickle")
    Deltas_preds = pickle_load("Deltas_preds.pickle")
    assert np.isnan(Deltas_means).sum() == 0
    assert np.isnan(Deltas_preds).sum() == 0
    
    if plot_figs:
        plt.rcParams["figure.figsize"] = (12,6)
        percentile_levels = [2.5, 97.5]
        conf_level = percentile_levels[-1] - percentile_levels[0]
        for i in range(d):
            obs = log_Ws_train[:,i,0]
            means = Deltas_means[:,:,i,0]
            means_avg = np.mean(means, axis=0)
            preds = Deltas_preds[:,:,i,0]
            percentiles = np.percentile(preds, np.array(percentile_levels), axis=0)
            lower = percentiles[0,:]
            upper = percentiles[1,:]
            plt.plot(s_test, log_Ws_test[:,i,0], label='full data',c='black', alpha=0.75, linestyle='dashed')
            plt.scatter(s_train, log_Ws_train[:,i,0], label='training data', c='g')
            plt.plot(s_test, means_avg, label='averaged mean prediction', c='r', alpha=0.75)
            plt.fill_between(s_test, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
            plt.xlabel(r"$s$")
            plt.legend()
            plt.vlines(s_train, 0.99*lower.min(), 1.01*upper.max(), colors='green', linestyles='dashed')
            plt.title(f"{i+1}th component of tangents")
            plt.show()
    
    Ws_means = pickle_load("Ws_means.pickle")
    Ws_preds = pickle_load("Ws_preds.pickle")
    assert np.isnan(Ws_means).sum() == 0
    assert np.isnan(Ws_preds).sum() == 0
        
    if plot_figs:
        plt.rcParams["figure.figsize"] = (12,6)
        percentile_levels = [2.5, 97.5]
        conf_level = percentile_levels[-1] - percentile_levels[0]
        for i in range(d):
            obs = Ws_train[:,i,0]
            means = Ws_means[:,:,i,0]
            means_avg = np.mean(means, axis=0)
            preds = Ws_preds[:,:,i,0]
            percentiles = np.percentile(preds, np.array(percentile_levels), axis=0)
            lower = percentiles[0,:]
            upper = percentiles[1,:]
            plt.plot(s_test, Ws_test[:,i,0], label='full data',c='black', alpha=0.75, linestyle='dashed')
            plt.scatter(s_train, Ws_train[:,i,0], label='training data', c='g')
            plt.plot(s_test, means_avg, label='averaged mean prediction', c='r', alpha=0.75)
            plt.fill_between(s_test, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
            plt.xlabel(r"$s$")
            plt.legend()
            plt.vlines(s_train, 0.99*lower.min(), 1.01*upper.max(), colors='green', linestyles='dashed')
            plt.title(f"{i+1}th component of projections")
            plt.show()
        
    alphas_means = pickle_load("alpha_means.pickle")
    alphas_preds = pickle_load("alpha_preds.pickle")
    
    if plot_figs:
        plt.rcParams["figure.figsize"] = (12,6)
        percentile_levels = [2.5, 97.5]
        conf_level = percentile_levels[-1] - percentile_levels[0]
        alphas_means_avg = np.mean(alphas_means, axis=0)
        percentiles = np.percentile(alphas_preds, np.array(percentile_levels), axis=0)
        lower = percentiles[0,:]
        upper = percentiles[1,:]
        plt.plot(s_test, alphas, label='full data',c='black', alpha=0.75, linestyle='dashed')
        plt.scatter(s_train, alphas_train, label='training data', c='g')
        plt.plot(s_test, alphas_means_avg, label='averaged mean prediction', c='r', alpha=0.75)
        plt.fill_between(s_test, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
        plt.xlabel(r"$s$")
        plt.ylabel("subspace angle")
        plt.legend()
        plt.vlines(s_train, 0, np.pi, colors='green', linestyles='dashed')
        plt.title(f"predictions for subspace angles")
        plt.show()
        
    test_means_mcmc_barycenters = pickle_load("test_means_mcmc_barycenters.pickle")
    test_preds_mcmc_barycenters = pickle_load("test_preds_mcmc_barycenters.pickle")
    
    out_sample_errors_df = pickle_load("out_sample_errors_df.pickle")
    
    out_sample_mean_errors = out_sample_errors_df["errors_mean"]
    out_sample_pred_errors = out_sample_errors_df["errors_pred"]
    sd_s_test_means = out_sample_errors_df["sd_mean"]
    sd_s_test_preds = out_sample_errors_df["sd_pred"]
    
    if plot_figs:
        plt.plot(s_test,out_sample_mean_errors, label='error using means')
        plt.plot(s_test,out_sample_pred_errors, label='error using preds')
        plt.vlines(s_train, 0, 1, colors="green", linestyles="dashed")
        plt.legend()
        plt.show()

# %% tags=[] jupyter={"outputs_hidden": true}
load_analyse(Config)

# %% tags=[]
train_analyse(Config)


# %%
def diagnostic_plots(cfg):
    # instantiate grass model
    model = instantiate(cfg.model).model
    svi_results = pickle_load('svi_results.pickle')
    
    # plt.plot(svi_results.losses)
    # plt.show()
    
    inference_data = pickle_load('inference_data.pickle')
    # print(inference_data.keys())
    
    # get initialisation from SVI results
    map_est = svi_results.params
    strip_val = len('_auto_loc')
    init_values = {key[:-strip_val]:value for (key, value) in map_est.items()}
        
    samples = dict(filter(lambda elem: 'initial_value' not in elem[0], inference_data.items()))
    initial_values = dict(filter(lambda elem: 'initial_value' in elem[0], inference_data.items()))
    assert set(samples.keys()).union(initial_values.keys()) == set(inference_data.keys())
    
    my_samples = flatten_samples(samples, ignore=[])
    trace_plot_vars = ['kernel_length']
    for key in my_samples.keys():
        if 'L_factor[0,1]' in key:
            trace_plot_vars.append(key)
        if 'L_factor[1,0]' in key:
            trace_plot_vars.append(key)
        if 'sigmas' in key:
            trace_plot_vars.append(key)

    fig, axs = plt.subplots(5,2,figsize=(20,30))
    samples_filtered = my_samples[trace_plot_vars]
    name_map = {"kernel_length": r"$\ell$", "L_factor[0,1]": r"$L_{12}$", "L_factor[1,0]": r"$L_{21}$", "sigmas[0]": r"$\sigma_{1}$", "sigmas[1]": r"$\sigma_{2}$"}
    for (i,var) in enumerate(trace_plot_vars):
        var_name = name_map[var]
        axs[i,0].plot(samples_filtered[var],c='k')
        axs[i,0].set_xlabel(r"$n$")
        axs[i,0].grid()
        axs[i,0].set_title(f"Traceplot for {var_name}")
        sm.graphics.tsa.plot_acf(samples_filtered[var], lags=Config.plots.acf_lags, ax=axs[i,1], title=f"Autocorrelation for {var_name}")
        axs[i,1].set_xlabel("lag")
        axs[i,1].grid()

    fig.savefig('diagnostic-plots.png',dpi=300,bbox_inches='tight',facecolor="w")
    plt.show()    
    


# %%
diagnostic_plots(Config)


# %%
def in_sample_plots(cfg):
    # instantiate grass model
    model = instantiate(cfg.model).model
        
    training_test_data = pickle_load('training_test_data.pickle')
    anchor_point = training_test_data['anchor_point']
    s_train = training_test_data['s_train']
    Ws_train = training_test_data['Ws_train']
    s_test = training_test_data['s_test']
    Ws_test = training_test_data['Ws_test']
    log_Ws_train = training_test_data['log_Ws_train']
    log_Ws_test = training_test_data['log_Ws_test']
    
    svi_key = random.PRNGKey(cfg.svi.seed)
    maxiter = cfg.svi.maxiter
    step_size = cfg.svi.step_size
    svi_results = pickle_load('svi_results.pickle')
    
    inference_data = pickle_load('inference_data.pickle')
    
    # get initialisation from SVI results
    map_est = svi_results.params
    strip_val = len('_auto_loc')
    init_values = {key[:-strip_val]:value for (key, value) in map_est.items()}
        
    samples = dict(filter(lambda elem: 'initial_value' not in elem[0], inference_data.items()))
    initial_values = dict(filter(lambda elem: 'initial_value' in elem[0], inference_data.items()))
    assert set(samples.keys()).union(initial_values.keys()) == set(inference_data.keys())
    
    tol=1e-5
    samples_Ws_train = pickle_load("samples_Ws_train.pickle")
    for ws in samples_Ws_train:
        assert vmap(lambda w: valid_grass_point(w, tol=tol))(ws).all()
        
        
    mcmc_barycenters = pickle_load("mcmc_barycenters.pickle")
    
    in_sample_errors_df = pickle_load("in_sample_errors_df.pickle")
    # plt.plot(s_train,in_sample_errors_df['errors'])
    # plt.show()

    samples_alphas_train = pickle_load("samples_alphas_train.pickle")
        
    percentile_levels = [2.5, 97.5]
    conf_level = percentile_levels[-1] - percentile_levels[0]
    percentiles = np.percentile(samples_alphas_train, np.array(percentile_levels), axis=0)
    lower = percentiles[0,:]
    upper = percentiles[1,:]
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111)
    ax.plot(s_test, alphas, c='black', alpha=0.5, label='test data')
    ax.scatter(s_train, alphas_train, label='train data', c='g')
    ax.scatter(s_train, samples_alphas_train.mean(axis=0), label='mean of HMC samples', c='r')
    ax.fill_between(s_train, lower, upper,  color='lightblue', alpha=0.75,label=f'{conf_level}% credible interval')
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$\alpha(s)$")
    ax.legend()
    ax.grid()
    fig.savefig('in-sample-subspace-angle-plot.png',dpi=300,bbox_inches='tight',facecolor="w")
    plt.show()

# %%
in_sample_plots(Config)

# %%
in_sample_errors_df = pickle_load("in_sample_errors_df.pickle")
plt.plot(s_train,in_sample_errors_df['errors'])
plt.show()

# %%
in_sample_errors_df.describe()


# %%
def out_of_sample_pred_plots(cfg):
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    Deltas_means = pickle_load("Deltas_means.pickle")
    Deltas_preds = pickle_load("Deltas_preds.pickle")
    assert np.isnan(Deltas_means).sum() == 0
    assert np.isnan(Deltas_preds).sum() == 0
    
    fig, axs = plt.subplots(2,1,figsize=(12,12),sharey=False)
    ordinals_dict = {1: 'st', 2: 'nd', 3: 'rd'}
    percentile_levels = [2.5, 97.5]
    conf_level = percentile_levels[-1] - percentile_levels[0]
    for i in range(d):
        obs = log_Ws_train[:,i,0]
        means = Deltas_means[:,:,i,0]
        means_avg = np.mean(means, axis=0)
        preds = Deltas_preds[:,:,i,0]
        percentiles = np.percentile(preds, np.array(percentile_levels), axis=0)
        lower = percentiles[0,:]
        upper = percentiles[1,:]
        axs[i].plot(s_test, log_Ws_test[:,i,0], label='test data',c='black', alpha=0.75, linestyle='dashed')
        axs[i].scatter(s_train, log_Ws_train[:,i,0], label='train data', c='g')
        axs[i].plot(s_test, means_avg, label='averaged mean prediction', c='r', alpha=0.75)
        axs[i].fill_between(s_test, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
        axs[i].set_xlabel(r"$s$")
        axs[i].grid()
        axs[i].legend()
        # axs[i].vlines(s_train, 0.99*lower.min(), 1.01*upper.max(), colors='green', linestyles='dashed')
        axs[i].set_title(f'{i+1}{ordinals_dict[i+1]} component of ' + r'$\mathbf{U}(s)$')
    
    axs[0].set_ylim([-0.01,0.01])
    fig.savefig('out-sample-tangent-predictions-plot.png',dpi=300,bbox_inches='tight',facecolor="w")
    plt.show()

    Ws_means = pickle_load("Ws_means.pickle")
    Ws_preds = pickle_load("Ws_preds.pickle")
    assert np.isnan(Ws_means).sum() == 0
    assert np.isnan(Ws_preds).sum() == 0
    
    alphas_means = pickle_load("alpha_means.pickle")
    alphas_preds = pickle_load("alpha_preds.pickle")
    
    percentile_levels = [2.5, 97.5]
    conf_level = percentile_levels[-1] - percentile_levels[0]
    alphas_means_avg = np.mean(alphas_means, axis=0)
    percentiles = np.percentile(alphas_preds, np.array(percentile_levels), axis=0)
    lower = percentiles[0,:]
    upper = percentiles[1,:]
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    ax.plot(s_test, alphas, label='test data',c='black', alpha=0.75, linestyle='dashed')
    ax.scatter(s_train, alphas_train, label='train data', c='g')
    ax.plot(s_test, alphas_means_avg, label='averaged mean prediction', c='r', alpha=0.75)
    ax.fill_between(s_test, lower, upper, color='lightblue', alpha=0.75, label=f'{conf_level}% credible interval')
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$\alpha(s)$")
    ax.legend()
    ax.grid()
    # ax.vlines(s_train, 0, np.pi, colors='green', linestyles='dashed')
    # ax.set_title(f"predictions for subspace angles")
    fig.savefig('out-sample-subspace-angle-plot.png',dpi=300,bbox_inches='tight',facecolor="w")
    plt.show()


# %%
out_of_sample_pred_plots(Config)

# %%
out_sample_errors_df = pickle_load("out_sample_errors_df.pickle")
    
out_sample_mean_errors = out_sample_errors_df["errors_mean"]
out_sample_pred_errors = out_sample_errors_df["errors_pred"]
sd_s_test_means = out_sample_errors_df["sd_mean"]
sd_s_test_preds = out_sample_errors_df["sd_pred"]
upper_mean = out_sample_mean_errors + sd_s_test_means
upper_pred = out_sample_pred_errors + sd_s_test_preds

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
ax.plot(s_test,out_sample_mean_errors, c='k', alpha=0.75, label='error using means')
ax.plot(s_test,out_sample_pred_errors, c='b', alpha=0.75, label='error using preds')
ax.vlines(s_train, 0, 1.2, colors="green", linestyles="dashed")
ax.fill_between(s_test, np.array(out_sample_pred_errors), np.array(upper_pred), color='lightblue', alpha=0.75, label=f'error + 1 std using means')
ax.fill_between(s_test, np.array(out_sample_mean_errors), np.array(upper_mean), color='coral', alpha=0.75, label=f'error + 1 std using preds')
ax.set_xlabel(r"$s$")
ax.legend()
ax.grid()
fig.savefig('out-sample-error-plot.png',dpi=300,bbox_inches='tight',facecolor="w")
plt.show()

# %%
out_sample_errors_df.describe()

# %%
