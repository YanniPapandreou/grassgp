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

from grassgp.utils import unvec, vec, kron_solve, kron_chol
from grassgp.grassmann import convert_to_projs, valid_grass_tangent


@chex.dataclass
class MatGP:
    d_in: int
    d_out: Tuple[int, int]
    mu: Callable = field(repr=False)
    k: Callable = field(repr=False)
    Omega_diag_chol: chex.ArrayDevice = field(repr=False)
    cov_jitter: float = field(default=1e-8, repr=False)

    def __post_init__(self):
        d, n = self.d_out
        d_n = d * n
        assert_shape(self.Omega_diag_chol, (d_n,),
                    custom_message=f"Omega_diag_chol has shape {self.Omega_diag_chol.shape}; expected shape {(d_n,)}")

    def model(self, s: chex.ArrayDevice) -> chex.ArrayDevice:
        d, n = self.d_out
        d_n = d * n
        assert_rank(s, self.d_in)
        N = s.shape[0]

        # compute mean matrix M = [mu(s[1]), mu(s[2]), ..., mu(s[N])]
        M = np.hstack(vmap(self.mu)(s))
        assert_shape(M, (d, n*N))
        # ! TODO: check this out
        vec_M = vec(M)

        # compute kernel matrix
        K = self.k(s, s)
        assert_shape(K, (N, N))
            
        K_chol = lin.cholesky(K + self.cov_jitter * np.eye(N))
        # Omega_diag_chol = np.sqrt(self.Omega_diag)

        # sample vec_Vs
        # Z = numpyro.sample("Z", dist.MultivariateNormal(covariance_matrix=np.eye(N*d_n)))
        Z = numpyro.sample("Z", dist.Normal().expand([N*d_n]))
        unvec_Z = unvec(Z, d_n, N)
        # vec_Vs = numpyro.deterministic("vec_Vs", vec(M + np.einsum('i,ij->ij', self.Omega_diag_chol, unvec_Z @ K_chol.T)))
        vec_Vs = numpyro.deterministic("vec_Vs", vec_M + vec(np.einsum('i,ij->ij', self.Omega_diag_chol, unvec_Z @ K_chol.T)))

        # form Vs
        Vs = numpyro.deterministic("Vs", vmap(lambda params: unvec(params, d, n))(np.array(vec_Vs.split(N))))
        return Vs

    def sample(self, seed: int, s: chex.ArrayDevice) -> chex.ArrayDevice:
        model = self.model
        seeded_model = handlers.seed(model, rng_seed=seed)
        return seeded_model(s)

    def predict(self, key: chex.ArrayDevice, s_test: chex.ArrayDevice, s_train: chex.ArrayDevice, Vs_train: chex.ArrayDevice, jitter: float = 1e-8) -> Tuple[chex.ArrayDevice, chex.ArrayDevice]:
        # TODO optimise this further to take advantage of diag structure of Omega
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
        Omega_diag = self.Omega_diag_chol ** 2
        Omega = np.diag(Omega_diag)
        K_test_train_Omega = np.kron(K_test_train, Omega)
        K_train_test_Omega = np.kron(K_train_test, Omega)
        K_test_test_Omega = np.kron(K_test_test, Omega)
        mean_sols = kron_solve(K_train_train, Omega, vec(np.hstack(Vs_train)) - vec(M_train))
        vec_post_mean = vec(M_test) + K_test_train_Omega @ mean_sols
        assert_shape(vec_post_mean, (d*n*N_test,),
                     custom_message=f"vec_post_mean should have shape {(d*n*N_test,)}; obtained {vec_post_mean.shape}")

        cov_sols = vmap(lambda v: kron_solve(K_train_train, Omega, v), in_axes=1, out_axes=1)(K_train_test_Omega)
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
    Omega_diag_chol: chex.ArrayDevice = field(repr=False)
    U: chex.ArrayDevice
    cov_jitter: float = field(default=1e-4, repr=False)

    def __post_init__(self):
        d, n = self.d_out
        d_n = d * n
        assert_shape(self.Omega_diag_chol, (d_n,),
                    custom_message=f"Omega_diag_chol has shape {self.Omega_diag_chol.shape}; expected shape {(d_n,)}")
        assert_shape(self.U, (d, n),
                    custom_message=f"U has shape {self.U.shape}; expected shape {(d, n)}")
        tol = 1e-06
        # assert valid_grass_point(self.U), f"U is not a valid point on Grassmann manifold G({d},{n}) at tolerance level {tol = }"

    @property
    def V(self) -> MatGP:
        mat_gp = MatGP(d_in=self.d_in, d_out=self.d_out, mu=self.mu, k=self.k, Omega_diag_chol=self.Omega_diag_chol, cov_jitter=self.cov_jitter)
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
