import pytest
import jax.numpy as np

from grassgp.kernels import rbf_covariance, rbf

def test_rbf_cov():
    var = 1.5
    length = 0.5
    noise = 0.1
    jitter = 1.0e-6
    N = 10
    x = np.linspace(0,1,N).reshape(-1,1)
    K = rbf_covariance(x, x, var, length, noise, jitter=jitter)
    k = lambda x,y: var * np.exp(-(0.5 / (length ** 2)) * ((x - y) ** 2)) + (noise + jitter) * (x == y)
    for i in range(N):
        for j in range(N):
            assert K[i,j] == k(x[i], x[j]), f"fails for index ({i},{j})"


def test_rbf_cov_no_noise():
    var = 1.5
    length = 0.5
    noise = 0.1
    jitter = 1.0e-6
    N = 10
    x = np.linspace(0,1,N).reshape(-1,1)
    K = rbf_covariance(x, x, var, length, noise, jitter=jitter, include_noise=False)
    k = lambda x,y: var * np.exp(-(0.5 / (length ** 2)) * ((x - y) ** 2))
    for i in range(N):
        for j in range(N):
            assert K[i,j] == k(x[i], x[j]), f"fails for index ({i},{j})"


def test_rbf():
    var = 1.5
    length = 0.5
    noise = 0.1
    jitter = 1.0e-6
    N = 10
    x = np.linspace(0,1,N).reshape(-1,1)
    params = {'var': var, 'length': length, 'noise': noise}
    K = rbf(x, x, params, jitter=jitter)
    k = lambda x,y: var * np.exp(-(0.5 / (length ** 2)) * ((x - y) ** 2)) + (noise + jitter) * (x == y)
    for i in range(N):
        for j in range(N):
            assert K[i,j] == k(x[i], x[j]), f"fails for index ({i},{j})"


def test_rbf_no_noise():
    var = 1.5
    length = 0.5
    noise = 0.1
    jitter = 1.0e-6
    N = 10
    x = np.linspace(0,1,N).reshape(-1,1)
    params = {'var': var, 'length': length, 'noise': noise}
    K = rbf(x, x, params, jitter=jitter, include_noise=False)
    k = lambda x,y: var * np.exp(-(0.5 / (length ** 2)) * ((x - y) ** 2))
    for i in range(N):
        for j in range(N):
            assert K[i,j] == k(x[i], x[j]), f"fails for index ({i},{j})"

