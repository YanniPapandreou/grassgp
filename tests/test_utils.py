import pytest
import jax.numpy as np
import jax.numpy.linalg as lin
from jax import random, vmap
from grassgp.utils import vec, unvec, multiprod, kron_solve, kron_chol

dims = [(3,3), (1,4), (4,1), (5,2), (5,5)]
Xs = [np.arange(dim[0] * dim[1]).reshape(dim) for dim in dims]

@pytest.mark.parametrize("Xs", Xs)
class TestVecUnvec:
    def test_vec_flattens(self, Xs):
        assert len(vec(Xs).shape) == 1
    
    def test_vec_flattens_to_correct_shape(self, Xs):
        assert vec(Xs).shape == (Xs.shape[0] * Xs.shape[1],)
    
    def test_unvec_inverse_of_vec(self, Xs):
        assert (unvec(vec(Xs), Xs.shape[0], Xs.shape[1]) == Xs).all()

def test_multiprod():
    tol = 1e-7
    M = 4
    N = 3
    P = 5
    Q = 2
    A = np.arange(M * N * P).reshape((M,N,P))
    B = np.arange(M * P * Q).reshape((M,P,Q))
    AB = multiprod(A, B)
    for i in range(M):
        assert lin.norm(A[i,:,:] @ B[i,:,:] - AB[i,:,:]) < tol 

def test_multiprod_single_mats():
    N = 5
    P = 3
    Q = 7
    A = np.arange(N*P).reshape((N,P))
    B = np.arange(P*Q).reshape((P,Q))
    assert (multiprod(A,B) == A@B).all()

def test_kron_solve():
    tol = 1e-4
    n1 = 100
    n2 = 20
    K1 = random.normal(random.PRNGKey(2345), (n1,n1)) + 1e-6 * np.eye(n1)
    K2 = random.normal(random.PRNGKey(25), (n2,n2)) + 1e-6 * np.eye(n2)
    K = np.kron(K1,K2)
    y = random.normal(random.PRNGKey(2314),(n1*n2,))

    sol_kron = kron_solve(K1,K2,y)

    assert lin.norm(K @ sol_kron - y) / lin.norm(y) < tol


def test_kron_chol():
    tol = 1e-5
    n1 = 100
    n2 = 20
    K1 = np.eye(n1) + 1e-6 * random.normal(random.PRNGKey(2345), (n1,n1))
    K2 = np.eye(n2) + 1e-6 * random.normal(random.PRNGKey(25), (n2,n2)) + 1e-6
    K1 = 0.5*(K1 + K1.T)
    K2 = 0.5*(K2 + K2.T)
    K = np.kron(K1,K2)
    chol_kron = kron_chol(K1, K2)

    assert lin.norm(chol_kron @ chol_kron.T - K) / lin.norm(K) < tol
