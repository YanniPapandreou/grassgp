from json import load
import os
import datetime
import jax.numpy as np
from jax import jit
import jax.numpy.linalg as lin
from functools import partial
from pathlib import Path

from hydra_zen import load_from_yaml

@jit
def vec(X):
    """returns the column-wise vectorization of matrix X

    Parameters
    ----------
    X : array
        M x N matrix

    Returns
    -------
    array
        MN-vector vec(X) obtained by stacking the columns of X on top of
        each other.
    """
    return np.ravel(X, order='F')


@partial(jit, static_argnums=(1, 2))
def unvec(x, m: int, n: int):
    """returns the m x n matrix X such that vec(X) = x

    Parameters
    ----------
    x : array
        mn-vector
    m : int
        number of rows
    n : int
        number of columns

    Returns
    -------
    array
        m x n matrix
    """
    return x.reshape((m, n), order='F')


@jit
def multiprod(A, B):
    """A and B are assumed to be arrays containing M matrices, that is, A and
    B have dimensions A: (M, N, P), B:(M, P, Q). multiprod multiplies each
    matrix in A with the corresponding matrix in B, using matrix
    multiplication. So multiprod(A, B) has dimensions (M, N, Q).

    Parameters
    ----------
    A : array
        M x N x P
    B : array
        M x P X Q

    Returns
    -------
    array
        M x N x Q
    """
    # First check if we have been given just one matrix
    if len(np.shape(A)) == 2:
        # ! not sure if this is the same as what docstring wants,
        # ! unless B is also just a matrix - if so should check this?
        assert len(B.shape) == 2
        return np.dot(A, B)

    # Old (slower) implementation:
    # a = A.reshape(np.hstack([np.shape(A), [1]]))
    # b = B.reshape(np.hstack([[np.shape(B)[0]], [1], np.shape(B)[1:]]))
    # return np.sum(a * b, axis=2)

    # Approx 5x faster, only supported by numpy version >= 1.6:
    return np.einsum('ijk,ikl->ijl', A, B)


@jit
def multitransp(A):
    """A is assumed to be an array containing M matrices, each of which has
    dimension N x P. That is, A is an M x N x P array. Multitransp then
    returns an array containing the M matrix transposes of the matrices
    in A, each of which will be P x N.

    Parameters
    ----------
    A : array
        M x N x P

    Returns
    -------
    array
        M x P x N
    """
    # First check if we have been given just one matrix
    if A.ndim == 2:
        return A.T
    return np.transpose(A, (0, 2, 1))


@jit
def kron_solve(K1, K2, y):
    """Efficiently solves the linear system (K1 ⊗ K2)x = y by solving two
    smaller systems. This works as follows: solution is x = inv((K1 ⊗ K2))y
    and inv(K1 ⊗ K2) = inv(K1) ⊗ inv(K2). So
    x = (inv(K1) ⊗ inv(K2))y = vec(inv(K2) Y inv(K1).T) where vec(Y) = y.
    We can thus solve systems specified by K1 and K2 separately.

    Parameters
    ----------
    K1 : array
        n1 x n1 invertible matrix
    K2 : array
        n2 x n2 invertible matrix
    y : array
        (n1 * n2)-vector

    Returns
    -------
    array
        (n1 * n2)-vector
    """
    n1 = K1.shape[0]
    n2 = K2.shape[0]
    Y = unvec(y, n2, n1)
    sol_mat = lin.solve(K2, lin.solve(K1, Y.T).T)
    return vec(sol_mat)


@jit
def kron_chol(K1, K2):
    """Efficiently computes the Cholesky decomposition of K1 ⊗ K2 via
    chol(K1 ⊗ K2) = chol(K1) ⊗ chol(K2).

    Parameters
    ----------
    K1 : array
    K2 : array

    Returns
    -------
    array
    """
    L1 = lin.cholesky(K1)
    L2 = lin.cholesky(K2)
    return np.kron(L1, L2)


def safe_save_jax_array_dict(path: str, jax_dict: dict):
    assert all([isinstance(value, type(np.zeros((1,))))
                for value in jax_dict.values()])

    if path[-4:] != '.npz':
        path += '.npz'

    if os.path.exists(path):
        raise FileExistsError("File exists, saving would overwrite; please use another path.")
    else:
        np.savez(path, **jax_dict)


def load_and_convert_to_samples_dict(path: str):
    """function to load a .npz file and convert back to a samples dictionary

    Parameters
    ----------
    path : str
        path to .npz file

    Returns
    -------
    dict
        samples dictionary
    """
    samples_data = np.load(path)

    samples = {}

    for name in samples_data.files:
        samples[name] = samples_data[name]

    return samples


# def get_save_path(head: str, main_name: str):
#     date = datetime.datetime.now()
#     suffix = f"{date.strftime('%Y-%m-%d--%H:%M')}"
#     suffix = f"{main_name}_{suffix}"
#     path = os.path.join(head, suffix)
#     return path

def get_save_path(head: str, main_name: str):
    # date = datetime.datetime.now()
    # suffix = f"{date.strftime('%Y-%m-%d--%H:%M')}"
    # suffix = f"{main_name}_{suffix}"
    path = os.path.join(head, main_name)
    return path


def subspace_angle(v):
    x, y = v
    alpha = np.arctan2(y[0], x[0])
    if alpha < 0:
        return alpha + np.pi
    else:
        return alpha


def get_config_and_data(path: Path):
    config = load_from_yaml(path / ".hydra" / "config.yaml")
    overrides = load_from_yaml(path / ".hydra" / "overrides.yaml")
    data = load_and_convert_to_samples_dict(str(path / "dataset.npz"))
    return {'config': config, 'overrides': overrides, 'data': data}
