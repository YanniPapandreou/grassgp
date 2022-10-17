import jax.numpy as np
from jax import jit, vmap, random
import jax.numpy.linalg as lin
from functools import partial
from grassgp.utils import multiprod, multitransp
from pymanopt.manifolds.grassmann import Grassmann
from pymanopt.optimizers.nelder_mead import compute_centroid

@partial(jit, static_argnums=1)
def valid_grass_point(X, tol=1.0e-6):
    """Checks if X is a valid point in the Grassmann manifold

    Parameters
    ----------
    X : array
        D x n array
    tol : float , optional
        tolerance, by default 1.0e-6

    Returns
    -------
    bool
        returns True if X is in G(D, n), otherwise returns False
    """
    D, n = X.shape
    return lin.norm(X.T @ X - np.eye(n)) / np.sqrt(n) < tol

@partial(jit, static_argnums=2)
def valid_grass_tangent(X, Delta, tol=1.0e-6):
    """Checks if Delta is a valid tangent vector at X in the Grasmmann manifold

    Parameters
    ----------
    X : array
        D x n base point in G(D, n)
    Delta : array
        D x n array
    tol : float, optional
        tolerance, by default 1.0e-6

    Returns
    -------
    bool
        returns True if Delta is in T_{X}G(D, n)
    """
    D, n = X.shape
    return lin.norm(X.T @ Delta) / np.sqrt(n) < tol


@partial(jit, static_argnums=2)
def grass_exp(X, Delta, reorthonormalize: bool = True):
    """function to compute the Grassmann Exponential

    Parameters
    ----------
    X : array
        Base point in the Grassmann Manifold
    Delta : array
        Tangent vector to the Grassmann Manifold at X
    reorthonormalize : bool
        Boolean controlling whether we should re-orthonormalize Y

    Returns
    -------
    array
        Exp_X(Delta)
    """
    u, s, vt = lin.svd(Delta, full_matrices=False)
    cos_s = np.expand_dims(np.cos(s), -2)  # ! purpose of this?
    sin_s = np.expand_dims(np.sin(s), -2)

    Y = (multiprod(multiprod(X, multitransp(vt) * cos_s), vt) +
         multiprod(u * sin_s, vt))

    # From numerical experiments, it seems necessary to
    # re-orthonormalize. This is not ideal, but appears to work
    # ! ???
    if reorthonormalize:
        Y, unused = np.linalg.qr(Y)

    return Y

@jit
def grass_dist(X, Y):
    """Computes the Riemannian distance between points X and Y in the Grassmann Manifold

    Parameters
    ----------
    X : array
        point in Grassmann manifold
    Y : array
        point in Grassmann manifold

    Returns
    -------
    float
        dist(X,Y)
    """
    u, s, v = lin.svd(multiprod(multitransp(X), Y))
    return lin.norm(np.arccos(np.where(s > 1, 1, s)))

@partial(jit, static_argnums=(1,2))
def rand_grass_point(key, D, n):
    """generates a random point on the Grassmann manifold G(D,n)

    Parameters
    ----------
    key : array
        random.PRNGkey
    D : int
        dimension of ambient space
    n : int
        dimension of subspace

    Returns
    -------
    array
        random point on G(D,n)
    """
    X = random.normal(key, (D, n))
    q, unused = lin.qr(X)
    return q

@jit
def grass_proj(X, U):
    """Projects U into the Grassmann tangent space at X

    Parameters
    ----------
    X : array
        point in Grassmann manifold
    U : array
        matrix

    Returns
    -------
    array
        proj_{X}(U)
    """
    assert X.shape == U.shape
    d = X.shape[0]
    projector = (np.eye(d) - X @ X.T)
    return projector @ U

@jit
def rand_grass_tangent(key, X):
    """generates a random tangent to X in the Grassmann manifold

    Parameters
    ----------
    key : array
        random.PRNGkey
    X : array
        point in the Grassmann manifold

    Returns
    -------
    array
        random tangent vector at X
    """ 
    d = X.shape[0]
    U = random.normal(key, X.shape)
    U = grass_proj(X, U)
    U = U / lin.norm(U)
    return U

@jit
def grass_log(X, Y):
    """function to compute the Grassmann logarithm

    Parameters
    ----------
    X : array
        point on Grassmann
    Y : array
        point on Grassmann

    Returns
    -------
    array
        tangent vector delta to X on Grassmann such that exp(X, delta) = Y
    """
    ytx = multiprod(multitransp(Y), X)
    psi, s, rt =lin.svd(ytx, full_matrices=False)
    Y_star = multiprod(Y, multiprod(psi, rt))
    L = Y_star - multiprod(X,multiprod(multitransp(X),Y_star))
    q, sigma, vt = lin.svd(L, full_matrices=False)
    arcsin_sigma = np.diag(np.arcsin(sigma))
    delta = multiprod(q, multiprod(arcsin_sigma, vt))
    return delta


@partial(jit, static_argnums=2)
def convert_to_projs(Deltas, anchor_point, reorthonormalize: bool = True):
    """converts tangent vectors to projections in Grassmann manifold

    Parameters
    ----------
    Deltas : array
        array containing tangent vectors at anchor_point
    anchor_point : array
        base point where tangents in Deltas are located
    reorthonormalize : bool, optional
        boolean controlling whether to reorthonormalize in the grass_exp, by default True

    Returns
    -------
    array
        array containing projections
    """

    # apply grass_exp to each Delta and return
    # ! Check which way args should go
    # return vmap(lambda delta: grass_exp(delta, anchor_point), out_axes=2)(Deltas)
    return vmap(lambda delta: grass_exp(anchor_point, delta, reorthonormalize=reorthonormalize))(Deltas)


def compute_karcher_mean(Ps_samples):
    """function to compute the karcher mean/barycenter of sampled projections

    Parameters
    ----------
    Ps_samples : array
        N x n_s x D x n array of sampled D x n projections for n_s times

    Returns
    -------
    array
        n_s x D x n array containing the n_s barycenters
    """
    N, n_s, D, n = Ps_samples.shape
    G = Grassmann(D, n)
    
    Ps_mean = np.zeros((n_s, D, n))
    for i in range(n_s):
        Ps = Ps_samples[:, i, :, :]
        points = []
        for j in range(N):
            proj = Ps[j, :, :]
            assert valid_grass_point(proj)
            points.append(proj)
        
        points_mean = compute_centroid(G, points)
        Ps_mean = Ps_mean.at[i,:,:].set(points_mean)

    return Ps_mean
