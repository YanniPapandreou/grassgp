import pytest
import jax.numpy as np
import jax.numpy.linalg as lin
from jax import random, vmap
from grassgp.grassmann import (
    valid_grass_point,
    valid_grass_tangent,
    grass_exp,
    grass_dist,
    rand_grass_point,
    rand_grass_tangent,
    grass_log,
    convert_to_projs
)

grass_params = [(3,1,3426), (3,2,1235), (5,1,346), (5,4,135)]

@pytest.mark.parametrize("dims_seeds", grass_params)
class TestGrassPtsTangents:
    def test_rand_grass_pts(self, dims_seeds):
        tol = 1e-6
        D, n, seed = dims_seeds
        key = random.PRNGKey(seed)
        X = rand_grass_point(key, D, n)
        assert valid_grass_point(X, tol=tol)

    def test_rand_tangents(self, dims_seeds):
        tol = 1e-6
        D, n, seed = dims_seeds
        key_pt, key_tan = random.split(random.PRNGKey(seed), 2)
        X = rand_grass_point(key_pt, D, n)
        U = rand_grass_tangent(key_tan, X)
        assert valid_grass_tangent(X, U, tol=tol)

    def test_grass_dist_gives_zero_for_same_point(self, dims_seeds):
        tol = 1e-3
        D, n, seed = dims_seeds
        key = random.PRNGKey(seed)
        X = rand_grass_point(key, D, n)
        dist = grass_dist(X, X)
        assert dist <= tol
    
    def test_grass_dist_gives_zero_for_point_and_its_negative(self, dims_seeds):
        tol = 1e-3
        D, n, seed = dims_seeds
        key = random.PRNGKey(seed)
        X = rand_grass_point(key, D, n)
        dist = grass_dist(X, -X)
        assert dist <= tol


@pytest.mark.parametrize("reortho", [True, False])
class TestGrassExpLogConvertToProjs:
    def test_grass_exp_gives_valid_pt(self, reortho):
        tol = 1e-6
        key_pt, key_tan = random.split(random.PRNGKey(2362475), 2)
        D = 5
        n = 2
        X = rand_grass_point(key_pt, D, n)
        U = rand_grass_tangent(key_tan, X)
        Y = grass_exp(X, U, reorthonormalize = reortho)
        # Y = grass_exp(U, X, reorthonormalize = reortho)
        assert valid_grass_point(Y)

    def test_grass_exp_invertible(self, reortho):
        tol = 1e-6
        key_pt, key_tan = random.split(random.PRNGKey(2362475), 2)
        D = 5
        n = 2
        X = rand_grass_point(key_pt, D, n)
        U = rand_grass_tangent(key_tan, X)
        Y = grass_exp(X, U, reorthonormalize=reortho)
        log_Y = grass_log(X, Y)
        assert valid_grass_tangent(X, log_Y)
        assert lin.norm(U - log_Y) / lin.norm(U) < tol
    

    def test_convert_to_projs(self, reortho):
        n_s = 10
        key = random.PRNGKey(232637)
        D = 3
        n = 1
        anchor_point = rand_grass_point(key, D, n)
        Deltas = np.zeros((n_s, D, n))
        keys = random.split(key, n_s)
        for i in range(n_s):
            sk = keys[i,:]
            tangent = rand_grass_tangent(sk, anchor_point)
            assert valid_grass_tangent(anchor_point, tangent), f"tangent generation failed"
            Deltas = Deltas.at[i,:,:].set(tangent)
        
        Ps = convert_to_projs(Deltas, anchor_point, reorthonormalize=reortho)

        assert all(vmap(valid_grass_point)(Ps))
        # for i in range(n_s):
        #     proj = Ps[:,:,i]
        #     assert valid_grass_point(proj), f"valid_grass_pt failed at i={i}"
    
    def test_convert_to_projs_works_for_1_time(self, reortho):
        tol = 1e-5
        n_s = 1
        sk1, sk2 = random.split(random.PRNGKey(57), 2)
        D = 2
        n = 1
        anchor_point = rand_grass_point(sk1, D, n)
        Deltas = np.zeros((n_s, D, n))
        tangent = rand_grass_tangent(sk2, anchor_point)
        assert valid_grass_tangent(anchor_point, tangent, tol=tol), f"tangent generation failed"
        Deltas = Deltas.at[0,:,:].set(tangent)
        Ps = convert_to_projs(Deltas, anchor_point, reorthonormalize=reortho)

        assert all(vmap(lambda proj: valid_grass_point(proj,tol=tol))(Ps))
    
