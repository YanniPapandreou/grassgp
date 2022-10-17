import pytest
import jax.numpy as np
from jax import random, vmap
from grassgp.grassmann import valid_grass_point
from grassgp.generate_data import gen_proj_from_grass_process

grass_params = [(2,1,1), (2,1,3), (3,1,1), (3,1,2), (3,2,1), (3,2,2), (3,2,3), (5,3,1), (5,3,2)]

@pytest.mark.parametrize("grass_params", grass_params)
class TestGrassProcess:
    def test_grass_process_gives_valid_points_reortho(self, grass_params):
        key = random.PRNGKey(0)
        tol = 1e-6
        D, n, n_s = grass_params
        s = np.linspace(0, 1, n_s)
        anchor_point = np.eye(D, n)
        grass_inputs = {'s': s, 'anchor_point': anchor_point, 'proj_jitter': 1e-4, 'require_noise': False, 'reorthonormalize': True}
        Ps = gen_proj_from_grass_process(key, **grass_inputs)
        assert Ps.shape == (n_s, D, n), f"Shape invalid, inputs: {grass_params}, reortho: {grass_inputs['reorthonormalize']}"
        assert all(vmap(lambda proj: valid_grass_point(proj, tol=tol))(Ps)), f"not valid pt, inputs: {grass_params}, reortho: {grass_inputs['reorthonormalize']}"

    def test_grass_process_gives_valid_points_no_reortho(self, grass_params):
        key = random.PRNGKey(0)
        tol = 1e-2
        D, n, n_s = grass_params
        s = np.linspace(0, 1, n_s)
        anchor_point = np.eye(D, n)
        grass_inputs = {'s': s, 'anchor_point': anchor_point, 'proj_jitter': 1e-4, 'require_noise': False, 'reorthonormalize': False}
        Ps = gen_proj_from_grass_process(key, **grass_inputs)
        assert Ps.shape == (n_s, D, n), f"Shape invalid, inputs: {grass_params}, reortho: {grass_inputs['reorthonormalize']}"
        assert all(vmap(lambda proj: valid_grass_point(proj, tol=tol))(Ps)), f"not valid pt, inputs: {grass_params}, reortho: {grass_inputs['reorthonormalize']}"
        
