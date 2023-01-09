from hydra_zen import builds
from grassgp.models import grassmann_process

grass_model_config_full_reortho_b_1 = {
    'anchor_point': [[1.0], [0.0]],
    'Omega' : None, 
    'proj_locs' : None,
    'var' : None,
    'length' : None,
    'noise' : None,
    'require_noise' : False,
    'jitter' : 1e-06,
    'proj_jitter' : 1e-4,
    'L_jitter' : 1e-8,
    'reorthonormalize' : True,
    'b' : 1.0
}

grass_model_config_full_reortho_b_5 = {
    'anchor_point': [[1.0], [0.0]],
    'Omega' : None, 
    'proj_locs' : None,
    'var' : None,
    'length' : None,
    'noise' : None,
    'require_noise' : False,
    'jitter' : 1e-06,
    'proj_jitter' : 1e-4,
    'L_jitter' : 1e-8,
    'reorthonormalize' : True,
    'b' : 5.0
}

GrassConfFullReortho_b_1 = builds(grassmann_process, grass_config = grass_model_config_full_reortho_b_1, zen_partial=True)
GrassConfFullReortho_b_5 = builds(grassmann_process, grass_config = grass_model_config_full_reortho_b_5, zen_partial=True)