from hydra_zen import builds
from grassgp.models import grassmann_process

grass_model_config_full_reortho_b_1 = {
    'anchor_point': [[1.0], [0.0]],
    'Omega' : None, 
    'proj_locs' : None,
    'var' : 1.0,
    'length' : None,
    'noise' : 0.1,
    'require_noise' : False,
    'jitter' : 1e-06,
    'proj_jitter' : 1e-4,
    'L_jitter' : 1e-8,
    'reorthonormalize' : True,
    'b' : 1.0
}

GrassConf = builds(grassmann_process, model_config = grass_model_config_full_reortho_b_1, zen_partial=True)