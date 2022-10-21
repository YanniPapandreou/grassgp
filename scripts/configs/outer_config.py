from hydra_zen import builds

from grassgp.means import zero_mean
from grassgp.kernels import rbf
from grassgp.generate_data import gen_from_univariate_gp

outer_gp_config = {
    'seed': 4357,
    'm': zero_mean,
    'k': rbf,
    'params': {'var': 1.0, 'length': 0.5, 'noise': 0.1},
    'jitter': 1e-08,
    'include_noise': True
}

OuterGPConf = builds(gen_from_univariate_gp, gp_config = outer_gp_config, zen_partial=True)