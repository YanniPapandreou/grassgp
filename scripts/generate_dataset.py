import hydra
from hydra.core.config_store import ConfigStore

from hydra_zen import make_custom_builds_fn, instantiate, to_yaml, builds, make_config

from configs.grass_config import GrassConf

# import jax.numpy as np
# from jax import random, vmap

# import numpyro
# import numpyro.distributions as dist
# from numpyro.infer import Predictive

# from grassgp.kernels import rbf
# from grassgp.models import grassmann_process


# set up config for Grassmann_model
# GrassConf = builds(grassmann_process, Omega=None, proj_locs=None, var=None,length=None, noise=None, require_noise=False, jitter=1e-06, proj_jitter=1e-4, L_jitter=1e-8, reorthonormalize=True, b=5.0, zen_partial=True)
# full_grass_model_reortho = GrassConf()
# full_grass_model_no_reortho = GrassConf(reorthonormalize=False)

Config = make_config(model=GrassConf)

cs = ConfigStore.instance()
cs.store(name="generate_dataset", node=Config)

# cs.store(group="model/grassmann", name="full_reortho", node=full_grass_model_reortho)
# cs.store(group="model/grassmann", name="full_reortho", node=full_grass_model_reortho)
# cs.store(group="model/grassmann", name="full_no_reortho", node=full_grass_model_no_reortho)

# Config = make_config("model", hydra_defaults=["__self__", {"grassmann": "full_reortho"}])

# cs.store(name="generate_dataset", node=Config)

@hydra.main(config_path=None, config_name="generate_dataset")
def generate_data(cfg):
    print(to_yaml(cfg))

if __name__ == "__main__":
    generate_data()
