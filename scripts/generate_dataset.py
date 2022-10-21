# %%
import os
# import hydra
from hydra.utils import get_original_cwd
# from hydra.core.config_store import ConfigStore
from hydra_zen import instantiate, builds, make_config

import jax.numpy as np
from jax import random

from numpyro.infer import Predictive

from grassgp.utils import unvec, get_save_path
from grassgp.generate_data import generate_input_data
from grassgp.utils import safe_save_jax_array_dict as safe_save


from configs.grass_config import GrassConf
from configs.outer_config import OuterGPConf

# %%
InputDataConf = builds(generate_input_data, populate_full_signature=True)

Config = make_config(
    input_data_conf = InputDataConf,
    inner_model = GrassConf,
    outer_model = OuterGPConf,
    inner_seed = 658769,
    gen_projs_from_prior = True,
    k = 2 * np.pi
)

# cs = ConfigStore.instance()
# cs.store(name="generate_dataset", node=Config)


# %%
# @hydra.main(version_base="1.1",config_path=None, config_name="generate_dataset")
def generate_dataset(cfg):
    # print(f"Current working directory : {os.getcwd()}")
    # print(f"Orig working directory    : {get_original_cwd()}")
    projs_key = random.PRNGKey(cfg.inner_seed)
    X, s = instantiate(cfg.input_data_conf)
    n_s = cfg.input_data_conf.n_s
    n_x_sqrt = cfg.input_data_conf.n_x_sqrt
    gen_projs_from_prior = cfg.gen_projs_from_prior
    data = {'X': X, 's': s}
    
    if gen_projs_from_prior:
        grass_model = instantiate(cfg.inner_model)
        anchor_point = np.array(grass_model.keywords['grass_config']['anchor_point'])
        prior = Predictive(grass_model, num_samples=1)
        pred = prior(projs_key, s=s)
        Ps = pred['Ps'][0]
        data['anchor_point'] = anchor_point
        data['Ps'] = Ps
    else:
        k = cfg.k
        x = np.cos(k * s).reshape(-1,1)
        y = np.sin(k * s).reshape(-1,1)
        Ps = np.hstack((x,y))[:,:,None]
        data['Ps'] = Ps
        
    X_projs = np.einsum('ij,ljk->lik', X, Ps)
    X_projs_all = np.vstack([X_projs[i,:,:] for i in range(n_s)])
    gen_from_gp_model = instantiate(cfg.outer_model)
    Y = gen_from_gp_model(X_projs_all)
    Ys = unvec(Y, n_x_sqrt**2, n_s)
    print('Data has been generated')
    data['Ys'] = Ys
    head = os.getcwd()
    main_name = "dataset"
    path = get_save_path(head, main_name)
    try:
        safe_save(path, data)
    except FileExistsError:
        print("File exists so not saving.")
    # return data

# %%
# if __name__ == "__main__":
#     generate_dataset()

# %%
