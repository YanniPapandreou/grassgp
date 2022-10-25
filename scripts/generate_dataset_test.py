# %%
import jax.numpy as np
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

# from grassgp.means import zero_mean
# from grassgp.kernels import rbf
# from grassgp.models import grassmann_process

from itertools import product
from grassgp.utils import subspace_angle, unvec
from grassgp.generate_data import generate_input_data
import matplotlib.pyplot as plt

from grassgp.configs.grass_config import GrassConfFullReortho_b_1
from grassgp.configs.outer_config import OuterGPConf

# from hydra_zen import builds, instantiate, to_yaml
from hydra_zen import make_custom_builds_fn, instantiate, to_yaml, builds, make_config, ZenField
from hydra.core.config_store import ConfigStore

# %%
# def show_var(var, var_name):
#     print(f"{var_name} = {var}")

# %%
# projs_key = random.PRNGKey(658769)
# # outer_key = random.PRNGKey(4357)
# outer_seed = 4357
# X, s = instantiate(InputDataConf)
# n_s = InputDataConf.n_s
# n_x_sqrt = InputDataConf.n_x_sqrt

# gen_projs_from_prior = False
# if gen_projs_from_prior:
#     grass_model = instantiate(GrassConf)
#     prior = Predictive(grass_model, num_samples=1)
#     pred = prior(projs_key, s=s)
#     show_var(pred.keys(), 'pred.keys()')
#     Ps = pred['Ps'][0]
# else:
#     k = 2 * np.pi
#     x = np.cos(k * s).reshape(-1,1)
#     y = np.sin(k * s).reshape(-1,1)
#     Ps = np.hstack((x,y))[:,:,None]

# alphas = []
# for i in range(n_s):
#     v = Ps[i]
#     alpha = subspace_angle(v)
#     alphas.append(alpha)

# alphas = np.array(alphas)

# # plt.plot(s_fine, alphas_fine)
# plt.scatter(s, alphas, c='red', lw=3)
# plt.grid()
# plt.xlabel('t')
# plt.ylabel(r'$\alpha$')
# plt.show()

# show_var(Ps.shape,'Ps.shape')

# X_projs = np.einsum('ij,ljk->lik', X, Ps)
# show_var(X_projs.shape,'X_projs.shape') 
# X_projs_all = np.vstack([X_projs[i,:,:] for i in range(n_s)])
# show_var(X_projs_all.shape,'X_projs_all.shape')

# gen_from_gp_model = instantiate(OuterGPConf)
# Y = gen_from_gp_model(X_projs_all)
# show_var(Y.shape,'Y.shape')
# Ys = unvec(Y, n_x_sqrt**2, n_s)
# show_var(Ys.shape, 'Ys.shape')
# print('Data has been generated')

# %%
InputDataConf = builds(generate_input_data, populate_full_signature=True)

Conf = make_config(
    input_data_conf = InputDataConf,
    inner_model = GrassConfFullReortho_b_1,
    outer_model = OuterGPConf,
    inner_seed = 658769,
    gen_projs_from_prior = True,
    k = 2 * np.pi
)

cs = ConfigStore.instance()
cs.store(name="generate_data", node=Conf)


# %%
def generate_dataset(cfg):
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
    return data


# %%
generate_dataset(Conf)

# %%
