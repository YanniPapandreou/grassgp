# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: grassgp
#     language: python
#     name: grassgp
# ---

# +
# # %load_ext autoreload
# # %autoreload 2

# +
from typing import Callable
import jax.numpy as np
from jax import random, vmap

import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive

# from grassgp.means import zero_mean
# from grassgp.kernels import rbf
# from grassgp.models import grassmann_process

from itertools import product
from grassgp.utils import subspace_angle, unvec
import matplotlib.pyplot as plt

from configs.grass_config import GrassConf
from configs.outer_config import OuterGPConf

# from hydra_zen import builds, instantiate, to_yaml
from hydra_zen import make_custom_builds_fn, instantiate, to_yaml, builds, make_config


# -
def show_var(var, var_name):
    print(f"{var_name} = {var}")


# +
projs_key = random.PRNGKey(658769)
# outer_key = random.PRNGKey(4357)
outer_seed = 4357
D = 2
active_dimension = 1
n_s = 10
s_lims = [0.0, 1.0]
s = np.linspace(s_lims[0],s_lims[1],n_s)
x_lims = [-2,2]
N_sqrt = 5
x_range = np.linspace(x_lims[0], x_lims[1], N_sqrt)
X = np.array([v for v in product(x_range, repeat=D)])

gen_projs_from_prior = False
if gen_projs_from_prior:
    grass_model = instantiate(GrassConf)
    prior = Predictive(grass_model, num_samples=1)
    pred = prior(projs_key, s=s)
    show_var(pred.keys(), 'pred.keys()')
    Ps = pred['Ps'][0]
else:
    k = 2 * np.pi
    x = np.cos(k * s).reshape(-1,1)
    y = np.sin(k * s).reshape(-1,1)
    Ps = np.hstack((x,y))[:,:,None]

# N = n_s
# s = np.linspace(0.2, 1.8, N) * np.pi
# x = np.cos(s).reshape(-1,1)
# y = np.sin(s).reshape(-1,1)
# Ps = np.hstack((x,y))[:,:,None]

alphas = []
for i in range(n_s):
    v = Ps[i]
    alpha = subspace_angle(v)
    alphas.append(alpha)

alphas = np.array(alphas)

# plt.plot(s_fine, alphas_fine)
plt.scatter(s, alphas, c='red', lw=3)
plt.grid()
plt.xlabel('t')
plt.ylabel(r'$\alpha$')
plt.show()

show_var(Ps.shape,'Ps.shape')

X_projs = np.einsum('ij,ljk->lik', X, Ps)
show_var(X_projs.shape,'X_projs.shape') 
X_projs_all = np.vstack([X_projs[i,:,:] for i in range(n_s)])
show_var(X_projs_all.shape,'X_projs_all.shape')

gen_from_gp_model = instantiate(OuterGPConf)
Y = gen_from_gp_model(X_projs_all)
show_var(Y.shape,'Y.shape')
Ys = unvec(Y, N_sqrt**2, n_s)
show_var(Ys.shape, 'Ys.shape')
print('Data has been generated')
# -
from grassgp.plot_utils import plot_projected_data


plot_projected_data(X_projs,s,Ys)
