# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import vmap, random, grad
import jax.numpy.linalg as lin
from functools import partial
from tqdm import tqdm

import chex
from typing import Tuple

from grassgp.grassmann import valid_grass_point, grass_dist, sample_karcher_mean, grass_exp, grass_log

import pickle
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,6)

# %% [markdown]
# $\newcommand{\R}{\mathbb{R}}$
# We are given a function
#
# $$
# f:\R^{d+1}\rightarrow\R
# $$
#
# We want to investigate how the computation of an active subspace varies as a function of a fixed input parameter. I.e. we want to choose $i\in\{1,...,d+1\}$ and consider the function $f^{(i)}:\R^{d}\rightarrow\R$ formed by fixing $x_{i}$ to some fixed value, i.e.:
#
# $$
# f^{(i)}(\mathbf{x}_{-i};x_{i}) := f(\mathbf{x})
# $$
#
# where $\mathbf{x}_{-i}\in\R^{d}$ denotes the vector $\mathbf{x}_{-i}:=(x_{1},...,x_{i-1},x_{i+1},...,x_{d+1})^{T}$
#
# Let $\mathbf{w}:=\mathbf{x}_{-i}$ and $s:=x_{i}$. For fixed $s\in\R$ we will compute a Monte Carlo approximation, $\hat{C}(S)$, of $C(s)$ defined below:
#
# $$
# C(s):=\int_{\R^{d}}\nabla f^{(i)}(\mathbf{w};s)\nabla f^{(i)}(\mathbf{w};s)^{T}\rho(\mathbf{w})\mathrm{d}\mathbf{w}
# $$
#
# $$
# \hat{C}(s):=\frac{1}{N}\sum_{j=1}^{N}\nabla f^{(i)}(\mathbf{w}_{j};s)\nabla f^{(i)}(\mathbf{w}_{j};s)^{T}
# $$
#
# where $\{\mathbf{w}_{j}\}_{j=1}^{N}$ are random i.i.d. samples from the density $\rho$.
#
# We will then eigen-decompose the $d\times d$ matrix $\hat{C}(s)$ and take the leading eigenvector (with largest eigenvalue). This yields an active subspace $W(s)$ represented as a skinny orthogonal matrix $W(s)\in\R^{d\times 1}$.
#
# We are interested in studying how $W(s)$ varies as a function of $s$.

# %% [markdown]
# # Generate dataset

# %%
wing_func = lambda Sw, Wfw, A, L, q, l, tc, Nz, Wdg, Wp: .036*Sw**.758*Wfw**.0035*A**.6*np.cos(L * np.pi/180)**-.9*q**.006*l**.04*100**-.3*tc**-.3*Nz**.49*Wdg**.49 + Sw*Wp

# %%
M = 1000 #This is the number of data points to use
#Sample the input space according to the distributions in the table above
key = random.PRNGKey(6753)
keys = random.split(key, 10)
Sw = random.uniform(keys[0], shape=(M,1), minval=150, maxval=200)
Wfw = random.uniform(keys[1], shape=(M,1), minval=220, maxval=300)
A = random.uniform(keys[2], shape=(M,1), minval=6, maxval=10)
L = random.uniform(keys[3], shape=(M,1), minval=-10, maxval=10)
q = random.uniform(keys[4], shape=(M,1), minval=16, maxval=45)
l = random.uniform(keys[5], shape=(M,1), minval=.5, maxval=1)
tc = random.uniform(keys[6], shape=(M,1), minval=.08, maxval=.18)
Nz = random.uniform(keys[7], shape=(M,1), minval=2.5, maxval=6)
Wdg = random.uniform(keys[8], shape=(M,1), minval=1700, maxval=2500)
Wp = random.uniform(keys[9], shape=(M,1), minval=.025, maxval=.08)

# %%
#Upper and lower limits for inputs
lb = np.array([150, 220, 6, -10, 16, .5, .08, 2.5, 1700, .025])
ub = np.array([200, 300, 10, 10, 45, 1, .18, 6, 2500, .08])

# %%
tau = lambda x: 0.5*((ub-lb) * x + ub + lb) # goes from [-1,1] -> [l,u]
tau_inv = lambda x: 2.0*(x - lb)/(ub - lb) - 1.0 # goes from [l, u] -> [-1,1]

# %%
x = np.hstack((Sw, Wfw, A, L, q, l, tc, Nz, Wdg, Wp))

x_norm = tau_inv(x)
Sw_norm, Wfw_norm, A_norm, L_norm, q_norm, l_norm, tc_norm, Nz_norm, Wdg_norm, Wp_norm = x_norm.T

# %%
assert (Sw_norm == x_norm[:,0]).all()
assert (Wfw_norm == x_norm[:,1]).all()
assert (A_norm == x_norm[:,2]).all()
assert (L_norm == x_norm[:,3]).all()
assert (q_norm == x_norm[:,4]).all()
assert (l_norm == x_norm[:,5]).all()
assert (tc_norm == x_norm[:,6]).all()
assert (Nz_norm == x_norm[:,7]).all()
assert (Wdg_norm == x_norm[:,8]).all()
assert (Wp_norm == x_norm[:,9]).all()


# %%
def get_AS(C):
    eigs, eig_vecs = lin.eigh(C)
    inds = eigs.argsort()[::-1]
    max_eig = eigs[inds[0]]
    prop = max_eig / eigs.sum()
    w = eig_vecs[:,inds[0]]
    return w, max_eig, prop


# %%
inds = {i: [i] + [x for x in range(10) if x != i] for i in range(10)}
for i, locs in inds.items():
    print(i, locs)

# %%
inds_inv = {i: [x for x in range(1,i+1)] + [0] + [y for y in range(i+1,10)] for i in range(10)}
for i, locs in inds_inv.items():
    print(i, locs)

# %%
for i in range(10):
    sigma = inds[i]
    sigma_inv = inds_inv[i]
    assert (x_norm[:,sigma][:,sigma_inv] == x_norm).all()
    assert (x_norm[:,sigma_inv][:,sigma] == x_norm).all()

# %%
var_dict = {0: Sw_norm, 1: Wfw_norm, 2: A_norm, 3: L_norm, 4: q_norm, 5: l_norm, 6: tc_norm, 7: Nz_norm, 8: Wdg_norm, 9: Wp_norm}
for i in range(10):
    my_inds = inds[i]
    x_norm_shuffled = x_norm[:,my_inds]
    for j in range(10):
        assert (x_norm_shuffled[:,j] == var_dict[my_inds[j]]).all()


# %%
def f(Sw_norm, Wfw_norm, A_norm, L_norm, q_norm, l_norm, tc_norm, Nz_norm, Wdg_norm, Wp_norm):
    x_norm = np.hstack((Sw_norm, Wfw_norm, A_norm, L_norm, q_norm, l_norm, tc_norm, Nz_norm, Wdg_norm, Wp_norm))
    assert x_norm.shape == (10,)
    x = tau(x_norm)
    Sw, Wfw, A, L, q, l, tc, Nz, Wdg, Wp = x
    return wing_func(Sw, Wfw, A, L, q, l, tc, Nz, Wdg, Wp)


# %%
for i in range(10):
    sigma = inds[i]
    sigma_inv = inds_inv[i]
    x_norm_shuffled = x_norm[:,sigma]
    def f_i(x_norm_shuffled):
        # unshuffle
        x_norm_unshuffled = x_norm_shuffled[np.array(sigma_inv)]
        x = tau(x_norm_unshuffled)
        Sw, Wfw, A, L, q, l, tc, Nz, Wdg, Wp = x
        return wing_func(Sw, Wfw, A, L, q, l, tc, Nz, Wdg, Wp)

    assert (vmap(f_i)(x_norm_shuffled) == vmap(f)(Sw_norm, Wfw_norm, A_norm, L_norm, q_norm, l_norm, tc_norm, Nz_norm, Wdg_norm, Wp_norm)).all()


# %%
def generate_dataset(i):
    sigma = inds[i]
    sigma_inv = inds_inv[i]
    x_norm_shuffled = x_norm[:,sigma]
    def f_i(x_norm_shuffled):
        # unshuffle
        x_norm_unshuffled = x_norm_shuffled[np.array(sigma_inv)]
        x = tau(x_norm_unshuffled)
        Sw, Wfw, A, L, q, l, tc, Nz, Wdg, Wp = x
        return wing_func(Sw, Wfw, A, L, q, l, tc, Nz, Wdg, Wp)
    
    W_res = []
    max_eig_res = []
    prop_res = []
    for s in tqdm(var_dict[i]):
        in_vars = x_norm_shuffled.copy()
        in_vars = in_vars.at[:,0].set(s)
        assert (in_vars[:,0] == s).all()
        grad_f_i = grad(f_i)
        grads = vmap(grad_f_i)(in_vars)[:,1:]
        C_i = vmap(lambda gradient: np.outer(gradient, gradient))(grads)
        C_hat_i = C_i.mean(axis=0)
        assert C_hat_i.shape == (9,9)
        W, max_eig, prop = get_AS(C_hat_i)
        W_res.append(W)
        max_eig_res.append(max_eig)
        prop_res.append(prop)
    
    return W_res, max_eig_res, prop_res


# %%
# datasets = {}
# for i in range(10):
#     s = var_dict[i]
#     W_res, max_eig_res, prop_res = generate_dataset(i)
#     Ws = np.vstack([v.T for v in W_res])[:,:,None]
#     datasets[i] = {'s': s, 'Ws': Ws, 'max_eig_res': max_eig_res, 'prop_res': prop_res}

# %%
# with open('parametrised_AS_datasets.pickle', 'wb') as f:
#     pickle.dump(datasets, f)

# %%
with open('parametrised_AS_datasets.pickle', 'rb') as f:
    datasets = pickle.load(f)

# %%
var_names_dict = {0: 'Sw_norm', 1: 'Wfw_norm', 2: 'A_norm', 3: 'L_norm', 4: 'q_norm', 5: 'l_norm', 6: 'tc_norm', 7: 'Nz_norm', 8: 'Wdg_norm', 9: 'Wp_norm'}

for i in range(10):
    s = datasets[i]['s']
    Ws = datasets[i]['Ws']
    sorted_inds = s.argsort()
    s_sorted = s[sorted_inds]
    Ws_sorted = Ws[sorted_inds,:,:]
    W0 = Ws_sorted[0]
    dists_from_start = vmap(lambda w: grass_dist(w,W0))(Ws_sorted)
    plt.plot(s_sorted, dists_from_start, label=f"{i = } : {var_names_dict[i]}")

plt.legend()
plt.title("Grass distance from starting AS")
plt.show()

# %% tags=[] jupyter={"outputs_hidden": true}
for i in range(10):
    s = datasets[i]['s']
    Ws = datasets[i]['Ws']
    sorted_inds = s.argsort()
    plt.plot(s[sorted_inds], Ws[sorted_inds,:,0])
    plt.title(f"{i = }")
    plt.show()

# %% tags=[] jupyter={"outputs_hidden": true}
i = 2
s = datasets[i]['s']
Ws = datasets[i]['Ws']
sorted_inds = s.argsort()
for j in range(9):
    plt.plot(s[sorted_inds], Ws[sorted_inds,j,0])
    plt.show()

# %%
for i in range(10):
    props = datasets[i]['prop_res']
    props = np.array(props)
    plt.plot(props, label=f'{i = }')
plt.legend()
plt.title("Proportion of spectrum explained")
# plt.ylim((0.9,1.0))
plt.show()


# %%
# var_names_dict = {0: 'Sw_norm', 1: 'Wfw_norm', 2: 'A_norm', 3: 'L_norm', 4: 'q_norm', 5: 'l_norm', 6: 'tc_norm', 7: 'Nz_norm', 8: 'Wdg_norm', 9: 'Wp_norm'}

# for i in range(10):
#     s = loaded_datasets[i]['s']
#     Ws = loaded_datasets[i]['Ws']
#     sorted_inds = s.argsort()
#     s_sorted = s[sorted_inds]
#     Ws_sorted = Ws[sorted_inds,:,:]
#     W0 = Ws_sorted[0]
#     dists_from_start = vmap(lambda w: grass_dist(w,W0))(Ws_sorted)
#     plt.plot(s_sorted, dists_from_start, label=f"{i = } : {var_names_dict[i]}")

# plt.legend()
# plt.title("Grass distance from starting AS")
# plt.show()

# %%
def plot_dist_from_start(i):
    s = datasets[i]['s']
    Ws = datasets[i]['Ws']
    sorted_inds = s.argsort()
    s_sorted = s[sorted_inds]
    Ws_sorted = Ws[sorted_inds,:,:]
    W0 = Ws_sorted[0]
    dists_from_start = vmap(lambda w: grass_dist(w,W0))(Ws_sorted)
    plt.plot(s_sorted, dists_from_start)
    plt.title(f"Grass distance from starting AS for variable {var_names_dict[i]}")
    plt.show()


# %%
plot_dist_from_start(2)

# %% tags=[] jupyter={"outputs_hidden": true}
i = 2
s = datasets[i]['s']
Ws = datasets[i]['Ws']
sorted_inds = s.argsort()
for j in range(9):
    plt.plot(s[sorted_inds], Ws[sorted_inds,j,0])
    plt.show()

# %%
??generate_dataset

# %%
i = 2
s = datasets[i]['s']
Ws = datasets[i]['Ws']

sorted_inds = s.argsort()
s_sorted = s[sorted_inds]
Ws_sorted = Ws[sorted_inds,:,:]

sigma = inds[i]
sigma_inv = inds_inv[i]
x_norm_shuffled = x_norm[:,sigma]
def f_i(x_norm_shuffled):
    # unshuffle
    x_norm_unshuffled = x_norm_shuffled[np.array(sigma_inv)]
    x = tau(x_norm_unshuffled)
    Sw, Wfw, A, L, q, l, tc, Nz, Wdg, Wp = x
    return wing_func(Sw, Wfw, A, L, q, l, tc, Nz, Wdg, Wp)

for j in range(M):
    if j % 200 != 0:
        continue
    s_fixed = s_sorted[j]
    W = Ws_sorted[j,:,:]
    in_vars = x_norm_shuffled.copy()
    in_vars = in_vars.at[:,0].set(s_fixed)
    assert (in_vars[:,0] == s_fixed).all()
    y = vmap(f_i)(in_vars)
    uni_in = in_vars[:,1:] @ W
    plt.scatter(uni_in.flatten(), y, alpha=0.5, label=f"A_norm = {s_fixed: .2f}")

s_fixed = s_sorted[j]
W = Ws_sorted[j,:,:]
in_vars = x_norm_shuffled.copy()
in_vars = in_vars.at[:,0].set(s_fixed)
assert (in_vars[:,0] == s_fixed).all()
y = vmap(f_i)(in_vars)
uni_in = in_vars[:,1:] @ W
plt.scatter(uni_in.flatten(), y, alpha=0.5, label=f"A_norm = {s_fixed: .2f}")
plt.legend()
plt.show()

# %%
