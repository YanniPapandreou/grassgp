# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: grassgp
#     language: python
#     name: grassgp
# ---

# %%
# # %load_ext autoreload
# # %autoreload 2

# %%
from hydra_zen import to_yaml, load_from_yaml
from pathlib import Path
import os
import jax.numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,8)

from grassgp.plot_utils import plot_projected_data
from grassgp.utils import get_config_and_data


# %%
def print_file(x: Path):
    with x.open("r") as f:
        print(f.read())


# %%
path = Path(os.getcwd()) / "multirun" / "2022-10-21" / "15-35-19" 

# %%
paths = [path / f"{i}" for i in range(8)]
paths.append(Path(os.getcwd()) / "outputs" / "2022-10-21" / "15-49-09")
paths

# %%
configs_and_datasets = {path.name:get_config_and_data(path) for path in paths}

# %%
for key, config_and_dataset in configs_and_datasets.items():
    config = config_and_dataset['config']
    overrides = config_and_dataset['overrides']
    print(f"key = {key}\n")
    for override in overrides:
        print(override)
    print("\n")

# %% jupyter={"outputs_hidden": true} tags=[]
for key, config_and_dataset in configs_and_datasets.items():
    config = config_and_dataset['config']
    overrides = config_and_dataset['overrides']
    data = config_and_dataset['data']
    data = {k:np.array(array) for k, array in data.items()}
    X = data['X']
    D = X.shape[1]
    s = data['s']
    Ps = data['Ps']
    Ys = data['Ys']
    X_projs = np.einsum('ij,ljk->lik', X, Ps)
    # print(f"Plot of projected data for key = {key}")
    # plot_projected_data(X_projs=X_projs, s=s, Ys=Ys)
    for i in range(D):
        plt.plot(s, Ps[:,i,0])
        plt.title(f"{key}:component {i}")
        plt.grid()
        plt.show()

# %%
