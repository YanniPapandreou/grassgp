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
# import os
# from pathlib import Path

from hydra_zen import launch, to_yaml, load_from_yaml
from pathlib import Path
from train import Config, train
import jax.numpy as np

# import numpyro.distributions as dist
# from grassgp.plot_utils import plot_densities

# import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (10,6)

# %%
# bs = [1.0, 1.5, 2.0, 5.0]
# d_dict = {f"LogNormal(0.0,{b:0.1f})":dist.LogNormal(0.0, b) for b in bs}
# fig, ax = plot_densities(d_dict, N=1000)
# plt.show()

# %%
def print_file(x: Path):
    with x.open("r") as f:
        print(f.read())


# %%
dataset_path = '/home/yanni/projects/grassgp/scripts/outputs/2022-11-02/16-27-53/'

# %%
# print(to_yaml(Config))

# %%
# dataset_config_path = Path(dataset_path) / ".hydra" / "config.yaml"
# assert dataset_config_path.exists()
# dataset_config = load_from_yaml(dataset_config_path)
# print("Config used to generate dataset:")
# print(to_yaml(dataset_config))
# dataset_overrides_path = Path(dataset_path) / ".hydra" / "overrides.yaml"
# assert dataset_overrides_path.exists()
# print("Overrides used to generate dataset:")
# print(to_yaml(load_from_yaml(dataset_overrides_path)))

# %%
(jobs,) = launch(
    Config,
    train,
    overrides=[
        f"train.n_subsample_gap=1",
        f"dataset_path={dataset_path}",
        "subsample_conf.random=True",
        "subsample_conf.n_x_samples=60",
        # "inner_model.grass_config.reorthonormalize=True,False",
        "inner_model.grass_config.reorthonormalize=False",
        "inner_model.grass_config.var=0.1",
        # "inner_model.grass_config.var=null",
        "inner_model.grass_config.b=0.25,0.5,1.0,5.0",
        "inner_model.grass_config.length=null",
        # "inner_model.grass_config.Omega=null",
        f"inner_model.grass_config.proj_locs=null",
        "outer_model.gp_config.params.var=1.0",
        "outer_model.gp_config.params.length=0.5",
        "outer_model.gp_config.params.noise=0.1"
    ],
    multirun=True,
    version_base="1.1"
)

# %%
